// Adapted from SemiAnalysisAI/InferenceX
// video-generation/evaluator/media.py@ea9e4f6f2e3073d55fe6f1dcd33557b7fe67f608
// (Apache-2.0; modified from the original): the acceptance doctrine of
// analyze_media, applied to perf receipts instead of video clips. A result
// counts only after its artifact is read back against the contract its
// producer declared; the file's size and mtime are compared before and after
// the read as evidence it did not change underneath; every check is recorded
// as passed, failed or not applicable; and the verdict names what was never
// established. The media decoding itself is not carried over.
package perfledger

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

// Artifact verdict statuses.
const (
	// ArtifactAccepted: nothing failed, and the size and at least one content
	// check were declared and held.
	ArtifactAccepted = "accepted"
	// ArtifactRefused: a declared check failed, the file changed during the
	// read, or it could not be read at all.
	ArtifactRefused = "refused"
	// ArtifactUnjudged: nothing failed, but too little was declared to say
	// the artifact is what it claims. It is not an acceptance.
	ArtifactUnjudged = "unjudged"
)

// Artifact check statuses.
const (
	ArtifactCheckPassed        = "passed"
	ArtifactCheckFailed        = "failed"
	ArtifactCheckNotApplicable = "not_applicable"
)

// Names used in ArtifactVerdict.Unverified.
const (
	UnverifiedDeclaredSize   = "declared_size"
	UnverifiedDeclaredSHA256 = "declared_sha256"
	UnverifiedMagicBytes     = "magic_bytes"
	UnverifiedContentType    = "content_type"
	UnverifiedDeclaredFields = "declared_fields"
	// UnverifiedMeasurement is always listed: the gate establishes that the
	// artifact is the declared one, never that its numbers are true.
	UnverifiedMeasurement = "measurement_correctness"
)

// ArtifactContentJSON is the one content type the gate can read back.
const ArtifactContentJSON = "application/json"

// MaxArtifactBytes bounds how much of an artifact the gate will read.
const MaxArtifactBytes = 64 << 20

// ArtifactContract is what a producer declares about its artifact. A field
// left at its zero value is not declared, is not checked, and is named in the
// verdict's Unverified list.
type ArtifactContract struct {
	Path string
	// SizeBytes is the declared size; nil when not declared.
	SizeBytes *int64
	// SHA256 is the declared lowercase or uppercase hex digest.
	SHA256 string
	// Magic is the declared leading bytes.
	Magic []byte
	// ContentType is the declared media type.
	ContentType string
	// RequiredFields are top-level JSON fields that must be present and
	// well-formed: not null, not an empty string, and a finite number where
	// numeric.
	RequiredFields []string
}

// ArtifactCheck is one recorded comparison.
type ArtifactCheck struct {
	Name     string `json:"name"`
	Status   string `json:"status"`
	Observed string `json:"observed"`
	Expected string `json:"expected"`
	Detail   string `json:"detail"`
}

// ArtifactVerdict is the outcome of reading an artifact back against its
// contract.
type ArtifactVerdict struct {
	Path     string          `json:"path"`
	Status   string          `json:"status"`
	ByteSize int64           `json:"byte_size"`
	SHA256   string          `json:"sha256,omitempty"`
	Checks   []ArtifactCheck `json:"checks"`
	// Unverified names, sorted, exactly what the gate did not establish.
	Unverified []string `json:"unverified"`
	Errors     []string `json:"errors,omitempty"`

	fields map[string]json.RawMessage
}

// Accepted reports whether a receipt built on this artifact may count.
func (v ArtifactVerdict) Accepted() bool { return v.Status == ArtifactAccepted }

// Metric returns a declared field's numeric value. It reports false for a
// field that was not declared, is absent, is not a number, or belongs to an
// artifact that was not accepted: an absent metric is never read as zero.
func (v ArtifactVerdict) Metric(field string) (float64, bool) {
	if !v.Accepted() {
		return 0, false
	}
	raw, ok := v.fields[field]
	if !ok || !isJSONNumber(raw) {
		return 0, false
	}
	var n json.Number
	if err := json.Unmarshal(raw, &n); err != nil {
		return 0, false
	}
	f, err := n.Float64()
	if err != nil {
		return 0, false
	}
	return f, true
}

// GateArtifact reads the artifact back against its contract.
func GateArtifact(contract ArtifactContract) ArtifactVerdict {
	return gateArtifact(contract, nil)
}

// gateArtifact is GateArtifact with a hook run between the read and the
// second stat, so a test can change the file inside that window.
func gateArtifact(contract ArtifactContract, afterRead func()) ArtifactVerdict {
	v := ArtifactVerdict{Path: contract.Path, Status: ArtifactRefused, Checks: []ArtifactCheck{}}
	v.Unverified = undeclared(contract)

	data, sum, size, err := readArtifact(contract.Path, afterRead)
	if err != nil {
		v.Errors = append(v.Errors, err.Error())
		v.Checks = append(v.Checks, ArtifactCheck{
			Name:     "artifact.preflight",
			Status:   ArtifactCheckFailed,
			Observed: err.Error(),
			Expected: "an unchanged regular file within the size bound",
			Detail:   "No measurement was accepted.",
		})
		return v
	}
	v.ByteSize = size
	v.SHA256 = sum

	if contract.SizeBytes != nil {
		v.check("artifact.size_bytes", size == *contract.SizeBytes,
			fmt.Sprint(size), fmt.Sprint(*contract.SizeBytes), "Size on disk against the declared size.")
	}
	if contract.SHA256 != "" {
		v.check("artifact.sha256", strings.EqualFold(sum, contract.SHA256),
			sum, strings.ToLower(contract.SHA256), "Digest of the bytes read against the declared digest.")
	}
	if len(contract.Magic) > 0 {
		head := data[:min(len(data), len(contract.Magic))]
		v.check("artifact.magic_bytes", bytes.Equal(head, contract.Magic),
			hex.EncodeToString(head), hex.EncodeToString(contract.Magic), "Leading bytes against the declared signature.")
	}

	var object map[string]json.RawMessage
	parsed := false
	if contract.ContentType == ArtifactContentJSON {
		parsed = json.Unmarshal(data, &object) == nil && object != nil
		observed := "a JSON object"
		if !parsed {
			observed = "not a JSON object"
		}
		v.check("artifact.content_type", parsed, observed, ArtifactContentJSON, "The whole artifact must parse as one JSON object.")
	}
	for _, field := range contract.RequiredFields {
		name := "field." + field
		switch {
		case contract.ContentType != ArtifactContentJSON:
			v.Checks = append(v.Checks, ArtifactCheck{
				Name: name, Status: ArtifactCheckNotApplicable, Expected: "present and well-formed",
				Detail: "Fields can only be read back from a declared " + ArtifactContentJSON + " artifact.",
			})
		case !parsed:
			v.check(name, false, "unreadable", "present and well-formed", "The artifact did not parse.")
		default:
			problem := fieldProblem(object, field)
			observed := "present and well-formed"
			if problem != "" {
				observed = problem
			}
			v.check(name, problem == "", observed, "present and well-formed", "Declared field read back from the artifact.")
			if problem == "" {
				if v.fields == nil {
					v.fields = map[string]json.RawMessage{}
				}
				v.fields[field] = object[field]
			}
		}
	}

	v.Status = v.judge(contract)
	return v
}

func (v *ArtifactVerdict) check(name string, passed bool, observed, expected, detail string) {
	status := ArtifactCheckFailed
	if passed {
		status = ArtifactCheckPassed
	}
	v.Checks = append(v.Checks, ArtifactCheck{Name: name, Status: status, Observed: observed, Expected: expected, Detail: detail})
}

func (v *ArtifactVerdict) judge(contract ArtifactContract) string {
	for _, c := range v.Checks {
		if c.Status == ArtifactCheckFailed {
			return ArtifactRefused
		}
	}
	contentChecked := contract.SHA256 != "" || len(contract.Magic) > 0 || contract.ContentType == ArtifactContentJSON
	if contract.SizeBytes == nil || !contentChecked {
		return ArtifactUnjudged
	}
	return ArtifactAccepted
}

// undeclared lists what the contract leaves unchecked.
func undeclared(contract ArtifactContract) []string {
	out := []string{UnverifiedMeasurement}
	if contract.SizeBytes == nil {
		out = append(out, UnverifiedDeclaredSize)
	}
	if contract.SHA256 == "" {
		out = append(out, UnverifiedDeclaredSHA256)
	}
	if len(contract.Magic) == 0 {
		out = append(out, UnverifiedMagicBytes)
	}
	if contract.ContentType != ArtifactContentJSON {
		// A declared type the gate cannot read back is as unverified as none.
		out = append(out, UnverifiedContentType)
	}
	if len(contract.RequiredFields) == 0 || contract.ContentType != ArtifactContentJSON {
		out = append(out, UnverifiedDeclaredFields)
	}
	sort.Strings(out)
	return out
}

// readArtifact returns the file's bytes, digest and size, refusing anything
// that is not a regular file, exceeds the bound, or whose size or mtime moved
// between the stat before the read and the stat after it.
func readArtifact(path string, afterRead func()) (data []byte, sum string, size int64, err error) {
	before, err := os.Stat(path)
	if err != nil {
		return nil, "", 0, fmt.Errorf("artifact is not readable: %w", err)
	}
	if !before.Mode().IsRegular() {
		return nil, "", 0, fmt.Errorf("artifact path is not a regular file")
	}
	if before.Size() > MaxArtifactBytes {
		return nil, "", 0, fmt.Errorf("artifact is %d bytes, over the %d byte read bound", before.Size(), MaxArtifactBytes)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, "", 0, fmt.Errorf("artifact is not readable: %w", err)
	}
	opened, statErr := f.Stat()
	data, err = io.ReadAll(io.LimitReader(f, MaxArtifactBytes+1))
	f.Close()
	if err != nil || statErr != nil {
		return nil, "", 0, fmt.Errorf("artifact read failed")
	}
	// The path could have been re-pointed between the stat and the open.
	if !os.SameFile(before, opened) {
		return nil, "", 0, fmt.Errorf("artifact changed while it was being analyzed")
	}
	if afterRead != nil {
		afterRead()
	}
	after, err := os.Stat(path)
	if err != nil {
		return nil, "", 0, fmt.Errorf("artifact vanished while it was being analyzed: %w", err)
	}
	if before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) || int64(len(data)) != before.Size() {
		return nil, "", 0, fmt.Errorf("artifact changed while it was being analyzed")
	}
	digest := sha256.Sum256(data)
	return data, hex.EncodeToString(digest[:]), before.Size(), nil
}

// fieldProblem says why a declared field is not usable, or "" when it is.
func fieldProblem(object map[string]json.RawMessage, field string) string {
	raw, ok := object[field]
	if !ok {
		return "absent"
	}
	trimmed := bytes.TrimSpace(raw)
	switch {
	case bytes.Equal(trimmed, []byte("null")):
		return "null"
	case bytes.Equal(trimmed, []byte(`""`)):
		return "empty string"
	}
	if isJSONNumber(trimmed) {
		if _, err := json.Number(trimmed).Float64(); err != nil {
			return "number out of range"
		}
	}
	return ""
}

// isJSONNumber reports whether a raw JSON value is a number literal. A quoted
// "41.5" is a string: decoding it as a number would turn text into a metric.
func isJSONNumber(raw []byte) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) > 0 && (raw[0] == '-' || (raw[0] >= '0' && raw[0] <= '9'))
}
