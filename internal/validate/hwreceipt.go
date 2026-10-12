package validate

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// Receipt admission is the sanctioned route for landing hardware-gated paths
// from a controller that cannot itself attest (a native Windows binary refuses
// UNSUPPORTED_VALIDATOR_BINARY). A stamped Linux/WSL controller runs
// `fak-dev amd-strix-validate --mine <the same paths>` against the same HEAD; its
// receipt binds the git tip and the sha256 of the exact candidate archive. The
// gate recomputes that archive itself from HEAD plus the owned working-tree
// bytes, so the receipt can only satisfy the gate for the byte-identical
// prospective tree. Admission is evidence, never promotion credit:
// CreditEligible stays reserved for in-process verifier authority.
//
// The file name deliberately avoids the GPU keywords isGPURelatedValidation
// matches, so evolving this gate never demands the hardware it gates.

const (
	hwReceiptEnv       = "FAK_VALIDATE_STRIX_RECEIPT"
	hwReceiptMaxAgeEnv = "FAK_VALIDATE_STRIX_RECEIPT_MAX_AGE"
	// hwReceiptDefaultMaxAge bounds how old a receipt may be. Binding to the tip
	// and archive digest is what makes it exact; the age bound keeps a long-lived
	// file from silently standing in for today's driver/firmware state.
	hwReceiptDefaultMaxAge = 24 * time.Hour
	hwReceiptFutureSkew    = 5 * time.Minute
	hwReceiptMaxBytes      = 8 << 20
	hwReceiptPhase         = "strix_receipt_admission"
	hwReceiptProducer      = "fak-dev amd-strix-validate"
	// hwReceiptVerifierRoot is the package that defines receipt semantics. When
	// the candidate changes it, HEAD's verifier is not the authority on the
	// candidate's receipt schema (the producing controller already ran the
	// candidate's own Validate before exiting 0), so admission falls back to the
	// schema-independent binding checks instead of HEAD's strict decode.
	hwReceiptVerifierRoot = "internal/amdgpu"
)

// Stable refusal codes for receipt admission.
const (
	hwReceiptUnreadable      = "STRIX_RECEIPT_UNREADABLE"
	hwReceiptSchema          = "STRIX_RECEIPT_SCHEMA"
	hwReceiptNotPass         = "STRIX_RECEIPT_NOT_PASS"
	hwReceiptProducerRefused = "STRIX_RECEIPT_PRODUCER"
	hwReceiptExpired         = "STRIX_RECEIPT_EXPIRED"
	hwReceiptStaleTip        = "STRIX_RECEIPT_STALE_TIP"
	hwReceiptArchiveMismatch = "STRIX_RECEIPT_ARCHIVE_MISMATCH"
	hwReceiptInvalid         = "STRIX_RECEIPT_INVALID"
	hwReceiptCandidate       = "STRIX_RECEIPT_CANDIDATE_UNAVAILABLE"
)

type hwReceiptAdmission struct {
	Path                string `json:"path"`
	FileSHA256          string `json:"file_sha256"`
	Schema              string `json:"schema"`
	Digest              string `json:"digest,omitempty"`
	GitTip              string `json:"git_tip"`
	SourceArchiveSHA256 string `json:"source_archive_sha256"`
	Timestamp           string `json:"timestamp"`
	Host                string `json:"host,omitempty"`
	// AdvancedTo is the HEAD the receipt was carried forward to when trunk moved
	// through only hardware-irrelevant commits after the controller ran.
	AdvancedTo string `json:"advanced_to,omitempty"`
	// Verifier names whose rules judged the receipt body: "head" (strict decode
	// plus HEAD's Validate) or "candidate" (the candidate evolves the verifier;
	// the producing controller ran it, the gate checks binding only).
	Verifier string `json:"verifier"`
	Admitted bool   `json:"admitted"`
	Code     string `json:"code,omitempty"`
	Detail   string `json:"detail,omitempty"`
}

type hwReceiptRefusal struct {
	code   string
	detail string
}

func (e *hwReceiptRefusal) Error() string { return e.code + ": " + e.detail }

func refuseHWReceipt(code, format string, args ...any) error {
	return &hwReceiptRefusal{code: code, detail: fmt.Sprintf(format, args...)}
}

// hwReceiptPath resolves the receipt from the flag, falling back to the
// environment so `fak commit` (which drives validate with fixed arguments) can
// carry it.
func hwReceiptPath(flagValue string) string {
	if p := strings.TrimSpace(flagValue); p != "" {
		return p
	}
	return strings.TrimSpace(os.Getenv(hwReceiptEnv))
}

func hwReceiptMaxAge() time.Duration {
	if raw := strings.TrimSpace(os.Getenv(hwReceiptMaxAgeEnv)); raw != "" {
		if d, err := time.ParseDuration(raw); err == nil && d > 0 {
			return d
		}
	}
	return hwReceiptDefaultMaxAge
}

// hwReceiptCommon is the schema-independent subset both receipt kinds share.
type hwReceiptCommon struct {
	Schema       string `json:"schema"`
	Timestamp    string `json:"timestamp"`
	Verdict      string `json:"verdict"`
	Verified     bool   `json:"verified"`
	Hardware     *bool  `json:"hardware"`
	EvidenceRung string `json:"evidence_rung"`
	Digest       string `json:"digest"`
	Target       struct {
		Host      string `json:"host"`
		Reachable bool   `json:"reachable"`
	} `json:"target"`
	Provenance struct {
		GitTip              string `json:"git_tip"`
		SourceArchiveSHA256 string `json:"source_archive_sha256"`
		GeneratedBy         string `json:"generated_by"`
		Command             string `json:"command"`
		CleanupObserved     bool   `json:"cleanup_observed"`
	} `json:"provenance"`
	Failures json.RawMessage `json:"failures"`
}

func hwReceiptCandidateEvolvesVerifier(mine []string) bool {
	for _, p := range mine {
		norm := strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
		if norm == hwReceiptVerifierRoot || strings.HasPrefix(norm, hwReceiptVerifierRoot+"/") {
			return true
		}
	}
	return false
}

// checkHWReceipt validates the receipt body and its binding to (tip,
// archiveSHA256). It is pure: callers supply the bytes, digest, and clock.
func checkHWReceipt(raw []byte, tip, archiveSHA256 string, mine []string, now time.Time, maxAge time.Duration) (hwReceiptAdmission, error) {
	adm := hwReceiptAdmission{}
	raw = bytes.TrimPrefix(raw, []byte("\xef\xbb\xbf"))
	var c hwReceiptCommon
	if err := json.Unmarshal(raw, &c); err != nil {
		return adm, refuseHWReceipt(hwReceiptUnreadable, "decode receipt: %v", err)
	}
	adm.Schema, adm.Digest, adm.GitTip = c.Schema, c.Digest, c.Provenance.GitTip
	adm.SourceArchiveSHA256, adm.Timestamp, adm.Host = c.Provenance.SourceArchiveSHA256, c.Timestamp, c.Target.Host

	switch c.Schema {
	case amdgpu.StrixSSDRestartSchema:
		if c.Hardware == nil || !*c.Hardware || c.EvidenceRung != "HW_WITNESSED" {
			return adm, refuseHWReceipt(hwReceiptNotPass, "native SSD restart receipt is not HW_WITNESSED hardware evidence")
		}
	case amdgpu.StrixValidationSchemaV2:
	default:
		return adm, refuseHWReceipt(hwReceiptSchema, "receipt schema %q is not %q or %q", c.Schema, amdgpu.StrixSSDRestartSchema, amdgpu.StrixValidationSchemaV2)
	}
	if c.Verdict != "PASS" || !c.Verified || !c.Target.Reachable || !c.Provenance.CleanupObserved {
		return adm, refuseHWReceipt(hwReceiptNotPass, "receipt verdict=%s verified=%t reachable=%t cleanup_observed=%t", c.Verdict, c.Verified, c.Target.Reachable, c.Provenance.CleanupObserved)
	}
	if trimmed := bytes.TrimSpace(c.Failures); len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null")) && !bytes.Equal(trimmed, []byte("[]")) {
		return adm, refuseHWReceipt(hwReceiptNotPass, "passing receipt records failures")
	}
	if !strings.HasPrefix(c.Provenance.GeneratedBy, hwReceiptProducer) || !strings.Contains(c.Provenance.Command, "amd-strix-validate") {
		return adm, refuseHWReceipt(hwReceiptProducerRefused, "receipt was generated by %q (command %q), not a stamped %s controller", c.Provenance.GeneratedBy, c.Provenance.Command, hwReceiptProducer)
	}

	stamp, err := time.Parse(time.RFC3339, c.Timestamp)
	if err != nil {
		return adm, refuseHWReceipt(hwReceiptExpired, "receipt timestamp %q is not RFC3339", c.Timestamp)
	}
	if stamp.After(now.Add(hwReceiptFutureSkew)) {
		return adm, refuseHWReceipt(hwReceiptExpired, "receipt timestamp %s is in the future", stamp.UTC().Format(time.RFC3339))
	}
	if age := now.Sub(stamp); age > maxAge {
		return adm, refuseHWReceipt(hwReceiptExpired, "receipt is %s old, beyond the %s bound (%s)", age.Round(time.Second), maxAge, hwReceiptMaxAgeEnv)
	}

	if !strings.EqualFold(strings.TrimSpace(c.Provenance.GitTip), strings.TrimSpace(tip)) || strings.TrimSpace(tip) == "" {
		return adm, refuseHWReceipt(hwReceiptStaleTip, "receipt git_tip %q does not match prospective base %q; rerun the controller at the current HEAD", c.Provenance.GitTip, tip)
	}
	if c.Provenance.SourceArchiveSHA256 == "" || c.Provenance.SourceArchiveSHA256 != archiveSHA256 {
		return adm, refuseHWReceipt(hwReceiptArchiveMismatch, "receipt source_archive_sha256 %q does not match the prospective tree %q; the controller validated different bytes or a different --mine set", c.Provenance.SourceArchiveSHA256, archiveSHA256)
	}

	if hwReceiptCandidateEvolvesVerifier(mine) {
		adm.Verifier = "candidate"
	} else {
		adm.Verifier = "head"
		if err := strictHWReceiptValidate(raw, c.Schema); err != nil {
			return adm, refuseHWReceipt(hwReceiptInvalid, "%v", err)
		}
	}
	adm.Admitted = true
	return adm, nil
}

// strictHWReceiptValidate holds a receipt whose schema HEAD fully owns to HEAD's
// complete verifier: unknown fields are refused and Validate must pass.
func strictHWReceiptValidate(raw []byte, schema string) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	switch schema {
	case amdgpu.StrixSSDRestartSchema:
		var r amdgpu.StrixSSDRestartReceipt
		if err := dec.Decode(&r); err != nil {
			return fmt.Errorf("strict decode: %w", err)
		}
		return r.Validate()
	default:
		var r amdgpu.StrixValidationReceipt
		if err := dec.Decode(&r); err != nil {
			return fmt.Errorf("strict decode: %w", err)
		}
		if err := r.Validate(); err != nil {
			return err
		}
		want, err := r.ComputeDigest()
		if err != nil || want != r.Digest {
			return errors.New("receipt digest mismatch")
		}
		return nil
	}
}

// admitHWReceipt runs the receipt-admission phase. It returns true only when
// the receipt satisfies hardware validation for exactly this prospective tree.
// A supplied but refused receipt fails the gate rather than falling back to
// in-process hardware execution, so a stale receipt is never silently ignored.
func admitHWReceipt(ctx context.Context, res *validateResult, recorder *validateRecorder, root, path string, mine []string) bool {
	phase := recorder.start(hwReceiptPhase)
	adm, err := loadAndCheckHWReceipt(ctx, root, res.Tip, path, mine)
	adm.Path = path
	if err != nil {
		var refusal *hwReceiptRefusal
		if errors.As(err, &refusal) {
			adm.Code, adm.Detail = refusal.code, refusal.detail
		} else {
			adm.Code, adm.Detail = hwReceiptUnreadable, err.Error()
		}
	}
	res.StrixReceipt = &adm
	if err != nil {
		phase.finish(err)
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-receipt-admission",
			Detail: fmt.Sprintf("hardware receipt refused: %s: %s", adm.Code, adm.Detail),
			Files:  []string{path},
		})
		return false
	}
	phase.finish(nil)
	return true
}

func loadAndCheckHWReceipt(ctx context.Context, root, tip, path string, mine []string) (hwReceiptAdmission, error) {
	f, err := os.Open(path)
	if err != nil {
		return hwReceiptAdmission{}, refuseHWReceipt(hwReceiptUnreadable, "open receipt: %v", err)
	}
	raw, err := io.ReadAll(io.LimitReader(f, hwReceiptMaxBytes+1))
	f.Close()
	if err != nil {
		return hwReceiptAdmission{}, refuseHWReceipt(hwReceiptUnreadable, "read receipt: %v", err)
	}
	if len(raw) > hwReceiptMaxBytes {
		return hwReceiptAdmission{}, refuseHWReceipt(hwReceiptUnreadable, "receipt exceeds %d bytes", hwReceiptMaxBytes)
	}
	sum := sha256.Sum256(raw)
	fileSHA := "sha256:" + hex.EncodeToString(sum[:])

	// Bind against the receipt's own tip when trunk has only moved on through
	// hardware-irrelevant commits since the controller ran; otherwise HEAD.
	base := tip
	var c hwReceiptCommon
	if json.Unmarshal(bytes.TrimPrefix(raw, []byte("\xef\xbb\xbf")), &c) == nil {
		receiptTip := strings.ToLower(strings.TrimSpace(c.Provenance.GitTip))
		if receiptTip != "" && !strings.EqualFold(receiptTip, tip) {
			if err := hwReceiptIntervalIrrelevant(ctx, root, receiptTip, tip, mine); err != nil {
				return hwReceiptAdmission{FileSHA256: fileSHA, GitTip: c.Provenance.GitTip, Schema: c.Schema}, err
			}
			base = receiptTip
		}
	}

	restore := pinCanonicalArchiveEOL()
	candidate, err := buildStrixCandidateArchiveFn(ctx, root, base, mine)
	restore()
	if err != nil {
		adm := hwReceiptAdmission{FileSHA256: fileSHA}
		return adm, refuseHWReceipt(hwReceiptCandidate, "rebuild prospective candidate archive: %v", err)
	}
	adm, err := checkHWReceipt(raw, base, candidate.SourceArchiveSHA256, mine, validateNow().UTC(), hwReceiptMaxAge())
	adm.FileSHA256 = fileSHA
	if base != tip {
		adm.AdvancedTo = tip
	}
	return adm, err
}

// pinCanonicalArchiveEOL makes the gate's `git archive` emit the LF bytes a
// Linux controller emits. git archive applies .gitattributes text conversion
// using core.eol, which is CRLF on native Windows git, so without the pin the
// same commit archives to different digests per controller OS and no Linux
// receipt could ever match a Windows gate. The pin rides git's GIT_CONFIG_*
// environment (inherited by the builder's child git) and is restored on return.
// TODO(#13802 follow-up): move this pin, and git-mode normalization of
// overlay entries, into amdgpu.BuildStrixCandidateArchive itself.
func pinCanonicalArchiveEOL() func() {
	keys := []string{"GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_CONFIG_KEY_1", "GIT_CONFIG_VALUE_1"}
	saved := map[string]*string{}
	for _, k := range keys {
		if v, ok := os.LookupEnv(k); ok {
			saved[k] = &v
		} else {
			saved[k] = nil
		}
	}
	if saved["GIT_CONFIG_COUNT"] != nil {
		// The caller already injects git config; do not renumber it.
		return func() {}
	}
	os.Setenv("GIT_CONFIG_COUNT", "2")
	os.Setenv("GIT_CONFIG_KEY_0", "core.eol")
	os.Setenv("GIT_CONFIG_VALUE_0", "lf")
	os.Setenv("GIT_CONFIG_KEY_1", "core.autocrlf")
	os.Setenv("GIT_CONFIG_VALUE_1", "false")
	return func() {
		for k, v := range saved {
			if v == nil {
				os.Unsetenv(k)
			} else {
				os.Setenv(k, *v)
			}
		}
	}
}

var hwReceiptGitFn = func(ctx context.Context, root string, args ...string) (string, error) {
	out, err := exec.CommandContext(ctx, "git", append([]string{"-C", root}, args...)...).Output()
	return strings.TrimSpace(string(out)), err
}

// hwReceiptIntervalIrrelevant admits a receipt bound to an ancestor of HEAD
// only when every commit since touched neither an owned path nor any path the
// hardware gate itself considers relevant. The candidate archive is then
// rebuilt at the receipt's tip, so the hardware still validated byte-exact
// owned content on a base whose device-relevant tree equals HEAD's.
func hwReceiptIntervalIrrelevant(ctx context.Context, root, receiptTip, head string, mine []string) error {
	if len(receiptTip) != 40 {
		return refuseHWReceipt(hwReceiptStaleTip, "receipt git_tip %q is not a full commit id", receiptTip)
	}
	if _, err := hwReceiptGitFn(ctx, root, "merge-base", "--is-ancestor", receiptTip, head); err != nil {
		return refuseHWReceipt(hwReceiptStaleTip, "receipt git_tip %s is not an ancestor of prospective base %s; rerun the controller at the current HEAD", receiptTip, head)
	}
	out, err := hwReceiptGitFn(ctx, root, "-c", "core.quotepath=off", "diff", "--no-renames", "--name-only", receiptTip, head)
	if err != nil {
		return refuseHWReceipt(hwReceiptStaleTip, "diff receipt git_tip %s..%s: %v", receiptTip, head, err)
	}
	owned := map[string]bool{}
	for _, p := range mine {
		owned[strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")] = true
	}
	for _, changed := range strings.Split(out, "\n") {
		if changed = strings.TrimSpace(changed); changed == "" {
			continue
		}
		hit := owned[changed]
		for p := range owned {
			if strings.HasPrefix(changed, p+"/") {
				hit = true
			}
		}
		if hit || isGPURelatedValidation([]string{changed}) {
			return refuseHWReceipt(hwReceiptStaleTip, "trunk moved from receipt git_tip %s to %s through hardware-relevant or owned path %s; rerun the controller at the current HEAD", receiptTip, head, changed)
		}
	}
	return nil
}
