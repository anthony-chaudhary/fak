package toolshape

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// ToolShapeSchema versions the derived record so a downstream corpus can tell
// which fingerprint discipline produced a row.
const ToolShapeSchema = "fak.toolshape.v1"

// Well-known OPEN-label keys this package reads from trajectory.Turn.Labels.
// They are the producer-side contract for input shape: a producer that stamps
// them gets full arg-shape fidelity; one that does not degrades cleanly to
// ArgClass=unknown / empty ArgKeys. Values carry NAMES and TYPES only — never
// an argument's raw value.
const (
	// LabelArgKeys is a comma-separated list of the call's top-level arg names
	// (e.g. "file_path,limit,offset").
	LabelArgKeys = "arg_keys"
	// LabelArgTypes is a comma-separated list of "name:type" pairs typing the
	// keys in LabelArgKeys (e.g. "file_path:string,limit:number"). A key with
	// no pair (or a type outside the closed set) types as "unknown".
	LabelArgTypes = "arg_types"
	// LabelTruncated marks the result as truncated by the producer ("true"/"1").
	LabelTruncated = "truncated"
	// LabelError marks the result as an error payload ("true"/"1") — the
	// producer-side complement of a kernel DENY/QUARANTINE verdict.
	LabelError = "error"
)

// ArgClass closed vocabulary: the dominant input class of a call, derived from
// arg NAMES only. An unrecognized shape maps to ArgClassUnknown, never to a
// silent miscount (the same closed-set discipline as internal/harnessprofile).
const (
	ArgClassPath    = "path"    // targets the filesystem (file_path, dir, ...)
	ArgClassPattern = "pattern" // selects by expression (pattern, glob, query, ...)
	ArgClassCommand = "command" // executes (command, cmd, script, ...)
	ArgClassContent = "content" // carries a payload (content, text, body, ...)
	ArgClassMixed   = "mixed"   // carries a payload AND targets/selects (a write-shaped call)
	ArgClassUnknown = "unknown" // no recognizable arg names (or none stamped)
)

// Output-size buckets: a closed log-scale vocabulary shared by OutBytesBucket
// and OutTokensBucket, so a rollup counts shapes instead of raw magnitudes.
const (
	BucketZero = "0"
	Bucket100  = "1-100"
	Bucket1K   = "101-1k"
	Bucket10K  = "1k-10k"
	BucketOver = "10k+"
)

// ToolShape is the compact, redaction-safe structural record of one tool call:
// what kind of input it took and how big/healthy its output was — never the
// content of either. The JSON tags are the stable export schema for
// ToolShapeSchema; new fields are additive (omitempty) so an older reader
// keeps parsing.
type ToolShape struct {
	Tool    string `json:"tool,omitempty"`
	Verdict string `json:"verdict,omitempty"`

	// Input shape (from LabelArgKeys/LabelArgTypes; empty when not stamped).
	ArgKeys   []string `json:"arg_keys,omitempty"`    // sorted, deduped top-level arg names
	ArgCount  int      `json:"arg_count,omitempty"`   // len(ArgKeys)
	ArgClass  string   `json:"arg_class"`             // closed set: ArgClass* consts
	ArgKeySig string   `json:"arg_key_sig,omitempty"` // sha256 over sorted key:type signature; "" when no keys

	// Output shape (from Turn cost fields + verdict + labels).
	OutBytesBucket  string `json:"out_bytes_bucket"`
	OutTokensBucket string `json:"out_tokens_bucket"`
	Truncated       bool   `json:"truncated,omitempty"`
	Empty           bool   `json:"empty,omitempty"`
	IsError         bool   `json:"is_error,omitempty"`
}

// Fingerprint derives the shape of one recorded turn. It is TOTAL and
// DETERMINISTIC: any Turn — including the zero value — yields a well-formed
// ToolShape with closed-vocabulary fields; same Turn in, same ToolShape out;
// no clock, no RNG, no I/O, no panic.
func Fingerprint(t trajectory.Turn) ToolShape {
	keys := argKeys(t.Labels)
	return ToolShape{
		Tool:    t.Tool,
		Verdict: t.Verdict,

		ArgKeys:   keys,
		ArgCount:  len(keys),
		ArgClass:  classify(keys),
		ArgKeySig: keySig(keys, t.Labels),

		OutBytesBucket:  bucket(t.Bytes),
		OutTokensBucket: bucket(int64(t.TokenEstimate)),
		Truncated:       truthy(t.Labels[LabelTruncated]),
		Empty:           t.Bytes == 0 && t.TokenEstimate == 0 && t.ResultDigest == "",
		IsError:         isError(t),
	}
}

// argKeys parses LabelArgKeys into a sorted, deduped name list. Nil labels, an
// absent key, or an empty value all yield nil — the clean "not stamped" default.
func argKeys(labels map[string]string) []string {
	raw := labels[LabelArgKeys] // nil-map read is safe and yields ""
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	seen := map[string]bool{}
	var keys []string
	for _, k := range strings.Split(raw, ",") {
		k = strings.TrimSpace(k)
		if k == "" || seen[k] {
			continue
		}
		seen[k] = true
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// classify folds the arg-name list into the closed ArgClass vocabulary. Each
// name votes for at most one class by exact token match (a "file_path" tokenizes
// to [file path] — substring matching would misread "description" as a script).
// The fold: a payload class combined with any targeting/selecting class is
// ArgClassMixed (a write-shaped call); otherwise the most specific selector wins
// (command > pattern > path); no recognized name is ArgClassUnknown.
func classify(keys []string) string {
	var path, pattern, command, content bool
	for _, k := range keys {
		switch classifyKey(k) {
		case ArgClassPath:
			path = true
		case ArgClassPattern:
			pattern = true
		case ArgClassCommand:
			command = true
		case ArgClassContent:
			content = true
		}
	}
	selector := path || pattern || command
	switch {
	case content && selector:
		return ArgClassMixed
	case command:
		return ArgClassCommand
	case pattern:
		return ArgClassPattern
	case path:
		return ArgClassPath
	case content:
		return ArgClassContent
	}
	return ArgClassUnknown
}

// classToken is the closed name→class vote table. Tokens, not substrings, and
// names only — a value never reaches this map.
var classToken = map[string]string{
	"command": ArgClassCommand, "cmd": ArgClassCommand, "script": ArgClassCommand,
	"shell": ArgClassCommand, "exec": ArgClassCommand,

	"pattern": ArgClassPattern, "regex": ArgClassPattern, "regexp": ArgClassPattern,
	"glob": ArgClassPattern, "query": ArgClassPattern, "search": ArgClassPattern,

	"path": ArgClassPath, "file": ArgClassPath, "filepath": ArgClassPath,
	"filename": ArgClassPath, "dir": ArgClassPath, "directory": ArgClassPath,
	"folder": ArgClassPath,

	"content": ArgClassContent, "text": ArgClassContent, "body": ArgClassContent,
	"data": ArgClassContent, "string": ArgClassContent, "message": ArgClassContent,
}

// classifyKey votes one arg name into a class ("" = no signal). The name is
// lowercased and split on non-alphanumeric runs so "file_path", "filePath",
// and "new_string" all tokenize predictably.
func classifyKey(name string) string {
	for _, tok := range tokens(name) {
		if c, ok := classToken[tok]; ok {
			return c
		}
	}
	return ""
}

// tokens lowercases and splits an arg name on non-alphanumeric runs.
func tokens(name string) []string {
	return strings.FieldsFunc(strings.ToLower(name), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	})
}

// keySig fingerprints the sorted key-set + type signature (genson/Avro
// fingerprint style): sha256 over "name:type" pairs under the schema version.
// Two turns with the same keys and types collide; adding a key or changing a
// type separates. No keys → "" (absent, not a fake fingerprint of nothing).
func keySig(keys []string, labels map[string]string) string {
	if len(keys) == 0 {
		return ""
	}
	types := argTypes(labels)
	pairs := make([]string, len(keys))
	for i, k := range keys {
		typ := types[k]
		if typ == "" {
			typ = "unknown" // untyped key: total default, never an empty type
		}
		pairs[i] = k + ":" + typ
	}
	sum := sha256.Sum256([]byte(ToolShapeSchema + "|" + strings.Join(pairs, ",")))
	return "sha256:" + hex.EncodeToString(sum[:])
}

// typeVocab is the closed Avro-style type vocabulary a producer may stamp in
// LabelArgTypes; anything else (or an untyped key) reads as "unknown".
var typeVocab = map[string]bool{
	"string": true, "number": true, "bool": true,
	"array": true, "object": true, "null": true,
}

// argTypes parses LabelArgTypes into a name→type map; a key outside the closed
// type vocabulary is dropped (it will sign as "unknown" in keySig).
func argTypes(labels map[string]string) map[string]string {
	out := map[string]string{}
	for _, pair := range strings.Split(labels[LabelArgTypes], ",") {
		name, typ, ok := strings.Cut(pair, ":")
		if !ok {
			continue
		}
		name = strings.TrimSpace(name)
		typ = strings.ToLower(strings.TrimSpace(typ))
		if name != "" && typeVocab[typ] {
			out[name] = typ
		}
	}
	return out
}

// bucket folds a size (bytes or tokens) into the closed log-scale vocabulary.
// Non-positive folds to BucketZero, keeping the fold total on defensive inputs.
func bucket(n int64) string {
	switch {
	case n <= 0:
		return BucketZero
	case n <= 100:
		return Bucket100
	case n <= 1000:
		return Bucket1K
	case n <= 10000:
		return Bucket10K
	}
	return BucketOver
}

// isError reads the error signal from the kernel verdict (a DENY or QUARANTINE
// is an errored turn) or the producer's LabelError stamp.
func isError(t trajectory.Turn) bool {
	switch t.Verdict {
	case "DENY", "QUARANTINE":
		return true
	}
	return truthy(t.Labels[LabelError])
}

// truthy reads a label toggle: "1", "true", "yes", "on" (case-insensitive)
// enable; everything else (including absent) is off — the same closed set as
// trajectory's env toggle.
func truthy(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}
