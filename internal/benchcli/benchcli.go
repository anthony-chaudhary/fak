// Package benchcli holds the small, identical helpers the benchmark-CLI mains
// (cmd/*bench and the demo/cert commands beside them) had each copy-pasted into
// their own file. Two functions covered the bulk of the duplication: reading a
// Hugging Face config.json into a model.Config, and writing a byte slice to a
// path while creating its parent directory. Both are pure, resource-free, and
// behaviour-identical across the callers, so one shared copy retires the clone
// family without changing any caller's observable output (#983, child of #775).
package benchcli

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ReadHFConfig loads the Hugging Face config.json from dir into a model.Config.
// HF Llama/Qwen2 configs omit head_dim (it is hidden_size/num_attention_heads),
// so it is derived when absent — matching what every bench main did by hand.
func ReadHFConfig(dir string) (model.Config, error) {
	var cfg model.Config
	cb, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("config.json: %w", err)
	}
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return cfg, fmt.Errorf("config.json parse: %w", err)
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

// WriteFile writes b to path, first creating path's parent directory tree. A
// bare filename (Dir is "." or "") needs no mkdir, so that no-op is skipped.
func WriteFile(path string, b []byte) error {
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(path, b, 0o644)
}

// LineageSchema tags the lineage record so a Go-emitted artifact and the external
// shell runner's sidecar (experiments/benchmark/.../lineage.json, the
// "fak-bench-lineage/1" schema) carry the same shape and converge rather than
// drift. Bump only on a breaking field change (#9).
const LineageSchema = "fak-bench-lineage/1"

// lineageUnknown is the fail-soft value for a field we could not derive (no git
// checkout, no hostname). A lineage record is never absent — a tarball build still
// emits one, with "unknown" where ground truth was unreachable.
const lineageUnknown = "unknown"

// Lineage is the four-axis provenance every benchmark emitter stamps so an
// artifact is traceable to the exact build that produced it: the application
// version, the wall-clock instant, the source commit, and the machine. The field
// names and the schema tag deliberately match the existing external sidecar
// (experiments/benchmark/runs/by-machine/.../lineage.json) so the Go-emitted
// record and the shell runner's record share one vocabulary (#9).
type Lineage struct {
	Schema     string `json:"lineage_schema"`
	AppVersion string `json:"app_version"`
	UTC        string `json:"utc"`
	GitCommit  string `json:"git_commit"`
	GoVersion  string `json:"go_version"`
	Node       string `json:"node"`
}

// Stamp derives the lineage for the current build/host/checkout. Every field is
// fail-soft so a lineage-free artifact can never ship: a tarball build with no
// .git still emits a record (git_commit "unknown"). Three fields honour an env
// override so a runner can pin a logical value or a test can be deterministic —
// FAK_BENCH_UTC, FAK_BENCH_COMMIT, FAK_BENCH_NODE — and app_version already
// resolves via appversion.Current() (VERSION file / FAK_APP_VERSION / build flag).
func Stamp() Lineage {
	return Lineage{
		Schema:     LineageSchema,
		AppVersion: appversion.Current(),
		UTC:        stampUTC(),
		GitCommit:  stampCommit(),
		GoVersion:  runtime.Version(),
		Node:       stampNode(),
	}
}

func stampUTC() string {
	if v := strings.TrimSpace(os.Getenv("FAK_BENCH_UTC")); v != "" {
		return v
	}
	return time.Now().UTC().Format(time.RFC3339)
}

func stampCommit() string {
	if v := strings.TrimSpace(os.Getenv("FAK_BENCH_COMMIT")); v != "" {
		return v
	}
	out, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return lineageUnknown
	}
	if c := strings.TrimSpace(string(out)); c != "" {
		return c
	}
	return lineageUnknown
}

func stampNode() string {
	if v := strings.TrimSpace(os.Getenv("FAK_BENCH_NODE")); v != "" {
		return v
	}
	h, err := os.Hostname()
	if err != nil {
		return lineageUnknown
	}
	if h = strings.TrimSpace(h); h != "" {
		return h
	}
	return lineageUnknown
}

// Into splices l as a "lineage" object that becomes the first member of the
// top-level JSON object in b, preserving b's existing bytes — key order and
// indentation — so a caller's report shape is untouched. It is a no-op (returns b
// unchanged) when b is not an indented JSON object or already carries a top-level
// lineage member, so re-stamping and non-object reports (a bare array/scalar, a
// JSONL line) are both safe. The lineage block is indented to match b's own
// members.
func (l Lineage) Into(b []byte) []byte {
	if len(b) == 0 || b[0] != '{' {
		return b
	}
	nl := bytes.IndexByte(b, '\n')
	if nl < 0 {
		return b // compact/single-line object: nothing to align to, leave it.
	}
	// The member indent is the leading whitespace of the line after "{\n".
	indent := leadingWhitespace(b[nl+1:])
	if indent == "" {
		indent = "  "
	}
	if bytes.Contains(b, []byte("\n"+indent+`"lineage":`)) {
		return b // already stamped.
	}
	blk, err := json.MarshalIndent(l, indent, indent)
	if err != nil {
		return b
	}
	var out bytes.Buffer
	out.Grow(len(b) + len(blk) + len(indent) + 16)
	out.Write(b[:nl+1])
	out.WriteString(indent)
	out.WriteString(`"lineage": `)
	out.Write(blk)
	// An empty object ("{\n}") has no following member, so no separating comma.
	if firstNonSpaceIsBrace(b[nl+1:]) {
		out.WriteByte('\n')
	} else {
		out.WriteString(",\n")
	}
	out.Write(b[nl+1:])
	return out.Bytes()
}

func leadingWhitespace(b []byte) string {
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t') {
		i++
	}
	return string(b[:i])
}

func firstNonSpaceIsBrace(b []byte) bool {
	for _, c := range b {
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			continue
		}
		return c == '}'
	}
	return false
}

// MarshalReport stamps the current lineage onto a benchmark report and returns the
// bytes. report is either a value to marshal with two-space indent, or
// already-marshalled bytes ([]byte / json.RawMessage) — pass the latter to
// preserve a struct's own .JSON() shape. The lineage block is spliced as the first
// member of the top-level object (see Lineage.Into); a non-object report is
// returned unchanged. This and WriteReport are the wiring points the lineage gate
// (tools/check_bench_lineage.py) requires every cmd/*bench* result emitter to use.
func MarshalReport(report any) ([]byte, error) {
	var b []byte
	switch r := report.(type) {
	case []byte:
		b = r
	case json.RawMessage:
		b = r
	default:
		m, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			return nil, err
		}
		b = m
	}
	return Stamp().Into(b), nil
}

// WriteReport marshals report (stamping lineage, see MarshalReport) and writes it
// to path, creating path's parent directory tree. It is the single-call form for
// emitters that always write to a file; emitters that also print to stdout use
// MarshalReport and branch on the bytes themselves.
func WriteReport(path string, report any) error {
	b, err := MarshalReport(report)
	if err != nil {
		return err
	}
	return WriteFile(path, b)
}
