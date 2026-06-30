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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
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

// BenchmarkArtifactSchema is the #416 scientific-rigor envelope stamped onto
// every benchmark report emitted through this package. It is additive beside the
// older lineage block: lineage stays the compact compatibility record, while
// benchmark_artifact carries version tracking, invalidation state, and witness
// fields for published-number traceability.
const BenchmarkArtifactSchema = "fak-benchmark-artifact/1"

// BenchmarkHarnessVersion versions the shared Go harness contract rather than a
// single benchmark's measurement logic. A benchmark runner can override it with
// FAK_BENCH_HARNESS_VERSION when its own harness has a stricter version.
const BenchmarkHarnessVersion = "1.0.0"

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

// BenchmarkArtifact is the durable benchmark-science record requested by #416.
// It deliberately mirrors the issue's vocabulary: exact fak code version,
// harness/model/config identity, invalidation metadata, and witness hooks. Fields
// that a generic emitter cannot infer are never omitted; they fail-soft to
// "unknown" (or CLAIM_UNWITNESSED for DOS verification) so absence is visible.
type BenchmarkArtifact struct {
	Schema         string            `json:"schema"`
	RunID          string            `json:"run_id"`
	Timestamp      string            `json:"timestamp"`
	FAKCommit      string            `json:"fak_commit"`
	FAKVersion     string            `json:"fak_version"`
	HarnessVersion string            `json:"harness_version"`
	Harness        HarnessInfo       `json:"harness"`
	Machine        MachineInfo       `json:"machine"`
	Model          ModelSnapshot     `json:"model"`
	Dependencies   map[string]string `json:"dependency_versions"`
	Build          BuildInfo         `json:"build"`
	Config         ConfigSnapshot    `json:"config"`
	Results        ResultSnapshot    `json:"results"`
	Invalidated    Invalidation      `json:"invalidated"`
	Witness        WitnessInfo       `json:"witness"`
	Lineage        LineageRefs       `json:"lineage"`
}

type HarnessInfo struct {
	Name       string   `json:"name"`
	Version    string   `json:"version"`
	BuildFlags []string `json:"build_flags"`
}

type MachineInfo struct {
	Hostname string `json:"hostname"`
	CPU      string `json:"cpu"`
	GPU      string `json:"gpu"`
	RAMGB    int    `json:"ram_gb"`
	OS       string `json:"os"`
	Arch     string `json:"arch"`
}

type ModelSnapshot struct {
	Name         string `json:"name"`
	Precision    string `json:"precision"`
	SourceCommit string `json:"source_commit"`
	SourceURL    string `json:"source_url"`
	Hash         string `json:"hash"`
}

type BuildInfo struct {
	Tags    []string `json:"tags"`
	GOOS    string   `json:"goos"`
	GOARCH  string   `json:"goarch"`
	GOFLAGS string   `json:"goflags"`
}

type ConfigSnapshot struct {
	Hash       string         `json:"hash"`
	Parameters map[string]any `json:"parameters"`
}

type ResultSnapshot struct {
	Metrics  map[string]any    `json:"metrics"`
	Units    map[string]string `json:"units"`
	Baseline map[string]any    `json:"baseline"`
}

type Invalidation struct {
	IsInvalid        bool    `json:"is_invalid"`
	Reason           *string `json:"reason"`
	ReplacementRunID *string `json:"replacement_run_id"`
}

type WitnessInfo struct {
	TestPath            string `json:"test_path"`
	DOSVerifyResult     string `json:"dos_verify_result"`
	DOSCommitAudit      string `json:"dos_commit_audit"`
	ReproductionCommand string `json:"reproduction_command"`
}

type LineageRefs struct {
	SourceArtifact string `json:"source_artifact"`
	Baseline       string `json:"baseline"`
}

// IndexedArtifact is one artifact entry in a lineage index. It is intentionally
// small: enough to answer "which run produced this published number, under what
// code/model/config identity, and is it still valid?"
type IndexedArtifact struct {
	Path           string        `json:"path"`
	RunID          string        `json:"run_id"`
	Timestamp      string        `json:"timestamp"`
	FAKCommit      string        `json:"fak_commit"`
	HarnessVersion string        `json:"harness_version"`
	Model          ModelSnapshot `json:"model"`
	ConfigHash     string        `json:"config_hash"`
	Invalidated    Invalidation  `json:"invalidated"`
	Witness        WitnessInfo   `json:"witness"`
}

type ArtifactIndex struct {
	Schema    string            `json:"schema"`
	Artifacts []IndexedArtifact `json:"artifacts"`
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
	out, err := exec.Command("git", "rev-parse", "HEAD").Output()
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

// ArtifactFromJSON derives the #416 artifact envelope from the already-rendered
// report bytes plus the lineage stamp for this run.
func ArtifactFromJSON(lin Lineage, report []byte) BenchmarkArtifact {
	root := decodeObject(report)
	hname := envOr("FAK_BENCH_HARNESS_NAME", executableName())
	hver := envOr("FAK_BENCH_HARNESS_VERSION", BenchmarkHarnessVersion)
	buildFlags := splitList(firstNonEmpty(os.Getenv("FAK_BENCH_BUILD_FLAGS"), os.Getenv("GOFLAGS")))
	model := extractModel(root)
	return BenchmarkArtifact{
		Schema:         BenchmarkArtifactSchema,
		RunID:          envOr("FAK_BENCH_RUN_ID", runID(lin.UTC, lin.GitCommit, hname)),
		Timestamp:      lin.UTC,
		FAKCommit:      lin.GitCommit,
		FAKVersion:     lin.AppVersion,
		HarnessVersion: hver,
		Harness: HarnessInfo{
			Name:       hname,
			Version:    hver,
			BuildFlags: buildFlags,
		},
		Machine: MachineInfo{
			Hostname: lin.Node,
			CPU:      envOr("FAK_BENCH_CPU", cpuLabel()),
			GPU:      envOr("FAK_BENCH_GPU", lineageUnknown),
			RAMGB:    envInt("FAK_BENCH_RAM_GB"),
			OS:       runtime.GOOS,
			Arch:     runtime.GOARCH,
		},
		Model: model,
		Dependencies: map[string]string{
			"go":   lin.GoVersion,
			"os":   runtime.GOOS,
			"arch": runtime.GOARCH,
			"cuda": envOr("FAK_BENCH_CUDA_VERSION", lineageUnknown),
		},
		Build: BuildInfo{
			Tags:    buildFlags,
			GOOS:    runtime.GOOS,
			GOARCH:  runtime.GOARCH,
			GOFLAGS: os.Getenv("GOFLAGS"),
		},
		Config:      extractConfig(root),
		Results:     extractResults(root),
		Invalidated: envInvalidation(),
		Witness: WitnessInfo{
			TestPath:            envOr("FAK_BENCH_WITNESS_TEST", lineageUnknown),
			DOSVerifyResult:     envOr("FAK_BENCH_DOS_VERIFY_RESULT", "CLAIM_UNWITNESSED"),
			DOSCommitAudit:      envOr("FAK_BENCH_DOS_COMMIT_AUDIT", "not_run"),
			ReproductionCommand: envOr("FAK_BENCH_REPRO_COMMAND", shellCommand(os.Args)),
		},
		Lineage: LineageRefs{
			SourceArtifact: envOr("FAK_BENCH_ARTIFACT", lineageUnknown),
			Baseline:       envOr("FAK_BENCH_BASELINE", lineageUnknown),
		},
	}
}

// Into splices the benchmark_artifact envelope into an indented top-level JSON
// object, preserving all original report members. It is idempotent and skips
// compact/non-object payloads the same way Lineage.Into does.
func (a BenchmarkArtifact) Into(b []byte) []byte {
	return spliceTopLevelObject(b, "benchmark_artifact", a)
}

func leadingWhitespace(b []byte) string {
	i := 0
	for i < len(b) && (b[i] == ' ' || b[i] == '\t') {
		i++
	}
	return string(b[:i])
}

func spliceTopLevelObject(b []byte, key string, value any) []byte {
	if len(b) == 0 || b[0] != '{' {
		return b
	}
	nl := bytes.IndexByte(b, '\n')
	if nl < 0 {
		return b
	}
	indent := leadingWhitespace(b[nl+1:])
	if indent == "" {
		indent = "  "
	}
	if bytes.Contains(b, []byte("\n"+indent+`"`+key+`":`)) {
		return b
	}
	blk, err := json.MarshalIndent(value, indent, indent)
	if err != nil {
		return b
	}
	var out bytes.Buffer
	out.Grow(len(b) + len(blk) + len(indent) + len(key) + 16)
	out.Write(b[:nl+1])
	out.WriteString(indent)
	out.WriteByte('"')
	out.WriteString(key)
	out.WriteString(`": `)
	out.Write(blk)
	if firstNonSpaceIsBrace(b[nl+1:]) {
		out.WriteByte('\n')
	} else {
		out.WriteString(",\n")
	}
	out.Write(b[nl+1:])
	return out.Bytes()
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
	lin := Stamp()
	b = ArtifactFromJSON(lin, b).Into(b)
	return lin.Into(b), nil
}

// WriteReport marshals report (stamping lineage, see MarshalReport) and writes it
// to path, creating path's parent directory tree. It is the single-call form for
// emitters that always write to a file; emitters that also print to stdout use
// MarshalReport and branch on the bytes themselves.
func WriteReport(path string, report any) error {
	prev := os.Getenv("FAK_BENCH_ARTIFACT")
	if prev == "" {
		_ = os.Setenv("FAK_BENCH_ARTIFACT", filepath.ToSlash(path))
		defer os.Unsetenv("FAK_BENCH_ARTIFACT")
	}
	b, err := MarshalReport(report)
	if err != nil {
		return err
	}
	return WriteFile(path, b)
}

// ManualInvalidation is the explicit superseded-result form: reason and optional
// replacement run id are recorded in the same shape as automatic invalidation.
func ManualInvalidation(reason, replacementRunID string) Invalidation {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = lineageUnknown
	}
	inv := Invalidation{IsInvalid: true, Reason: &reason}
	if r := strings.TrimSpace(replacementRunID); r != "" {
		inv.ReplacementRunID = &r
	}
	return inv
}

// DetectInvalidation implements the #416 automatic invalidation rules over a
// previous and replacement artifact plus the changed paths in the commit/window
// being evaluated.
func DetectInvalidation(prev, next BenchmarkArtifact, changedPaths []string) Invalidation {
	var reasons []string
	if p := firstInvalidatingCodePath(changedPaths); p != "" {
		reasons = append(reasons, "code change touched "+p)
	}
	if p := firstHarnessPath(prev.Harness.Name, changedPaths); p != "" {
		reasons = append(reasons, "harness change touched "+p)
	}
	if modelIdentity(prev.Model) != modelIdentity(next.Model) {
		reasons = append(reasons, "model weights or quantization changed")
	}
	if prev.Config.Hash != "" && next.Config.Hash != "" && prev.Config.Hash != next.Config.Hash {
		reasons = append(reasons, "benchmark config changed")
	}
	if len(reasons) == 0 {
		return Invalidation{}
	}
	reason := strings.Join(reasons, "; ")
	inv := Invalidation{IsInvalid: true, Reason: &reason}
	if next.RunID != "" {
		inv.ReplacementRunID = &next.RunID
	}
	return inv
}

// BuildLineageIndex scans root for committed JSON benchmark artifacts and folds
// their benchmark_artifact records into a deterministic index. Legacy lineage and
// benchmark/run-manifest.v1 documents are also admitted so older runs remain
// traceable while new emitters move to the #416 envelope.
func BuildLineageIndex(root string) (ArtifactIndex, error) {
	var out ArtifactIndex
	out.Schema = "fak-benchmark-lineage-index/1"
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(strings.ToLower(d.Name()), ".json") {
			return nil
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		art, ok := DecodeArtifact(raw)
		if !ok {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			rel = path
		}
		out.Artifacts = append(out.Artifacts, IndexedArtifact{
			Path:           filepath.ToSlash(rel),
			RunID:          art.RunID,
			Timestamp:      art.Timestamp,
			FAKCommit:      art.FAKCommit,
			HarnessVersion: art.HarnessVersion,
			Model:          art.Model,
			ConfigHash:     art.Config.Hash,
			Invalidated:    art.Invalidated,
			Witness:        art.Witness,
		})
		return nil
	})
	sort.Slice(out.Artifacts, func(i, j int) bool { return out.Artifacts[i].Path < out.Artifacts[j].Path })
	return out, err
}

// DecodeArtifact extracts the #416 envelope from a report, or adapts older
// lineage/manifest records into the same in-memory shape for indexing.
func DecodeArtifact(raw []byte) (BenchmarkArtifact, bool) {
	root := decodeObject(raw)
	if len(root) == 0 {
		return BenchmarkArtifact{}, false
	}
	if v, ok := root["benchmark_artifact"]; ok {
		b, _ := json.Marshal(v)
		var art BenchmarkArtifact
		if json.Unmarshal(b, &art) == nil && art.RunID != "" {
			return art, true
		}
	}
	if v, ok := root["lineage"]; ok {
		b, _ := json.Marshal(v)
		var lin Lineage
		if json.Unmarshal(b, &lin) == nil && (lin.GitCommit != "" || lin.UTC != "") {
			art := ArtifactFromJSON(lin, raw)
			art.RunID = firstNonEmpty(stringField(root, "run_id"), art.RunID)
			return art, true
		}
	}
	if schema, _ := root["$schema"].(string); schema == "benchmark/run-manifest.v1" {
		return artifactFromRunManifest(root), true
	}
	return BenchmarkArtifact{}, false
}

func decodeObject(raw []byte) map[string]any {
	var root map[string]any
	if json.Unmarshal(raw, &root) != nil {
		return nil
	}
	return root
}

func artifactFromRunManifest(root map[string]any) BenchmarkArtifact {
	git := mapField(root, "git")
	harness := mapField(root, "harness")
	model := mapField(root, "model")
	config := mapField(root, "config")
	hv := stringField(harness, "version")
	return BenchmarkArtifact{
		Schema:         BenchmarkArtifactSchema,
		RunID:          stringField(root, "run_id"),
		Timestamp:      stringField(root, "timestamp"),
		FAKCommit:      firstNonEmpty(stringField(git, "rev"), lineageUnknown),
		FAKVersion:     lineageUnknown,
		HarnessVersion: firstNonEmpty(hv, lineageUnknown),
		Harness: HarnessInfo{
			Name:    firstNonEmpty(stringField(harness, "name"), lineageUnknown),
			Version: firstNonEmpty(hv, lineageUnknown),
		},
		Machine: MachineInfo{
			Hostname: firstNonEmpty(stringField(root, "machine_id"), lineageUnknown),
			CPU:      lineageUnknown,
			GPU:      lineageUnknown,
			OS:       lineageUnknown,
			Arch:     lineageUnknown,
		},
		Model: ModelSnapshot{
			Name:      firstNonEmpty(stringField(model, "name"), lineageUnknown),
			Precision: firstNonEmpty(stringField(model, "precision"), lineageUnknown),
		},
		Dependencies: map[string]string{},
		Config: ConfigSnapshot{
			Hash:       hashAny(config),
			Parameters: config,
		},
		Results:     ResultSnapshot{Metrics: map[string]any{}, Units: map[string]string{}, Baseline: map[string]any{}},
		Invalidated: Invalidation{},
		Witness: WitnessInfo{
			TestPath:            lineageUnknown,
			DOSVerifyResult:     "CLAIM_UNWITNESSED",
			DOSCommitAudit:      "not_run",
			ReproductionCommand: lineageUnknown,
		},
		Lineage: LineageRefs{SourceArtifact: lineageUnknown, Baseline: lineageUnknown},
	}
}

func extractModel(root map[string]any) ModelSnapshot {
	name := firstNonEmpty(os.Getenv("FAK_BENCH_MODEL_NAME"), recursiveString(root, "model"), recursiveString(root, "engine_model"), recursiveString(root, "model_name"))
	precision := firstNonEmpty(os.Getenv("FAK_BENCH_MODEL_PRECISION"), recursiveString(root, "precision"), recursiveString(root, "quantization"))
	source := firstNonEmpty(os.Getenv("FAK_BENCH_MODEL_SOURCE_URL"), recursiveString(root, "source_url"), recursiveString(root, "source"), recursiveString(root, "model_filename"))
	commit := firstNonEmpty(os.Getenv("FAK_BENCH_MODEL_COMMIT"), recursiveString(root, "source_commit"), recursiveString(root, "hf_commit"), snapshotCommitFromPath(source))
	hash := firstNonEmpty(os.Getenv("FAK_BENCH_MODEL_HASH"), recursiveString(root, "gguf_sha256"), recursiveString(root, "model_sha256"), recursiveString(root, "model_hash"), recursiveString(root, "sha256"))
	return ModelSnapshot{
		Name:         firstNonEmpty(name, lineageUnknown),
		Precision:    firstNonEmpty(precision, lineageUnknown),
		SourceCommit: firstNonEmpty(commit, lineageUnknown),
		SourceURL:    firstNonEmpty(source, lineageUnknown),
		Hash:         firstNonEmpty(hash, lineageUnknown),
	}
}

func extractConfig(root map[string]any) ConfigSnapshot {
	params := map[string]any{}
	for _, k := range []string{"config", "cost_model", "workload", "parameters"} {
		if v, ok := root[k]; ok {
			params[k] = v
		}
	}
	if prov := mapField(root, "provenance"); len(prov) > 0 {
		for _, k := range []string{"command", "slice_id", "workload_hash"} {
			if v, ok := prov[k]; ok {
				params["provenance."+k] = v
			}
		}
	}
	if len(params) == 0 {
		return ConfigSnapshot{Hash: lineageUnknown, Parameters: map[string]any{}}
	}
	return ConfigSnapshot{Hash: hashAny(params), Parameters: params}
}

func extractResults(root map[string]any) ResultSnapshot {
	metrics := map[string]any{}
	units := map[string]string{}
	for _, section := range []string{"results", "kpis", "net"} {
		if m := mapField(root, section); len(m) > 0 {
			flattenScalars(metrics, section, m)
		}
	}
	for k, v := range root {
		if isScalar(v) && looksMetric(k) {
			metrics[k] = v
		}
	}
	for k := range metrics {
		if u := metricUnit(k); u != "" {
			units[k] = u
		}
	}
	base := firstMap(root, "baseline", "spawned_hook_baseline")
	return ResultSnapshot{Metrics: metrics, Units: units, Baseline: base}
}

func envInvalidation() Invalidation {
	if !truthy(os.Getenv("FAK_BENCH_INVALIDATED")) {
		return Invalidation{}
	}
	return ManualInvalidation(os.Getenv("FAK_BENCH_INVALIDATION_REASON"), os.Getenv("FAK_BENCH_REPLACEMENT_RUN_ID"))
}

func firstInvalidatingCodePath(paths []string) string {
	for _, p := range normalizedPaths(paths) {
		for _, pref := range []string{"internal/model/", "internal/compute/", "internal/radixkv/"} {
			if strings.HasPrefix(p, pref) {
				return pref
			}
		}
	}
	return ""
}

func firstHarnessPath(harness string, paths []string) string {
	harness = strings.TrimSuffix(strings.ToLower(filepath.Base(harness)), ".exe")
	cmdPrefix := "cmd/" + harness + "/"
	for _, p := range normalizedPaths(paths) {
		if strings.HasPrefix(p, cmdPrefix) || strings.HasPrefix(p, "internal/bench/") ||
			strings.HasPrefix(p, "internal/benchcli/") || p == "BENCHMARK-TEMPLATE.md" {
			return p
		}
	}
	return ""
}

func normalizedPaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		p = filepath.ToSlash(strings.ReplaceAll(strings.TrimSpace(p), "\\", "/"))
		p = strings.TrimPrefix(p, "./")
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func modelIdentity(m ModelSnapshot) string {
	return strings.Join([]string{m.Name, m.Precision, m.SourceCommit, m.SourceURL, m.Hash}, "\x00")
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envInt(key string) int {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return 0
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0
	}
	return n
}

func firstNonEmpty(xs ...string) string {
	for _, x := range xs {
		if s := strings.TrimSpace(x); s != "" {
			return s
		}
	}
	return ""
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' || r == '\n' }) {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func executableName() string {
	if len(os.Args) == 0 {
		return lineageUnknown
	}
	base := filepath.Base(os.Args[0])
	base = strings.TrimSuffix(base, ".exe")
	if base == "" {
		return lineageUnknown
	}
	return base
}

func runID(utc, commit, harness string) string {
	stamp := strings.NewReplacer(":", "", "-", "", "T", "T").Replace(utc)
	stamp = strings.TrimSuffix(stamp, "Z")
	if stamp == "" || stamp == lineageUnknown {
		stamp = "unknown-time"
	}
	c := commit
	if len(c) > 12 {
		c = c[:12]
	}
	if c == "" {
		c = lineageUnknown
	}
	return sanitizeID(stamp + "-" + harness + "-" + c)
}

func sanitizeID(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
		} else {
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}

func cpuLabel() string {
	for _, k := range []string{"PROCESSOR_IDENTIFIER", "PROCESSOR_ARCHITECTURE"} {
		if v := strings.TrimSpace(os.Getenv(k)); v != "" {
			return v
		}
	}
	return runtime.GOARCH
}

func shellCommand(args []string) string {
	if len(args) == 0 {
		return lineageUnknown
	}
	parts := make([]string, len(args))
	for i, a := range args {
		parts[i] = shellQuote(a)
	}
	return strings.Join(parts, " ")
}

func shellQuote(s string) string {
	if s == "" {
		return `""`
	}
	if !strings.ContainsAny(s, " \t\n\"'") {
		return s
	}
	return `"` + strings.ReplaceAll(s, `"`, `\"`) + `"`
}

func mapField(m map[string]any, key string) map[string]any {
	v, ok := m[key]
	if !ok {
		return nil
	}
	mv, ok := v.(map[string]any)
	if ok {
		return mv
	}
	return nil
}

func stringField(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if s, ok := m[key].(string); ok {
		return s
	}
	return ""
}

func firstMap(root map[string]any, keys ...string) map[string]any {
	for _, k := range keys {
		if m := mapField(root, k); len(m) > 0 {
			return m
		}
	}
	return map[string]any{}
}

func recursiveString(v any, keys ...string) string {
	wanted := map[string]bool{}
	for _, k := range keys {
		wanted[k] = true
	}
	return recursiveStringAny(v, wanted)
}

func recursiveStringAny(v any, wanted map[string]bool) string {
	switch x := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			if wanted[k] {
				if s, ok := x[k].(string); ok && strings.TrimSpace(s) != "" {
					return s
				}
			}
		}
		for _, k := range keys {
			if s := recursiveStringAny(x[k], wanted); s != "" {
				return s
			}
		}
	case []any:
		for _, e := range x {
			if s := recursiveStringAny(e, wanted); s != "" {
				return s
			}
		}
	}
	return ""
}

func snapshotCommitFromPath(s string) string {
	const marker = "/snapshots/"
	p := filepath.ToSlash(s)
	i := strings.Index(p, marker)
	if i < 0 {
		return ""
	}
	rest := p[i+len(marker):]
	if j := strings.IndexByte(rest, '/'); j >= 0 {
		rest = rest[:j]
	}
	if len(rest) >= 7 {
		return rest
	}
	return ""
}

func hashAny(v any) string {
	if v == nil {
		return lineageUnknown
	}
	b, err := json.Marshal(v)
	if err != nil {
		return lineageUnknown
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func flattenScalars(out map[string]any, prefix string, m map[string]any) {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := m[k]
		name := prefix + "." + k
		if isScalar(v) {
			out[name] = v
		}
	}
}

func isScalar(v any) bool {
	switch v.(type) {
	case string, float64, bool, nil:
		return true
	default:
		return false
	}
}

func looksMetric(k string) bool {
	k = strings.ToLower(k)
	return strings.Contains(k, "tok_per_sec") || strings.Contains(k, "_ns") ||
		strings.Contains(k, "_ms") || strings.Contains(k, "rate") ||
		strings.Contains(k, "speedup") || strings.Contains(k, "latency")
}

func metricUnit(k string) string {
	k = strings.ToLower(k)
	switch {
	case strings.Contains(k, "tok_per_sec"):
		return "tokens/s"
	case strings.HasSuffix(k, "_ns") || strings.Contains(k, "_ns."):
		return "ns"
	case strings.HasSuffix(k, "_ms") || strings.Contains(k, "_ms."):
		return "ms"
	case strings.Contains(k, "rate"):
		return "fraction"
	case strings.Contains(k, "speedup"):
		return "x"
	default:
		return ""
	}
}

func truthy(s string) bool {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "1", "t", "true", "yes", "y", "on":
		return true
	default:
		return false
	}
}
