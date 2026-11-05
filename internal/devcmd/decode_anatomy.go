// decode-anatomy — runtime launch-hygiene gate over per-token decode anatomy.
//
// At small batch sizes decode is launch-bound or host-overhead-bound: engines
// with CUDA-graph replay and zero-copy device-side metadata beat engines doing
// per-layer host setup even with identical math kernels. The pinned incident
// corpus behind this gate (fak-private selection note
// LAUNCH-OVERHEAD-PREVENTION-SELECTION-2026-11-04 §1) shows the existing
// surface caught 1 of 5 incidents; this verb adds the four previously
// unguarded shapes as fail-closed gate signals over data the runtime already
// pays for:
//
//   - launches_per_token: hard cap calibrated from the launch-census math
//     (the closed dispatch inventory's looped-variant total for the
//     documented glm-dsa shape) plus drift vs a committed --snapshot
//     baseline. Witnesses: the v3 receipt's kernel_launch_count over
//     decode_tokens, or the kernel profile's selected-kernel
//     calls_per_token sum.
//   - syncs_per_token: cap plus the per-site top-N ladder (sync_sites),
//     catching stream-sync fence storms (48 fences/tok) that sit far below
//     any host_share line.
//   - barriers_elided_while_armed: the guard-drift assert — hazards armed
//     (declared via --hazards-armed or a recorded hazards_armed=true) with
//     barriers/token > 0 and elided/token = 0 means the elision counter is
//     inert; mirrors the llama.cpp guard-drift asserts.
//   - profile_receipt_missing: a decode-performance claim row that presents
//     profiling evidence (a kernel-profile section or a v3 receipt's
//     host-overhead decomposition) which landed empty or incomplete
//     (Complete=false, no positive us-scale fields) is an ESTIMATED band and
//     fails the gate. Rows carrying no profile section at all are not flagged
//     here — committed ABBA control arms deliberately run without structured
//     profiles, and their ms-scale fields stay visible through the other five
//     signals and the regime column.
//   - capture_eligibility_drift: the capture-eligibility predicates in
//     internal/model/cuda_graph_eligibility.go re-read as data (go/ast
//     func-span scan, the graph-audit method) — a vanished KV-grow guard,
//     QSA-threshold guard, or hybrid branch is DRIFT — joined with the
//     receipt-side witnesses: the "Instantiate: out of memory" signature and
//     graph_replay_degraded_tokens > 0 (replay disengaged to the per-layer
//     host loop).
//   - host_share: the existing 0.30 launch-bound line, read through the
//     internal/hostshare reader (reused, never forked).
//
// The regime reader (internal/regime) classifies each record for the
// machine-readable receipts table. Output is the pinned
// fak.decode-anatomy/v1 document a private gate consumes; --fail-on restricts
// which signals may fail the run. Exit codes: 0 PASS, 1 gate FAIL,
// 2 usage/IO error.

package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostshare"
	"github.com/anthony-chaudhary/fak/internal/regime"
)

// DecodeAnatomySchema versions the --json document.
const DecodeAnatomySchema = "fak.decode-anatomy/v1"

// Gate signal ids: the closed vocabulary, emitted in this order.
const (
	SignalLaunchesPerToken         = "launches_per_token"
	SignalSyncsPerToken            = "syncs_per_token"
	SignalBarriersElidedWhileArmed = "barriers_elided_while_armed"
	SignalProfileReceiptMissing    = "profile_receipt_missing"
	SignalCaptureEligibilityDrift  = "capture_eligibility_drift"
	SignalHostShare                = "host_share"
)

// decodeAnatomySignalIDs is the closed set --fail-on validates against.
var decodeAnatomySignalIDs = []string{
	SignalLaunchesPerToken,
	SignalSyncsPerToken,
	SignalBarriersElidedWhileArmed,
	SignalProfileReceiptMissing,
	SignalCaptureEligibilityDrift,
	SignalHostShare,
}

// AnatomyStatus is one signal's closed verdict.
type AnatomyStatus string

const (
	AnatomyPass    AnatomyStatus = "pass"
	AnatomyFail    AnatomyStatus = "fail"
	AnatomyWarn    AnatomyStatus = "warn"
	AnatomyUnknown AnatomyStatus = "unknown"
)

// AnatomySignal is one gate signal in the pinned document. Value/Cap are nil
// when the signal carries no numeric witness/line (never a fabricated zero).
type AnatomySignal struct {
	ID     string        `json:"id"`
	Value  *float64      `json:"value"`
	Cap    *float64      `json:"cap"`
	Status AnatomyStatus `json:"status"`
	Detail string        `json:"detail"`
}

// AnatomyReceipt is one scanned record's per-token anatomy.
type AnatomyReceipt struct {
	Path                   string   `json:"path"`
	Model                  string   `json:"model,omitempty"`
	Backend                string   `json:"backend,omitempty"`
	LaunchesPerToken       *float64 `json:"launches_per_token,omitempty"`
	SyncsPerToken          *float64 `json:"syncs_per_token,omitempty"`
	BarriersElidedPerToken *float64 `json:"barriers_elided_per_token,omitempty"`
	Regime                 string   `json:"regime,omitempty"`
	OOMSignature           bool     `json:"oom_signature,omitempty"`
}

// DecodeAnatomyResult is the pinned output document.
type DecodeAnatomyResult struct {
	Schema     string           `json:"schema"`
	ObservedAt string           `json:"observed_at"`
	Receipts   []AnatomyReceipt `json:"receipts"`
	Signals    []AnatomySignal  `json:"signals"`
	Verdict    string           `json:"verdict"` // pass | fail
	Failures   []string         `json:"failures"`
}

const (
	// defaultSyncsPerTokenCap is the stream-sync fence line: 4x headroom over
	// the committed aegis baseline (~2.06 syncs/tok), well under the witnessed
	// fence-storm incident (48/tok). Overridable via --syncs-cap.
	defaultSyncsPerTokenCap = 8.0
	// defaultHostShareLine mirrors the internal/hostshare launch-bound line.
	defaultHostShareLine = 0.30
	// defaultLaunchesDriftTol is the relative drift tolerance vs the
	// --snapshot baseline, matching the glue-census cross-check tolerance.
	defaultLaunchesDriftTol = 0.20
	// anatomyMaxFileBytes bounds one scanned receipt file.
	anatomyMaxFileBytes = 16 << 20
	// topSyncSitesN caps the per-site ladder rendered into the syncs detail.
	topSyncSitesN = 5
	// maxDetailPaths caps the paths named in a signal detail.
	maxDetailPaths = 5
)

// anatomySyncSite is one (site, stage) host-sync cell per decode step.
type anatomySyncSite struct {
	site, stage   string
	syncsPerToken float64
}

// anatomyRecord is the per-record witness set extracted from one receipt or
// profile record. Nil pointers mean unmeasured, never zero.
type anatomyRecord struct {
	path             string
	model, backend   string
	launchesPerToken *float64
	syncsPerToken    *float64
	barriersElidedPT *float64
	barriersPerToken *float64
	hazardsArmed     bool
	replayDegraded   uint64
	usWitness        bool
	profileOwed      bool
	claimRow         bool
	oomSignature     bool
	regime           regime.Regime
	syncSites        []anatomySyncSite
}

// capturePin is one version-controlled pin on the capture-eligibility
// predicates: Pattern must occur in the enclosing function's source text (or
// anywhere in the file when Func is empty). Verification re-checks every pin
// against the live tree on each run — a vanished guard is DRIFT and fails.
type capturePin struct {
	File    string
	Func    string
	Pattern string
	Note    string
}

// captureEligibilityPins is the closed pin set on
// internal/model/cuda_graph_eligibility.go: the static predicate, the
// session-ready guards (KV-grow, QSA threshold, session failure), the hybrid
// branch, and the stable no-capture-backend vocabulary.
var captureEligibilityPins = []capturePin{
	{File: "internal/model/cuda_graph_eligibility.go",
		Pattern: "func cudaGraphCaptureEligible(", Note: "static capture predicate"},
	{File: "internal/model/cuda_graph_eligibility.go", Func: "cudaGraphCaptureReady",
		Pattern: "GraphCaptureReady()", Note: "KV-grow guard: the KV store must hold the fixed graph capacity"},
	{File: "internal/model/cuda_graph_eligibility.go", Func: "cudaGraphCaptureReady",
		Pattern: "QSADynamicGatingThreshold", Note: "QSA sparse-gather threshold guard (that gather reads the host)"},
	{File: "internal/model/cuda_graph_eligibility.go", Func: "cudaGraphCaptureReady",
		Pattern: "halFailure", Note: "session-failure guard"},
	{File: "internal/model/cuda_graph_eligibility.go",
		Pattern: "qwen35HybridGraphEligible", Note: "hybrid-model eligibility branch"},
	{File: "internal/model/cuda_graph_eligibility.go",
		Pattern: "CUDAGraphNoCaptureBackend", Note: "stable no-capture-backend verdict vocabulary"},
}

// verifyCaptureEligibility re-checks every pin against the live tree (reusing
// the glue-census go/ast scanner) and returns one finding per broken pin.
func verifyCaptureEligibility(root string) []string {
	sc, _ := scanGoFile(root, "internal/model/cuda_graph_eligibility.go")
	if sc.src == nil {
		return []string{"internal/model/cuda_graph_eligibility.go: file-missing"}
	}
	if sc.parseErr != nil {
		return []string{"internal/model/cuda_graph_eligibility.go: parse-error"}
	}
	var out []string
	for _, p := range captureEligibilityPins {
		var text string
		switch {
		case p.Func == "":
			text = string(sc.src)
		default:
			ft, ok := sc.funcText(p.Func)
			if !ok {
				out = append(out, fmt.Sprintf("%s[%s]: func-missing (%s)", p.File, p.Pattern, p.Note))
				continue
			}
			text = ft
		}
		if strings.Count(text, p.Pattern) == 0 {
			out = append(out, fmt.Sprintf("%s[%s]: pattern-absent (%s)", p.File, p.Pattern, p.Note))
		}
	}
	return out
}

// defaultLaunchesPerTokenCap calibrates the hard cap from the launch-census
// math: the looped-variant total for the documented glm-dsa shape is the
// static upper bound of the closed dispatch inventory, so a runtime
// observation above it means launch sites exist outside the inventory.
func defaultLaunchesPerTokenCap() int {
	shape, ok := launchCensusPresets["glm-dsa"]
	if !ok {
		return 0
	}
	normed, err := shape.normalize()
	if err != nil {
		return 0
	}
	return computeLaunches(launchCensusInventory, normed, variantPerExpert)
}

// --- record extraction ------------------------------------------------------

// scanAnatomyRecords walks dir (or a single receipt file) for *.json/*.jsonl
// records and returns their extracted anatomy, sorted by path. Malformed lines
// are skipped (a shared receipts tree routinely holds peers' half-written
// files); I/O errors fail the scan.
func scanAnatomyRecords(dir string) ([]anatomyRecord, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	var paths []string
	if !info.IsDir() {
		paths = []string{dir}
	} else {
		walkErr := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			ext := strings.ToLower(filepath.Ext(p))
			if ext != ".json" && ext != ".jsonl" {
				return nil
			}
			paths = append(paths, p)
			return nil
		})
		if walkErr != nil {
			return nil, walkErr
		}
		sort.Strings(paths)
	}
	var out []anatomyRecord
	for _, p := range paths {
		recs, err := readAnatomyRecords(p)
		if err != nil {
			return nil, err
		}
		out = append(out, recs...)
	}
	return out, nil
}

// readAnatomyRecords parses one file into zero or more records: a single JSON
// object is one record; a JSONL file contributes one record per parseable
// line. raw keeps the exact bytes the record came from (the regime reader and
// the OOM-signature scan work on them).
func readAnatomyRecords(path string) ([]anatomyRecord, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) > anatomyMaxFileBytes {
		return nil, fmt.Errorf("decode-anatomy: %s exceeds %d bytes", path, anatomyMaxFileBytes)
	}
	var out []anatomyRecord
	if rec, ok := parseJSONObject(data); ok {
		out = append(out, extractAnatomyRecord(path, rec, data))
		return out, nil
	}
	for _, line := range splitJSONLLines(data) {
		rec, ok := parseJSONObject(line)
		if !ok {
			continue
		}
		out = append(out, extractAnatomyRecord(path, rec, line))
	}
	return out, nil
}

func parseJSONObject(data []byte) (map[string]any, bool) {
	var rec map[string]any
	if err := json.Unmarshal(data, &rec); err != nil || rec == nil {
		return nil, false
	}
	return rec, true
}

func splitJSONLLines(data []byte) [][]byte {
	var lines [][]byte
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, []byte(line))
		}
	}
	return lines
}

// extractAnatomyRecord pulls every gate witness out of one parsed record.
// Absent witnesses stay nil/false — the gate never fabricates a zero.
func extractAnatomyRecord(path string, rec map[string]any, raw []byte) anatomyRecord {
	r := anatomyRecord{path: path}
	r.model = strAny(rec, "model")
	if r.model == "" {
		r.model = strAny(rec, "envelope_id")
	}
	r.backend = strAny(rec, "backend")
	native, _ := rec["native_timing"].(map[string]any)
	kernels, _ := nestedObject(rec, native, "kernels")

	// launches/token: an explicit kernel_launch_count witness wins; otherwise
	// the kernel profile's selected-kernel calls_per_token sum IS the launch
	// cadence (every row is one kernel x group shape invoked per token).
	if v, ok := explicitLaunchesPerToken(rec); ok {
		r.launchesPerToken = &v
	} else if v, ok := kernelCallsPerToken(kernels); ok {
		r.launchesPerToken = &v
	}
	// syncs/token: the gateway record's measured rate wins; the v3 receipt's
	// sync_roundtrips over decode_tokens is next; a bare top-level field last.
	if v, ok := numAny(native, "syncs_per_token", nil); ok {
		r.syncsPerToken = &v
	} else if v, ok := v3SyncsPerToken(rec); ok {
		r.syncsPerToken = &v
	} else if v, ok := numAny(rec, "syncs_per_token", nil); ok {
		r.syncsPerToken = &v
	}
	// barrier elision rate: the kernel profile's whole-window elided count
	// over its decode steps.
	if be, ok := numAny(kernels, "barriers_elided", nil); ok {
		if steps, ok2 := numAny(kernels, "decode_steps", nil); ok2 && steps > 0 {
			v := be / steps
			r.barriersElidedPT = &v
		}
	}
	for _, key := range []string{"barriers_per_token", "barriers_issued_per_token"} {
		if v, ok := numAny(rec, key, nil); ok {
			r.barriersPerToken = &v
			break
		}
		if v, ok := numAny(native, key, nil); ok {
			r.barriersPerToken = &v
			break
		}
	}
	r.hazardsArmed = boolAny(rec, "hazards_armed") || boolAny(native, "hazards_armed")
	if d, ok := degradedReplayTokens(rec); ok {
		r.replayDegraded = d
	}
	r.usWitness = hasUsWitness(rec, native, kernels)
	r.profileOwed = owesUsWitness(rec, native, kernels)
	r.claimRow = isClaimRow(rec, native)
	low := strings.ToLower(string(raw))
	r.oomSignature = strings.Contains(low, "instantiate") && strings.Contains(low, "out of memory")
	if p, err := regime.DecodeProfile(raw); err == nil {
		r.regime = regime.Classify(p).Regime
	} else {
		r.regime = regime.RegimeUnknown
	}
	if sites, _ := nestedSlice(native, "sync_sites"); sites != nil {
		for _, sv := range sites {
			sm, _ := sv.(map[string]any)
			if sm == nil {
				continue
			}
			sp, ok := numAny(sm, "syncs_per_token", nil)
			if !ok {
				continue
			}
			r.syncSites = append(r.syncSites, anatomySyncSite{site: strAny(sm, "site"), stage: strAny(sm, "stage"), syncsPerToken: sp})
		}
	}
	return r
}

// nestedObject finds key under native first, then at the record top level.
func nestedObject(rec, native map[string]any, key string) (map[string]any, bool) {
	if m, ok := native[key].(map[string]any); ok {
		return m, true
	}
	m, ok := rec[key].(map[string]any)
	return m, ok
}

func nestedSlice(m map[string]any, key string) ([]any, bool) {
	s, ok := m[key].([]any)
	return s, ok
}

func strAny(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	s, _ := m[key].(string)
	return strings.TrimSpace(s)
}

// numAny reads a finite numeric field; when dst is non-nil it also stores the
// value there. Absent or non-finite fields leave dst untouched.
func numAny(m map[string]any, key string, dst *float64) (float64, bool) {
	if m == nil {
		return 0, false
	}
	v, ok := m[key].(float64)
	if !ok || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	if dst != nil {
		*dst = v
	}
	return v, true
}

func boolAny(m map[string]any, key string) bool {
	if m == nil {
		return false
	}
	b, _ := m[key].(bool)
	return b
}

// controlsDecodeTokens resolves the workload's decode-token count: the v3
// receipt carries it under controls; ad-hoc records may carry it top-level.
func controlsDecodeTokens(rec map[string]any) (float64, bool) {
	if c, ok := rec["controls"].(map[string]any); ok {
		if v, ok2 := numAny(c, "decode_tokens", nil); ok2 && v > 0 {
			return v, true
		}
	}
	return numAny(rec, "decode_tokens", nil)
}

// explicitLaunchesPerToken reads the recorded kernel_launch_count witness:
// averaged over the repetitions that record it, over the workload's decode
// tokens; a bare top-level pair is the single-record form.
func explicitLaunchesPerToken(rec map[string]any) (float64, bool) {
	toks, ok := controlsDecodeTokens(rec)
	if !ok {
		return 0, false
	}
	reps, _ := rec["repetitions"].([]any)
	if len(reps) > 0 {
		var sum float64
		n := 0
		for _, rv := range reps {
			rm, _ := rv.(map[string]any)
			if rm == nil {
				continue
			}
			lc, ok := numAny(rm, "kernel_launch_count", nil)
			if !ok {
				continue
			}
			sum += lc
			n++
		}
		if n == 0 {
			return 0, false
		}
		return sum / float64(n) / toks, true
	}
	lc, ok := numAny(rec, "kernel_launch_count", nil)
	if !ok {
		return 0, false
	}
	return lc / toks, true
}

// kernelCallsPerToken sums the selected-kernel calls_per_token rows: the
// kernel invocations per decoded token the timestamp profile witnessed.
func kernelCallsPerToken(kernels map[string]any) (float64, bool) {
	rows, ok := nestedSlice(kernels, "selected_kernels")
	if !ok {
		return 0, false
	}
	var sum float64
	seen := false
	for _, rv := range rows {
		rm, _ := rv.(map[string]any)
		if rm == nil {
			continue
		}
		if c, ok := numAny(rm, "calls_per_token", nil); ok {
			sum += c
			seen = true
		}
	}
	if !seen || sum <= 0 {
		return 0, false
	}
	return sum, true
}

// v3SyncsPerToken reads the v3 receipt's sync_roundtrips over decode tokens,
// averaged over the repetitions that record them.
func v3SyncsPerToken(rec map[string]any) (float64, bool) {
	toks, ok := controlsDecodeTokens(rec)
	if !ok {
		return 0, false
	}
	reps, _ := rec["repetitions"].([]any)
	var sum float64
	n := 0
	for _, rv := range reps {
		rm, _ := rv.(map[string]any)
		if rm == nil {
			continue
		}
		sr, ok := numAny(rm, "sync_roundtrips", nil)
		if !ok {
			continue
		}
		sum += sr
		n++
	}
	if n == 0 {
		return 0, false
	}
	return sum / float64(n) / toks, true
}

// degradedReplayTokens reports the P3 engagement witness: tokens served by
// the per-layer host loop on a capture-once-capable backend (the v3 receipt's
// graph_replay_degraded_tokens), top-level or maxed over the repetitions.
func degradedReplayTokens(rec map[string]any) (uint64, bool) {
	if v, ok := rec["graph_replay_degraded_tokens"].(float64); ok && v > 0 {
		return uint64(v), true
	}
	var best uint64
	found := false
	reps, _ := rec["repetitions"].([]any)
	for _, rv := range reps {
		rm, _ := rv.(map[string]any)
		if rm == nil {
			continue
		}
		if v, ok := rm["graph_replay_degraded_tokens"].(float64); ok && v > 0 {
			d := uint64(v)
			if d > best {
				best = d
			}
			found = true
		}
	}
	return best, found
}

// hasUsWitness reports whether the record carries a µs-scale witness for its
// decode claim: a complete kernel profile with positive per-token GPU time,
// the inter-kernel gap/execution us pair, or the v3 host_phase_us
// decomposition. Everything else is an ESTIMATED band.
func hasUsWitness(rec, native, kernels map[string]any) bool {
	if kernels != nil {
		complete, _ := kernels["complete"].(bool)
		if g, ok := numAny(kernels, "gpu_ms_per_token", nil); ok && complete && g > 0 {
			return true
		}
	}
	if gap, ok := numAny(rec, "inter_kernel_gap_sum_us", nil); ok && gap > 0 {
		if ex, ok2 := numAny(rec, "kernel_execution_sum_us", nil); ok2 && ex > 0 {
			return true
		}
	}
	if h, ok := numAny(rec, "host_phase_us", nil); ok && h > 0 {
		return true
	}
	if h, ok := numAny(native, "host_phase_us", nil); ok && h > 0 {
		return true
	}
	reps, _ := rec["repetitions"].([]any)
	for _, rv := range reps {
		rm, _ := rv.(map[string]any)
		if rm == nil {
			continue
		}
		if h, ok := numAny(rm, "host_phase_us", nil); ok && h > 0 {
			return true
		}
	}
	return false
}

// owesUsWitness reports whether the record presents profiling evidence that
// must be complete to back its claim: a kernel-profile section is present
// (kernels object / selected_kernels / kernel_profile_source), or it is a v3
// native performance receipt with repetitions (whose host-overhead
// decomposition is part of the schema promise). Rows with no profile section
// at all are not flagged by this signal — their ms-scale ledger fields stay
// visible through the other five signals and the regime column. This keeps
// committed ABBA control arms (which deliberately carry no structured
// profile) out of the ESTIMATED-band failure while still catching a profile
// that was promised but landed empty or incomplete.
func owesUsWitness(rec, native, kernels map[string]any) bool {
	if kernels != nil {
		return true
	}
	if _, ok := rec["selected_kernels"]; ok {
		return true
	}
	if strAny(rec, "kernel_profile_source") != "" || strAny(native, "kernel_profile_source") != "" {
		return true
	}
	if schema, _ := rec["schema"].(string); strings.HasPrefix(schema, "fak-native-performance-receipt/") {
		if reps, _ := rec["repetitions"].([]any); len(reps) > 0 {
			return true
		}
	}
	return false
}

// isClaimRow reports whether the record makes a decode-performance claim: a
// closed set of per-rate/per-time fields, a v3 native performance receipt
// with repetitions, or a native_timing decode phase.
func isClaimRow(rec, native map[string]any) bool {
	for _, k := range []string{"decode_tps", "decode_ms", "ms_per_token", "wall_ms_per_token", "tokens_per_second", "decode_ms_per_token"} {
		if _, ok := rec[k]; ok {
			return true
		}
	}
	if schema, _ := rec["schema"].(string); strings.HasPrefix(schema, "fak-native-performance-receipt/") {
		if reps, _ := rec["repetitions"].([]any); len(reps) > 0 {
			return true
		}
	}
	if dm, ok := numAny(native, "decode_ms", nil); ok && dm > 0 {
		return true
	}
	return false
}

// --- signal computations ----------------------------------------------------

func fptr(v float64) *float64 { return &v }

// launchesSignal gates the observed worst-case launches/tok against the hard
// cap and, when a snapshot baseline exists, against the drift tolerance.
func launchesSignal(obs *float64, cap int, baseline *float64, tol float64) AnatomySignal {
	s := AnatomySignal{ID: SignalLaunchesPerToken}
	c := float64(cap)
	s.Cap = &c
	if obs == nil {
		s.Status = AnatomyUnknown
		s.Detail = "no launch-count witness in scanned receipts (kernel_launch_count or selected_kernels calls_per_token)"
		return s
	}
	s.Value = obs
	var reasons []string
	if *obs > c {
		reasons = append(reasons, fmt.Sprintf("%.1f/tok exceeds hard cap %d", *obs, cap))
	}
	if baseline != nil && *baseline > 0 {
		limit := *baseline * (1 + tol)
		if *obs > limit {
			reasons = append(reasons, fmt.Sprintf("%.1f/tok drifted above snapshot baseline %.1f (+%.0f%% tolerance)", *obs, *baseline, tol*100))
		}
	}
	if len(reasons) > 0 {
		s.Status = AnatomyFail
		s.Detail = strings.Join(reasons, "; ")
	} else {
		s.Status = AnatomyPass
		s.Detail = fmt.Sprintf("%.1f/tok at or under cap %d", *obs, cap)
	}
	return s
}

// syncsSignal gates the observed worst-case syncs/tok against the cap and
// renders the per-site top-N ladder into the detail.
func syncsSignal(obs *float64, cap float64, sites []anatomySyncSite) AnatomySignal {
	s := AnatomySignal{ID: SignalSyncsPerToken}
	c := cap
	s.Cap = &c
	if obs == nil {
		s.Status = AnatomyUnknown
		s.Detail = "no syncs-per-token witness in scanned receipts"
		return s
	}
	s.Value = obs
	siteDetail := ""
	if top := topNSyncSites(sites, topSyncSitesN); len(top) > 0 {
		parts := make([]string, 0, len(top))
		for _, t := range top {
			parts = append(parts, fmt.Sprintf("%s/%s=%.1f", t.site, t.stage, t.syncsPerToken))
		}
		siteDetail = "; top sites: " + strings.Join(parts, ", ")
	}
	if *obs > cap {
		s.Status = AnatomyFail
		s.Detail = fmt.Sprintf("%.1f syncs/tok exceeds cap %.1f%s", *obs, cap, siteDetail)
	} else {
		s.Status = AnatomyPass
		s.Detail = fmt.Sprintf("%.1f syncs/tok at or under cap %.1f%s", *obs, cap, siteDetail)
	}
	return s
}

// topNSyncSites returns the N busiest (site, stage) cells, busiest first,
// ties broken lexicographically so the ladder is deterministic.
func topNSyncSites(sites []anatomySyncSite, n int) []anatomySyncSite {
	out := append([]anatomySyncSite(nil), sites...)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].syncsPerToken != out[j].syncsPerToken {
			return out[i].syncsPerToken > out[j].syncsPerToken
		}
		if out[i].site != out[j].site {
			return out[i].site < out[j].site
		}
		return out[i].stage < out[j].stage
	})
	if len(out) > n {
		out = out[:n]
	}
	return out
}

// barriersSignal is the guard-drift assert: hazards armed with barrier
// activity but zero elided barriers means the elision counter is inert.
// Without a declared arming state the signal cannot judge and stays unknown.
func barriersSignal(armed bool, elided, barriers *float64) AnatomySignal {
	s := AnatomySignal{ID: SignalBarriersElidedWhileArmed}
	if elided != nil {
		s.Value = elided
	}
	switch {
	case !armed:
		s.Status = AnatomyUnknown
		s.Detail = "hazard arming state not declared (pass --hazards-armed or record hazards_armed=true); armed-but-zero-elided cannot be judged"
	case barriers == nil || *barriers <= 0:
		s.Status = AnatomyUnknown
		s.Detail = "hazards armed but no barriers-per-token witness recorded; nothing to elide was observed"
	case elided == nil || *elided == 0:
		s.Status = AnatomyFail
		s.Detail = fmt.Sprintf("hazards armed with %.1f barriers/tok but 0 elided/tok: the elision counter is inert (guard drift)", *barriers)
	default:
		s.Status = AnatomyPass
		s.Detail = fmt.Sprintf("hazards armed: %.1f barriers/tok, %.1f elided/tok", *barriers, *elided)
	}
	return s
}

// profileSignal fails when any decode-performance claim row presents
// profiling evidence that is missing or incomplete (fields empty /
// Complete=false / ESTIMATED band).
func profileSignal(claims, missing int, missingPaths []string) AnatomySignal {
	s := AnatomySignal{ID: SignalProfileReceiptMissing}
	s.Cap = fptr(0)
	if claims == 0 {
		s.Status = AnatomyUnknown
		s.Detail = "no decode-performance claim rows found in scanned receipts"
		return s
	}
	s.Value = fptr(float64(missing))
	if missing > 0 {
		s.Status = AnatomyFail
		paths := missingPaths
		if len(paths) > maxDetailPaths {
			paths = paths[:maxDetailPaths]
		}
		s.Detail = fmt.Sprintf("%d of %d claim row(s) lack a µs witness (ESTIMATED band): %s", missing, claims, strings.Join(paths, ", "))
	} else {
		s.Status = AnatomyPass
		s.Detail = fmt.Sprintf("%d claim row(s), none with promised-but-unmet profiling evidence", claims)
	}
	return s
}

// captureDriftSignal fails on any source-pin drift, Instantiate-OOM signature,
// or degraded-replay witness.
func captureDriftSignal(findings []string) AnatomySignal {
	s := AnatomySignal{ID: SignalCaptureEligibilityDrift}
	s.Cap = fptr(0)
	s.Value = fptr(float64(len(findings)))
	if len(findings) > 0 {
		s.Status = AnatomyFail
		list := findings
		if len(list) > maxDetailPaths {
			list = list[:maxDetailPaths]
		}
		s.Detail = fmt.Sprintf("%d finding(s): %s", len(findings), strings.Join(list, "; "))
	} else {
		s.Status = AnatomyPass
		s.Detail = "capture-eligibility pins intact; no Instantiate-OOM or degraded-replay witness"
	}
	return s
}

// hostShareSignal applies the existing 0.30 line through the internal/hostshare
// reader: the most recent reading decides; UNKNOWN never fails.
func hostShareSignal(gated hostshare.Reading, exceeded, ok bool, threshold float64) AnatomySignal {
	s := AnatomySignal{ID: SignalHostShare}
	s.Cap = fptr(threshold)
	if !ok || gated.HostShare == nil {
		s.Status = AnatomyUnknown
		s.Detail = "hostshare reader found no device-busy evidence in scanned receipts (UNKNOWN, never a fabricated zero)"
		return s
	}
	s.Value = gated.HostShare
	if exceeded {
		s.Status = AnatomyFail
		s.Detail = fmt.Sprintf("most recent reading %s host_share=%.3f (%s) exceeds the %.2f line -> LAUNCH-BOUND", gated.Path, *gated.HostShare, gated.Source, threshold)
	} else {
		s.Status = AnatomyPass
		s.Detail = fmt.Sprintf("most recent reading %s host_share=%.3f (%s) at or under the %.2f line", gated.Path, *gated.HostShare, gated.Source, threshold)
	}
	return s
}

// --- runner -----------------------------------------------------------------

var decodeAnatomyUsage = "usage: fak-dev decode-anatomy [--receipts DIR] [--root DIR] [--json] [--fail-on SIG[,SIG...]] [--snapshot PATH]\n" +
	"                               [--launches-cap N] [--launches-drift F] [--syncs-cap F] [--threshold F] [--hazards-armed]\n" +
	"\n" +
	"Runtime launch-hygiene gate over per-token decode anatomy. Scans receipt/profile\n" +
	"JSON under --receipts (default docs/benchmarks/receipts), extracts the per-token\n" +
	"anatomy, and emits six gate signals:\n" +
	"\n" +
	"  launches_per_token          hard cap (calibrated from the launch-census math) + drift vs --snapshot baseline\n" +
	"  syncs_per_token             cap + per-site top-N ladder\n" +
	"  barriers_elided_while_armed armed-but-zero-elided guard-drift assert\n" +
	"  profile_receipt_missing     claim rows whose promised profile evidence is missing/incomplete (ESTIMATED band)\n" +
	"  capture_eligibility_drift   cuda_graph_eligibility.go pins + Instantiate-OOM / degraded-replay witnesses\n" +
	"  host_share                  the existing 0.30 line via the internal/hostshare reader\n" +
	"\n" +
	"Flags:\n" +
	"  --receipts DIR      receipt/profile-note JSON dir (or a single file) to scan\n" +
	"  --root DIR          repository root for the capture-eligibility source scan (default: auto-discovered)\n" +
	"  --json              emit the pinned fak.decode-anatomy/v1 document\n" +
	"  --fail-on SIG       restrict gating to the named signal(s), comma-separated\n" +
	"                      (closed vocabulary: " + strings.Join(decodeAnatomySignalIDs, ", ") + ")\n" +
	"  --snapshot PATH     write the machine-readable snapshot; when a prior snapshot\n" +
	"                      exists at PATH its launches_per_token value is the drift baseline\n" +
	"  --launches-cap N    override the calibrated launches/tok hard cap\n" +
	"  --launches-drift F  relative drift tolerance vs the snapshot baseline (default " + fmt.Sprintf("%.2f", defaultLaunchesDriftTol) + ")\n" +
	"  --syncs-cap F       syncs/tok cap (default " + fmt.Sprintf("%.0f", defaultSyncsPerTokenCap) + ")\n" +
	"  --threshold F       host_share line (default " + fmt.Sprintf("%.2f", defaultHostShareLine) + ")\n" +
	"  --hazards-armed     declare that the scanned run had batch hazards armed\n" +
	"\n" +
	"Exit codes: 0 PASS, 1 gate FAIL, 2 usage/IO error."

// RunDecodeAnatomy implements the fak-dev decode-anatomy verb.
func RunDecodeAnatomy(stdout, stderr io.Writer, argv []string) int {
	root := repoRoot()
	receiptsDir := "docs/benchmarks/receipts"
	asJSON := false
	failOn := ""
	snapshot := ""
	launchesCap := -1
	launchesDrift := -1.0
	syncsCap := -1.0
	threshold := -1.0
	hazardsArmed := false
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			fmt.Fprintln(stdout, decodeAnatomyUsage)
			return 0
		case arg == "--root":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "decode-anatomy: --root needs a value")
				return 2
			}
			i++
			root = argv[i]
		case strings.HasPrefix(arg, "--root="):
			root = strings.TrimPrefix(arg, "--root=")
		case arg == "--receipts":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "decode-anatomy: --receipts needs a value")
				return 2
			}
			i++
			receiptsDir = argv[i]
		case strings.HasPrefix(arg, "--receipts="):
			receiptsDir = strings.TrimPrefix(arg, "--receipts=")
		case arg == "--json":
			asJSON = true
		case arg == "--fail-on":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "decode-anatomy: --fail-on needs a value")
				return 2
			}
			i++
			failOn = argv[i]
		case strings.HasPrefix(arg, "--fail-on="):
			failOn = strings.TrimPrefix(arg, "--fail-on=")
		case arg == "--snapshot":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "decode-anatomy: --snapshot needs a value")
				return 2
			}
			i++
			snapshot = argv[i]
		case strings.HasPrefix(arg, "--snapshot="):
			snapshot = strings.TrimPrefix(arg, "--snapshot=")
		case arg == "--launches-cap":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "decode-anatomy: --launches-cap needs a value")
				return 2
			}
			i++
			v, err := strconv.Atoi(argv[i])
			if err != nil || v <= 0 {
				fmt.Fprintln(stderr, "decode-anatomy: --launches-cap needs a positive integer")
				return 2
			}
			launchesCap = v
		case strings.HasPrefix(arg, "--launches-cap="):
			v, err := strconv.Atoi(strings.TrimPrefix(arg, "--launches-cap="))
			if err != nil || v <= 0 {
				fmt.Fprintln(stderr, "decode-anatomy: --launches-cap needs a positive integer")
				return 2
			}
			launchesCap = v
		case arg == "--launches-drift":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "decode-anatomy: --launches-drift needs a value")
				return 2
			}
			i++
			v, err := strconv.ParseFloat(argv[i], 64)
			if err != nil || v < 0 {
				fmt.Fprintln(stderr, "decode-anatomy: --launches-drift needs a non-negative number")
				return 2
			}
			launchesDrift = v
		case strings.HasPrefix(arg, "--launches-drift="):
			v, err := strconv.ParseFloat(strings.TrimPrefix(arg, "--launches-drift="), 64)
			if err != nil || v < 0 {
				fmt.Fprintln(stderr, "decode-anatomy: --launches-drift needs a non-negative number")
				return 2
			}
			launchesDrift = v
		case arg == "--syncs-cap":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "decode-anatomy: --syncs-cap needs a value")
				return 2
			}
			i++
			v, err := strconv.ParseFloat(argv[i], 64)
			if err != nil || v <= 0 {
				fmt.Fprintln(stderr, "decode-anatomy: --syncs-cap needs a positive number")
				return 2
			}
			syncsCap = v
		case strings.HasPrefix(arg, "--syncs-cap="):
			v, err := strconv.ParseFloat(strings.TrimPrefix(arg, "--syncs-cap="), 64)
			if err != nil || v <= 0 {
				fmt.Fprintln(stderr, "decode-anatomy: --syncs-cap needs a positive number")
				return 2
			}
			syncsCap = v
		case arg == "--threshold":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "decode-anatomy: --threshold needs a value")
				return 2
			}
			i++
			v, err := strconv.ParseFloat(argv[i], 64)
			if err != nil || v < 0 || v > 1 {
				fmt.Fprintln(stderr, "decode-anatomy: --threshold must be in [0,1]")
				return 2
			}
			threshold = v
		case strings.HasPrefix(arg, "--threshold="):
			v, err := strconv.ParseFloat(strings.TrimPrefix(arg, "--threshold="), 64)
			if err != nil || v < 0 || v > 1 {
				fmt.Fprintln(stderr, "decode-anatomy: --threshold must be in [0,1]")
				return 2
			}
			threshold = v
		case arg == "--hazards-armed":
			hazardsArmed = true
		default:
			fmt.Fprintln(stderr, decodeAnatomyUsage)
			return 2
		}
	}
	if launchesCap < 0 {
		launchesCap = defaultLaunchesPerTokenCap()
	}
	if launchesDrift < 0 {
		launchesDrift = defaultLaunchesDriftTol
	}
	if syncsCap < 0 {
		syncsCap = defaultSyncsPerTokenCap
	}
	if threshold < 0 {
		threshold = defaultHostShareLine
	}

	active := map[string]bool{}
	if failOn == "" {
		for _, id := range decodeAnatomySignalIDs {
			active[id] = true
		}
	} else {
		for _, tok := range strings.Split(failOn, ",") {
			tok = strings.TrimSpace(tok)
			if tok == "" {
				continue
			}
			if !knownSignal(tok) {
				fmt.Fprintf(stderr, "decode-anatomy: --fail-on supports only %s (got %q)\n", strings.Join(decodeAnatomySignalIDs, ", "), tok)
				return 2
			}
			active[tok] = true
		}
		if len(active) == 0 {
			fmt.Fprintln(stderr, "decode-anatomy: --fail-on names no signal")
			return 2
		}
	}

	recs, err := scanAnatomyRecords(receiptsDir)
	if err != nil {
		if os.IsNotExist(err) {
			fmt.Fprintf(stderr, "decode-anatomy: receipts dir %q does not exist (pass --receipts DIR)\n", receiptsDir)
		} else {
			fmt.Fprintf(stderr, "decode-anatomy: %v\n", err)
		}
		return 2
	}

	// Snapshot baseline: a prior snapshot at the same path pins the launches
	// trend line (read before the new snapshot overwrites it).
	var baseline *float64
	if snapshot != "" {
		baseline = snapshotLaunchesBaseline(snapshot)
	}

	// Aggregate the per-record witnesses.
	var (
		maxLaunches, maxSyncs, maxElided, maxBarriers *float64
		sites                                         []anatomySyncSite
		claims, missing                               int
		missingPaths                                  []string
		armed                                         = hazardsArmed
	)
	for _, r := range recs {
		if v := r.launchesPerToken; v != nil && (maxLaunches == nil || *v > *maxLaunches) {
			maxLaunches = v
		}
		if v := r.syncsPerToken; v != nil && (maxSyncs == nil || *v > *maxSyncs) {
			maxSyncs = v
		}
		if v := r.barriersElidedPT; v != nil && (maxElided == nil || *v > *maxElided) {
			maxElided = v
		}
		if v := r.barriersPerToken; v != nil && (maxBarriers == nil || *v > *maxBarriers) {
			maxBarriers = v
		}
		sites = append(sites, r.syncSites...)
		if r.claimRow {
			claims++
			if r.profileOwed && !r.usWitness {
				missing++
				missingPaths = append(missingPaths, r.path)
			}
		}
		if r.hazardsArmed {
			armed = true
		}
	}

	// Capture-eligibility drift: source pins plus the receipt-side witnesses.
	findings := verifyCaptureEligibility(root)
	for _, r := range recs {
		if r.oomSignature {
			findings = append(findings, fmt.Sprintf("%s: Instantiate-OOM signature (graph instantiate failed at VRAM fit)", r.path))
		}
		if r.replayDegraded > 0 {
			findings = append(findings, fmt.Sprintf("%s: graph replay degraded %d token(s) to the per-layer host loop", r.path, r.replayDegraded))
		}
	}

	// host_share through the reused internal/hostshare reader.
	rep, err := hostshare.ScanReport([]string{receiptsDir})
	if err != nil {
		fmt.Fprintf(stderr, "decode-anatomy: hostshare scan: %v\n", err)
		return 2
	}
	gated, exceeded, hsOK := hostshare.Evaluate(rep.Readings, threshold)

	signals := []AnatomySignal{
		launchesSignal(maxLaunches, launchesCap, baseline, launchesDrift),
		syncsSignal(maxSyncs, syncsCap, sites),
		barriersSignal(armed, maxElided, maxBarriers),
		profileSignal(claims, missing, missingPaths),
		captureDriftSignal(findings),
		hostShareSignal(gated, exceeded, hsOK, threshold),
	}

	result := DecodeAnatomyResult{
		Schema:     DecodeAnatomySchema,
		ObservedAt: time.Now().UTC().Format(time.RFC3339),
		Verdict:    "pass",
	}
	for _, r := range recs {
		row := AnatomyReceipt{Path: r.path, Model: r.model, Backend: r.backend, OOMSignature: r.oomSignature}
		if r.regime != regime.RegimeUnknown {
			row.Regime = string(r.regime)
		}
		row.LaunchesPerToken = r.launchesPerToken
		row.SyncsPerToken = r.syncsPerToken
		row.BarriersElidedPerToken = r.barriersElidedPT
		result.Receipts = append(result.Receipts, row)
	}
	result.Signals = signals
	for _, sig := range signals {
		if sig.Status == AnatomyFail && active[sig.ID] {
			result.Verdict = "fail"
			result.Failures = append(result.Failures, sig.ID)
		}
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "decode-anatomy: encode: %v\n", err)
			return 2
		}
	} else {
		writeDecodeAnatomyText(stdout, &result, receiptsDir, claims, missing)
	}

	if snapshot != "" {
		body, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "decode-anatomy: marshal snapshot: %v\n", err)
			return 2
		}
		if err := os.MkdirAll(filepath.Dir(snapshot), 0o755); err != nil {
			fmt.Fprintf(stderr, "decode-anatomy: mkdir snapshot dir: %v\n", err)
			return 2
		}
		if err := os.WriteFile(snapshot, append(body, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "decode-anatomy: write snapshot: %v\n", err)
			return 2
		}
		if !asJSON {
			fmt.Fprintf(stdout, "snapshot written: %s\n", snapshot)
		}
	}

	if result.Verdict == "fail" {
		return 1
	}
	return 0
}

func knownSignal(id string) bool {
	for _, s := range decodeAnatomySignalIDs {
		if s == id {
			return true
		}
	}
	return false
}

// snapshotLaunchesBaseline reads the prior snapshot's launches_per_token value
// (nil when absent or unreadable — a bad prior snapshot never blocks a run).
func snapshotLaunchesBaseline(path string) *float64 {
	body, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var prior DecodeAnatomyResult
	if json.Unmarshal(body, &prior) != nil {
		return nil
	}
	for _, s := range prior.Signals {
		if s.ID == SignalLaunchesPerToken {
			return s.Value
		}
	}
	return nil
}

func writeDecodeAnatomyText(w io.Writer, r *DecodeAnatomyResult, receiptsDir string, claims, missing int) {
	fmt.Fprintf(w, "DECODE-ANATOMY receipts=%s observed_at=%s\n", receiptsDir, r.ObservedAt)
	fmt.Fprintf(w, "%-28s %-10s %-10s %-8s %s\n", "signal", "value", "cap", "status", "detail")
	for _, s := range r.Signals {
		val, capS := "null", "null"
		if s.Value != nil {
			val = fmt.Sprintf("%.2f", *s.Value)
		}
		if s.Cap != nil {
			capS = fmt.Sprintf("%.2f", *s.Cap)
		}
		fmt.Fprintf(w, "%-28s %-10s %-10s %-8s %s\n", s.ID, val, capS, s.Status, s.Detail)
	}
	fmt.Fprintf(w, "receipts: %d record(s), %d claim row(s), %d unwitnessed\n", len(r.Receipts), claims, missing)
	if r.Verdict == "fail" {
		fmt.Fprintf(w, "verdict: FAIL (%s)\n", strings.Join(r.Failures, ", "))
	} else {
		fmt.Fprintln(w, "verdict: PASS")
	}
}
