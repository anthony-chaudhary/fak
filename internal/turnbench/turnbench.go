// Package turnbench is the TURN-TAX benchmark — the dedicated A/B that isolates
// the one cost the microsecond-transport bench (internal/bench) does NOT measure:
// the EXTRA MODEL TURN a SOTA agent loop must fire when a tool call fails its
// first attempt and comes back as an error code, versus fak's "1-shot" path,
// where the kernel resolves the very same condition INSIDE the syscall the call
// arrived on — no second round-trip to the model.
//
// Why a separate benchmark. internal/bench answers "how cheap is one in-process
// adjudication vs a spawned hook?" (a sub-microsecond transport question, the
// call-mix-independent gate). It deliberately does NOT price the dominant cost of
// an agent loop, which is not the adjudication — it is the MODEL ROUND-TRIP. A
// frontier turn is ~10^9× the adjudication: seconds of latency, hundreds-to-
// thousands of tokens, real dollars. The thing worth measuring is therefore how
// many of those turns the kernel DELETES, and by which lever. That is this file.
//
// The honest construction (no self-grading). The fak side is REAL: we replay a
// frozen, class-labeled trace through the actual kernel (k.Syscall) and read the
// kernel's OWN counters — Transforms, VDSOHits, Quarantines, Denies. Nothing
// about our own savings is modeled. The BASELINE side is a transparent,
// configurable turn-cost model, and the saved turns split into two HONESTLY
// DIFFERENT kinds (do not blend them either):
//
//   - FORCED (grammar repair + tier-2 dedup): the baseline demonstrably pays the
//     round-trip — the model already issued the call. An aliased call ERRORS and a
//     SOTA loop re-prompts to fix it (+1 turn, the documented ReAct / function-
//     calling convention; we charge the conservative 1, not a flail loop); a
//     re-issued duplicate read is a wasted round-trip the model already chose to
//     make. fak resolves both in the same syscall, so the turn never fires.
//   - ELISION (tier-1 pure + tier-3 static): the model CALLED a tool whose result
//     needs no engine. The baseline pays a round-trip ONLY because the model chose
//     to make a call a stronger agent could have elided (compute 240+60 inline,
//     recall the airport list). fak serves it locally regardless. This is a
//     capability/optimization win, not a forced error-recovery turn — counted
//     toward the total but reported separately so the claim is not overstated.
//
// Either way the benchmark counts ONLY calls actually present in the trace: a
// model that never aliases yields Transforms=0, and one that never calls a pure
// tool yields no pure serves — so the cost model cannot over-count a turn the
// workload did not contain. The "1-shot" win is literal: the kernel produced the
// corrected/served result in the same syscall the call arrived on.
//
// Two axes, never blended (this is the integrity move). The HAPPY-PATH turn-tax
// (grammar repair + vDSO local serve) is reported separately from the SAFETY
// FLOOR (quarantine + deny). On a clean trace with no errors, dups, or poison the
// turn-tax is ZERO — and the benchmark ships a happy-path slice that proves it,
// so the headline cannot be inflated. The safety floor (a poisoned result kept
// out of context, a destructive op never dispatched) is the deterministic moat
// and is a *completion/integrity* delta, not a turn count.
//
// Relationship to the fleet-scale ROI (inline_tool_roi.py / frontier_sensitivity.py
// at the repo root): those model the GPU-side re-prefill tax of two-pass loops at
// fleet scale (the upper bound, "what if it saves 250 turns?"). This is the
// grounded floor: the turns the kernel verifiably eliminates on a concrete trace,
// per lever, with a cost model the reader can re-knob.
package turnbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

const (
	BenchmarkConceptVersion = "fak.benchmark-concept.v1"
	CostModelVersion        = "fak.cost-model.v1"
	FanoutCostModelVersion  = "fak.fanout-cost-model.v1"
)

// adjReps is the calibration-loop count for the in-process adjudication-latency
// measurement (the µs cost of the "1-shot"), mirroring internal/bench: an
// in-process Decide is faster than the OS clock granularity, so we time a batch
// and divide.
const adjReps = 2000

// Call is one trace entry. Class/Note are documentation + an assertion target
// (the EXPECTED class); the ACTUAL class is derived from the live kernel verdict,
// so a mislabeled fixture is caught, not trusted.
type Call struct {
	Tool  string            `json:"tool"`
	Args  json.RawMessage   `json:"args"`
	Meta  map[string]string `json:"meta,omitempty"`
	Class string            `json:"class,omitempty"`
	Note  string            `json:"note,omitempty"`
}

// Trace is a frozen, replayable, class-labeled tool-call slice.
type Trace struct {
	SliceID string `json:"slice_id"`
	Calls   []Call `json:"calls"`
}

// LoadTrace reads a trace JSON file.
func LoadTrace(path string) (*Trace, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var t Trace
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, err
	}
	return &t, nil
}

// WorkloadHash is a stable hash of the trace's calls (tool+args+meta, in order).
// The Class/Note documentation fields are deliberately excluded — relabeling a
// call's expected class does not change the workload the kernel actually runs.
func (t *Trace) WorkloadHash() string {
	h := sha256.New()
	h.Write([]byte(t.SliceID))
	for _, c := range t.Calls {
		h.Write([]byte(c.Tool))
		h.Write([]byte{0})
		h.Write(c.Args)
		h.Write([]byte{0})
		keys := make([]string, 0, len(c.Meta))
		for k := range c.Meta {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			h.Write([]byte(k + "=" + c.Meta[k] + ";"))
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// CostModel converts a saved model turn into tokens, dollars, and latency. Every
// field is a knob (the defaults are illustrative, not billed); the sensitivity
// grid sweeps the two that matter so a reader recomputes for their own stack.
type CostModel struct {
	Version                 string  `json:"version,omitempty"`
	PromptTokensPerTurn     int     `json:"prompt_tokens_per_turn"`
	CompletionTokensPerTurn int     `json:"completion_tokens_per_turn"`
	DollarsPerMTokIn        float64 `json:"dollars_per_mtok_in"`
	DollarsPerMTokOut       float64 `json:"dollars_per_mtok_out"`
	ModelTurnLatencyMs      float64 `json:"model_turn_latency_ms"`
}

// DefaultCostModel is a documented, conservative mid-range agent turn: a ~1200-tok
// growing prompt (system + tool schemas + history) and a ~120-tok tool-call
// completion, at the repo's blended $3/$15-per-Mtok rate, with a ~1.5s round-trip
// (a hosted "flash"-class model). All overridable on the CLI.
func DefaultCostModel() CostModel {
	return CostModel{
		Version:                 CostModelVersion,
		PromptTokensPerTurn:     1200,
		CompletionTokensPerTurn: 120,
		DollarsPerMTokIn:        3.0,
		DollarsPerMTokOut:       15.0,
		ModelTurnLatencyMs:      1500,
	}
}

func (cm CostModel) tokensPerTurn() int { return cm.PromptTokensPerTurn + cm.CompletionTokensPerTurn }

func (cm CostModel) dollarsPerTurn() float64 {
	return float64(cm.PromptTokensPerTurn)/1e6*cm.DollarsPerMTokIn +
		float64(cm.CompletionTokensPerTurn)/1e6*cm.DollarsPerMTokOut
}

// ClassBreakdown is the per-call disposition, derived from live kernel verdicts.
// The first three rows are turn-tax (each a saved model turn); quarantine/deny
// are the safety floor; pass is the control (a call both arms pay identically).
type ClassBreakdown struct {
	Grammar    int `json:"grammar"`     // TRANSFORM repaired in-syscall (alias->canonical)
	VDSOPure   int `json:"vdso_pure"`   // tier-1 pure local serve
	VDSODedup  int `json:"vdso_dedup"`  // tier-2 content-cache hit (duplicate read)
	VDSOStatic int `json:"vdso_static"` // tier-3 static local serve
	Quarantine int `json:"quarantine"`  // poisoned result held out of context (safety)
	Deny       int `json:"deny"`        // refused by the capability floor (safety)
	Pass       int `json:"pass"`        // allow + engine dispatch — no saving (control)
}

func (c ClassBreakdown) vdsoTotal() int { return c.VDSOPure + c.VDSODedup + c.VDSOStatic }

// turnsSaved is the deterministic happy-path headline: every grammar repair and
// every local serve is one model round-trip the baseline pays and fak does not.
func (c ClassBreakdown) turnsSaved() int { return c.Grammar + c.vdsoTotal() }

// forcedTurns are the turns the baseline DEMONSTRABLY pays: a re-issued duplicate
// read (tier-2) and an aliased call that errors and must be re-prompted (grammar).
// The model already made these calls; fak resolves them in-syscall.
func (c ClassBreakdown) forcedTurns() int { return c.Grammar + c.VDSODedup }

// elisionTurns are tool calls the model made that need no engine (tier-1 pure,
// tier-3 static). The baseline pays a round-trip only because the model chose to
// call a tool a stronger agent could elide; fak serves it locally either way. A
// capability win, reported apart from forcedTurns so the headline isn't overstated.
func (c ClassBreakdown) elisionTurns() int { return c.VDSOPure + c.VDSOStatic }

// KernelCounters mirrors the kernel's own tallies for the consistency check (the
// report asserts these equal the per-call ClassBreakdown — a bucketing guard, not
// an independent oracle; see consistencyCheck).
type KernelCounters struct {
	Submits     int64 `json:"submits"`
	VDSOHits    int64 `json:"vdso_hits"`
	EngineCalls int64 `json:"engine_calls"`
	Denies      int64 `json:"denies"`
	Transforms  int64 `json:"transforms"`
	Quarantines int64 `json:"quarantines"`
}

// Net is the converted savings of the 1-shot path over the SOTA baseline.
type Net struct {
	TurnsSaved     int     `json:"turns_saved"`
	TokensSaved    int     `json:"tokens_saved"`
	DollarsSaved   float64 `json:"dollars_saved"`
	LatencySavedMs float64 `json:"latency_saved_ms"`
}

// TurnKinds decomposes turns_saved into the two honestly-different kinds (see the
// package doc): FORCED turns the baseline demonstrably pays (grammar retry +
// duplicate read) vs ELISION turns for optional tool calls a stronger model could
// omit (pure + static). Forced + Elision == Net.TurnsSaved.
type TurnKinds struct {
	Forced  int `json:"forced"`  // grammar + tier-2 dedup
	Elision int `json:"elision"` // tier-1 pure + tier-3 static
}

// CallDisposition is one replayed call's live-kernel outcome, in trace order —
// the per-call stream the demo renders. It is derived from the SAME verdict/result
// the aggregate ClassBreakdown is (classifyDisposition mirrors classify exactly),
// so there is one source of truth for "what did the kernel do with this call".
type CallDisposition struct {
	Index     int    `json:"index"`             // position in the trace (0-based)
	Tool      string `json:"tool"`              // the tool the call invoked
	Class     string `json:"class"`             // grammar|vdso_pure|vdso_dedup|vdso_static|quarantine|deny|pass
	By        string `json:"by,omitempty"`      // which adjudicator decided (forensics)
	Tier      string `json:"tier,omitempty"`    // vDSO tier tag ("1"|"2"|"3") when By==vdso
	Axis      string `json:"axis"`              // "turn-tax" | "safety-floor" | "control"
	SavedTurn bool   `json:"saved_turn"`        // true iff this call deleted a baseline model turn
	Forced    bool   `json:"forced,omitempty"`  // turn-tax: baseline DEMONSTRABLY pays (grammar + tier-2 dedup)
	Elision   bool   `json:"elision,omitempty"` // turn-tax: optional call a stronger model could elide (pure + static)
	Reason    string `json:"reason"`            // one-line human label: why the baseline pays (or why it's safe)
}

// SafetyFloor is the deterministic moat axis (NOT a turn count): what the
// unmediated baseline would have done that fak structurally prevents.
type SafetyFloor struct {
	InjectionsAdmittedBaseline  int `json:"injections_admitted_baseline"`
	InjectionsAdmittedFak       int `json:"injections_admitted_fak"`
	DestructiveExecutedBaseline int `json:"destructive_executed_baseline"`
	DestructiveExecutedFak      int `json:"destructive_executed_fak"`
}

// Lever is one row of the ablation table.
type Lever struct {
	Version    string `json:"version,omitempty"`
	Name       string `json:"name"`
	Mechanism  string `json:"mechanism"`
	Axis       string `json:"axis"` // "turn-tax" | "safety-floor"
	TurnsSaved int    `json:"turns_saved"`
	Witness    string `json:"witness"`
}

// SensitivityRow recomputes the net under a different model/stack assumption.
type SensitivityRow struct {
	Scenario            string  `json:"scenario"`
	ModelTurnLatencyMs  float64 `json:"model_turn_latency_ms"`
	PromptTokensPerTurn int     `json:"prompt_tokens_per_turn"`
	TokensSaved         int     `json:"tokens_saved"`
	DollarsSaved        float64 `json:"dollars_saved"`
	LatencySavedSec     float64 `json:"latency_saved_sec"`
}

// Provenance pins what produced the report.
type Provenance struct {
	AppVersion   string `json:"app_version"`
	Command      string `json:"command"`
	SliceID      string `json:"slice_id"`
	WorkloadHash string `json:"workload_hash"`
	GoVersion    string `json:"go_version"`
	OS           string `json:"os"`
	GeneratedBy  string `json:"generated_by"`
}

// Report is the full turn-tax artifact.
type Report struct {
	Provenance   Provenance       `json:"provenance"`
	Calls        int              `json:"calls"`
	Cost         CostModel        `json:"cost_model"`
	Class        ClassBreakdown   `json:"class_breakdown"`
	Counters     KernelCounters   `json:"kernel_counters"`
	LocalServeNs int64            `json:"local_serve_ns"` // in-process adjudication p50: the µs cost of the 1-shot
	Net          Net              `json:"net"`
	TurnKinds    TurnKinds        `json:"turn_kinds"`   // forced vs elision split of net.turns_saved
	VDSOOffNet   Net              `json:"vdso_off_net"` // same trace, vDSO disabled (the real ON/OFF path-swap ablation)
	Safety       SafetyFloor      `json:"safety_floor"`
	Levers       []Lever          `json:"ablation_levers"`
	Sensitivity  []SensitivityRow `json:"latency_sensitivity"`
	// ConsistencyCheck is "ok" iff the kernel's aggregate counters agree with the
	// per-call classification. NOTE: this is a CONSISTENCY guard (it catches a
	// bucketing/wiring drift in the bench), NOT an independent oracle — classify()
	// reads the same live verdicts the counters were incremented from, so it cannot
	// by itself prove the kernel's verdict was correct. The grounding that the
	// numbers are real events comes from the kernel emitting them, not from this
	// equality.
	ConsistencyCheck string `json:"consistency_check"`
}

// JSON renders the report.
func (r *Report) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// rawSafety accumulates the per-replay safety observations.
type rawSafety struct {
	poison              int
	destructiveExecuted int
}

// destructiveTool reports whether a denied tool is write-shaped — i.e. the kind
// the unmediated baseline would have actually EXECUTED (the safety harm), as
// opposed to a benign misroute. Mirrors vdso.destructive's name heuristic.
func destructiveTool(name string) bool {
	for _, p := range []string{"write", "edit", "delete", "patch", "exec", "run", "book", "update", "cancel", "send"} {
		if strings.Contains(strings.ToLower(name), p) {
			return true
		}
	}
	return false
}

// classify buckets one replayed call from its LIVE verdict + result. Quarantine is
// read from the result meta (it is an admit-time outcome, independent of the
// allow verdict); deny/vdso/grammar from the verdict; everything else is a
// control "pass".
func classify(cb *ClassBreakdown, sf *rawSafety, c Call, r *abi.Result, v abi.Verdict) {
	if r != nil && r.Meta != nil && r.Meta["admit"] == "quarantined" {
		cb.Quarantine++
		sf.poison++
		return
	}
	switch {
	case v.Kind == abi.VerdictDeny:
		cb.Deny++
		if destructiveTool(c.Tool) {
			sf.destructiveExecuted++ // the baseline runs it unmediated
		}
	case v.By == "vdso":
		tier := ""
		if r != nil && r.Meta != nil {
			tier = r.Meta["tier"]
		}
		switch tier {
		case "1":
			cb.VDSOPure++
		case "2":
			cb.VDSODedup++
		case "3":
			cb.VDSOStatic++
		default:
			cb.VDSOPure++ // defensive: an untagged local serve still counts as a saved turn
		}
	case v.Kind == abi.VerdictTransform && v.By == "grammar":
		cb.Grammar++
	default:
		cb.Pass++
	}
}

// classifyDisposition derives the ordered, per-call CallDisposition from the SAME
// live verdict + result classify() buckets — the single-source-of-truth the demo
// renders. It mirrors classify's branch order exactly (quarantine read from result
// meta first, then deny/vdso/grammar from the verdict, else a control pass), so the
// stream and the aggregate ClassBreakdown can never disagree about a call's class.
func classifyDisposition(idx int, c Call, r *abi.Result, v abi.Verdict) CallDisposition {
	d := CallDisposition{Index: idx, Tool: c.Tool, By: v.By, Axis: "control", Class: "pass", Reason: "engine round-trip — both arms pay it"}
	if r != nil && r.Meta != nil && r.Meta["admit"] == "quarantined" {
		d.Class, d.Axis = "quarantine", "safety-floor"
		d.Reason = "baseline admits poison to context → derail; fak pages it out"
		return d
	}
	switch {
	case v.Kind == abi.VerdictDeny:
		d.Class, d.Axis = "deny", "safety-floor"
		if destructiveTool(c.Tool) {
			d.Reason = "baseline executes the destructive op; fak refuses it (deny-as-value)"
		} else {
			d.Reason = "refused by the capability floor (deny-as-value)"
		}
	case v.By == "vdso":
		tier := ""
		if r != nil && r.Meta != nil {
			tier = r.Meta["tier"]
		}
		d.Tier, d.Axis, d.SavedTurn = tier, "turn-tax", true
		switch tier {
		case "2":
			d.Class, d.Forced = "vdso_dedup", true
			d.Reason = "baseline re-issues a duplicate read it already made; fak serves it from the content-cache"
		case "3":
			d.Class, d.Elision = "vdso_static", true
			d.Reason = "static answer the model could recall; fak serves it from a local table"
		case "1":
			d.Class, d.Elision = "vdso_pure", true
			d.Reason = "pure fn the model could inline; fak computes it locally"
		default:
			d.Class, d.Elision, d.Tier = "vdso_pure", true, "1"
			d.Reason = "local serve (untagged) — no engine needed"
		}
	case v.Kind == abi.VerdictTransform && v.By == "grammar":
		d.Class, d.Axis, d.SavedTurn, d.Forced = "grammar", "turn-tax", true, true
		d.Reason = "aliased arg errors → baseline re-prompts (+1 turn); fak repairs it in-syscall (TRANSFORM)"
	}
	return d
}

// replayOpts carry the OPTIONAL per-replay overrides. The zero value is the v0.1
// behavior (walk the process-global adjudicator registry), so every existing caller
// is unchanged; the policy-replay driver supplies adj to give each concurrent arm its
// OWN monitor instead of mutating the shared adjudicator.Default (issue #500).
type replayOpts struct {
	adj []abi.Adjudicator // explicit per-kernel adjudicator chain; nil => global registry
}

// ReplayOption is an additive functional option for replay() (and the policy-replay
// driver). Existing callers pass none and get the v0.1 global-registry path.
type ReplayOption func(*replayOpts)

// withAdjudicators makes the replay's kernel fold an EXPLICIT adjudicator chain rather
// than the process-global registry, so K policy arms can replay CONCURRENTLY without
// colliding on adjudicator.Default's mutable policy (issue #500).
func withAdjudicators(chain []abi.Adjudicator) ReplayOption {
	return func(o *replayOpts) { o.adj = chain }
}

// replay runs the whole trace through one fresh kernel (vDSO on/off per the arm)
// and returns the live counters, the independently-derived classification, the
// safety observations, the in-process adjudication p50 (the 1-shot's µs cost,
// measured with the same calibration loop internal/bench uses), and — when
// collectDisp is set — the ordered per-call dispositions (the demo's stream).
//
// An optional explicit adjudicator chain (withAdjudicators) makes the kernel fold
// THAT chain instead of the process-global registry; with no option the kernel reads
// the global registry exactly as before, so the turn-tax / stochastic / fleet callers
// are unaffected.
func replay(ctx context.Context, t *Trace, vdsoOn, calibrate, collectDisp bool, opts ...ReplayOption) (KernelCounters, ClassBreakdown, rawSafety, int64, []CallDisposition, error) {
	var ro replayOpts
	for _, opt := range opts {
		opt(&ro)
	}

	// Each arm is an ISOLATED session. Reset the two process-global pieces of
	// cross-call state so a replay is reproducible regardless of any prior run in
	// this process: the vDSO tier-2 cache (world bump => first read of each key is
	// a cold miss / real engine call, so the duplicate-read savings are honest),
	// and the IFC control-flow taint ledger for the default trace (a fresh session
	// has not yet "seen untrusted content").
	vdso.Default.BumpWorld()
	ifc.Default.Reset("")

	var kopts []kernel.Option
	if len(ro.adj) > 0 {
		kopts = append(kopts, kernel.WithAdjudicators(ro.adj))
	}
	k := kernel.New("localtools", kopts...)
	k.SetVDSO(vdsoOn)
	res := abi.ActiveResolver()
	var cb ClassBreakdown
	var sf rawSafety
	var h metrics.Hist
	var disp []CallDisposition
	if collectDisp {
		disp = make([]CallDisposition, 0, len(t.Calls))
	}

	for idx, c := range t.Calls {
		args := []byte(c.Args)
		if len(args) == 0 {
			args = []byte("{}")
		}
		ref, err := res.Put(ctx, args)
		if err != nil {
			return KernelCounters{}, cb, sf, 0, nil, err
		}

		// Measure the pure in-process adjudication boundary (calibration loop).
		// Decide touches no engine/network, so this is the cost the 1-shot adds in
		// place of a model round-trip. Skipped (calibrate=false) by callers that need
		// only the turn counts — e.g. the stochastic harness, which runs thousands of
		// replays where the per-call µs latency is irrelevant.
		if calibrate {
			tcDecide := &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta}
			t0 := time.Now()
			for i := 0; i < adjReps; i++ {
				_ = k.Decide(ctx, tcDecide)
			}
			h.RecordNs(int64(time.Since(t0)) / adjReps)
		}

		// Drive the full syscall once for the real verdict/result + counters. A
		// fresh Ref per call (Submit may rewrite c.Args on a TRANSFORM).
		ref2, err := res.Put(ctx, args)
		if err != nil {
			return KernelCounters{}, cb, sf, 0, nil, err
		}
		tc := &abi.ToolCall{Tool: c.Tool, Args: ref2, Meta: c.Meta}
		r, v := k.Syscall(ctx, tc)
		classify(&cb, &sf, c, r, v)
		if collectDisp {
			disp = append(disp, classifyDisposition(idx, c, r, v))
		}
	}

	cc := k.Counters()
	kc := KernelCounters{
		Submits: cc.Submits, VDSOHits: cc.VDSOHits, EngineCalls: cc.EngineCalls,
		Denies: cc.Denies, Transforms: cc.Transforms, Quarantines: cc.Quarantines,
	}
	return kc, cb, sf, h.P50(), disp, nil
}

// Run executes the full turn-tax A/B: a real vDSO-on replay (the 1-shot arm), a
// real vDSO-off replay (the path-swap ablation that proves the vDSO lever), the
// cost-model conversion, the safety floor, the ablation table, and the
// sensitivity grid — with a consistency check that the live kernel counters
// agree with the independently-derived per-call classification.
func Run(ctx context.Context, t *Trace, cm CostModel) (*Report, error) {
	rep, _, err := RunWithCalls(ctx, t, cm)
	return rep, err
}

// RunWithCalls is Run plus the ordered per-call dispositions (the live stream the
// turntax-demo renders). The Report is byte-identical to Run's — the dispositions
// are derived from the SAME on-arm replay via classifyDisposition, so there is one
// source of truth for "what the kernel did with each call". Callers that need only
// the rolled-up Report use Run; the demo uses this.
func RunWithCalls(ctx context.Context, t *Trace, cm CostModel) (*Report, []CallDisposition, error) {
	cm = withCostModelVersion(cm)
	// Install the agent's policy/grammar/engine world (idempotent) so the trace's
	// tools trigger the REAL rungs: convert_currency aliases -> grammar TRANSFORM,
	// fetch_policy refund doc -> ctxmmu quarantine, delete_account -> policy deny.
	// (replay isolates the per-arm cross-call state — vDSO cache + IFC ledger.)
	agent.Configure()

	// The on-arm calibrates (its p50 is the reported 1-shot serve cost) and collects
	// the per-call dispositions; the off-arm never needs either, so it skips both.
	on, onClass, onSafety, localNs, disp, err := replay(ctx, t, true, true, true)
	if err != nil {
		return nil, nil, err
	}

	_, offClass, _, _, _, err := replay(ctx, t, false, false, false)
	if err != nil {
		return nil, nil, err
	}

	rep := &Report{
		Provenance: Provenance{
			AppVersion:   appversion.Current(),
			Command:      "fak turntax --suite " + t.SliceID,
			SliceID:      t.SliceID,
			WorkloadHash: t.WorkloadHash(),
			GoVersion:    runtime.Version(),
			OS:           runtime.GOOS,
			GeneratedBy:  "fak/internal/turnbench",
		},
		Calls:        len(t.Calls),
		Cost:         cm,
		Class:        onClass,
		Counters:     on,
		LocalServeNs: localNs,
		Net:          netFor(onClass.turnsSaved(), cm),
		TurnKinds:    TurnKinds{Forced: onClass.forcedTurns(), Elision: onClass.elisionTurns()},
		VDSOOffNet:   netFor(offClass.turnsSaved(), cm),
		Safety: SafetyFloor{
			InjectionsAdmittedBaseline:  onSafety.poison,
			InjectionsAdmittedFak:       0,
			DestructiveExecutedBaseline: onSafety.destructiveExecuted,
			DestructiveExecutedFak:      0,
		},
		Levers:      levers(onClass),
		Sensitivity: sensitivity(onClass.turnsSaved(), cm),
	}
	rep.ConsistencyCheck = consistencyCheck(on, onClass)
	return rep, disp, nil
}

// scoreTrace runs ONE vDSO-on replay with NO latency calibration and returns the
// turn-tax split (total / forced / elision). It is the fast path the stochastic
// harness uses across thousands of trials: it needs only the turns the kernel
// deletes, not the µs serve cost, the off-arm ablation, or the full Report — so it
// skips the adjReps calibration loop that dominates Run's cost. The caller MUST
// have run agent.Configure() once (to register the localtools engine + policy).
func scoreTrace(ctx context.Context, t *Trace) (turns, forced, elision int, err error) {
	_, cb, _, _, _, e := replay(ctx, t, true, false, false)
	if e != nil {
		return 0, 0, 0, e
	}
	return cb.turnsSaved(), cb.forcedTurns(), cb.elisionTurns(), nil
}

func withCostModelVersion(cm CostModel) CostModel {
	if cm.Version == "" {
		cm.Version = CostModelVersion
	}
	return cm
}

func netFor(turns int, cm CostModel) Net {
	return Net{
		TurnsSaved:     turns,
		TokensSaved:    turns * cm.tokensPerTurn(),
		DollarsSaved:   float64(turns) * cm.dollarsPerTurn(),
		LatencySavedMs: float64(turns) * cm.ModelTurnLatencyMs,
	}
}

// consistencyCheck asserts the kernel's aggregate counters agree with the
// per-call classification. It catches a BUCKETING or WIRING drift in the bench (a
// class mapped to the wrong counter, a miscount) — but it is NOT an independent
// oracle: classify() reads the same live verdicts the counters were incremented
// from, so the two are correlated by construction. The grounding that the numbers
// are real events is that the KERNEL emits them; this equality only guards the
// bench's own bookkeeping against itself.
func consistencyCheck(kc KernelCounters, cb ClassBreakdown) string {
	var bad []string
	if int(kc.Transforms) != cb.Grammar {
		bad = append(bad, fmt.Sprintf("transforms %d != grammar %d", kc.Transforms, cb.Grammar))
	}
	if int(kc.VDSOHits) != cb.vdsoTotal() {
		bad = append(bad, fmt.Sprintf("vdso_hits %d != vdso classes %d", kc.VDSOHits, cb.vdsoTotal()))
	}
	if int(kc.Quarantines) != cb.Quarantine {
		bad = append(bad, fmt.Sprintf("quarantines %d != quarantine %d", kc.Quarantines, cb.Quarantine))
	}
	if int(kc.Denies) != cb.Deny {
		bad = append(bad, fmt.Sprintf("denies %d != deny %d", kc.Denies, cb.Deny))
	}
	if len(bad) > 0 {
		return "MISMATCH: " + strings.Join(bad, "; ")
	}
	return "ok"
}

func levers(cb ClassBreakdown) []Lever {
	return []Lever{
		{
			Version: BenchmarkConceptVersion,
			Name:    "grammar-repair", Mechanism: "TRANSFORM in-syscall (alias->canonical)",
			Axis: "turn-tax", TurnsSaved: cb.Grammar,
			Witness: "kernel Counters.Transforms == grammar-classified calls",
		},
		{
			Version: BenchmarkConceptVersion,
			Name:    "vdso", Mechanism: "3-tier local serve (pure / content-cache / static)",
			Axis: "turn-tax", TurnsSaved: cb.vdsoTotal(),
			Witness: "real ON/OFF path swap: net.turns_saved(on) - net.turns_saved(off) == VDSOHits",
		},
		{
			Version: BenchmarkConceptVersion,
			Name:    "quarantine", Mechanism: "context-MMU result admission (poison paged out)",
			Axis: "safety-floor", TurnsSaved: 0,
			Witness: "kernel Counters.Quarantines; injection kept out of context",
		},
		{
			Version: BenchmarkConceptVersion,
			Name:    "deny", Mechanism: "capability-floor adjudication (deny-as-value)",
			Axis: "safety-floor", TurnsSaved: 0,
			Witness: "kernel Counters.Denies; destructive op never dispatched",
		},
		{
			Version: BenchmarkConceptVersion,
			Name:    "NET (1-shot)", Mechanism: "all levers — turns the baseline pays and fak does not",
			Axis: "turn-tax", TurnsSaved: cb.turnsSaved(),
			Witness: "grammar + vdso (sum of the turn-tax levers)",
		},
	}
}

// sensitivity sweeps the two cost knobs that actually move the net — model
// round-trip latency and prompt size — across a small, labeled grid (the turns
// saved are fixed by the kernel; only the per-turn price varies).
func sensitivity(turns int, base CostModel) []SensitivityRow {
	type scen struct {
		name      string
		latencyMs float64
		promptTok int
	}
	scens := []scen{
		{"local-fast (400ms, 600tok)", 400, 600},
		{"hosted-flash (1.5s, 1200tok)", 1500, 1200},
		{"frontier (4s, 4000tok)", 4000, 4000},
	}
	out := make([]SensitivityRow, 0, len(scens))
	for _, s := range scens {
		cm := base
		cm.ModelTurnLatencyMs = s.latencyMs
		cm.PromptTokensPerTurn = s.promptTok
		out = append(out, SensitivityRow{
			Scenario:            s.name,
			ModelTurnLatencyMs:  s.latencyMs,
			PromptTokensPerTurn: s.promptTok,
			TokensSaved:         turns * cm.tokensPerTurn(),
			DollarsSaved:        float64(turns) * cm.dollarsPerTurn(),
			LatencySavedSec:     float64(turns) * s.latencyMs / 1000,
		})
	}
	return out
}
