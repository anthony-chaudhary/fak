package quality

import (
	"encoding/json"
	"fmt"
	"sort"
)

// scheduler_parity.go is the scheduler-policy child of the quality spine (#4537):
// it proves that a request's output is IDENTICAL across scheduler policies. A
// scheduling policy — fcfs, priority, longest-first — may change the ORDER a
// batch's requests execute in, but never any request's own tokens: reordering is
// a throughput decision, not a semantics decision. The reference is the batch
// decoded under one policy (fcfs); the engine is the same batch under another.
// A scheduler whose reordering lets one request's tokens leak into another's
// output (the classic shared-buffer / un-reset-slab defect) fails here, with the
// first corrupted token localized to its request, step, and flat index, and the
// offending policy named in the Detail.

// schedVocab is the small fixed vocabulary the deterministic per-request decoder
// emits from. It is disjoint from the other spine vocabularies so a scheduler
// trace is never confused with a sibling oracle's trace in a failure bundle.
var schedVocab = []string{"quill", "raft", "slate", "tarn", "umber", "vale", "wren", "yoke"}

// schedRequest is one request in the scheduled batch: a stable ID, the priority
// the priority policy sorts by, the number of decode steps, and the seed its
// tokens are drawn under. A request's decode is a pure function of (Seed, step)
// — nothing about its output may depend on when the scheduler ran it.
type schedRequest struct {
	ID       string `json:"id"`
	Priority int    `json:"priority"`
	Steps    int    `json:"steps"`
	Seed     int64  `json:"seed"`
}

// schedWorkload is the batch a case schedules, serialized as JSON into the
// case's Prompt (the additive seam for per-case data richer than Tokens/Logits).
// Requests appear in SUBMISSION order — the order fcfs executes in and the
// canonical order every trace reports outputs in.
type schedWorkload struct {
	Requests []schedRequest `json:"requests"`
}

// schedOutput is one request's decoded tokens as a runner delivered them.
type schedOutput struct {
	ID     string   `json:"id"`
	Tokens []string `json:"tokens"`
}

// schedBody is the structured result a scheduler runner serializes into
// Trace.Text: the policy it executed under and every request's output in
// SUBMISSION order (canonical — never execution order, so two traces are
// comparable request by request regardless of how each policy reordered).
type schedBody struct {
	Policy  string        `json:"policy"`
	Outputs []schedOutput `json:"outputs"`
}

// schedDraw maps (seed, step) to one pseudo-random draw via a splitmix64-style
// finalizer. Pure function of its inputs: token i of a request depends only on
// the request's seed and the step index — the property scheduler parity rests on.
func schedDraw(seed int64, step int) uint64 {
	z := uint64(seed)*0x9e3779b97f4a7c15 + (uint64(step)+1)*0xd6e8feb86659fd93
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// schedDecode decodes one request in isolation — the ground-truth output every
// policy must deliver for that request, byte for byte.
func schedDecode(r schedRequest) []string {
	toks := make([]string, 0, r.Steps)
	for i := 0; i < r.Steps; i++ {
		toks = append(toks, schedVocab[schedDraw(r.Seed, i)%uint64(len(schedVocab))])
	}
	return toks
}

// SchedPolicies is the closed set of scheduler policies this child qualifies:
// fcfs (submission order), priority (highest Priority first), and longest-first
// (most Steps first). Sorts are stable so every policy's order is deterministic.
var SchedPolicies = []string{"fcfs", "priority", "longest-first"}

// schedOrder returns the execution order policy schedules reqs in. Unknown
// policies fall back to fcfs — the oracle judges outputs, not names, so a
// mislabeled policy still has to deliver correct outputs to pass.
func schedOrder(policy string, reqs []schedRequest) []schedRequest {
	out := append([]schedRequest(nil), reqs...)
	switch policy {
	case "priority":
		sort.SliceStable(out, func(i, j int) bool { return out[i].Priority > out[j].Priority })
	case "longest-first":
		sort.SliceStable(out, func(i, j int) bool { return out[i].Steps > out[j].Steps })
	}
	return out
}

// schedJSON marshals a value that cannot fail (plain structs of strings/ints);
// a marshal error here is a programming bug, not a runtime condition.
func schedJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic("quality: scheduler body marshal: " + err.Error())
	}
	return string(b)
}

// schedFlatten concatenates every output's tokens in the body's canonical
// (submission) order — the flat token stream Trace.Tokens carries and the
// index space FirstDivergence localizes into.
func schedFlatten(outputs []schedOutput) []string {
	var flat []string
	for _, o := range outputs {
		flat = append(flat, o.Tokens...)
	}
	return flat
}

// schedTrace assembles the Trace for a scheduled batch run: the canonical flat
// token stream plus the structured per-request body in Text.
func schedTrace(policy string, outputs []schedOutput) Trace {
	return Trace{Tokens: schedFlatten(outputs), Text: schedJSON(schedBody{Policy: policy, Outputs: outputs})}
}

// SchedulerRunner executes the case's workload (parsed from the Prompt) under
// one scheduling policy. The zero defect is a faithful scheduler: it may run
// requests in any policy order because each request decodes into its own fresh
// buffer. The defect field (set via SchedulerEngine) injects the shared-buffer
// bug this child exists to catch.
type SchedulerRunner struct {
	Label  string
	Policy string
	defect string
}

func (s SchedulerRunner) Name() string {
	if s.Label != "" {
		return s.Label
	}
	return "scheduler-engine"
}

func (s SchedulerRunner) policyName() string {
	if s.Policy == "" {
		return "fcfs"
	}
	return s.Policy
}

func (s SchedulerRunner) Run(c QualityCase) (Trace, error) {
	var wl schedWorkload
	if err := json.Unmarshal([]byte(c.Prompt), &wl); err != nil {
		return Trace{}, fmt.Errorf("scheduler workload in case prompt: %w", err)
	}
	if len(wl.Requests) == 0 {
		return Trace{}, fmt.Errorf("scheduler workload declares no requests")
	}
	slot := make(map[string]int, len(wl.Requests))
	for i, r := range wl.Requests {
		slot[r.ID] = i
	}

	// Outputs are stored by SUBMISSION slot regardless of execution order, so
	// every trace reports requests in the same canonical order.
	outputs := make([]schedOutput, len(wl.Requests))
	switch s.defect {
	case "shared-buffer":
		// The injected defect: one shared token slab reused across the whole
		// batch and never cleared between requests. Each request's decode is
		// written over the slab's PREFIX and the requester is handed the WHOLE
		// slab — so whenever the policy schedules a longer request before a
		// shorter one, the shorter request's output carries the longer
		// request's stale tail. Under fcfs on a submission-length-ascending
		// batch the slab never outgrows the current request and the bug is
		// invisible; a reordering policy is exactly what exposes it.
		var slab []string
		for _, r := range schedOrder(s.Policy, wl.Requests) {
			toks := schedDecode(r)
			if len(toks) > len(slab) {
				slab = append(slab, make([]string, len(toks)-len(slab))...)
			}
			copy(slab, toks)
			outputs[slot[r.ID]] = schedOutput{ID: r.ID, Tokens: append([]string(nil), slab...)}
		}
	default:
		// Faithful scheduler: still executes in policy order, but each request
		// decodes into its own fresh buffer — order changes, outputs cannot.
		for _, r := range schedOrder(s.Policy, wl.Requests) {
			outputs[slot[r.ID]] = schedOutput{ID: r.ID, Tokens: schedDecode(r)}
		}
	}

	t := schedTrace(s.policyName(), outputs)
	t.Runner = s.Name()
	return t, nil
}

// SchedulerEngine returns a scheduler runner for policy with an optional
// injected defect: "" schedules faithfully (isolated per-request buffers);
// "shared-buffer" reuses one un-cleared slab across the batch so a reordering
// policy corrupts the shorter requests' outputs with a longer request's stale
// tail. This is the deterministic mutant source the tests use to prove the
// parity gate trips.
func SchedulerEngine(policy, defect string) SchedulerRunner {
	label := "engine-sched-" + policy
	if defect != "" {
		label += "-" + defect
	}
	return SchedulerRunner{Label: label, Policy: policy, defect: defect}
}

// schedDemoRequests is the built-in batch: submission order ascends in Steps
// (so fcfs leaves no residue in the defective slab), while both the priority
// order (req-b, req-c, req-a) and the longest-first order (req-c, req-b, req-a)
// genuinely reorder execution and schedule a longer request before a shorter
// one — the exact shape that makes the shared-buffer defect bite.
func schedDemoRequests() []schedRequest {
	return []schedRequest{
		{ID: "req-a", Priority: 1, Steps: 3, Seed: 101},
		{ID: "req-b", Priority: 3, Steps: 5, Seed: 202},
		{ID: "req-c", Priority: 2, Steps: 7, Seed: 303},
	}
}

// SchedulerParityCase builds the scheduler-parity case: the demo workload as
// the Prompt, and a reference trace produced by a faithful fcfs run — the "one
// policy" every other policy's outputs are compared against. The case says
// nothing about WHICH policy the engine runs; parity must hold for all of them.
func SchedulerParityCase() QualityCase {
	reqs := schedDemoRequests()
	outputs := make([]schedOutput, len(reqs))
	total := 0
	for i, r := range reqs {
		outputs[i] = schedOutput{ID: r.ID, Tokens: schedDecode(r)}
		total += r.Steps
	}
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "scheduler-parity-demo",
		Version:   1,
		Prompt:    schedJSON(schedWorkload{Requests: reqs}),
		Params:    SamplingParams{Temperature: 0, MaxTokens: total},
		Reference: schedTrace("fcfs", outputs),
		Oracles:   []string{"scheduler-parity"},
	}
}

// schedParseBody decodes a trace's structured scheduler body. The oracle fails
// closed on a trace that carries none — a scheduler run that cannot show its
// per-request outputs is not a passing run.
func schedParseBody(text string) (schedBody, error) {
	var b schedBody
	if err := json.Unmarshal([]byte(text), &b); err != nil {
		return schedBody{}, err
	}
	if len(b.Outputs) == 0 {
		return schedBody{}, fmt.Errorf("scheduler body carries no request outputs")
	}
	return b, nil
}

// schedFindOutput resolves a request's output by ID.
func schedFindOutput(outputs []schedOutput, id string) (schedOutput, bool) {
	for _, o := range outputs {
		if o.ID == id {
			return o, true
		}
	}
	return schedOutput{}, false
}

// schedTokenAt mirrors tokenAt for a per-request token slice.
func schedTokenAt(toks []string, i int) string {
	if i < len(toks) {
		return toks[i]
	}
	return "<end>"
}

// SchedulerParity is the differential oracle for scheduler-policy output parity
// (#4537): every request's output under the engine's policy must equal its
// output under the reference policy token for token — policy may reorder
// EXECUTION, never a request's own tokens. Outputs are compared request by
// request in the reference body's canonical (submission) order, so the first
// corrupted token localizes to a request, a step within it, and a flat index
// into the canonical stream, and the Detail names the policy that corrupted it.
type SchedulerParity struct{}

func (SchedulerParity) Name() string { return "scheduler-parity" }
func (SchedulerParity) Kind() string { return "differential" }

func init() { Register(SchedulerParity{}) }

func (SchedulerParity) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "scheduler-parity", Kind: "differential", Pass: true}
	refBody, err := schedParseBody(ref.Text)
	if err != nil {
		v.Pass = false
		v.Detail = fmt.Sprintf("reference trace carries no scheduler body: %v", err)
		return v
	}
	engBody, err := schedParseBody(eng.Text)
	if err != nil {
		v.Pass = false
		v.Detail = fmt.Sprintf("engine trace carries no scheduler body: %v", err)
		return v
	}

	flat := 0
	for _, ro := range refBody.Outputs {
		eo, ok := schedFindOutput(engBody.Outputs, ro.ID)
		if !ok {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: flat, Reference: schedTokenAt(ro.Tokens, 0), Engine: "<missing>"}
			v.Detail = fmt.Sprintf("policy %q dropped request %q entirely (flat token %d)", engBody.Policy, ro.ID, flat)
			return v
		}
		n := len(ro.Tokens)
		if len(eo.Tokens) < n {
			n = len(eo.Tokens)
		}
		for i := 0; i < n; i++ {
			if ro.Tokens[i] != eo.Tokens[i] {
				v.Pass = false
				v.FirstDivergence = &Divergence{Index: flat + i, Reference: ro.Tokens[i], Engine: eo.Tokens[i]}
				v.Detail = fmt.Sprintf("policy %q corrupted request %q at step %d (flat token %d): reference %q, engine %q — scheduling may reorder execution, never a request's own output",
					engBody.Policy, ro.ID, i, flat+i, ro.Tokens[i], eo.Tokens[i])
				return v
			}
		}
		if len(ro.Tokens) != len(eo.Tokens) {
			v.Pass = false
			v.FirstDivergence = &Divergence{Index: flat + n, Reference: schedTokenAt(ro.Tokens, n), Engine: schedTokenAt(eo.Tokens, n)}
			v.Detail = fmt.Sprintf("policy %q changed request %q's length at step %d (flat token %d): reference has %d tokens, engine has %d — a shared buffer leaked another request's tokens or dropped this one's",
				engBody.Policy, ro.ID, n, flat+n, len(ro.Tokens), len(eo.Tokens))
			return v
		}
		flat += len(ro.Tokens)
	}
	if len(engBody.Outputs) != len(refBody.Outputs) {
		v.Pass = false
		extra := len(engBody.Outputs) - len(refBody.Outputs)
		v.Detail = fmt.Sprintf("policy %q emitted %d extra request output(s) the reference never scheduled", engBody.Policy, extra)
		return v
	}
	v.Detail = fmt.Sprintf("policy %q preserved every request's output: %d requests, %d tokens identical to the %q reference",
		engBody.Policy, len(refBody.Outputs), flat, refBody.Policy)
	return v
}
