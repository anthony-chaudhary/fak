// policyreplay.go — the POLICY-REPLAY SPINE: score K policies against ONE recorded
// trajectory, instead of running the whole agent+model K times.
//
// WHY THIS IS THE STRUCTURAL WIN. The expensive factor in a policy comparison is the
// MODEL: re-running a coding agent through SGLang/a frontier API once per candidate
// policy is K full multi-turn decode runs (seconds-and-dollars each; ~10^9× the cost
// of an in-process adjudication — see the package doc). But the kernel's verdict on a
// recorded call is a DETERMINISTIC function of (the call's tool+args, the produced
// result, the policy). So if you FREEZE the trajectory the model produced once, you
// can re-adjudicate it through K different policies as model-free kernel replays:
//
//	naive:   K policies × (full agent+model run)              — a product
//	replay:  1 recording  +  K model-free kernel replays      — a sum
//
// This file does the second. It reuses the SAME blessed replay() the turn-tax/
// stochastic harness uses (real k.Syscall, real counters, per-arm vDSO-world + IFC
// isolation), swapping only the monitor's policy between arms.
//
// THE HONESTY GATE (the load-bearing half — read before quoting any number). Replay
// is sound only where the candidate policy would not have changed what the MODEL SAW.
// The moment a policy denies/quarantines a call the recorded run served (or serves one
// it denied), a live run would observe a different result and BRANCH — every recorded
// call after that point is counterfactual. So every arm carries a divergence witness:
//
//	exact      — every call's model-observed result class matches the reference, so the
//	             recorded trajectory is valid under this policy; even RESOLVE-RATE replays
//	             soundly. (Two DIFFERENT policies can be exact on the same trace — they
//	             only differ on calls this trajectory never made.)
//	bounded@i  — the observed result first diverged at call i. The verdict/floor COUNTERS
//	             are still real kernel events for every recorded call (you may honestly
//	             say "this policy would have denied N of these calls"), but resolve-rate
//	             past call i is fiction the frozen trace cannot produce — it needs a live
//	             re-run from the frontier.
//
// This is the same measured-vs-modeled wall the rest of the package guards: the
// verdict comparison is MEASURED for all arms; resolve-rate is sound only for EXACT
// arms. The repo's own SWE-bench astropy result (1/1 raw → 0 through fak) is a maximal
// divergence — a deny at an early call — which is exactly why the gate is not optional.
//
// SCOPE / FENCES. This proves the spine on TRACE FIXTURES, whose Call carries its Args
// payload, so deny/arg-rule/allow policies re-adjudicate faithfully. It does NOT yet
// replay a PRODUCTION corpus: the journal stores arg/result DIGESTS, not payloads, so
// "replay last month's sessions" needs a payload-bearing trace sink first (filed
// follow-up). The policy axis here is the adjudicator decision table (Allow / Deny /
// arg-rules / self-modify / redact); vDSO, grammar repair, and ctx-MMU quarantine are
// registered by OTHER drivers and are constant across arms.
//
// REDACT divergence (issue #501). A monitor REDACT transform still buckets as a served
// result in classifyDisposition (it IS served — the call runs, just with rewritten
// args), so observedClass alone reads two redact-differing arms as both "served" and
// would call them exact. That is a FALSE exact: a live model would observe the rewritten
// args/results and could branch. RunPolicyReplay closes this by carrying a RAW monitor
// verdict alongside the disposition — it re-queries adjudicator.Default for each call's
// VerdictTransform and resolves the rewritten args — and treats a redact whose rewritten
// args DIFFER from the reference arm as a divergence, so the arm comes out bounded@i, not
// a false exact. The raw-verdict path runs ONLY for monitor redacts; every other class is
// witnessed by the observed-result class exactly as before.
package turnbench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// PolicyArm is one policy variant in a same-trace comparison. Name is a label; Policy
// is the adjudicator decision table swapped onto the process-global monitor for this
// arm's replay.
type PolicyArm struct {
	Name   string
	Policy adjudicator.Policy
}

// PolicyArmResult is one arm's outcome: the live kernel counters + the independently
// derived per-call class breakdown from replaying the SAME trace under this arm's
// policy, the measured replay wall-time, and the divergence witness vs the reference.
type PolicyArmResult struct {
	Name     string         `json:"name"`
	Class    ClassBreakdown `json:"class_breakdown"`
	Counters KernelCounters `json:"kernel_counters"`
	ReplayNs int64          `json:"replay_ns"`

	// FirstDivergence is the first call index whose MODEL-OBSERVED RESULT CLASS
	// (served | denied | quarantined) differs from the reference arm, or -1 if the
	// replay is EXACT (every observed result matches — the recorded trajectory is
	// valid under this policy and resolve-rate replays soundly).
	FirstDivergence int    `json:"first_divergence"`
	ExactPrefix     int    `json:"exact_prefix_calls"`
	Replayability   string `json:"replayability"` // "exact" | "bounded@<i>"
	DivergenceTool  string `json:"divergence_tool,omitempty"`
	DivergenceNote  string `json:"divergence_note,omitempty"`

	// ResolveRateEstimate is the MODELED off-policy estimate of this arm's resolve-rate
	// WITH a confidence interval (issue #505). It lives ALONGSIDE the measured floor
	// counters above, NEVER replacing them: the Counters/Class fields and the bounded@i
	// measured-refusal of the MEASURED resolve-rate stand exactly as before. For an EXACT
	// arm this collapses to the measured served-fraction with a zero-width CI; for a
	// bounded@i arm it is the bounded-doubly-robust projection whose CI widens with the
	// post-frontier depth. Modeled=true on the estimate is the measured/modeled wall.
	ResolveRateEstimate ResolveRateEstimate `json:"resolve_rate_estimate"`
}

// PolicyReplayReport is the spine artifact: K policies scored against ONE recorded
// trajectory, with the cube-collapse accounting and the per-arm divergence witness.
type PolicyReplayReport struct {
	Provenance Provenance        `json:"provenance"`
	Calls      int               `json:"calls"`
	Reference  string            `json:"reference_arm"`
	Cost       CostModel         `json:"cost_model"`
	Arms       []PolicyArmResult `json:"arms"`

	// Cube-collapse accounting (the product→sum). A naive K-policy comparison runs the
	// whole agent+model K times (K·calls model turns); replay runs it ONCE (the
	// recording) and scores the other K-1 policies as model-free kernel replays.
	Policies          int     `json:"policies"`
	ModelTurnsNaive   int     `json:"model_turns_naive"`       // K · calls
	ModelTurnsReplay  int     `json:"model_turns_replay"`      // calls (the one recording)
	ModelTurnsAvoided int     `json:"model_turns_avoided"`     // (K-1) · calls — for the VERDICT comparison
	DollarsAvoided    float64 `json:"dollars_avoided_modeled"` // MODELED via the CostModel knobs
	ReplayWallNs      int64   `json:"replay_wall_ns"`          // MEASURED: total wall-time of the K kernel replays

	// Honesty split. The verdict/floor comparison is replay-valid for ALL arms (real
	// kernel events on the recorded calls); resolve-rate is replay-valid ONLY for EXACT
	// arms — a bounded arm needs a live re-run from its divergence frontier.
	ExactArms   int `json:"exact_arms"`
	BoundedArms int `json:"bounded_arms"`
}

// JSON renders the report.
func (r *PolicyReplayReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// observedClass maps a replayed call's disposition to the RESULT CLASS the model would
// observe: "served" (a usable result — engine dispatch, a vDSO local serve, or an
// in-syscall grammar repair that still returns the real answer), "denied" (a
// deny-as-value error), or "quarantined" (the result paged out of context). The
// divergence witness keys on this: two policies that produce the same observed-result
// class for every recorded call are replay-equivalent on this trajectory.
//
// NOTE: a monitor REDACT transform lands in "served" here — it IS served, just with
// rewritten args. observedClass deliberately keeps it "served"; the redact-induced
// divergence (where two arms rewrite the SAME call's args differently) is caught by the
// raw-verdict path in RunPolicyReplay (redactFingerprint + firstRawDivergence), NOT by
// this coarse result class. See the file doc (issue #501).
func observedClass(d CallDisposition) string {
	switch d.Class {
	case "deny":
		return "denied"
	case "quarantine":
		return "quarantined"
	default:
		return "served"
	}
}

// servedCount is the MEASURED served-fraction numerator for an arm: how many of the
// arm's replayed calls the model would observe as "served" (a usable result), across the
// WHOLE frozen trace. It is the direct-method plug-in the OPE estimator (ope.go) projects
// the resolve-rate from — a real per-call count of the frozen replay, NOT a modeled number.
// For an exact arm served/calls IS the measured resolve-rate; for a bounded arm it is the
// max-likelihood continuation plug-in the bounded-DR estimate uses past the frontier.
func servedCount(disp []CallDisposition) int {
	n := 0
	for _, d := range disp {
		if observedClass(d) == "served" {
			n++
		}
	}
	return n
}

// firstDivergence returns the first call index where arm's observed-result class
// differs from the reference's (with the offending tool), or (-1, "") if the arm
// replays EXACTLY. The two slices cover the same trace, so they are equal length.
func firstDivergence(ref, arm []CallDisposition) (int, string) {
	n := len(ref)
	if len(arm) < n {
		n = len(arm)
	}
	for i := 0; i < n; i++ {
		if observedClass(arm[i]) != observedClass(ref[i]) {
			return i, arm[i].Tool
		}
	}
	return -1, ""
}

// redactFingerprint is the RAW monitor verdict captured per call for the redact-aware
// divergence path (issue #501). isRedact is true iff the monitor emitted a VerdictTransform
// for this call (a redact rewrite of the args); RewrittenArgs is the canonical bytes the
// model would observe AFTER the rewrite (empty when isRedact is false). Comparing this
// across arms catches a redact-only policy difference that observedClass buckets as a
// uniform "served" and would otherwise read as a false exact.
type redactFingerprint struct {
	isRedact      bool
	rewrittenArgs []byte
}

// captureRedactFingerprints re-queries THIS ARM'S monitor (the adjudicator the arm's
// kernel folded) for the raw verdict of every call, recording the monitor REDACT
// rewrite where one fires. It does NOT run a second kernel replay — it is a read-only
// adjudication query (the same Adjudicate the kernel's monitor link calls), so it adds no
// engine/network work and stays deterministic. Args are Put through the active resolver
// exactly as replay() does, so the verdict matches the one the replay's kernel saw.
//
// It takes the arm's OWN *adjudicator.Adjudicator rather than reading the process-global
// adjudicator.Default, so K arms can capture their fingerprints CONCURRENTLY without any
// arm depending on a global the others are also using (issue #500). adj.Adjudicate reads
// the policy under the adjudicator's RLock, so the query is itself race-safe.
func captureRedactFingerprints(ctx context.Context, t *Trace, adj *adjudicator.Adjudicator) ([]redactFingerprint, error) {
	res := abi.ActiveResolver()
	out := make([]redactFingerprint, len(t.Calls))
	for i, c := range t.Calls {
		args := []byte(c.Args)
		if len(args) == 0 {
			args = []byte("{}")
		}
		ref, err := res.Put(ctx, args)
		if err != nil {
			return nil, err
		}
		v := adj.Adjudicate(ctx, &abi.ToolCall{Tool: c.Tool, Args: ref, Meta: c.Meta})
		// Only a MONITOR redact transform rewrites the model-observed args. A grammar
		// TRANSFORM (By=="grammar") is a constant-across-arms in-syscall repair, not a
		// policy lever, so it is never a redact divergence point.
		if v.Kind == abi.VerdictTransform && v.By == "monitor" {
			if tp, ok := v.Payload.(abi.TransformPayload); ok {
				b, err := res.Resolve(ctx, tp.NewArgs)
				if err != nil {
					return nil, err
				}
				out[i] = redactFingerprint{isRedact: true, rewrittenArgs: append([]byte(nil), b...)}
			}
		}
	}
	return out, nil
}

// firstRawDivergence finds the first call where this arm's redact verdict DIFFERS from the
// reference's: either exactly one of them redacted the call, or both redacted it but to
// DIFFERENT rewritten args. Such a call is served under both arms (so observedClass calls
// it a match), yet the model would observe different args/results and could branch — a
// divergence observedClass cannot see. Returns (-1, "") when the redact verdicts agree on
// every call.
func firstRawDivergence(t *Trace, ref, arm []redactFingerprint) (int, string) {
	n := len(ref)
	if len(arm) < n {
		n = len(arm)
	}
	for i := 0; i < n; i++ {
		if ref[i].isRedact != arm[i].isRedact {
			return i, t.Calls[i].Tool
		}
		if ref[i].isRedact && arm[i].isRedact && !bytes.Equal(ref[i].rewrittenArgs, arm[i].rewrittenArgs) {
			return i, t.Calls[i].Tool
		}
	}
	return -1, ""
}

// swapMonitor returns a COPY of the registered adjudicator chain with the rank-100
// reference monitor (adjudicator.Default) replaced by mon, the arm's own monitor. Every
// other rung (grammar, preflight, the IFC sink gate, …) is carried through unchanged and
// in the same rank order, so a kernel folding the returned chain produces the IDENTICAL
// verdict the old adjudicator.Default.SetPolicy(arm.Policy) swap produced on the live
// chain — only the policy table the monitor consults differs, which is the whole point of
// an arm. The copy shares no mutable state with the global registry or with another arm's
// chain, so K arms fold concurrently without colliding. If the chain has no Default rung
// (a stripped registration), mon is appended so the arm is still monitored (fail-closed).
func swapMonitor(base []abi.Adjudicator, mon *adjudicator.Adjudicator) []abi.Adjudicator {
	out := make([]abi.Adjudicator, 0, len(base)+1)
	swapped := false
	for _, a := range base {
		if a == abi.Adjudicator(adjudicator.Default) {
			out = append(out, mon)
			swapped = true
			continue
		}
		out = append(out, a)
	}
	if !swapped {
		out = append(out, mon)
	}
	return out
}

// RunPolicyReplay scores K policy arms against ONE frozen trajectory by replaying the
// same trace per arm — the structural collapse of the K-policy comparison from "K full
// agent+model runs" to "1 recording + K model-free kernel replays" (see the file doc).
// The reference arm (refName, or arms[0] when empty) is the recorded trajectory's
// policy; every arm carries a divergence witness against it, so no replayed number
// silently crosses the measured/modeled wall.
//
// The arms replay CONCURRENTLY (issue #500). Each arm builds its OWN monitor
// (adjudicator.New(arm.Policy)) and injects it into its kernel via
// kernel.WithAdjudicators, so an arm NEVER mutates the process-global
// adjudicator.Default — the per-kernel adjudicator injection is what lets K arms fan out
// across goroutines instead of serializing on the shared monitor's policy. The result is
// IDENTICAL to a serial run: each arm's verdict is a deterministic function of (its own
// policy, the frozen trace), and the only remaining shared state (the localtools engine,
// the CAS resolver, the vDSO world counter) is stateless or mutex-guarded and CONSTANT
// across arms (the policy axis is the only thing that varies). agent.Configure() is
// called once (idempotent) to install the engine/grammar/schemas; no global policy is
// swapped, so there is nothing to restore.
func RunPolicyReplay(ctx context.Context, t *Trace, arms []PolicyArm, refName string, cm CostModel) (*PolicyReplayReport, error) {
	if t == nil || len(t.Calls) == 0 {
		return nil, fmt.Errorf("turnbench: RunPolicyReplay needs a non-empty trace")
	}
	if len(arms) == 0 {
		return nil, fmt.Errorf("turnbench: RunPolicyReplay needs at least one policy arm")
	}
	cm = withCostModelVersion(cm)

	agent.Configure()

	refIdx := 0
	if refName != "" {
		refIdx = -1
		for i, a := range arms {
			if a.Name == refName {
				refIdx = i
				break
			}
		}
		if refIdx < 0 {
			return nil, fmt.Errorf("turnbench: reference arm %q not found among %d arms", refName, len(arms))
		}
	}

	results := make([]PolicyArmResult, len(arms))
	disps := make([][]CallDisposition, len(arms))
	fps := make([][]redactFingerprint, len(arms))
	errs := make([]error, len(arms))

	// The base chain is the FULL registered adjudicator chain (rank-sorted) — every rung
	// the global-registry replay would fold: grammar repair, preflight, the IFC sink gate,
	// and the rank-100 monitor (adjudicator.Default). Per arm we copy it and swap ONLY the
	// monitor for the arm's own adjudicator.New(arm.Policy). That is exactly what the old
	// adjudicator.Default.SetPolicy(arm.Policy) swap did to the live chain — every OTHER
	// rung is unchanged and constant across arms — so the injected per-arm chain is
	// verdict-identical to the serial global-swap path, just without the shared mutable.
	baseChain := abi.Adjudicators()

	// Fan out: each arm runs in its own goroutine against its OWN injected monitor.
	// Every goroutine writes ONLY its own index of the pre-sized result slices, so there
	// is no shared mutable state between arms — the race detector run (the acceptance
	// gate) is clean by construction. ReplayWallNs is the WALL SPAN of the whole fan-out
	// (one clock read around wg.Wait): with the arms running concurrently the honest "how
	// long did the K replays take" is the span they overlapped in, NOT the sum of the
	// per-arm intervals (which would over-count concurrent time). The per-arm ReplayNs
	// stays per-arm. NOTE: when the trace is tiny and the process warm, the whole fan-out
	// can complete inside one tick of a coarse OS monotonic clock, so the span may read
	// 0ns — a legitimate "sub-tick" measurement, not a missing one.
	fanStart := time.Now()
	var wg sync.WaitGroup
	for i := range arms {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			arm := arms[i]
			// The arm's own monitor — the per-kernel adjudicator chain. This replaces the
			// old process-global adjudicator.Default.SetPolicy(arm.Policy) swap: build a
			// fresh chain that is the registered chain with the monitor rung substituted.
			adj := adjudicator.New(arm.Policy)
			chain := swapMonitor(baseChain, adj)
			t0 := time.Now()
			kc, cb, _, _, disp, err := replay(ctx, t, true, false, true, withAdjudicators(chain))
			dt := time.Since(t0).Nanoseconds()
			if err != nil {
				errs[i] = fmt.Errorf("turnbench: replay arm %q: %w", arm.Name, err)
				return
			}
			// Raw monitor verdict per call for the redact-aware divergence path (#501),
			// queried against THIS arm's own monitor (not the global), so the fingerprint
			// matches the replay the arm's kernel just ran; a read-only adjudication, not a
			// second replay.
			fp, err := captureRedactFingerprints(ctx, t, adj)
			if err != nil {
				errs[i] = fmt.Errorf("turnbench: capture redact fingerprints arm %q: %w", arm.Name, err)
				return
			}
			disps[i] = disp
			fps[i] = fp
			results[i] = PolicyArmResult{Name: arm.Name, Class: cb, Counters: kc, ReplayNs: dt}
		}(i)
	}
	wg.Wait()
	wall := time.Since(fanStart).Nanoseconds()
	for i := range arms {
		if errs[i] != nil {
			return nil, errs[i]
		}
	}

	// Divergence witness: each arm vs the reference. Two witnesses fold together and
	// the EARLIER divergence wins:
	//   (1) the observed-result CLASS flip (served|denied|quarantined) — a deny where
	//       the reference served, etc.; and
	//   (2) the RAW redact verdict differing (#501) — a call both arms serve, but with
	//       different rewritten args, which class (1) cannot see.
	refDisp := disps[refIdx]
	refFp := fps[refIdx]
	exact, bounded := 0, 0
	for i := range results {
		idx, tool := firstDivergence(refDisp, disps[i])
		ridx, rtool := firstRawDivergence(t, refFp, fps[i])
		// Pick the earlier of the class-flip and the redact divergence. A redact
		// divergence carries a distinct note so the witness names WHY it is bounded.
		redact := false
		if ridx >= 0 && (idx < 0 || ridx < idx) {
			idx, tool, redact = ridx, rtool, true
		}
		results[i].FirstDivergence = idx
		// The MODELED off-policy resolve-rate estimate (issue #505), computed from the
		// arm's MEASURED served-fraction and its divergence frontier. This is a SEPARATE,
		// clearly-labeled projection (Modeled=true) — it does NOT touch the measured
		// Counters/Class above, and a bounded arm's MEASURED resolve-rate stays refused.
		// frontier idx<0 means exact (depth 0) ⇒ the estimate collapses to the measured
		// value with a zero-width CI.
		results[i].ResolveRateEstimate = EstimateResolveRate(resolveRateInputs{
			served:   servedCount(disps[i]),
			calls:    len(t.Calls),
			frontier: idx,
		})
		if idx < 0 {
			results[i].ExactPrefix = len(t.Calls)
			results[i].Replayability = "exact"
			exact++
			continue
		}
		results[i].ExactPrefix = idx
		results[i].Replayability = fmt.Sprintf("bounded@%d", idx)
		results[i].DivergenceTool = tool
		if redact {
			results[i].DivergenceNote = fmt.Sprintf(
				"call %d (%s) is REDACTED differently under this policy than under reference %q "+
					"(the model would observe different args/results); a live run would branch here — "+
					"verdict counters are real, resolve-rate past this call is counterfactual",
				idx, tool, arms[refIdx].Name)
		} else {
			results[i].DivergenceNote = fmt.Sprintf(
				"call %d (%s) observed %q under this policy vs %q under reference %q; a live run "+
					"would branch here — verdict counters are real, resolve-rate past this call is counterfactual",
				idx, tool, observedClass(disps[i][idx]), observedClass(refDisp[idx]), arms[refIdx].Name)
		}
		bounded++
	}

	k := len(arms)
	return &PolicyReplayReport{
		Provenance: Provenance{
			AppVersion:   appversion.Current(),
			Command:      "turnbench.RunPolicyReplay",
			SliceID:      t.SliceID,
			WorkloadHash: t.WorkloadHash(),
			GoVersion:    runtime.Version(),
			OS:           runtime.GOOS,
			GeneratedBy:  "fak/internal/turnbench (policy-replay spine)",
		},
		Calls:             len(t.Calls),
		Reference:         arms[refIdx].Name,
		Cost:              cm,
		Arms:              results,
		Policies:          k,
		ModelTurnsNaive:   k * len(t.Calls),
		ModelTurnsReplay:  len(t.Calls),
		ModelTurnsAvoided: (k - 1) * len(t.Calls),
		DollarsAvoided:    float64((k-1)*len(t.Calls)) * cm.dollarsPerTurn(),
		ReplayWallNs:      wall,
		ExactArms:         exact,
		BoundedArms:       bounded,
	}, nil
}
