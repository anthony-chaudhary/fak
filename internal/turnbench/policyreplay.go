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
// registered by OTHER drivers and are constant across arms. A monitor REDACT transform
// currently classifies as a served result (classifyDisposition buckets it as a pass),
// so a redact-induced divergence is NOT yet detected — a documented follow-up, not a
// silent gap.
package turnbench

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"time"

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
// NOTE (documented limitation): a monitor REDACT transform lands in "served" today —
// classifyDisposition buckets it as a pass — so a redact-induced divergence is not yet
// detected. Surfacing it needs the raw monitor verdict (a filed follow-up).
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

// RunPolicyReplay scores K policy arms against ONE frozen trajectory by swapping the
// policy on the process-global monitor and replaying the same trace per arm — the
// structural collapse of the K-policy comparison from "K full agent+model runs" to
// "1 recording + K model-free kernel replays" (see the file doc). The reference arm
// (refName, or arms[0] when empty) is the recorded trajectory's policy; every arm
// carries a divergence witness against it, so no replayed number silently crosses the
// measured/modeled wall.
//
// It calls agent.Configure() itself (idempotent — installs the localtools engine +
// grammar + schemas) and restores the canonical bench policy on the way out. It is NOT
// safe to call concurrently with another replay in the same process: the monitor's
// policy is process-global (the same single-process discipline RunFleetSweep /
// RunStochastic already follow).
func RunPolicyReplay(ctx context.Context, t *Trace, arms []PolicyArm, refName string, cm CostModel) (*PolicyReplayReport, error) {
	if t == nil || len(t.Calls) == 0 {
		return nil, fmt.Errorf("turnbench: RunPolicyReplay needs a non-empty trace")
	}
	if len(arms) == 0 {
		return nil, fmt.Errorf("turnbench: RunPolicyReplay needs at least one policy arm")
	}
	cm = withCostModelVersion(cm)

	agent.Configure()
	// The monitor policy is process-global; leave it as we found it for the next caller.
	defer agent.Configure()

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
	var wall int64
	for i, arm := range arms {
		adjudicator.Default.SetPolicy(arm.Policy)
		t0 := time.Now()
		kc, cb, _, _, disp, err := replay(ctx, t, true, false, true)
		dt := time.Since(t0).Nanoseconds()
		if err != nil {
			return nil, fmt.Errorf("turnbench: replay arm %q: %w", arm.Name, err)
		}
		wall += dt
		disps[i] = disp
		results[i] = PolicyArmResult{Name: arm.Name, Class: cb, Counters: kc, ReplayNs: dt}
	}

	// Divergence witness: each arm vs the reference's per-call observed-result class.
	refDisp := disps[refIdx]
	exact, bounded := 0, 0
	for i := range results {
		idx, tool := firstDivergence(refDisp, disps[i])
		results[i].FirstDivergence = idx
		if idx < 0 {
			results[i].ExactPrefix = len(t.Calls)
			results[i].Replayability = "exact"
			exact++
			continue
		}
		results[i].ExactPrefix = idx
		results[i].Replayability = fmt.Sprintf("bounded@%d", idx)
		results[i].DivergenceTool = tool
		results[i].DivergenceNote = fmt.Sprintf(
			"call %d (%s) observed %q under this policy vs %q under reference %q; a live run "+
				"would branch here — verdict counters are real, resolve-rate past this call is counterfactual",
			idx, tool, observedClass(disps[i][idx]), observedClass(refDisp[idx]), arms[refIdx].Name)
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
