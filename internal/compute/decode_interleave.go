package compute

import (
	"os"
	"runtime"
	"strconv"
	"strings"
)

// This is the in-process production seam for #4974: reproduce the witnessed
// `numactl --interleave=all` weight placement WITHOUT requiring the operator to wrap
// the process. On a dual EPYC-7742 (8 NUMA nodes, 256 threads) the exact Qwen3.6-27B
// Q4_K decode cell at NUMA interleave + 64 workers was 1.337 tok/s vs 0.511 for the
// default all-workers-node0 cell (2.61x), but the default first-touch places every
// resident weight page on the single loader node, so the ordinary path never reproduced
// it. The 64-worker half is already selected automatically (Q4KDecodeWorkers, #4625); this
// is the placement half. Witness: experiments/qwen36/cpu-decode-int8-q4k-numa-witness-2026-07-14.md.
//
// The decision is a pure verdict over the runtime host snapshot (node count, task policy,
// core count, GOOS/GOARCH) — no host topology is hardcoded — and it stays overrideable via
// FAK_NUMA_INTERLEAVE. The apply itself is an mbind(MPOL_INTERLEAVE) over the weight regions,
// linux/amd64 only, no-op elsewhere. Placement is applied by address (mbind), not by task
// mempolicy: a process-wide set_mempolicy would only bind the calling OS thread's future
// faults, which the Go runtime's thread pool does not reliably inherit — so a per-region
// mbind is the only reliable in-process equivalent of numactl's pre-exec policy.

// DecodeInterleaveReason is the closed verdict vocabulary for whether the ordinary CPU
// decode path should apply NUMA interleave placement to the resident weights. A reason
// other than the two apply reasons (eligible, override_on) is a refusal: no mbind runs.
type DecodeInterleaveReason string

const (
	// DecodeInterleaveEligible: the auto heuristic fired — a witnessed-like host.
	DecodeInterleaveEligible DecodeInterleaveReason = "eligible"
	// DecodeInterleaveOverrideOn: FAK_NUMA_INTERLEAVE=on forced apply past the auto heuristic.
	DecodeInterleaveOverrideOn DecodeInterleaveReason = "override_on"
	// DecodeInterleaveOverrideOff: FAK_NUMA_INTERLEAVE=off — operator opted out.
	DecodeInterleaveOverrideOff DecodeInterleaveReason = "override_off"
	// DecodeInterleaveUnsupported: no mbind apply on this platform (not linux/amd64).
	DecodeInterleaveUnsupported DecodeInterleaveReason = "unsupported"
	// DecodeInterleaveSingleNode: fewer than 2 online NUMA nodes — nothing to stripe across.
	DecodeInterleaveSingleNode DecodeInterleaveReason = "single_node"
	// DecodeInterleaveConstrained: a strict membind/cpuset owns placement. Interleaving would
	// fight it and can OOM-kill the bound node (see mempolicy.go) — never override, even on force.
	DecodeInterleaveConstrained DecodeInterleaveReason = "constrained_policy"
	// DecodeInterleaveAlreadyPlaced: the operator already set a non-default task policy
	// (e.g. numactl --interleave=all). The witnessed regime is already present; do not
	// double-apply. This is a success state, not a failure.
	DecodeInterleaveAlreadyPlaced DecodeInterleaveReason = "already_placed"
	// DecodeInterleaveBelowManycore: linux/amd64 with >=2 nodes but too few cores for the
	// witnessed manycore regime. Auto stays conservative here (untested); force with =on.
	DecodeInterleaveBelowManycore DecodeInterleaveReason = "below_manycore"
)

// decodeInterleaveManycoreFloor mirrors the resident-Q4_K worker cap's manycore gate
// (budget.go: amd64 && workers>=64): auto placement only fires where that cap also bites,
// i.e. the witnessed regime. Operators force smaller boxes with FAK_NUMA_INTERLEAVE=on.
const decodeInterleaveManycoreFloor = 64

// decodeInterleaveOverride is the operator knob parsed from FAK_NUMA_INTERLEAVE.
type decodeInterleaveOverride uint8

const (
	interleaveAuto decodeInterleaveOverride = iota
	interleaveForceOn
	interleaveForceOff
)

// decodeInterleaveOverrideFromEnv parses FAK_NUMA_INTERLEAVE. "" / "auto" ⇒ auto,
// "on"/"1"/"true"/"yes" ⇒ force on, "off"/"0"/"false"/"no" ⇒ force off. An unrecognized
// value is treated as auto (fail-open to the safe heuristic, not a hard error).
func decodeInterleaveOverrideFromEnv(getenv func(string) string) decodeInterleaveOverride {
	switch strings.ToLower(strings.TrimSpace(getenv("FAK_NUMA_INTERLEAVE"))) {
	case "on", "1", "true", "yes":
		return interleaveForceOn
	case "off", "0", "false", "no":
		return interleaveForceOff
	default:
		return interleaveAuto
	}
}

// decodeInterleaveSnapshot is the pure input to the planner: everything the verdict reads,
// captured once so the decision is a total function of the host snapshot and the override.
type decodeInterleaveSnapshot struct {
	goos, goarch string
	online       []int  // online NUMA node ids (from /sys/devices/system/node/online)
	cores        int    // runtime.NumCPU()
	constrained  bool   // strict membind/cpuset confinement
	policyLabel  string // raw task policy token: "", "default", "interleave:0-7", "bind:0", ...
}

// DecodeInterleavePlan is a pre-apply verdict. Nodes is populated only when Apply is true.
type DecodeInterleavePlan struct {
	Apply  bool
	Reason DecodeInterleaveReason
	Nodes  []int
}

// planDecodeInterleave is the pure decision core. Refusal order is deliberate: an operator
// opt-out wins over everything; a strict bind is never overridden (even by force-on); the
// platform/topology gates come before the auto-only heuristics so force-on cannot claim an
// apply the mbind shim cannot perform.
func planDecodeInterleave(s decodeInterleaveSnapshot, ov decodeInterleaveOverride) DecodeInterleavePlan {
	refuse := func(r DecodeInterleaveReason) DecodeInterleavePlan {
		return DecodeInterleavePlan{Reason: r}
	}
	apply := func(r DecodeInterleaveReason) DecodeInterleavePlan {
		return DecodeInterleavePlan{Apply: true, Reason: r, Nodes: append([]int(nil), s.online...)}
	}

	if ov == interleaveForceOff {
		return refuse(DecodeInterleaveOverrideOff)
	}
	// A strict membind/cpuset owns placement; interleaving would fight it and can OOM the
	// bound node. Refuse for BOTH auto and force-on — force forces the heuristic, not safety.
	if s.constrained {
		return refuse(DecodeInterleaveConstrained)
	}
	// The mbind apply exists only on linux/amd64; refuse elsewhere so force-on cannot claim
	// an apply that would immediately fail in the shim.
	if s.goos != "linux" || s.goarch != "amd64" {
		return refuse(DecodeInterleaveUnsupported)
	}
	if len(s.online) < 2 {
		return refuse(DecodeInterleaveSingleNode)
	}
	if ov == interleaveForceOn {
		// Force bypasses the manycore floor and the already-placed deference: the operator
		// has explicitly asked for in-process interleave on this box.
		return apply(DecodeInterleaveOverrideOn)
	}
	// Auto: defer to an operator policy already in place (e.g. numactl --interleave=all) —
	// the regime is present, re-applying would be redundant.
	if p := normalizeDecodePolicyLabel(s.policyLabel); p != "" && p != "default" {
		return refuse(DecodeInterleaveAlreadyPlaced)
	}
	// Auto only fires on the witnessed manycore regime; smaller boxes are untested for this
	// lever, so stay conservative and let operators force with =on.
	if s.cores < decodeInterleaveManycoreFloor {
		return refuse(DecodeInterleaveBelowManycore)
	}
	return apply(DecodeInterleaveEligible)
}

// normalizeDecodePolicyLabel trims a numa_maps policy token to its head word so
// "interleave:0-7" and "bind:0" compare as "interleave"/"bind"; "" and "default" both
// mean an unmanaged first-touch policy.
func normalizeDecodePolicyLabel(label string) string {
	label = strings.TrimSpace(strings.ToLower(label))
	if i := strings.IndexByte(label, ':'); i >= 0 {
		label = label[:i]
	}
	return label
}

// liveDecodeInterleaveSnapshot reads the current host into a planner snapshot.
func liveDecodeInterleaveSnapshot() decodeInterleaveSnapshot {
	st := ReadHostMemStatus()
	return decodeInterleaveSnapshot{
		goos:        runtime.GOOS,
		goarch:      runtime.GOARCH,
		online:      onlineNUMANodes(),
		cores:       runtime.NumCPU(),
		constrained: st.Constrained,
		policyLabel: st.PolicyLabel,
	}
}

// PlanDecodeInterleave returns the live verdict for whether the ordinary CPU decode path
// should apply NUMA interleave placement to the resident weights on this host.
func PlanDecodeInterleave() DecodeInterleavePlan {
	return planDecodeInterleave(liveDecodeInterleaveSnapshot(), decodeInterleaveOverrideFromEnv(os.Getenv))
}

// DecodeInterleaveResult is the outcome of applying (or declining to apply) interleave
// placement. RegionsPlaced counts regions successfully mbind'd; Err is the first mbind
// failure, if any (fail-visible — a partial apply is reported, never silently swallowed).
type DecodeInterleaveResult struct {
	Plan          DecodeInterleavePlan
	RegionsPlaced int
	Err           error
}

// Label renders the decision for a decode witness / bench line, e.g.
// "interleave=applied(reason=eligible,nodes=0-7,regions=339)" or
// "interleave=skipped(reason=single_node)" or "interleave=error(reason=eligible,err=...)".
func (r DecodeInterleaveResult) Label() string {
	if r.Err != nil {
		return "interleave=error(reason=" + string(r.Plan.Reason) + ",err=" + r.Err.Error() + ")"
	}
	if !r.Plan.Apply {
		return "interleave=skipped(reason=" + string(r.Plan.Reason) + ")"
	}
	return "interleave=applied(reason=" + string(r.Plan.Reason) +
		",nodes=" + formatNodeList(r.Plan.Nodes) +
		",regions=" + strconv.Itoa(r.RegionsPlaced) + ")"
}

// ApplyDecodeInterleave runs the live verdict and, when it says apply, mbinds each weight
// region to MPOL_INTERLEAVE across the online nodes (the in-process numactl --interleave=all).
// Empty regions are skipped. On the first mbind error it stops and reports it via Result.Err
// so a partial placement is never mistaken for a full one.
func ApplyDecodeInterleave(regions [][]byte) DecodeInterleaveResult {
	plan := PlanDecodeInterleave()
	res := DecodeInterleaveResult{Plan: plan}
	if !plan.Apply {
		return res
	}
	for _, region := range regions {
		if len(region) == 0 {
			continue
		}
		if err := mbindInterleave(region, plan.Nodes); err != nil {
			res.Err = err
			return res
		}
		res.RegionsPlaced++
	}
	return res
}
