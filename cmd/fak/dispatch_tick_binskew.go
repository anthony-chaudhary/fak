package main

// dispatch_tick_binskew.go — the impure shell that turns the preflight's ALREADY-MEASURED
// fak-bin provenance into a spawn refusal (#6508, Done condition 3).
//
// tools/dispatch_preflight.py has reported `FAK_BIN_DISAGREEMENT` for a while: the repo-root
// binary adjudicating the gate was an `+uncommitted` compile on a different revision than the
// tools/.bin copy fronting every worker it admitted. That warning changed no verdict, so the
// tick answered SPAWN_OK and unreviewed code kept deciding who may run.
//
// Everything decidable lives in internal/selfinstall.ApplyGateSkew, which has real tests. This
// file only does what a shell may do: read the payload the gate already produced, read one env
// knob, and hand both to the fold. No new process is spawned — the Python gate paid for the
// `fak version` probes already, so the refusal costs a map lookup.

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// allowBinSkewEnv is the operator escape hatch. The repo-root gate binary is routinely a
// hand-build in a shared maintainer checkout, so a refusal with no override would freeze the
// fleet exactly when someone is mid-test on it. An overridden tick still SAYS so in its reason.
const allowBinSkewEnv = "FAK_PREFLIGHT_ALLOW_BIN_SKEW"

// dispatchGateProvenance lifts the preflight payload's `fak_bin` block into the measured shape
// the fold consumes. A payload without the block (an older gate, a probe that did not run)
// yields the zero value, which never refuses.
//
// It reads `build` — the real VCS revision — and never `build_key`, which is
// `<size>-<mtime_ns>-<basename>`: two byte-identical copies have different keys, so keying skew
// off it would report every host as divergent.
func dispatchGateProvenance(pre map[string]any, env func(string) string) selfinstall.GateProvenance {
	block := mapAt(pre, "fak_bin")
	if len(block) == 0 {
		return selfinstall.GateProvenance{}
	}
	resolvers := mapAt(block, "resolvers")
	gate, worker := mapAt(resolvers, "preflight_gate"), mapAt(resolvers, "worker_guard")
	p := selfinstall.GateProvenance{Probed: true, Allow: dispatchAllowBinSkew(env)}
	p.GatePath, p.GateResolved = dispatchMapString(gate, "path"), dispatchMapBool(gate, "resolved")
	p.GateBuild = strings.TrimSpace(dispatchMapString(gate, "build"))
	p.GateAttested, p.GateDirty = p.GateBuild != "", dispatchMapBool(gate, "dirty")
	p.WorkerPath, p.WorkerResolved = dispatchMapString(worker, "path"), dispatchMapBool(worker, "resolved")
	p.WorkerBuild = strings.TrimSpace(dispatchMapString(worker, "build"))
	p.WorkerAttested = p.WorkerBuild != ""
	return p
}

// dispatchAllowBinSkew reads the override. Absent env lookup (nil) means os.Getenv.
func dispatchAllowBinSkew(env func(string) string) bool {
	if env == nil {
		env = os.Getenv
	}
	switch strings.ToLower(strings.TrimSpace(env(allowBinSkewEnv))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// dispatchApplyBinSkew folds the refusal into the preflight payload in place, returning whether
// the verdict changed so the caller can annotate. The payload keys it rewrites are exactly the
// two the tick already branches on, so a refusal here flows through the SAME `!preOK` path as
// every other refusal rather than opening a second exit.
func dispatchApplyBinSkew(pre map[string]any, okVerdict string) bool {
	if pre == nil {
		return false
	}
	prov := dispatchGateProvenance(pre, nil)
	was := dispatchMapString(pre, "verdict")
	verdict, reason := selfinstall.ApplyGateSkew(was, dispatchMapString(pre, "reason"), okVerdict, prov)
	pre["verdict"], pre["reason"] = verdict, reason
	return verdict != was
}
