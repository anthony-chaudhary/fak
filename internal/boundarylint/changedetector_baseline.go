package boundarylint

import (
	"path/filepath"
)

// changedetector_baseline.go — the CHANGE_DETECTOR_TEST ratchet baseline.
//
// The rule (rules_changedetector.go) flags _test.go assertions that freeze a current
// value instead of asserting an invariant. The tree already carries dozens of such
// tells — some genuine change-detectors, some deliberate fixed-width invariants the
// heuristic can't yet distinguish (a sha256 hex width, a 16-byte id). Turning the
// gate on as a hard failure over the whole tree would red the trunk on day one, the
// same self-defeating mistake as a Python ratchet that banned all Python.
//
// So the gate is a RATCHET, exactly like internal/pythongate: it does not fail on the
// grandfathered set below — the files that carried a change-detector tell the day the
// rule landed — it fails only on a NEW file that introduces one. The suite can burn
// the backlog down (convert a tell to an invariant, or //boundarylint:ignore a
// deliberate fixed-width check) and refreeze this list SMALLER; it can never grow. A
// file dropping out of the tracked set simply stops appearing in the live scan and is
// harmlessly stale here. That makes the change-detector surface monotonically
// decrease while stopping the suite from accreting new ones as the kernel grows —
// which is the whole point of #2867.
//
// # Regenerating (only ever to TIGHTEN, after a burn-down)
//
// After converting the last tell in a file to an invariant (or suppressing a genuine
// fixed-width check in place), refreeze from the live scan so the now-smaller set is
// captured:
//
//	go test ./internal/boundarylint/ -run TestTestSuitePolicy -count=1 2>&1 \
//	  | grep -oE '(cmd|internal)/[A-Za-z0-9_/]+_test\.go' | sort -u | sed 's/.*/\t"&",/'
//
// then paste the result between the braces below. Because the recipe reads the live
// findings, the regenerated list is always a subset of this one.
var changeDetectorBaseline = []string{
	"cmd/batchbench/batchbench_test.go",
	"cmd/ctxplandemo/main_test.go",
	"cmd/fak/ablate_test.go",
	"cmd/fak/attest_test.go",
	"cmd/fak/computetarget_test.go",
	"cmd/fak/dispatch_tick_test.go",
	"cmd/fak/frontierswe_cachewitness_test.go",
	"cmd/fak/frontierswe_test.go",
	"cmd/fak/info_visual_test.go",
	"cmd/fak/lab_test.go",
	"cmd/fak/release_ship_test.go",
	"cmd/fak/swebench_smoke_contract_test.go",
	"cmd/fak/test_test.go",
	"cmd/fak/vcache_observe_test.go",
	"cmd/fak/version_identity_test.go",
	"cmd/fanbench/main_test.go",
	"cmd/fleetserve/workload_test.go",
	"cmd/ggufprobe/ggufprobe_test.go",
	"cmd/guarddemo/main_test.go",
	"cmd/lensviz/main_test.go",
	"cmd/tokendemo/main_test.go",
	"internal/agent/ctxplan_session_test.go",
	"internal/ailuminate/contract_test.go",
	"internal/auditusage/auditusage_test.go",
	"internal/benchcli/lineage_test.go",
	"internal/benchpost/benchpost_test.go",
	"internal/cachemeta/prefix_transcript_test.go",
	"internal/cadencereport/cadencereport_test.go",
	"internal/closebatch/closebatch_test.go",
	"internal/cmdutil/cmdutil_test.go",
	"internal/compute/kvreplay_trace_test.go",
	"internal/ctxmmu/ctxmmu_test.go",
	"internal/devindex/devindex_test.go",
	"internal/dispatchtick/priority_test.go",
	"internal/dormancysim/dormancysim_test.go",
	"internal/fleet/fleet_test.go",
	"internal/fleetpane/fleetpane_test.go",
	"internal/fleettrend/fleettrend_test.go",
	"internal/frontierswe/task_test.go",
	"internal/gateway/harness_coherence_test.go",
	"internal/gateway/openai_parity_test.go",
	"internal/ggufload/dequant_reuse_test.go",
	"internal/ggufload/gemma4_test.go",
	"internal/ggufload/gguf_test.go",
	"internal/journal/rotate_test.go",
	"internal/learningdebt/learningdebt_test.go",
	"internal/memview/memview_test.go",
	"internal/metrics/metrics_test.go",
	"internal/milestonereport/scorecard_test.go",
	"internal/popularizationtickets/tickets_test.go",
	"internal/randhex/randhex_test.go",
	"internal/releasestale/releasestale_test.go",
	"internal/resume/stopped/stopped_test.go",
	"internal/selfinstall/reap_test.go",
	"internal/session/revision_test.go",
	"internal/sessionimage/rehydrate_gate_test.go",
	"internal/superloop/superloop_test.go",
	"internal/supportmaturity/supportmaturity_test.go",
	"internal/taskidentity/taskidentity_test.go",
	"internal/toolproc/toolproc_test.go",
	"internal/trajctl/trajctl_test.go",
	"internal/workflow/workflow_test.go",
}

// changeDetectorBaselineSet materializes the grandfathered slice into a lookup set,
// keyed by slash-separated repo-relative path.
func changeDetectorBaselineSet() map[string]bool {
	set := make(map[string]bool, len(changeDetectorBaseline))
	for _, p := range changeDetectorBaseline {
		set[p] = true
	}
	return set
}

// ScanNewChangeDetectors runs DefaultTestRules over root's cmd/ and internal/ test
// trees and returns only findings in files NOT grandfathered in changeDetectorBaseline
// — the NEW change-detectors the ratchet fails on. Each returned Finding.File is
// rewritten to a repo-relative slash path so it reads the same as the baseline. This
// is the single shared entrypoint for both TestTestSuitePolicy (the gate) and
// `fak boundary` (the report), so the shrink-only ratchet is defined in exactly one
// place.
func ScanNewChangeDetectors(root string) ([]Finding, error) {
	findings, err := ScanTests(
		[]string{filepath.Join(root, "cmd"), filepath.Join(root, "internal")},
		DefaultTestRules(),
	)
	if err != nil {
		return nil, err
	}
	baseline := changeDetectorBaselineSet()
	var out []Finding
	for _, f := range findings {
		rel, err := filepath.Rel(root, filepath.FromSlash(f.File))
		if err != nil {
			continue
		}
		relSlash := filepath.ToSlash(rel)
		if baseline[relSlash] {
			continue
		}
		f.File = relSlash
		out = append(out, f)
	}
	return out, nil
}
