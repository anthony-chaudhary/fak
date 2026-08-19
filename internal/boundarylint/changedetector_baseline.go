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
//
// # When the ratchet reds on a NEW tell (do NOT add it below)
//
// TestTestSuitePolicy fails in THIS package, but the file it names almost never lives
// here — a NEW tell is introduced under cmd/… or internal/<other>/…, whose author runs
// their own lane's suite and so never sees this gate. The red is therefore inherited by
// every clone until someone triages it, and the tempting clear — pasting the named file
// into the list below — is the one move the ratchet forbids: it is shrink-only, and
// grandfathering a NEW tell retires the signal instead of paying it. Pay it at the
// source file instead, in one of exactly two ways:
//
//   - assert the relation the frozen value stands for, so the assertion still holds
//     after a legitimate change to the count — #5312 converted a frozen `len(sel) != 5`
//     into `const k = 5; len(sel) != k`, whose invariant is "sampleTopK returns exactly
//     k distinct experts";
//   - or, when the width is a genuine algorithm invariant rather than a growable total,
//     mark it in place with a trailing //boundarylint:ignore CHANGE_DETECTOR_TEST and
//     the reason it cannot drift — #5312 suppressed an 8-hex scrubbed-hash width, the
//     same class as sha256 hex being 64.
//
// Either way the fix lands in the source file's own lane, not this one.
var changeDetectorBaseline = []string{
	"cmd/batchbench/batchbench_test.go",
	"cmd/ctxplandemo/main_test.go",
	"cmd/disambiguationdemo/main_test.go",
	"cmd/dispatchworker/guard_test.go",
	"cmd/fak/ablate_test.go",
	"cmd/fak/attest_test.go",
	"cmd/fak/budget_readout_test.go",
	"cmd/fak/computetarget_test.go",
	"cmd/fak/disambiguation_claims_source_test.go",
	"cmd/fak/disambiguation_fleet_source_test.go",
	"cmd/fak/disambiguation_policy_source_test.go",
	"cmd/fak/disambiguation_reason_source_test.go",
	"cmd/fak/disambiguation_runtime_source_test.go",
	"cmd/fak/disambiguation_session_source_test.go",
	"cmd/fak/disambiguation_version_test.go",
	"cmd/fak/dispatch_sessions_audit_test.go",
	"cmd/fak/dispatch_tick_test.go",
	"cmd/fak/doctor_launch_posture_test.go",
	"cmd/fak/doomloop_test.go",
	"cmd/fak/frontierswe_cachewitness_test.go",
	"cmd/fak/frontierswe_test.go",
	"cmd/fak/guard_profiles_test.go",
	"cmd/fak/guard_response_profile_test.go",
	"cmd/fak/info_tabs_test.go",
	"cmd/fak/info_visual_test.go",
	"cmd/fak/info_work_coverage_test.go",
	"cmd/fak/lab_test.go",
	"cmd/fak/milestone_test.go",
	"cmd/fak/multisubmit_test.go",
	"cmd/fak/recovery_crash_matrix_test.go",
	"cmd/fak/release_ship_test.go",
	"cmd/fak/resume_watchdog_candidates_test.go",
	"cmd/fak/route_place_citation_test.go",
	"cmd/fak/session_new_test.go",
	"cmd/fak/swebench_smoke_contract_test.go",
	"cmd/fak/test_test.go",
	"cmd/fak/toolcall_control_test.go",
	"cmd/fak/vcache_observe_test.go",
	"cmd/fak/version_identity_test.go",
	"cmd/fanbench/main_test.go",
	"cmd/fleetserve/workload_test.go",
	"cmd/ggufprobe/ggufprobe_test.go",
	"cmd/guarddemo/main_test.go",
	"cmd/lensviz/main_test.go",
	"cmd/microcontextdemo/filter_tool_scheduler_test.go",
	"cmd/microcontextdemo/natural_traffic_test.go",
	"cmd/microcontextdemo/semantic_residual_test.go",
	"cmd/portability-adapter-selfcheck/main_test.go",
	"cmd/tokendemo/main_test.go",
	"cmd/toolcallcontroldemo/main_test.go",
	"internal/ablate/cachelevers_test.go",
	"internal/accounts/credfamily_test.go",
	"internal/accounts/login_test.go",
	"internal/agent/codetools_loop_test.go",
	"internal/agent/ctxplan_session_test.go",
	"internal/ailuminate/contract_test.go",
	"internal/archreport/report_test.go",
	"internal/armbench/caveman_factorial_test.go",
	"internal/armbench/passthrough_test.go",
	"internal/armbench/ponytail_gates_test.go",
	"internal/armbench/ponytail_test.go",
	"internal/armbench/promptfoo_test.go",
	"internal/auditusage/auditusage_test.go",
	"internal/bench/cpumemstress_test.go",
	"internal/bench/ctxadmission_test.go",
	"internal/bench/workspaceselectivity_test.go",
	"internal/benchcli/lineage_test.go",
	"internal/benchpost/benchpost_test.go",
	"internal/cachemeta/prefix_transcript_test.go",
	"internal/cachevalue/cachevalue_test.go",
	"internal/cadencereport/cadencereport_test.go",
	"internal/closebatch/closebatch_test.go",
	"internal/cmdutil/cmdutil_test.go",
	"internal/codetools/engines_test.go",
	"internal/compute/decode_occupancy_test.go",
	"internal/compute/kvreplay_trace_test.go",
	"internal/ctxmmu/compare_test.go",
	"internal/ctxmmu/ctxmmu_test.go",
	"internal/cubicquanteval/contract_test.go",
	"internal/deepseekv4moe/cache_trace_test.go",
	"internal/deliverystages/registry_test.go",
	"internal/deploymanifest/deploymanifest_test.go",
	"internal/devindex/devindex_test.go",
	"internal/devindex/ownership_baseline_test.go",
	"internal/disambiguation/claims_source_test.go",
	"internal/disambiguation/fleet_source_test.go",
	"internal/disambiguation/policy_source_test.go",
	"internal/disambiguation/reason_source_test.go",
	"internal/disambiguation/runtime_source_test.go",
	"internal/disambiguation/session_source_test.go",
	"internal/disambiguation/version_test.go",
	"internal/dispatchaudit/signatures_test.go",
	"internal/dispatchconservation/tail_test.go",
	"internal/dispatchtick/priority_test.go",
	"internal/dojo/claims_test.go",
	"internal/dormancysim/dormancysim_test.go",
	"internal/experiments/showcase_contract_test.go",
	"internal/fleet/fleet_test.go",
	"internal/fleetpane/fleetpane_test.go",
	"internal/fleettrend/fleettrend_test.go",
	"internal/flowmetrics/gather_test.go",
	"internal/flowmetrics/report_test.go",
	"internal/flowmetrics/score_test.go",
	"internal/frontierswe/task_test.go",
	"internal/gateway/harness_coherence_test.go",
	"internal/gateway/mcp_batch_test.go",
	"internal/gateway/native_code_health_test.go",
	"internal/gateway/native_serve_loop_test.go",
	"internal/gateway/openai_parity_test.go",
	"internal/gateway/session_move_test.go",
	"internal/ggufload/dequant_reuse_test.go",
	"internal/ggufload/gguf_bonsai_arch_test.go",
	"internal/ggufload/gguf_test.go",
	"internal/glm52prefillsweep/glm52prefillsweep_test.go",
	"internal/harnesscompose/receipts_test.go",
	"internal/harnessdiscover/harnessdiscover_test.go",
	"internal/harnessinit/harnessinit_test.go",
	"internal/harnessprotocol/protocol_test.go",
	"internal/harnessrelease/witness_test.go",
	"internal/harnessweb/web_test.go",
	"internal/harnessweb/workspace_status_test.go",
	"internal/issuecentrality/issuecentrality_test.go",
	"internal/issuecohort/issuecohort_test.go",
	"internal/issuehygiene/issuehygiene_test.go",
	"internal/journal/forest_test.go",
	"internal/journal/rotate_test.go",
	"internal/journal/witness_bound_test.go",
	"internal/kvint2eval/contract_test.go",
	"internal/launchlatency/compare_test.go",
	"internal/learningdebt/learningdebt_test.go",
	"internal/leaseref/reap_batch_test.go",
	"internal/lightgapport/lightgapport_test.go",
	"internal/lightgapscore/score_test.go",
	"internal/livecodebench/normalizeproblem_test.go",
	"internal/llamacppinterop/contract_test.go",
	"internal/loopmgr/rotate_test.go",
	"internal/memq/starve_test.go",
	"internal/memview/memview_test.go",
	"internal/metrics/metrics_test.go",
	"internal/microagent/qualityledger_test.go",
	"internal/milestonereport/scorecard_test.go",
	"internal/model/hiddentap_test.go",
	"internal/model/rope_yarn_longcontext_test.go",
	"internal/model/v4_expert_batch_test.go",
	"internal/model/v4_expert_parallel_test.go",
	"internal/model/v4_sharded_expert_source_test.go",
	"internal/model/v4_sharded_quant_test.go",
	"internal/model/vision_splice_test.go",
	"internal/modelaccept/modelaccept_test.go",
	"internal/modelroute/compare_test.go",
	"internal/modelroute/crossaudit_accidental_corpus_test.go",
	"internal/modelroute/evidencelog_trail_test.go",
	"internal/popularizationtickets/tickets_test.go",
	"internal/portability/adapter_test.go",
	"internal/portabilitylab/lab_test.go",
	"internal/quality/nightly_matrix_test.go",
	"internal/randhex/randhex_test.go",
	"internal/releasestale/releasestale_test.go",
	"internal/releasestatus/status_triage_test.go",
	"internal/resume/stopped/stopped_test.go",
	"internal/rotationmeta/contract_test.go",
	"internal/selfinstall/roles_test.go",
	"internal/session/revision_test.go",
	"internal/sessiondiag/watchdog_candidates_test.go",
	"internal/sessionimage/rehydrate_gate_test.go",
	"internal/sessionjournal/worksession_test.go",
	"internal/sessionread/mcpresources/mcpresources_test.go",
	"internal/sessionread/transcriptfeed/transcriptfeed_test.go",
	"internal/sessionrecovery/recovery_test.go",
	"internal/superloop/superloop_test.go",
	"internal/supportmaturity/supportmaturity_test.go",
	"internal/syspromptmmu/style_test.go",
	"internal/taskidentity/taskidentity_test.go",
	"internal/toolcallcontrol/replay_test.go",
	"internal/toolproc/toolproc_test.go",
	"internal/tooltrend/tooltrend_test.go",
	"internal/trajctl/detour_test.go",
	"internal/trajctl/quote_test.go",
	"internal/trajctl/trajctl_test.go",
	"internal/trajectory/runtime_event_test.go",
	"internal/workaccount/registry_test.go",
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
