package issueorchestrator

import "strings"

// hardwareValidationGuidance routes software workers through existing requests and
// source-bound operator receipts; prompt text does not grant hardware execution.
func hardwareValidationGuidance(issue Issue) string {
	if !needsHaloValidation(issue) {
		return ""
	}
	return "\nPhysical hardware validation:\n" +
		"- Software validation and landing precede physical qualification gates (author deterministic test witness, implement atomic fix, pass package tests, and land green software increment before physical device qualification).\n" +
		"- During investigation, read the current hardware catalog (experiments/benchmark/catalog.json; docs/notes/HARDWARE-CATALOG.md) and existing request/results before proposing hardware work. For V4.1 CUDA use docs/notes/REQUEST-DS41-RAIDER-HARDWARE-VERIFICATION-2026-10-28.md and its existing #12640/#13993 request links and receipt schema. For other work, use its already-authorized target request; do not invent an endpoint or duplicate tracker. Reuse historical evidence within its exact source/model/build/device scope and request only missing changed-path qualification.\n" +
		"- Halo targets are retired. Do not probe or request their maintenance/execution. A coordinator desktop connection or desktop quota is not a prerequisite. Software/cloud workers prepare the bounded request; only an independently authorized hardware operator claims a currently admitted target. This guidance grants no access, spending, reservation or execution authority.\n" +
		"- The operator must follow the actual target's ownership/admission protocol: shared lease only for compatible low-impact inspection, exclusive lease across baseline/candidate performance measurements, and do not nest locks. Busy or unavailable hardware means an exact blocker and independent software work, never bypassing ownership.\n" +
		"- Before physical qualification is marked done, the operator must execute the changed code on the admitted device and return a source-bound receipt: exact source revision plus dirty patch digest when applicable, built executable digest, actual engine/device identity, command, lease mode, exit status, observed output and teardown. Keep native execution fak-native; a historical binary or generic probe cannot prove the changed path ran.\n" +
		"- Performance claims additionally require matched baseline/candidate workload and configuration with measured timings under exclusive access. A correctness PASS is not a speedup measurement. If hardware is unavailable/busy or current source binding is missing, report hardware validation PENDING (PENDING_HARDWARE) with the existing request and exact next operator action. Unit tests and out-of-scope historical receipts cannot satisfy the missing physical witness. The green software increment must still land while physical qualification is pending.\n"
}

func needsHaloValidation(issue Issue) bool {
	for _, p := range append(append([]string(nil), issue.Paths...), "internal/"+issue.Lane) {
		p = strings.ToLower(strings.ReplaceAll(p, "\\", "/"))
		p = strings.TrimPrefix(p, "./")
		for _, root := range []string{"internal/amdgpu", "internal/compute", "internal/roofline", "internal/nativeperf"} {
			if p == root || strings.HasPrefix(p, root+"/") {
				return true
			}
		}
		if strings.Contains(p, "vulkan") || strings.Contains(p, "strix") {
			return true
		}
	}
	text := strings.ToLower(issue.Title + " " + strings.Join(issue.Labels, " "))
	if strings.Contains(text, "perf") || strings.Contains(text, "benchmark") || strings.Contains(text, "throughput") || strings.Contains(text, "latency") {
		for _, p := range append(append([]string(nil), issue.Paths...), "internal/"+issue.Lane) {
			p = strings.ToLower(strings.ReplaceAll(p, "\\", "/"))
			p = strings.TrimPrefix(p, "./")
			if p == "internal/model" || strings.HasPrefix(p, "internal/model/") {
				return true
			}
		}
	}
	for _, marker := range []string{"strix", "vulkan", "gfx1151", "amdgpu", "nativeperf", "amd gpu", "amd performance"} {
		if strings.Contains(text, marker) {
			return true
		}
	}
	for _, word := range strings.Fields(strings.ReplaceAll(text, "/", " ")) {
		if word == "halo" {
			return true
		}
	}
	return false
}
