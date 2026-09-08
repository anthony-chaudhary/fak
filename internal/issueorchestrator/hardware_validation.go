package issueorchestrator

import "strings"

// hardwareValidationGuidance makes the LAN appliance part of investigation and
// acceptance for relevant workers, without treating prompt text as an execution gate.
func hardwareValidationGuidance(issue Issue) string {
	if !needsHaloValidation(issue) {
		return ""
	}
	return "\nPhysical hardware validation:\n" +
		"- Software validation and landing precede physical qualification gates (author deterministic test witness, implement atomic fix, pass package tests, and land green software increment before physical device qualification).\n" +
		"- During investigation, run `fak-dev amd-strix-probe` early to discover the LAN AMD Strix Halo and choose a bounded device correctness witness. Read docs/fleet-compute-nodes.md for the sanctioned route.\n" +
		"- Coordinate access on the appliance using its canonical GPU lock (`FAK_GPU_LEASE`, default /tmp/fak-gpu.lease). Use a shared lease for low-impact inspection or explicitly compatible bounded correctness checks; hold an exclusive lease across the entire baseline/candidate performance comparison. External test executables can use `flock -w 30 -x /tmp/fak-gpu.lease <command>` remotely; use the configured lock path when overridden. Commands that already lease internally, such as modelbench, do not nest locks. Busy hardware means bounded waiting or independent local work, never bypassing the lease.\n" +
		"- Before done, execute the changed code on the physical device and capture a source-bound receipt: exact source revision plus dirty patch digest when applicable, built executable digest, actual engine/device identity, command, lease mode, exit status, and observed output. Keep native execution fak-native. `fak validate --strix` alone does not establish that a pre-existing remote executable contains your change.\n" +
		"- Performance claims additionally require matched baseline/candidate workload and configuration with measured timings under exclusive access. A correctness PASS is not a speedup measurement. If hardware is unavailable/busy or source binding is missing, report hardware validation PENDING (PENDING_HARDWARE) with the exact next command; unit tests or historical receipts cannot satisfy the physical witness. The green software increment must still land while physical qualification is pending.\n"
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
