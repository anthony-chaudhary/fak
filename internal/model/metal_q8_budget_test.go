package model

import "testing"

// TestQ8UploadFits pins the #1087-OOM budget gate: the additive Q8 GPU copy is declined when the
// resident weights plus that copy would breach metalQ8UploadFraction of the device working-set
// budget, and admitted when there is room — with the FAK_METAL_Q8_UPLOAD override winning either way.
func TestQ8UploadFits(t *testing.T) {
	const GiB = int64(1) << 30
	cases := []struct {
		name         string
		resident, q8 int64
		deviceTotal  int64
		forceEnv     string
		want         bool
	}{
		{
			// The measured OOM: 27B q4_k_m resident ~23 GiB + ~4 GiB Q8 projection copy on an
			// M3 Pro whose Metal working-set budget is ~27 GiB. 27 GiB projected vs 0.90*27 =
			// 24.3 GiB → does NOT fit → decline (un-regresses the SIGKILL).
			name: "27b_on_36g_mac_declines", resident: 23 * GiB, q8: 4 * GiB,
			deviceTotal: 27 * GiB, want: false,
		},
		{
			// A roomy device (e.g. a 96 GiB working set) easily absorbs the same footprint → allow,
			// so #1087's Metal-Q8 prefill win is kept where it is safe.
			name: "roomy_device_allows", resident: 23 * GiB, q8: 4 * GiB,
			deviceTotal: 96 * GiB, want: true,
		},
		{
			// Exactly at the fraction boundary fits (<=): projected 27 GiB == 0.90*30 GiB.
			name: "at_boundary_fits", resident: 23 * GiB, q8: 4 * GiB,
			deviceTotal: 30 * GiB, want: true,
		},
		{
			// One byte over the boundary declines.
			name: "one_over_boundary_declines", resident: 23 * GiB, q8: 4*GiB + 1,
			deviceTotal: 30 * GiB, want: false,
		},
		{
			// Unknown device budget (probe unavailable) is conservative: decline.
			name: "unknown_device_declines", resident: 1 * GiB, q8: 1 * GiB,
			deviceTotal: 0, want: false,
		},
		{
			// Force ON overrides an over-budget estimate (operator opt-in on a box they trust).
			name: "force_on_overrides_over_budget", resident: 23 * GiB, q8: 4 * GiB,
			deviceTotal: 27 * GiB, forceEnv: "1", want: true,
		},
		{
			// Force OFF overrides a comfortable fit (operator wants CPU Q8 regardless).
			name: "force_off_overrides_fit", resident: 1 * GiB, q8: 1 * GiB,
			deviceTotal: 96 * GiB, forceEnv: "0", want: false,
		},
		{
			// Force ON even when the device budget is unknown (bypasses the conservative default).
			name: "force_on_with_unknown_device", resident: 1 * GiB, q8: 1 * GiB,
			deviceTotal: 0, forceEnv: "on", want: true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := q8UploadFits(c.resident, c.q8, c.deviceTotal, c.forceEnv); got != c.want {
				t.Fatalf("q8UploadFits(resident=%d, q8=%d, dev=%d, env=%q) = %v, want %v",
					c.resident, c.q8, c.deviceTotal, c.forceEnv, got, c.want)
			}
		})
	}
}
