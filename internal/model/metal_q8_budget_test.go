package model

import (
	"errors"
	"testing"
)

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

// TestQ6KAliasFitsAdmitsUnifiedResidentBand pins the unified-memory admission for the Q6_K no-copy
// alias: the band is judged only against activation/KV headroom (metalQ6AliasFraction), NOT the
// additive resident+q6 sum. The witnessed failure was a 5.89 GiB Q6_K band (fused-MLP down_proj +
// Q6_K head) refused by the additive gate while 15.92 GiB was already resident on a 36 GiB Mac; the
// alias adds no device bytes, so it must be ADMITTED.
func TestQ6KAliasFitsAdmitsUnifiedResidentBand(t *testing.T) {
	const GiB = int64(1) << 30
	gib := float64(GiB)
	resident := int64(15.92 * gib)
	q6 := int64(5.89 * gib)
	device := int64(36 * GiB)
	// The additive #1087 model on the same box is DECLINED at a 24 GiB working set, proving the two
	// gates disagree — the alias is the path that admits.
	if q6kUploadFits(resident, q6, int64(24*GiB)) {
		t.Fatal("additive q6kUploadFits admitted 15.92+5.89 GiB on a 24 GiB budget; want refusal")
	}
	if err := q6kAliasFits(resident, device); err != nil {
		t.Fatalf("q6kAliasFits(resident=%.2fGiB, device=36GiB) = %v, want nil (aliased band is not additive)", float64(resident)/gib, err)
	}
	// An already-saturating owner still fails closed and carries the concrete reason.
	err := q6kAliasFits(int64(35.5*gib), device)
	if err == nil {
		t.Fatal("q6kAliasFits admitted a 35.5/36 GiB owner; want headroom refusal")
	}
	var unavailable *MetalQ6ResidencyUnavailableError
	if !errors.As(err, &unavailable) || unavailable.Reason == "" {
		t.Fatalf("q6kAliasFits decline = %#v, want *MetalQ6ResidencyUnavailableError with a reason", err)
	}
	if err := q6kAliasFits(resident, 0); err == nil {
		t.Fatal("q6kAliasFits admitted an unknown device budget; want fail-closed refusal")
	}
}

// TestQ6KAdditiveGateStillGuardsCopies pins that an unaligned/borrowed (copied) payload keeps the
// exact #1087 additive behaviour: refused when resident+copy would breach 0.90*device, admitted
// with room. The alias predicate must not be reachable for such a payload.
func TestQ6KAdditiveGateStillGuardsCopies(t *testing.T) {
	const GiB = int64(1) << 30
	gib := float64(GiB)
	resident := int64(15.92 * gib)
	q6 := int64(5.89 * gib)
	if q6kUploadFits(resident, q6, 36*GiB) == false {
		t.Fatal("additive copy with room on a 36 GiB budget should fit")
	}
	if q6kUploadFits(resident, q6, int64(24*GiB)) == true {
		t.Fatal("additive copy must be refused when resident+copy breaches 0.90*device")
	}
	if q6kUploadFits(resident, q6, 0) == true {
		t.Fatal("additive copy must be refused when the device budget is unknown")
	}
}
