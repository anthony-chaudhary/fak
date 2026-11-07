package ggufload

import "testing"

// engram_device_fit_test.go — V41C-19 phase-1 witnesses for the pure Engram device-admission
// model. The anchor vector is the measured H200 baseline, rtp-test-h200 run 20261106T023306Z
// (vcruz Q2_K c4a0855..., pin e2a6b10642, --native-context-tokens 16384, explicit 32 GiB ring).
//
// Exact byte derivations from the reported figures:
//   - device total 143,771 MiB            -> 143771 x 1024^2      = 150754820096 B (exact)
//   - dense core (fit_confirm weights)    -> 3,230,237,376 B      (reported exactly)
//   - ring bytes                          -> 34,359,738,368 B     (explicit 32 GiB, exact)
//   - KV pool 16384 x 240 KiB             -> 4,026,531,840 B     (exact; = 3.75 GiB)
//   - host_gathered 60.083 GiB            -> int64(60.083*1024^3) = 64513630011 B
//   - working set 2.863 GiB               -> int64(2.863*1024^3)  = 3074122842 B
//
// The two float-derived constants are display-rounded measurements, so their constants are the
// exact truncated products of the displayed values, not raw checkpoint byte counts.

const (
	h200DeviceTotalBytes = int64(150754820096) // 143,771 MiB
	h200DenseCoreBytes   = int64(3230237376)   // fit_confirm weights
	h200RingBytes        = int64(34359738368)  // 32 GiB explicit ring
	h200KVPoolBytes      = int64(4026531840)   // 16384 x 240 KiB
	h200EngramBytes      = int64(64513630011)  // 60.083 x 1024^3, truncated
	h200WorksetBytes     = int64(3074122842)   // 2.863 x 1024^3, truncated

	// Derived placement totals for the measured envelope (checked by the table below):
	h200FixedNoHeadroomBytes = h200DenseCoreBytes + h200RingBytes + h200KVPoolBytes + h200WorksetBytes // 44690630426
	h200FullResidentBytes    = h200FixedNoHeadroomBytes + h200EngramBytes                              // 109204260437
)

func TestModelEngramDeviceAdmission(t *testing.T) {
	gib := int64(1) << 30
	cases := []struct {
		name string
		b    EngramAdmissionBudget
		want EngramDeviceFit
	}{
		{
			// The measured H200 envelope: does the FULL 60.083 GiB Engram band admit to
			// device under 143,771 MiB with the 32 GiB ring, the 16384-token KV pool and
			// the 2.863 GiB per-token working set? The math says yes, with ~38.70 GiB of
			// slack left after the full placement — the admission blocker today is the
			// fit plan dropping the band by design, not the geometry.
			name: "h200 measured envelope admits the full engram band",
			b: EngramAdmissionBudget{
				DeviceTotalBytes: h200DeviceTotalBytes,
				DenseCoreBytes:   h200DenseCoreBytes,
				EngramBytes:      h200EngramBytes,
				RingBytes:        h200RingBytes,
				KVPoolBytes:      h200KVPoolBytes,
				WorkingSetBytes:  h200WorksetBytes,
			},
			want: EngramDeviceFit{
				AdmissibleBytes:     h200EngramBytes,
				Fits:                true,
				DeviceResidentBytes: h200FullResidentBytes,                        // 109204260437
				HeadroomBytes:       h200DeviceTotalBytes - h200FullResidentBytes, // 41550559659
			},
		},
		{
			// Same envelope with a 4 GiB required headroom reservation: still fits, and
			// the reserved headroom comes out of the slack, not the residency.
			name: "h200 envelope with 4GiB required headroom",
			b: EngramAdmissionBudget{
				DeviceTotalBytes: h200DeviceTotalBytes,
				DenseCoreBytes:   h200DenseCoreBytes,
				EngramBytes:      h200EngramBytes,
				RingBytes:        h200RingBytes,
				KVPoolBytes:      h200KVPoolBytes,
				WorkingSetBytes:  h200WorksetBytes,
				RequiredHeadroom: 4 * gib,
			},
			want: EngramDeviceFit{
				AdmissibleBytes:     h200EngramBytes,
				Fits:                true,
				DeviceResidentBytes: h200FullResidentBytes,
				HeadroomBytes:       h200DeviceTotalBytes - h200FullResidentBytes - 4*gib, // 37255592363
			},
		},
		{
			// No Engram tables: the verdict prices the remaining placement only.
			name: "engram zero fits when the rest fits",
			b: EngramAdmissionBudget{
				DeviceTotalBytes: h200DeviceTotalBytes,
				DenseCoreBytes:   h200DenseCoreBytes,
				RingBytes:        h200RingBytes,
				KVPoolBytes:      h200KVPoolBytes,
				WorkingSetBytes:  h200WorksetBytes,
			},
			want: EngramDeviceFit{
				AdmissibleBytes:     0,
				Fits:                true,
				DeviceResidentBytes: h200FixedNoHeadroomBytes,                        // 44690630426
				HeadroomBytes:       h200DeviceTotalBytes - h200FixedNoHeadroomBytes, // 106064189670
			},
		},
		{
			// Engram zero AND the fixed placement itself exceeds the total: nothing is
			// admissible, the verdict is false, and the negative HeadroomBytes names the
			// shortfall instead of hiding it.
			name: "engram zero but fixed demand exceeds total",
			b: EngramAdmissionBudget{
				DeviceTotalBytes: 40 * gib,
				DenseCoreBytes:   h200DenseCoreBytes,
				RingBytes:        h200RingBytes,
				KVPoolBytes:      h200KVPoolBytes,
				WorkingSetBytes:  h200WorksetBytes,
			},
			want: EngramDeviceFit{
				AdmissibleBytes:     0,
				Fits:                false,
				DeviceResidentBytes: h200FixedNoHeadroomBytes,
				HeadroomBytes:       40*gib - h200FixedNoHeadroomBytes, // -1740957466
			},
		},
		{
			// Ring + KV alone exceed the budget: no Engram slice can be admitted.
			name: "ring plus kv alone exceed budget",
			b: EngramAdmissionBudget{
				DeviceTotalBytes: 100 * gib,
				EngramBytes:      10 * gib,
				RingBytes:        80 * gib,
				KVPoolBytes:      40 * gib,
			},
			want: EngramDeviceFit{
				AdmissibleBytes:     0,
				Fits:                false,
				DeviceResidentBytes: 120 * gib,
				HeadroomBytes:       -20 * gib,
			},
		},
		{
			// The dense core alone exceeds the budget: fail-closed, zero admissible.
			name: "dense core alone exceeds budget",
			b: EngramAdmissionBudget{
				DeviceTotalBytes: 100 * gib,
				DenseCoreBytes:   110 * gib,
				EngramBytes:      10 * gib,
			},
			want: EngramDeviceFit{
				AdmissibleBytes:     0,
				Fits:                false,
				DeviceResidentBytes: 110 * gib,
				HeadroomBytes:       -10 * gib,
			},
		},
		{
			// Partial admission: the budget leaves exactly 62 GiB of room for Engram
			// (100 GiB total minus 38 GiB of fixed demand incl. 1 GiB headroom), so the
			// maximal admissible slice is that room and the slack drains to the headroom
			// floor.
			name: "partial admission is the leftover room",
			b: EngramAdmissionBudget{
				DeviceTotalBytes: 100 * gib,
				DenseCoreBytes:   10 * gib,
				EngramBytes:      80 * gib,
				RingBytes:        20 * gib,
				KVPoolBytes:      5 * gib,
				WorkingSetBytes:  2 * gib,
				RequiredHeadroom: 1 * gib,
			},
			want: EngramDeviceFit{
				AdmissibleBytes:     62 * gib,
				Fits:                false,
				DeviceResidentBytes: 10*gib + 62*gib + 20*gib + 5*gib + 2*gib, // 99 * gib
				HeadroomBytes:       0,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ModelEngramDeviceAdmission(tc.b)
			if err != nil {
				t.Fatalf("ModelEngramDeviceAdmission(%+v) error = %v, want nil", tc.b, err)
			}
			if got != tc.want {
				t.Fatalf("ModelEngramDeviceAdmission(%+v) = %+v, want %+v", tc.b, got, tc.want)
			}
			// Fail-closed invariant: a reported fit never over-commits the budget.
			if got.Fits && got.DeviceResidentBytes+tc.b.RequiredHeadroom > tc.b.DeviceTotalBytes {
				t.Fatalf("Fits=true but placement %d + headroom %d exceeds total %d",
					got.DeviceResidentBytes, tc.b.RequiredHeadroom, tc.b.DeviceTotalBytes)
			}
			// Admissible count is always within the requested band.
			if got.AdmissibleBytes < 0 || got.AdmissibleBytes > tc.b.EngramBytes {
				t.Fatalf("AdmissibleBytes = %d outside [0, %d]", got.AdmissibleBytes, tc.b.EngramBytes)
			}
		})
	}
}

// TestModelEngramDeviceAdmissionMaximalAdmissible pins the maximality of the partial-admission
// verdict: one byte more Engram than the room refuses, and requesting exactly the room fits.
func TestModelEngramDeviceAdmissionMaximalAdmissible(t *testing.T) {
	gib := int64(1) << 30
	base := EngramAdmissionBudget{
		DeviceTotalBytes: 100 * gib,
		DenseCoreBytes:   10 * gib,
		EngramBytes:      80 * gib,
		RingBytes:        20 * gib,
		KVPoolBytes:      5 * gib,
		WorkingSetBytes:  2 * gib,
		RequiredHeadroom: 1 * gib,
	}
	fit, err := ModelEngramDeviceAdmission(base)
	if err != nil {
		t.Fatal(err)
	}
	if fit.Fits || fit.AdmissibleBytes != 62*gib {
		t.Fatalf("partial admission = %+v, want Fits=false AdmissibleBytes=%d", fit, 62*gib)
	}
	over := base
	over.EngramBytes = fit.AdmissibleBytes + 1
	gotOver, err := ModelEngramDeviceAdmission(over)
	if err != nil {
		t.Fatal(err)
	}
	if gotOver.Fits || gotOver.AdmissibleBytes != fit.AdmissibleBytes {
		t.Fatalf("requesting one byte more: fits=%v admissible=%d, want false/%d",
			gotOver.Fits, gotOver.AdmissibleBytes, fit.AdmissibleBytes)
	}
	exact := base
	exact.EngramBytes = fit.AdmissibleBytes
	gotExact, err := ModelEngramDeviceAdmission(exact)
	if err != nil {
		t.Fatal(err)
	}
	if !gotExact.Fits || gotExact.AdmissibleBytes != fit.AdmissibleBytes {
		t.Fatalf("requesting exactly the room: fits=%v admissible=%d, want true/%d",
			gotExact.Fits, gotExact.AdmissibleBytes, fit.AdmissibleBytes)
	}
}

// TestModelEngramDeviceAdmissionRefusesNegativeInputs is the fail-closed input gate: every
// negative field is a miscomputed input and is refused with an error, never sized against.
func TestModelEngramDeviceAdmissionRefusesNegativeInputs(t *testing.T) {
	mutators := map[string]func(*EngramAdmissionBudget){
		"DeviceTotalBytes": func(b *EngramAdmissionBudget) { b.DeviceTotalBytes = -1 },
		"DenseCoreBytes":   func(b *EngramAdmissionBudget) { b.DenseCoreBytes = -1 },
		"EngramBytes":      func(b *EngramAdmissionBudget) { b.EngramBytes = -1 },
		"RingBytes":        func(b *EngramAdmissionBudget) { b.RingBytes = -1 },
		"KVPoolBytes":      func(b *EngramAdmissionBudget) { b.KVPoolBytes = -1 },
		"WorkingSetBytes":  func(b *EngramAdmissionBudget) { b.WorkingSetBytes = -1 },
		"RequiredHeadroom": func(b *EngramAdmissionBudget) { b.RequiredHeadroom = -1 },
	}
	for name, mutate := range mutators {
		t.Run(name, func(t *testing.T) {
			b := EngramAdmissionBudget{
				DeviceTotalBytes: h200DeviceTotalBytes,
				DenseCoreBytes:   h200DenseCoreBytes,
				EngramBytes:      h200EngramBytes,
				RingBytes:        h200RingBytes,
				KVPoolBytes:      h200KVPoolBytes,
				WorkingSetBytes:  h200WorksetBytes,
			}
			mutate(&b)
			if _, err := ModelEngramDeviceAdmission(b); err == nil {
				t.Fatalf("negative %s accepted, want refusal", name)
			}
		})
	}
}

// TestModelEngramDeviceAdmissionMonotone witnesses the monotone contract on the measured H200
// envelope: the admissible set is the prefix [0, room], where room is what the budget leaves
// after the fixed demand — admitting less always fits when admitting more fits. Every request
// at or below the room fits with its full amount admissible; every request above it refuses
// and reports the same room as the maximal admissible slice.
func TestModelEngramDeviceAdmissionMonotone(t *testing.T) {
	room := h200DeviceTotalBytes - h200FixedNoHeadroomBytes // 106064189670: the Engram slice the budget leaves
	if room <= h200EngramBytes {
		t.Fatalf("test setup: expected the H200 budget to leave more room (%d) than the full band (%d)", room, h200EngramBytes)
	}
	full, err := ModelEngramDeviceAdmission(EngramAdmissionBudget{
		DeviceTotalBytes: h200DeviceTotalBytes,
		DenseCoreBytes:   h200DenseCoreBytes,
		EngramBytes:      h200EngramBytes,
		RingBytes:        h200RingBytes,
		KVPoolBytes:      h200KVPoolBytes,
		WorkingSetBytes:  h200WorksetBytes,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !full.Fits || full.AdmissibleBytes != h200EngramBytes {
		t.Fatalf("measured H200 envelope: full admission = %+v, want Fits=true AdmissibleBytes=%d", full, h200EngramBytes)
	}
	requests := []int64{0, 1 << 30, h200EngramBytes / 4, h200EngramBytes / 2, h200EngramBytes - 1, h200EngramBytes, room, room + 1}
	for _, req := range requests {
		b := EngramAdmissionBudget{
			DeviceTotalBytes: h200DeviceTotalBytes,
			DenseCoreBytes:   h200DenseCoreBytes,
			EngramBytes:      req,
			RingBytes:        h200RingBytes,
			KVPoolBytes:      h200KVPoolBytes,
			WorkingSetBytes:  h200WorksetBytes,
		}
		got, err := ModelEngramDeviceAdmission(b)
		if err != nil {
			t.Fatalf("request %d: %v", req, err)
		}
		wantFits := req <= room
		if got.Fits != wantFits {
			t.Errorf("request %d: Fits = %v, want %v (prefix property)", req, got.Fits, wantFits)
		}
		wantAdmissible := min(req, room)
		if got.AdmissibleBytes != wantAdmissible {
			t.Errorf("request %d: AdmissibleBytes = %d, want %d", req, got.AdmissibleBytes, wantAdmissible)
		}
	}
}
