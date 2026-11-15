// fak-test:justify why=contract when=changed:internal/perfledger/**
// fak-test:runtime fast est=50ms lane=default
package perfledger

import (
	"errors"
	"math"
	"testing"
	"time"
)

func TestNVBandwidthPlans(t *testing.T) {
	t.Run("named_copy_plan_shape", func(t *testing.T) {
		plan, err := NewCopyPlan(DirectionD2D, InitiatorTMA, AccessRead)
		if err != nil {
			t.Fatalf("NewCopyPlan: %v", err)
		}
		if plan.PlanName() != "device_to_device_copy_tma_read" {
			t.Fatalf("PlanName = %q, want device_to_device_copy_tma_read", plan.PlanName())
		}
		back, err := KnownPlan(plan.PlanName())
		if err != nil {
			t.Fatalf("KnownPlan(%q): %v", plan.PlanName(), err)
		}
		if back != plan {
			t.Fatalf("round-trip = %+v, want %+v", back, plan)
		}
		plans := NamedCopyPlans()
		if len(plans) != 4*3*2 {
			t.Fatalf("NamedCopyPlans = %d plans, want 24 (4 directions x 3 initiators x 2 accesses)", len(plans))
		}
		seen := make(map[string]bool, len(plans))
		for _, p := range plans {
			if _, err := KnownPlan(p.PlanName()); err != nil {
				t.Fatalf("plan %q does not parse back: %v", p.PlanName(), err)
			}
			seen[p.PlanName()] = true
		}
		for _, want := range []string{
			"device_to_device_copy_ce_read",
			"device_to_device_copy_sm_write",
			"host_to_device_copy_tma_read",
			"device_to_host_copy_ce_write",
			"host_to_host_copy_sm_read",
		} {
			if !seen[want] {
				t.Fatalf("NamedCopyPlans missing %q", want)
			}
		}
	})

	t.Run("unknown_plan_fails_closed", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			plan CopyPlan
		}{
			{"bad_direction", CopyPlan{Direction: "gpu_to_gpu", Initiator: InitiatorCE, Access: AccessRead}},
			{"bad_initiator", CopyPlan{Direction: DirectionD2D, Initiator: "dma", Access: AccessRead}},
			{"bad_access", CopyPlan{Direction: DirectionD2D, Initiator: InitiatorCE, Access: "copy"}},
		} {
			if _, err := NewCopyPlan(tc.plan.Direction, tc.plan.Initiator, tc.plan.Access); !errors.Is(err, ErrUnknownCopyPlan) {
				t.Errorf("%s: err = %v, want ErrUnknownCopyPlan", tc.name, err)
			}
		}
		for _, name := range []string{"", "device_to_device_copy_ce_read_extra", "d2d_copy_ce_read", "device_to_device_copy_ce", "DEVICE_TO_DEVICE_COPY_CE_READ", "device_to_device_copy_ce_read_write"} {
			if _, err := KnownPlan(name); !errors.Is(err, ErrUnknownCopyPlan) {
				t.Errorf("KnownPlan(%q): err = %v, want ErrUnknownCopyPlan", name, err)
			}
		}
	})

	t.Run("d2d_h2d_matrix", func(t *testing.T) {
		m, err := NewTransportMatrix(3) // host + 2 devices
		if err != nil {
			t.Fatalf("NewTransportMatrix: %v", err)
		}
		if err := m.Set(1, 2, 480.5); err != nil {
			t.Fatalf("Set(1,2): %v", err)
		}
		if err := m.Set(2, 1, 470.25); err != nil {
			t.Fatalf("Set(2,1): %v", err)
		}
		if err := m.Set(0, 1, 56.0); err != nil { // host -> device
			t.Fatalf("Set(0,1): %v", err)
		}
		if err := m.Set(1, 0, 55.5); err != nil { // device -> host
			t.Fatalf("Set(1,0): %v", err)
		}
		if got, ok := m.Pair(1, 2); !ok || got != 480.5 {
			t.Fatalf("Pair(1,2) = %v,%v, want 480.5,true", got, ok)
		}
		if got, ok := m.Pair(2, 1); !ok || got != 470.25 {
			t.Fatalf("Pair(2,1) = %v,%v, want 470.25,true (matrix is ordered, not symmetric)", got, ok)
		}
		if _, ok := m.Pair(0, 2); ok {
			t.Fatal("Pair(0,2) observed without a Set")
		}
		stats := m.Stats()
		wantSum := 480.5 + 470.25 + 56.0 + 55.5
		if stats.Sum != wantSum || stats.Max != 480.5 || stats.Min != 55.5 || stats.Average != wantSum/4 {
			t.Fatalf("Stats = %+v, want sum %v max 480.5 min 55.5 avg %v", stats, wantSum, wantSum/4)
		}
		for _, tc := range []struct {
			name string
			src, dst int
			bw   float64
		}{
			{"src_out_of_range", 3, 0, 1},
			{"dst_out_of_range", 0, 3, 1},
			{"negative_src", -1, 0, 1},
			{"nan", 0, 1, math.NaN()},
			{"inf", 0, 1, math.Inf(+1)},
			{"negative_bw", 0, 1, -1},
		} {
			if err := m.Set(tc.src, tc.dst, tc.bw); !errors.Is(err, ErrInvalidMatrixCell) {
				t.Errorf("%s: Set(%d,%d,%v) err = %v, want ErrInvalidMatrixCell", tc.name, tc.src, tc.dst, tc.bw, err)
			}
		}
		if got, ok := m.Pair(0, 1); !ok || got != 56.0 {
			t.Fatalf("rejected Set changed Pair(0,1) = %v,%v", got, ok)
		}
		if _, err := NewTransportMatrix(0); !errors.Is(err, ErrInvalidMatrixCell) {
			t.Fatalf("NewTransportMatrix(0): err = %v, want ErrInvalidMatrixCell", err)
		}
	})

	t.Run("stability_metric", func(t *testing.T) {
		m, err := NewTransportMatrix(2)
		if err != nil {
			t.Fatalf("NewTransportMatrix: %v", err)
		}
		for _, v := range []float64{100, 102, 98} {
			if err := m.Set(0, 1, v); err != nil {
				t.Fatalf("Set: %v", err)
			}
		}
		cv := CoefficientOfVariation(m.Samples[0][1])
		if math.IsNaN(cv) || cv <= 0 {
			t.Fatalf("CoefficientOfVariation = %v, want positive", cv)
		}
		if got := m.WorstCV(); math.IsNaN(got) || got != cv {
			t.Fatalf("WorstCV = %v, want %v", got, cv)
		}
		if !math.IsNaN(CoefficientOfVariation([]float64{100})) {
			t.Fatal("single sample must be unmeasured (NaN), not zero")
		}
		if !math.IsNaN(CoefficientOfVariation(nil)) {
			t.Fatal("empty sample set must be unmeasured (NaN)")
		}
		if !math.IsNaN(CoefficientOfVariation([]float64{0, 0})) {
			t.Fatal("zero mean must be unmeasured (NaN)")
		}
	})

	t.Run("moment_in_time_receipt", func(t *testing.T) {
		t0 := time.Date(2026, 11, 6, 12, 0, 0, 0, time.UTC)
		r := NewReceipt("fak-perfledger", "host-a", t0)
		if r.MomentMS != t0.UnixMilli() {
			t.Fatalf("receipt moment = %d, want %d", r.MomentMS, t0.UnixMilli())
		}
		d2d, _ := NewTransportMatrix(2)
		if err := d2d.Set(0, 1, 480); err != nil {
			t.Fatalf("Set: %v", err)
		}
		h2d, _ := NewTransportMatrix(2)
		if err := h2d.Set(0, 1, 56); err != nil {
			t.Fatalf("Set: %v", err)
		}
		d2dPlan, _ := NewCopyPlan(DirectionD2D, InitiatorCE, AccessRead)
		h2dPlan, _ := NewCopyPlan(DirectionH2D, InitiatorCE, AccessWrite)
		t1 := t0.Add(30 * time.Second)
		if err := r.Record(d2dPlan, d2d, ReceiptPassed, t1); err != nil {
			t.Fatalf("Record d2d: %v", err)
		}
		t2 := t1.Add(30 * time.Second)
		if err := r.Record(h2dPlan, h2d, ReceiptPassed, t2); err != nil {
			t.Fatalf("Record h2d: %v", err)
		}
		if len(r.Entries) != 2 {
			t.Fatalf("entries = %d, want 2", len(r.Entries))
		}
		if r.Entries[0].MomentMS != t1.UnixMilli() || r.Entries[1].MomentMS != t2.UnixMilli() {
			t.Fatalf("entry moments = %d,%d, want %d,%d", r.Entries[0].MomentMS, r.Entries[1].MomentMS, t1.UnixMilli(), t2.UnixMilli())
		}
		if r.Entries[0].Stats.Max != 480 || r.Entries[1].Stats.Max != 56 {
			t.Fatalf("entry stats = %+v / %+v", r.Entries[0].Stats, r.Entries[1].Stats)
		}
		if err := r.Validate(); err != nil {
			t.Fatalf("Validate: %v", err)
		}
		// Unknown plan names fail closed on record and on validate.
		bad, _ := NewCopyPlan(DirectionD2D, InitiatorCE, AccessRead)
		if err := r.Record(CopyPlan{Direction: "gpu_to_gpu", Initiator: InitiatorCE, Access: AccessRead}, d2d, ReceiptPassed, t2); !errors.Is(err, ErrUnknownCopyPlan) {
			t.Fatalf("Record unknown plan: err = %v, want ErrUnknownCopyPlan", err)
		}
		if err := r.Record(bad, nil, ReceiptPassed, t2); !errors.Is(err, ErrInvalidMatrixCell) {
			t.Fatalf("Record nil matrix: err = %v, want ErrInvalidMatrixCell", err)
		}
		if len(r.Entries) != 2 {
			t.Fatal("rejected records mutated the receipt")
		}
		broken := *r
		broken.Entries[0].Plan = "not_a_plan"
		if err := broken.Validate(); !errors.Is(err, ErrUnknownCopyPlan) {
			t.Fatalf("Validate unknown entry: err = %v, want ErrUnknownCopyPlan", err)
		}
		unstamped := *r
		unstamped.Entries[1].MomentMS = 0
		if err := unstamped.Validate(); !errors.Is(err, ErrUnknownCopyPlan) {
			t.Fatalf("Validate unstamped entry: err = %v, want ErrUnknownCopyPlan", err)
		}
	})
}
