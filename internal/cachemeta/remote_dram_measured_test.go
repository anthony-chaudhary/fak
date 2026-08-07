package cachemeta

import "testing"

// remote_dram_measured_test.go is the witness for the ingest half of #5066: the 2-node
// RDMA hardware run itself needs an RDMA fabric and is dispatched to sanctioned lab
// compute, but WHAT FAK DOES WITH THE RETURNED NUMBERS is pure and provable here. These
// tests pin the fail-closed posture (no measurement, no MEASURED claim), the adjudication
// against the model, and the failure class the whole reconciler exists to catch: hardware
// that does NOT back the paging preference the model encodes.

// lentRegion is the neighbor's registered lendable DRAM used across these tests.
const lentRegion = 200 << 30

// kvPage is the representative KV page both paths are measured at (the lab recipe's
// ib_read_lat -s 262144 / fio --bs=256k).
const kvPage = 256 << 10

// labResult is a plausible complete return from the lab operator: borrowed peer DRAM
// paging in ~2x faster than the local NVMe baseline it displaces, with attribution.
func labResult() RemoteDRAMPageInMeasurement {
	return RemoteDRAMPageInMeasurement{
		PageBytes:           kvPage,
		RemoteDRAMP50Nanos:  26_000,
		RemoteDRAMP99Nanos:  41_000,
		LocalNVMeP50Nanos:   52_000,
		LocalNVMeP99Nanos:   180_000,
		BorrowedBytesGained: lentRegion,
		Commit:              "1719f3006",
		Source:              "private lab CPU/DC target (scrubbed)",
	}
}

// TestReconcileRemoteDRAMPageInPendingUntilHardwareReturns is the fail-closed default and
// the honest state of this repo today: with no returned run — or with a run missing any
// quantity #5066's done-condition names, attribution included — the witness is PENDING,
// the provenance stays MODELED, and no borrowed pool capacity is claimed. A partial or
// anonymous number must never read as hardware proof (Law A2).
func TestReconcileRemoteDRAMPageInPendingUntilHardwareReturns(t *testing.T) {
	probe := CapacityProbe{RemoteDRAMPresent: true, RemoteDRAMBytes: lentRegion}

	incomplete := func(mutate func(*RemoteDRAMPageInMeasurement)) RemoteDRAMPageInMeasurement {
		m := labResult()
		mutate(&m)
		return m
	}
	cases := []struct {
		name string
		m    RemoteDRAMPageInMeasurement
	}{
		{"nothing returned at all", RemoteDRAMPageInMeasurement{}},
		{"no page size", incomplete(func(m *RemoteDRAMPageInMeasurement) { m.PageBytes = 0 })},
		{"no remote-DRAM latency", incomplete(func(m *RemoteDRAMPageInMeasurement) { m.RemoteDRAMP50Nanos = 0 })},
		{"no NVMe baseline", incomplete(func(m *RemoteDRAMPageInMeasurement) { m.LocalNVMeP50Nanos = 0 })},
		{"latency but no pool number", incomplete(func(m *RemoteDRAMPageInMeasurement) { m.BorrowedBytesGained = 0 })},
		{"unattributed: no commit", incomplete(func(m *RemoteDRAMPageInMeasurement) { m.Commit = "" })},
		{"unattributed: no source", incomplete(func(m *RemoteDRAMPageInMeasurement) { m.Source = "" })},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := ReconcileRemoteDRAMPageIn(tc.m, probe)
			if w.Verdict != RemoteDRAMWitnessPending {
				t.Fatalf("verdict = %q, want PENDING (%s is not a hardware witness)", w.Verdict, tc.name)
			}
			if w.Provenance != "MODELED" {
				t.Fatalf("provenance = %q, want MODELED until a complete run is folded in", w.Provenance)
			}
			if w.BorrowedBytesGained != 0 {
				t.Fatalf("pending witness claims %d borrowed bytes; an unproven borrow gains nothing", w.BorrowedBytesGained)
			}
			if w.MeasuredSpeedupX != 0 || w.MeasuredTailSpeedupX != 0 {
				t.Fatalf("pending witness reports speedups %.3fx/%.3fx, want none", w.MeasuredSpeedupX, w.MeasuredTailSpeedupX)
			}
			if w.Reason == "" {
				t.Fatal("a pending witness must name the quantity it is waiting on")
			}
		})
	}
}

// TestReconcileRemoteDRAMPageInConfirmsTheModel: a complete, attributed lab return where
// borrowed peer DRAM beat local NVMe promotes the record to MEASURED and CONFIRMS the
// paging order the model already acts on — reporting both the measured ratio and the
// modeled prediction at the same page size, plus the effective KV-pool bytes gained.
func TestReconcileRemoteDRAMPageInConfirmsTheModel(t *testing.T) {
	probe := CapacityProbe{RemoteDRAMPresent: true, RemoteDRAMBytes: lentRegion}
	w := ReconcileRemoteDRAMPageIn(labResult(), probe)

	if w.Verdict != RemoteDRAMWitnessConfirms {
		t.Fatalf("verdict = %q (%s), want CONFIRMS", w.Verdict, w.Reason)
	}
	if w.Provenance != "MEASURED" {
		t.Fatalf("provenance = %q, want MEASURED once real hardware returned", w.Provenance)
	}
	if w.MeasuredSpeedupX <= 1 {
		t.Fatalf("measured speedup = %.3fx, want > 1 for a CONFIRMS", w.MeasuredSpeedupX)
	}
	if w.MeasuredTailSpeedupX <= 1 {
		t.Fatalf("measured p99 speedup = %.3fx, want the tail to win too", w.MeasuredTailSpeedupX)
	}
	// The modeled side is recomputed at the MEASURED page size, so the two ratios are
	// directly comparable — the point of the reconciliation.
	if w.Modeled.SpeedupX <= 1 || w.Modeled.Provenance != "MODELED" {
		t.Fatalf("modeled side = %.3fx / %q, want a MODELED prediction > 1 alongside the measurement",
			w.Modeled.SpeedupX, w.Modeled.Provenance)
	}
	// The pool half of the done-condition: the lent region is the effective KV-pool gain.
	if w.BorrowedBytesGained != lentRegion {
		t.Fatalf("KV-pool bytes gained = %d, want the lent %d", w.BorrowedBytesGained, int64(lentRegion))
	}
}

// TestReconcileRemoteDRAMPageInRefutesWhenHardwareLoses is the failure class this seam
// exists for: hardware that does NOT beat local NVMe must REFUTE the rung's preference —
// never be rounded up to a confirmation because the model predicted a win. A fabric that
// wins on the median but loses on the p99 tail is refuted too: a paging tier is chosen for
// the worst case a stalled decode actually waits on. The run still happened, so the record
// stays MEASURED and the borrowed capacity it proved is still reported.
func TestReconcileRemoteDRAMPageInRefutesWhenHardwareLoses(t *testing.T) {
	probe := CapacityProbe{RemoteDRAMPresent: true, RemoteDRAMBytes: lentRegion}

	slow := labResult()
	slow.RemoteDRAMP50Nanos = 60_000 // slower than the 52us NVMe baseline
	slow.RemoteDRAMP99Nanos = 90_000
	w := ReconcileRemoteDRAMPageIn(slow, probe)
	if w.Verdict != RemoteDRAMWitnessRefutes {
		t.Fatalf("verdict = %q (%s), want REFUTES when the measured page-in loses to NVMe", w.Verdict, w.Reason)
	}
	if w.Provenance != "MEASURED" {
		t.Fatalf("provenance = %q, want MEASURED — a refutation is a real hardware result", w.Provenance)
	}
	if w.MeasuredSpeedupX >= 1 {
		t.Fatalf("measured speedup = %.3fx, want < 1 in this case", w.MeasuredSpeedupX)
	}
	if w.BorrowedBytesGained != lentRegion {
		t.Fatalf("a refuted latency claim must still report the %d bytes the borrow added, got %d",
			int64(lentRegion), w.BorrowedBytesGained)
	}
	// The model is not quietly kept: a refutation reports the modeled prediction it beat.
	if w.Modeled.SpeedupX <= 1 {
		t.Fatalf("modeled prediction = %.3fx, want the >1 claim the measurement refutes", w.Modeled.SpeedupX)
	}

	tail := labResult()
	tail.RemoteDRAMP99Nanos = 400_000 // median wins, tail collapses
	w = ReconcileRemoteDRAMPageIn(tail, probe)
	if w.Verdict != RemoteDRAMWitnessRefutes {
		t.Fatalf("verdict = %q (%s), want REFUTES when the p99 tail loses", w.Verdict, w.Reason)
	}
	if w.MeasuredSpeedupX <= 1 {
		t.Fatalf("the median still won (%.3fx); the tail is what refutes here", w.MeasuredSpeedupX)
	}

	// A run that reported no tail at all is judged on the median alone, not failed for it.
	noTail := labResult()
	noTail.RemoteDRAMP99Nanos, noTail.LocalNVMeP99Nanos = 0, 0
	if w := ReconcileRemoteDRAMPageIn(noTail, probe); w.Verdict != RemoteDRAMWitnessConfirms || w.MeasuredTailSpeedupX != 0 {
		t.Fatalf("no-tail run: verdict %q tail %.3fx, want CONFIRMS on the median with no tail claim",
			w.Verdict, w.MeasuredTailSpeedupX)
	}
}

// TestRemoteDRAMSpeedupRoundingIsShared pins the reason both halves call one helper: a
// measurement that lands exactly on the modeled nanos must produce exactly the modeled
// ratio. If the two sides ever round differently, a hardware run that precisely matched the
// prediction would read as a divergence.
func TestRemoteDRAMSpeedupRoundingIsShared(t *testing.T) {
	probe := CapacityProbe{RemoteDRAMPresent: true, RemoteDRAMBytes: lentRegion}
	mod := ModelRemoteDRAMPageInAdvantage(kvPage, probe)

	onModel := labResult()
	onModel.RemoteDRAMP50Nanos = mod.RemoteDRAMPageInNanos
	onModel.LocalNVMeP50Nanos = mod.LocalDiskPageInNanos
	w := ReconcileRemoteDRAMPageIn(onModel, probe)

	if w.MeasuredSpeedupX != w.Modeled.SpeedupX {
		t.Fatalf("measured %.3fx != modeled %.3fx on identical inputs; the two halves must round alike",
			w.MeasuredSpeedupX, w.Modeled.SpeedupX)
	}
	if w.Verdict != RemoteDRAMWitnessConfirms {
		t.Fatalf("verdict = %q, want CONFIRMS when hardware lands exactly on the model", w.Verdict)
	}
}
