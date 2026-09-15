//go:build darwin && arm64 && cgo

package metalgemm

import (
	"sort"
	"testing"
)

// q4kCrossoverReceipt is the moment-in-time record the ticket asks for: the date, commit SHA,
// device name, OS version and the measured candidate/scalar ratio at each panel shape. It is the
// non-fabricable artifact behind the pinned q4kM5CrossoverTable row — the routing gate is only
// ever opened against numbers captured here on real silicon, and the test below re-derives the
// same table from this receipt so the row and its witness cannot drift apart.
type q4kCrossoverReceipt struct {
	Date       string
	Commit     string
	DeviceName string
	OSVersion  string
	Ratios     map[int]float64 // prompt shape -> median candidate/scalar on-GPU ratio
	Floor      float64         // smallest measured ratio, the conservative gate value
}

// q4kCrossoverRepeats is the number of timed candidate/scalar pairs per shape. Odd so the median
// is an observed sample; 7 keeps the whole witness fast (sub-second on an M3 Pro) while still
// resisting a single scheduler hiccup.
const (
	q4kCrossoverRepeats = 7
	q4kCrossoverDate    = "2026-09-15"
	q4kCrossoverCommit  = "97cae3629"
)

// medianFloat returns the median of a sample; it copies so the caller's slice is untouched.
func medianFloat(xs []float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[len(s)/2]
}

// TestQ4KCrossoverReceiptCandidateVsScalar is the on-silicon witness for the pinned crossover. It
// times the scalar kernel (mode 0) against the wide-tile cooperative-SMEM candidate (mode 2) at
// the widened panel shapes the ticket names (P in {64,128}), derives the median candidate/scalar
// on-GPU ratio over repeats, records the receipt (date, commit SHA, device, OS, ratios), and
// asserts the production table pins exactly one M3 Pro row that clears the fak#9937 >=1.10x gate.
//
// The on-GPU window (LastGEMMGPUMs) is the right metric: it excludes the CPU-side
// encode/commit/sync/H2D round-trip, which is identical for both kernels and would otherwise
// dilute a compute-side crossover. Physical ratios are [HW-WITNESSED]: they are only meaningful
// on the pinned device, so the test skips (rather than asserts) anywhere else — a run on a
// different Mac is [SW-VERIFIED] at best and must not promote the row.
func TestQ4KCrossoverReceiptCandidateVsScalar(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable; the crossover receipt is [HW-WITNESSED] and needs on-silicon execution")
	}
	if got := DeviceName(); got != "Apple M3 Pro" {
		t.Skipf("crossover receipt is pinned to Apple M3 Pro; this device is %q ([SW-VERIFIED] only)", got)
	}
	if got := OSVersion(); got[0:2] != "26" {
		t.Skipf("crossover receipt is pinned to macOS 26; this host reports %q", got)
	}
	defer ResetQ4K()

	// Real projection geometry (out,in) = (4096,4096) — the FFN projection width of the resident
	// 27B q4_k_m model this lane serves, so the timing reflects the production panel GEMM.
	const out, in = 4096, 4096
	w := UploadQ4K(q4kTestRaw(out, in, 0x9937), out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	receipt := q4kCrossoverReceipt{
		Date:       q4kCrossoverDate,
		Commit:     q4kCrossoverCommit,
		DeviceName: DeviceName(),
		OSVersion:  OSVersion(),
		Ratios:     make(map[int]float64),
	}
	receipt.Floor = -1

	for _, prompt := range []int{64, 128} {
		x := make([]float32, prompt*in)
		for i := range x {
			x[i] = float32((i*37)%251-125) / 127
		}
		y := make([]float32, prompt*out)

		// Warm both kernels once so pipeline compile/TSO setup is outside the timed window.
		if id := w.GEMMWithEventsMode(x, prompt, y, nil, Q4KGEMMModeScalar); id.Executed != Q4KGEMMExecutedScalar {
			t.Fatalf("P=%d scalar warmup executed=%v", prompt, id.Executed)
		}
		wantCandidate := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedM5CooperativeSMEM, Executed: Q4KGEMMExecutedM5CooperativeSMEM}
		if id := w.GEMMWithEventsMode(x, prompt, y, nil, Q4KGEMMModeM5CooperativeSMEM); id != wantCandidate {
			t.Fatalf("P=%d candidate warmup identity=%+v want %+v", prompt, id, wantCandidate)
		}

		scalarMs := make([]float64, 0, q4kCrossoverRepeats)
		candidateMs := make([]float64, 0, q4kCrossoverRepeats)
		for rep := 0; rep < q4kCrossoverRepeats; rep++ {
			if id := w.GEMMWithEventsMode(x, prompt, y, nil, Q4KGEMMModeScalar); id.Executed != Q4KGEMMExecutedScalar {
				t.Fatalf("P=%d scalar rep=%d executed=%v", prompt, rep, id.Executed)
			}
			scalarMs = append(scalarMs, LastGEMMGPUMs())
			if id := w.GEMMWithEventsMode(x, prompt, y, nil, Q4KGEMMModeM5CooperativeSMEM); id != wantCandidate {
				t.Fatalf("P=%d candidate rep=%d identity=%+v", prompt, rep, id)
			}
			candidateMs = append(candidateMs, LastGEMMGPUMs())
		}
		s, c := medianFloat(scalarMs), medianFloat(candidateMs)
		if s <= 0 || c <= 0 {
			t.Fatalf("P=%d non-positive on-GPU window: scalar=%g candidate=%g", prompt, s, c)
		}
		ratio := s / c
		receipt.Ratios[prompt] = ratio
		if receipt.Floor < 0 || ratio < receipt.Floor {
			receipt.Floor = ratio
		}
		t.Logf("[HW-WITNESSED] date=%s commit=%s device=%q os=%q P=%d scalar_gpu_ms=%.4f candidate_gpu_ms=%.4f ratio=%.4f",
			receipt.Date, receipt.Commit, receipt.DeviceName, receipt.OSVersion, prompt, s, c, ratio)
	}

	// Routing gate: the measured floor must clear the fak#9937 >=1.10x margin for the row to be
	// pinned. Below the floor the table must stay empty (the quarantined-fallback branch), so a
	// regression in the candidate kernel fails this witness rather than silently mis-routing.
	rows := q4kM5CrossoverTable
	if receipt.Floor < Q4KM5CrossoverMinimumRatio {
		if len(rows) != 0 {
			t.Fatalf("measured floor=%.4f below gate %.2f but the table pins %d row(s): %+v",
				receipt.Floor, Q4KM5CrossoverMinimumRatio, len(rows), rows)
		}
		t.Logf("[HW-WITNESSED] floor=%.4f below %.2f: table correctly empty; file the finding",
			receipt.Floor, Q4KM5CrossoverMinimumRatio)
		return
	}
	if len(rows) != 1 {
		t.Fatalf("measured floor=%.4f clears the gate but the table has %d rows: %+v",
			receipt.Floor, len(rows), rows)
	}
	row := rows[0]
	if row.Family != receipt.DeviceName {
		t.Fatalf("pinned row family=%q, measured on %q", row.Family, receipt.DeviceName)
	}
	if row.OSVersion != receipt.OSVersion[0:2] {
		t.Fatalf("pinned row os=%q, measured on %q", row.OSVersion, receipt.OSVersion)
	}
	if row.MinRatio < Q4KM5CrossoverMinimumRatio || row.MinRatio > receipt.Floor {
		t.Fatalf("pinned row MinRatio=%.4f must lie in [%.2f, measured floor %.4f]",
			row.MinRatio, Q4KM5CrossoverMinimumRatio, receipt.Floor)
	}
	if !Q4KM5CrossoverPredicate(receipt.DeviceName, receipt.OSVersion) {
		t.Fatal("the pinned row does not admit mode 2 on the device it was measured on")
	}
	if !Q4KM5CrossoverAdmits() {
		t.Fatal("the live device/OS reports the crossover admits false despite a measured, pinned row")
	}
	t.Logf("[HW-WITNESSED] pinned exactly one row: family=%q os=%q minRatio=%.2f witness=%q",
		row.Family, row.OSVersion, row.MinRatio, row.Witness)
}
