//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// q4kCrossoverRepeats is the number of timed candidate/scalar pairs per shape. Odd so the median
// is an observed sample; 7 keeps the whole witness fast (sub-second on an M3 Pro) while still
// resisting a single scheduler hiccup.
const (
	q4kCrossoverRepeats = 7
	q4kCrossoverDate    = "2026-09-15"
	q4kCrossoverCommit  = "97cae3629"
)

// q4kCrossoverReceiptPathEnv names the env var that directs the on-silicon witness to write its
// machine-admissible banded receipt to a real path. Unset, the test still measures and validates
// in memory but writes nothing, so an ordinary `go test` never mutates the tracked tree.
const q4kCrossoverReceiptPathEnv = "FAK_Q4K_M5_CROSSOVER_RECEIPT"

// q4kCrossoverLeaseWaitEnv names the env var that lets a re-capture QUEUE for the GPU lease for a
// bounded number of milliseconds instead of skipping. Absent/zero keeps the fast no-wait skip.
const q4kCrossoverLeaseWaitEnv = "FAK_Q4K_M5_LEASE_WAIT_MS"

// q4kCrossoverMeasuredShapes is the prompt-length set the on-silicon witness measures. It is the
// widened-panel regime's measured envelope (fak#13041 P>=64); the band's endpoints are its first
// and last entry.
var q4kCrossoverMeasuredShapes = []int{64, 128}

// TestQ4KCrossoverReceiptCandidateVsScalar is the on-silicon witness for the P-band-pinned
// crossover. It times the scalar kernel (mode 0) against the wide-tile cooperative-SMEM candidate
// (mode 2) at the widened panel shapes the ticket names (P in {64,128}), derives the median
// candidate/scalar on-GPU ratio over balanced repeats, records the raw samples and the band's
// measured floor, validates the receipt fail-closed, optionally writes the JSON artifact, and
// asserts the production table pins exactly one M3 Pro row whose band covers the measured shapes
// and whose MinRatio is the MEASURED floor (not merely the gate floor) while still clearing the
// fak#9937 >=1.10x gate.
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

	// The ratio floor is only attributable on an uncontended GPU. A resident
	// `fak-native up` holder, a concurrent modelbench run, or any other GPU-heavy
	// process sharing this box can drag a single timed rep below the >=1.10x gate
	// even though the kernel is correct (observed 2026-09-15 on this M3 Pro box).
	// Take the machine-wide lease NoWait: if another exclusive GPU holder owns it,
	// skip with the contention named rather than asserting a ratio measured under
	// noise. A quiet box still asserts the strict floor below.
	//
	// A promoted lane that must re-capture the physical receipt can set
	// FAK_Q4K_M5_LEASE_WAIT_MS to QUEUE behind the holder for that many
	// milliseconds instead of skipping, so a re-measurement is possible without
	// killing an active managed server. A zero/absent value preserves the fast
	// no-wait skip that keeps the ordinary suite hermetic.
	leaseOpts := gpulease.Options{NoWait: true}
	if waitMS := os.Getenv(q4kCrossoverLeaseWaitEnv); waitMS != "" {
		ms, parseErr := strconv.Atoi(waitMS)
		if parseErr != nil || ms < 0 {
			t.Fatalf("%s=%q must be a non-negative integer of milliseconds", q4kCrossoverLeaseWaitEnv, waitMS)
		}
		if ms > 0 {
			leaseOpts = gpulease.Options{Timeout: time.Duration(ms) * time.Millisecond}
		}
	}
	lease, err := gpulease.Acquire(leaseOpts)
	if err != nil {
		t.Skipf("crossover receipt is [HW-WITNESSED] only on an uncontended GPU; exclusive GPU lease unavailable (%v)", err)
	}
	defer lease.Release()

	defer ResetQ4K()

	// Real projection geometry (out,in) = (4096,4096) — the FFN projection width of the resident
	// 27B q4_k_m model this lane serves, so the timing reflects the production panel GEMM.
	const out, in = 4096, 4096
	w := UploadQ4K(q4kTestRaw(out, in, 0x9937), out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	band := Q4KM5CrossoverBand{
		MinP:           q4kCrossoverMeasuredShapes[0],
		MaxP:           q4kCrossoverMeasuredShapes[len(q4kCrossoverMeasuredShapes)-1],
		Measured:       make(map[int]float64),
		ScalarGPUMS:    make(map[int]float64),
		CandidateGPUMS: make(map[int]float64),
		Samples:        make(map[int]ArmSamples),
	}
	floor := math.Inf(1)

	for _, prompt := range q4kCrossoverMeasuredShapes {
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
		band.Measured[prompt] = ratio
		band.ScalarGPUMS[prompt] = s
		band.CandidateGPUMS[prompt] = c
		band.Samples[prompt] = ArmSamples{ScalarMS: scalarMs, CandidateMS: candidateMs}
		if ratio < floor {
			floor = ratio
		}
		t.Logf("[HW-WITNESSED] date=%s commit=%s device=%q os=%q P=%d scalar_gpu_ms=%.4f candidate_gpu_ms=%.4f ratio=%.4f",
			q4kCrossoverDate, q4kCrossoverCommit, DeviceName(), OSVersion(), prompt, s, c, ratio)
	}
	// The band's MinRatio is the MEASURED floor over its endpoints — the lever magnitude fak#13124
	// paid hardware time for — not the gate floor. A regression below the gate fails validation.
	band.Floor = floor

	receipt := Q4KM5CrossoverReceipt{
		Schema:        Q4KM5CrossoverReceiptSchema,
		ContractIssue: "13133",
		PairedHarness: Q4KM5CrossoverReceiptHarness,
		GeneratedAt:   time.Now().UTC().Format(time.RFC3339),
		EvidenceKind:  Q4KReceiptEvidenceHW,
		Measurement:   Q4KMeasurementFresh,
		DeviceName:    DeviceName(),
		OSVersion:     OSVersion(),
		SourceCommit:  q4kCrossoverCommit,
		SourceTest:    "TestQ4KCrossoverReceiptCandidateVsScalar",
		GeometryOut:   out,
		GeometryIn:    in,
		Metric:        "on_gpu_ms",
		Repeats:       q4kCrossoverRepeats,
		GateMargin:    Q4KM5CrossoverMinimumRatio,
		Bands:         []Q4KM5CrossoverBand{band},
		Notes: []string{
			"candidate/scalar ratio is scalar_gpu_ms / candidate_gpu_ms; >1 means the wide tile is faster",
			"on_gpu_ms is the cb.GPUEndTime-cb.GPUStartTime window and excludes the host round-trip",
			"the band's measured floor is the row's MinRatio; the P band, not the ratio, changes routing",
		},
	}

	// Validate fail-closed before pinning: an unbalanced, below-gate, or self-inconsistent
	// measurement must fail the witness rather than silently keep the row.
	if err := ValidateQ4KM5CrossoverReceipt(receipt); err != nil {
		t.Fatalf("measured receipt is not admissible: %v", err)
	}
	if path := os.Getenv(q4kCrossoverReceiptPathEnv); path != "" {
		if err := WriteQ4KM5CrossoverReceipt(path, receipt); err != nil {
			t.Fatalf("write receipt %s: %v", path, err)
		}
		t.Logf("[HW-WITNESSED] wrote machine-admissible receipt to %s", path)
	}

	// Routing gate: the measured floor must clear the fak#9937 >=1.10x margin for the row to be
	// pinned. Below the floor the table must stay empty (the quarantined-fallback branch), so a
	// regression in the candidate kernel fails this witness rather than silently mis-routing.
	rows := q4kM5CrossoverTable
	if receipt.Bands[0].Floor < Q4KM5CrossoverMinimumRatio {
		if len(rows) != 0 {
			t.Fatalf("measured floor=%.4f below gate %.2f but the table pins %d row(s): %+v",
				receipt.Bands[0].Floor, Q4KM5CrossoverMinimumRatio, len(rows), rows)
		}
		t.Logf("[HW-WITNESSED] floor=%.4f below %.2f: table correctly empty; file the finding",
			receipt.Bands[0].Floor, Q4KM5CrossoverMinimumRatio)
		return
	}
	if len(rows) != 1 {
		t.Fatalf("measured floor=%.4f clears the gate but the table has %d rows: %+v",
			receipt.Bands[0].Floor, len(rows), rows)
	}
	row := rows[0]
	if row.Family != receipt.DeviceName {
		t.Fatalf("pinned row family=%q, measured on %q", row.Family, receipt.DeviceName)
	}
	if row.OSVersion != receipt.OSVersion[0:2] {
		t.Fatalf("pinned row os=%q, measured on %q", row.OSVersion, receipt.OSVersion)
	}
	// The pinned band must cover exactly the measured shapes: the lower edge is the smallest
	// measured shape and the upper edge is the largest.
	if row.MinP != band.MinP || row.MaxP != band.MaxP {
		t.Fatalf("pinned row band=[%d,%d], measured band=[%d,%d]", row.MinP, row.MaxP, band.MinP, band.MaxP)
	}
	// MinRatio is the MEASURED band floor (fak#13133), not the gate floor; it must clear the gate,
	// and the tombstoned pin must be CONSERVATIVE with respect to this fresh measurement — the row
	// may record a floor at or below what a re-run just measured (never above it), so a re-run whose
	// ratio comes in low fails the witness instead of silently keeping an optimistic pin.
	if row.MinRatio < Q4KM5CrossoverMinimumRatio {
		t.Fatalf("pinned row MinRatio=%.4f is below the fak#9937 gate %.2f", row.MinRatio, Q4KM5CrossoverMinimumRatio)
	}
	if row.MinRatio > band.Floor*(1+1e-9) {
		t.Fatalf("pinned row MinRatio=%.6f exceeds the freshly measured floor %.6f; a fresh measurement below the pin is a regression",
			row.MinRatio, band.Floor)
	}
	if !q4kFinite(row.MinRatio) {
		t.Fatalf("pinned row MinRatio=%v is not finite", row.MinRatio)
	}
	for _, P := range q4kCrossoverMeasuredShapes {
		if !Q4KM5CrossoverPredicate(receipt.DeviceName, receipt.OSVersion, P) {
			t.Fatalf("the pinned row does not admit mode 2 at measured P=%d", P)
		}
	}
	// A P above the measured band is unmeasured and must stay fail-closed scalar.
	if Q4KM5CrossoverPredicate(receipt.DeviceName, receipt.OSVersion, band.MaxP+1) {
		t.Fatalf("crossover admitted P=%d above the measured band; an unmeasured P must execute scalar", band.MaxP+1)
	}
	if !Q4KM5CrossoverAdmits(band.MinP) {
		t.Fatal("the live device/OS reports the crossover admits false at a measured, pinned P")
	}
	t.Logf("[HW-WITNESSED] pinned exactly one row: family=%q os=%q band=[%d,%d] minRatio=%.4f (measured floor %.4f) witness=%q",
		row.Family, row.OSVersion, row.MinP, row.MaxP, row.MinRatio, band.Floor, row.Witness)
}
