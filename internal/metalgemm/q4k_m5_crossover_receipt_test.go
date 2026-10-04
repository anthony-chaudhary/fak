//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// q4kCrossoverRepeats is the number of timed candidate/scalar pairs per shape. Odd so the median
// is an observed sample; 7 keeps the whole witness fast while still resisting a single scheduler
// hiccup.
const (
	q4kCrossoverRepeats = 7
	q4kCrossoverDate    = "2026-10-04"
	q4kCrossoverCommit  = "37e7d7293+fak#13692"
)

// q4kCrossoverReceiptPathEnv names the env var that directs the on-silicon witness to write its
// machine-admissible banded receipt to a real path. Unset, the test still measures and validates
// in memory but writes nothing, so an ordinary `go test` never mutates the tracked tree.
const q4kCrossoverReceiptPathEnv = "FAK_Q4K_M5_CROSSOVER_RECEIPT"

// q4kCrossoverLeaseWaitEnv names the env var that lets a re-capture QUEUE for the GPU lease for a
// bounded number of milliseconds instead of skipping. Absent/zero keeps the fast no-wait skip.
const q4kCrossoverLeaseWaitEnv = "FAK_Q4K_M5_LEASE_WAIT_MS"

// q4kCrossoverMeasuredShapes is the prompt-length sweep the on-silicon witness measures. The
// admitted band starts at its first entry (the wide-tile envelope's P>=64 floor, fak#13041) and
// extends over the longest run of consecutive shapes whose ratios all clear the gate (fak#13692:
// the sweep reaches 2048 so the band's upper edge is measured, not assumed).
var q4kCrossoverMeasuredShapes = []int{64, 128, 256, 512, 1024, 2048}

// q4kOSMajor returns the leading macOS major of a product version ("27.0.1" -> "27").
func q4kOSMajor(v string) string {
	major, _, _ := strings.Cut(v, ".")
	return major
}

// TestQ4KCrossoverReceiptCandidateVsScalar is the on-silicon witness for the P-band-pinned
// crossover. It times the scalar kernel (mode 0) against the wide-tile cooperative-SMEM candidate
// (mode 2) over P in {64..2048}, derives the median candidate/scalar on-GPU ratio over balanced
// repeats, and admits the longest band from P=64 whose ratios all clear the fak#9937 >=1.10x gate.
// It records the raw samples and the band's measured floor, validates the receipt fail-closed,
// optionally writes the JSON artifact, and asserts that the production row pinned for THIS device
// and macOS major starts at the band's MinP, does not extend past the measured band, and records a
// MinRatio no higher than the fresh floor. If even P=64 misses the gate, the table must pin no row
// for this device/OS major.
//
// Since fak#13692 the witness runs on any macOS major (it previously skipped off macOS 26, which is
// how macOS 27 fell through to scalar unnoticed). The scalar arm is the kernel as shipped with
// fak#13692 (single 2D dispatch, padded xbuf).
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
	osMajor := q4kOSMajor(OSVersion())
	if osMajor == "" {
		t.Skipf("host reports no macOS version (%q); the row's OS pin cannot be resolved", OSVersion())
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

	type measured struct {
		ratio, scalar, candidate float64
		samples                  ArmSamples
	}
	results := make(map[int]measured, len(q4kCrossoverMeasuredShapes))
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
		results[prompt] = measured{ratio: s / c, scalar: s, candidate: c,
			samples: ArmSamples{ScalarMS: scalarMs, CandidateMS: candidateMs}}
		t.Logf("[HW-WITNESSED] date=%s commit=%s device=%q os=%q P=%d scalar_gpu_ms=%.4f candidate_gpu_ms=%.4f ratio=%.4f",
			q4kCrossoverDate, q4kCrossoverCommit, DeviceName(), OSVersion(), prompt, s, c, s/c)
	}

	// The table rows that apply to this live device and macOS major.
	var rows []q4kM5CrossoverRow
	for _, row := range q4kM5CrossoverTable {
		if strings.HasPrefix(DeviceName(), row.Family) && row.OSVersion == osMajor {
			rows = append(rows, row)
		}
	}

	// The admitted band is the longest run from P=64 whose ratios all clear the gate.
	admitted := 0
	for _, P := range q4kCrossoverMeasuredShapes {
		if results[P].ratio < Q4KM5CrossoverMinimumRatio {
			break
		}
		admitted++
	}
	if admitted == 0 {
		// Quarantined-fallback branch: no band clears the gate, so this OS major must pin no row.
		if len(rows) != 0 {
			t.Fatalf("P=%d ratio=%.4f misses the %.2f gate on macOS %s, but the table pins %d row(s): %+v",
				q4kCrossoverMeasuredShapes[0], results[q4kCrossoverMeasuredShapes[0]].ratio,
				Q4KM5CrossoverMinimumRatio, osMajor, len(rows), rows)
		}
		t.Logf("[HW-WITNESSED] P=%d ratio=%.4f below %.2f: no row for macOS %s, correctly; no receipt validated or written",
			q4kCrossoverMeasuredShapes[0], results[q4kCrossoverMeasuredShapes[0]].ratio, Q4KM5CrossoverMinimumRatio, osMajor)
		return
	}
	bandShapes := q4kCrossoverMeasuredShapes[:admitted]
	band := Q4KM5CrossoverBand{
		MinP:           bandShapes[0],
		MaxP:           bandShapes[len(bandShapes)-1],
		Measured:       make(map[int]float64),
		ScalarGPUMS:    make(map[int]float64),
		CandidateGPUMS: make(map[int]float64),
		Samples:        make(map[int]ArmSamples),
	}
	floor := math.Inf(1)
	for _, P := range bandShapes {
		m := results[P]
		band.Measured[P] = m.ratio
		band.ScalarGPUMS[P] = m.scalar
		band.CandidateGPUMS[P] = m.candidate
		band.Samples[P] = m.samples
		floor = math.Min(floor, m.ratio)
	}
	// The band's MinRatio is the MEASURED floor over its shapes — the lever magnitude, not the gate
	// floor. A regression below the gate fails validation.
	band.Floor = floor

	notes := []string{
		"candidate/scalar ratio is scalar_gpu_ms / candidate_gpu_ms; >1 means the wide tile is faster",
		"on_gpu_ms is the cb.GPUEndTime-cb.GPUStartTime window and excludes the host round-trip",
		"the band's measured floor is the row's MinRatio; the P band, not the ratio, changes routing",
		"fak#13692: the scalar arm is q4k_gemm as shipped with fak#13692 (single 2D dispatch, padded xbuf); the band is the longest run from P=64 clearing the gate",
	}
	for _, P := range q4kCrossoverMeasuredShapes[admitted:] {
		m := results[P]
		notes = append(notes, fmt.Sprintf("outside the band: P=%d ratio=%.4f scalar_gpu_ms=%.4f candidate_gpu_ms=%.4f scalar_samples=%v candidate_samples=%v",
			P, m.ratio, m.scalar, m.candidate, m.samples.ScalarMS, m.samples.CandidateMS))
	}
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
		Notes:         notes,
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

	// The band clears the gate, so exactly one row must be pinned for this device and OS major.
	if len(rows) != 1 {
		t.Fatalf("measured band [%d,%d] floor=%.4f clears the gate on macOS %s but the table pins %d matching row(s): %+v",
			band.MinP, band.MaxP, band.Floor, osMajor, len(rows), rows)
	}
	row := rows[0]
	// The pinned band starts at the measured lower edge and never extends past the measured band.
	if row.MinP != band.MinP {
		t.Fatalf("pinned row MinP=%d, measured band MinP=%d", row.MinP, band.MinP)
	}
	if row.MaxP == 0 || row.MaxP > band.MaxP {
		t.Fatalf("pinned row MaxP=%d extends past the measured band MaxP=%d", row.MaxP, band.MaxP)
	}
	// MinRatio is the MEASURED floor (fak#13133), not the gate floor; it must clear the gate and be
	// CONSERVATIVE with respect to this fresh measurement over the shapes the row covers — a re-run
	// whose ratio comes in below the pin fails the witness instead of keeping an optimistic pin.
	// (The row may be narrower than the measured band; shapes it does not route do not bound it.)
	rowFloor := math.Inf(1)
	for _, P := range bandShapes {
		if P <= row.MaxP {
			rowFloor = math.Min(rowFloor, band.Measured[P])
		}
	}
	if row.MinRatio < Q4KM5CrossoverMinimumRatio {
		t.Fatalf("pinned row MinRatio=%.4f is below the fak#9937 gate %.2f", row.MinRatio, Q4KM5CrossoverMinimumRatio)
	}
	if !q4kFinite(row.MinRatio) {
		t.Fatalf("pinned row MinRatio=%v is not finite", row.MinRatio)
	}
	if row.MinRatio > rowFloor*(1+1e-9) {
		t.Fatalf("pinned row MinRatio=%.6f exceeds the freshly measured floor %.6f over its band [%d,%d]; a fresh measurement below the pin is a regression",
			row.MinRatio, rowFloor, row.MinP, row.MaxP)
	}
	for _, P := range bandShapes {
		if P > row.MaxP {
			break
		}
		if !Q4KM5CrossoverPredicate(receipt.DeviceName, receipt.OSVersion, P) {
			t.Fatalf("the pinned row does not admit mode 2 at measured P=%d", P)
		}
	}
	// A P above the pinned band is unrouted and must stay fail-closed scalar.
	if Q4KM5CrossoverPredicate(receipt.DeviceName, receipt.OSVersion, row.MaxP+1) {
		t.Fatalf("crossover admitted P=%d above the pinned band; an unpinned P must execute scalar", row.MaxP+1)
	}
	if !Q4KM5CrossoverAdmits(band.MinP) {
		t.Fatal("the live device/OS reports the crossover admits false at a measured, pinned P")
	}
	t.Logf("[HW-WITNESSED] pinned row: family=%q os=%q band=[%d,%d] minRatio=%.4f (fresh floor over row band %.4f, measured band [%d,%d] floor %.4f) witness=%q",
		row.Family, row.OSVersion, row.MinP, row.MaxP, row.MinRatio, rowFloor, band.MinP, band.MaxP, band.Floor, row.Witness)
}
