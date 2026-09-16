//go:build darwin && arm64 && cgo

package metalgemm

import (
	"slices"
	"strconv"
	"testing"
)

// withPinnedCrossoverRow installs one crossover row for the duration of fn and restores the
// prior table, so the routing witness can exercise the >=1.10x gate without permanently
// promoting an unwitnessed row into the production table.
func withPinnedCrossoverRow(t *testing.T, row q4kM5CrossoverRow, fn func()) {
	t.Helper()
	prior := q4kM5CrossoverTable
	q4kM5CrossoverTable = []q4kM5CrossoverRow{row}
	defer func() { q4kM5CrossoverTable = prior }()
	fn()
}

// TestQ4KM5ProductionSeamDefaultsToWideTile pins the end-to-end production seam that fak#13124
// left dormant: with the model-layer opt-in now default-ON (FAK_Q4K_M5 unset), the exact call the
// prefill walk makes — SetQ4KGEMMMode(Q4KGEMMModeForPrompt(P)) at the production panel width
// PromptPanelMaxTokens (128) — must select mode 2 on the receipted Apple M3 Pro / macOS 26 box, and
// the graph must report that mode. Off the receipted device the crossover gate keeps the request
// inert, so the same call yields scalar and the scalar identity stays executed (fail-closed).
func TestQ4KM5ProductionSeamDefaultsToWideTile(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()
	const in, out = 256, 64
	w := UploadQ4K(q4kTestRaw(out, in, 0x13124), out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	// Simulate the model layer's default-on resolution without an env dependency: the prefill walk
	// calls SetGEMMUseM5(true) on FAK_Q4K_M5-unset. Restore whatever the ambient process had.
	priorM5 := q4kUseM5.Swap(true)
	defer q4kUseM5.Store(priorM5)

	P := PromptPanelMaxTokens // 128: the production panel width (P>=64 wide-tile envelope)
	g, err := BeginProjectionGraph(make([]float32, P*in), nil, nil, P, in)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()

	mode := Q4KGEMMModeForPrompt(P)
	admitted := Q4KM5CrossoverAdmits()
	if admitted {
		// Receipted box: the production seam must actually select the wide-tile kernel.
		if mode != Q4KGEMMModeM5CooperativeSMEM {
			t.Fatalf("admitted device: Q4KGEMMModeForPrompt(%d)=%v, want mode 2", P, mode)
		}
		if !g.SetQ4KGEMMMode(mode) {
			t.Fatal("graph refused the wide-tile mode on an admitted P=128 panel")
		}
		if got := g.Q4KGEMMMode(); got != Q4KGEMMModeM5CooperativeSMEM {
			t.Fatalf("graph mode=%v, want mode 2", got)
		}
		if _, err := g.EncodeQ4K(w); err != nil {
			t.Fatalf("wide-tile P=128 encode: %v", err)
		}
	} else {
		// Unreceipted device: the gate keeps default-on inert, so the scalar identity stays executed.
		if mode != Q4KGEMMModeScalar {
			t.Fatalf("unadmitted device: Q4KGEMMModeForPrompt(%d)=%v, want scalar", P, mode)
		}
		t.Logf("device %q is not the receipted M3 Pro/macOS 26 box; default-on is inert as designed", DeviceName())
	}
}

func TestQ4KM5CooperativeSMEMCandidateMatchesIndependentCPUOracle(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	const out, in, prompt = 96, 512, 64
	raw := q4kTestRaw(out, in, 0x9937)
	x := make([]float32, prompt*in)
	for i := range x {
		x[i] = float32((i*37)%251-125) / 127
	}
	w := UploadQ4K(raw, out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	control := make([]float32, prompt*out)
	controlID := w.GEMMWithEventsMode(x, prompt, control, nil, Q4KGEMMModeScalar)
	if controlID != (Q4KGEMMIdentity{Requested: Q4KGEMMExecutedScalar, Executed: Q4KGEMMExecutedScalar}) {
		t.Fatalf("scalar identity=%+v", controlID)
	}

	candidate := make([]float32, prompt*out)
	candidateID := w.GEMMWithEventsMode(x, prompt, candidate, nil, Q4KGEMMModeM5CooperativeSMEM)
	wantCandidate := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedM5CooperativeSMEM, Executed: Q4KGEMMExecutedM5CooperativeSMEM}
	if candidateID != wantCandidate {
		t.Fatalf("candidate identity=%+v want %+v", candidateID, wantCandidate)
	}

	reference := make([]float32, prompt*out)
	for p := 0; p < prompt; p++ {
		copy(reference[p*out:(p+1)*out], q4kVectorizedReference(raw, out, in, x[p*in:(p+1)*in]))
	}
	for name, got := range map[string][]float32{"control": control, "candidate": candidate} {
		cosine, maxRel := q4kTestCosineMaxRel(reference, got)
		if cosine < 0.999999 || maxRel > 5e-3 {
			t.Fatalf("%s vs independent CPU oracle: cosine=%g maxRel=%g", name, cosine, maxRel)
		}
	}
	cosine, maxRel := q4kTestCosineMaxRel(control, candidate)
	if cosine < 0.999999 || maxRel > 5e-3 {
		t.Fatalf("candidate vs scalar fallback: cosine=%g maxRel=%g", cosine, maxRel)
	}

	untouched := make([]float32, prompt*out)
	for i := range untouched {
		untouched[i] = 9937
	}
	before := slices.Clone(untouched)
	unavailable := w.GEMMWithEventsMode(x, prompt, untouched, nil, Q4KGEMMModeM5CooperativeSMEMUnavailable)
	if unavailable != (Q4KGEMMIdentity{Requested: Q4KGEMMExecutedM5CooperativeSMEM, Executed: Q4KGEMMNotExecuted}) {
		t.Fatalf("unavailable identity=%+v", unavailable)
	}
	if !slices.Equal(untouched, before) {
		t.Fatal("unavailable candidate changed output instead of failing closed")
	}
}

// TestQ4KM5CrossoverGatesPanelRegime is the routing witness for the pinned crossover the ticket
// asks for: mode 2 is requested for the widened panel shapes P>=64 ONLY where a pinned row both
// matches the device/OS and clears the fak#9937 >=1.10x margin; below the margin, on a mismatched
// device/OS, and for P<64 the typed requested identity stays scalar. It is pure Go (no Metal work),
// so it runs on any host and pins the gate itself rather than a magic literal.
func TestQ4KM5CrossoverGatesPanelRegime(t *testing.T) {
	// Exactly one row is pinned: the on-silicon Apple M3 Pro / macOS 26 receipt from
	// TestQ4KCrossoverReceiptCandidateVsScalar. A second row would be an unpinned device reaching
	// mode 2, and zero rows would mean the gate never opened despite the measured margin.
	if n := Q4KM5CrossoverRowCount(); n != 1 {
		t.Fatalf("production crossover table has %d rows; exactly one on-silicon M3 Pro row is pinned", n)
	}
	if Q4KM5CrossoverMinimumRatio < 1.10 {
		t.Fatalf("crossover gate=%g, want >= 1.10", Q4KM5CrossoverMinimumRatio)
	}
	// q4kGEMMModeForPrompt must never reach mode 2 without the SetGEMMUseM5 opt-in, even for
	// P>=64 on a device the pinned table admits.
	priorM5 := q4kUseM5.Swap(false)
	defer q4kUseM5.Store(priorM5)
	for _, P := range []int{32, 64, 128, 256} {
		if got := q4kGEMMModeForPrompt(P); got == Q4KGEMMModeM5CooperativeSMEM {
			t.Fatalf("P=%d selected mode 2 without the SetGEMMUseM5 opt-in", P)
		}
	}

	const device, osVersion = "Apple M3 Pro", "26.6.2"
	// On the device the pinned production row was measured on, the opt-in must reach mode 2 for
	// the widened panel shapes. This is the end-to-end gate the row exists to open.
	if Q4KM5CrossoverPredicate(device, osVersion) {
		q4kUseM5.Store(true)
		for _, P := range []int{64, 128, 256} {
			if got := q4kGEMMModeForPrompt(P); got != Q4KGEMMModeM5CooperativeSMEM {
				t.Fatalf("P=%d on the pinned device selected %v, want mode 2", P, got)
			}
		}
		q4kUseM5.Store(false)
	}
	// A row below the >=1.10x gate must NOT admit mode 2.
	withPinnedCrossoverRow(t, q4kM5CrossoverRow{Family: device, OSVersion: "26", MinRatio: 1.09, Witness: "witness://below-gate"}, func() {
		if Q4KM5CrossoverPredicate(device, osVersion) {
			t.Fatal("a row gated below 1.10 admitted mode 2")
		}
	})
	// A row above the gate admits only for the pinned device/OS.
	withPinnedCrossoverRow(t, q4kM5CrossoverRow{Family: device, OSVersion: "26", MinRatio: 1.10, Witness: "witness://oracle-parity"}, func() {
		if !Q4KM5CrossoverPredicate(device, osVersion) {
			t.Fatal("a matching >=1.10 row did not admit mode 2")
		}
		if Q4KM5CrossoverPredicate("Apple M1 Max", osVersion) {
			t.Fatal("crossover admitted a device outside its pin")
		}
		if Q4KM5CrossoverPredicate(device, "25.4.0") {
			t.Fatal("crossover admitted an OS outside its pin")
		}
		q4kUseM5.Store(true)
		defer q4kUseM5.Store(false)
		// The encode-time selector consults the LIVE device, so the mode-2 assertions only hold
		// where Metal reports a matching device; on any other host the selector stays scalar.
		if Available() && DeviceName() == device {
			if got := q4kGEMMModeForPrompt(64); got != Q4KGEMMModeM5CooperativeSMEM {
				t.Fatalf("P=64 with an admitting row selected %v, want mode 2", got)
			}
			if got := q4kGEMMModeForPrompt(128); got != Q4KGEMMModeM5CooperativeSMEM {
				t.Fatalf("P=128 with an admitting row selected %v, want mode 2", got)
			}
			// The typed requested identity must report mode 2 (not scalar) for the admitted shape.
			if req := q4kGEMMRequestedExecution(64, q4kGEMMModeForPrompt(64)); req != Q4KGEMMExecutedM5CooperativeSMEM {
				t.Fatalf("P=64 requested identity=%v, want M5CooperativeSMEM", req)
			}
		}
		// P<64 is outside the wide-tile envelope even under an admitting row.
		if got := q4kGEMMModeForPrompt(32); got == Q4KGEMMModeM5CooperativeSMEM {
			t.Fatalf("P=32 selected mode 2 outside the wide-tile envelope")
		}
		if req := q4kGEMMRequestedExecution(64, Q4KGEMMModeScalar); req != Q4KGEMMExecutedScalar {
			t.Fatalf("scalar requested identity=%v, want scalar", req)
		}
	})
}

// TestQ4KM5CooperativeSMEMOraclePanelShapes extends the #9937 CPU-oracle parity witness to the
// widened panel shapes the ticket names (P in {64,128}) against the real [out,in] projection
// geometry, and asserts the candidate-vs-scalar fallback stays within tolerance at each shape.
func TestQ4KM5CooperativeSMEMOraclePanelShapes(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	const out, in = 96, 512
	raw := q4kTestRaw(out, in, 0x13088)
	w := UploadQ4K(raw, out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	for _, prompt := range []int{64, 128} {
		t.Run("P"+strconv.Itoa(prompt), func(t *testing.T) {
			x := make([]float32, prompt*in)
			for i := range x {
				x[i] = float32((i*37)%251-125) / 127
			}
			control := make([]float32, prompt*out)
			if id := w.GEMMWithEventsMode(x, prompt, control, nil, Q4KGEMMModeScalar); id != (Q4KGEMMIdentity{Requested: Q4KGEMMExecutedScalar, Executed: Q4KGEMMExecutedScalar}) {
				t.Fatalf("scalar identity=%+v", id)
			}
			candidate := make([]float32, prompt*out)
			want := Q4KGEMMIdentity{Requested: Q4KGEMMExecutedM5CooperativeSMEM, Executed: Q4KGEMMExecutedM5CooperativeSMEM}
			if id := w.GEMMWithEventsMode(x, prompt, candidate, nil, Q4KGEMMModeM5CooperativeSMEM); id != want {
				t.Fatalf("candidate identity=%+v want %+v", id, want)
			}
			reference := make([]float32, prompt*out)
			for p := 0; p < prompt; p++ {
				copy(reference[p*out:(p+1)*out], q4kVectorizedReference(raw, out, in, x[p*in:(p+1)*in]))
			}
			for name, got := range map[string][]float32{"control": control, "candidate": candidate} {
				cosine, maxRel := q4kTestCosineMaxRel(reference, got)
				if cosine < 0.999999 || maxRel > 5e-3 {
					t.Fatalf("P=%d %s vs independent CPU oracle: cosine=%g maxRel=%g", prompt, name, cosine, maxRel)
				}
			}
			if cosine, maxRel := q4kTestCosineMaxRel(control, candidate); cosine < 0.999999 || maxRel > 5e-3 {
				t.Fatalf("P=%d candidate vs scalar fallback: cosine=%g maxRel=%g", prompt, cosine, maxRel)
			}
		})
	}
}

// TestQ4KM5CooperativeSMEMGraphFailsClosedBelow64 pins the graph-side selector contract: requesting
// the wide-tile candidate on a panel below the kernel's 64-token eligibility is refused before any
// encode, the graph keeps the scalar identity, and an ineligible request never mutates state.
func TestQ4KM5CooperativeSMEMGraphFailsClosedBelow64(t *testing.T) {
	if !Available() {
		t.Skip("Metal unavailable")
	}
	defer ResetQ4K()

	const in, out = 256, 64
	w := UploadQ4K(q4kTestRaw(out, in, 0x13088), out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	defer w.Release()

	g, err := BeginProjectionGraph(make([]float32, 32*in), nil, nil, 32, in)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()

	if g.SetQ4KGEMMMode(Q4KGEMMModeM5CooperativeSMEM) {
		t.Fatal("P=32 graph accepted the P>=64 wide-tile candidate")
	}
	if got := g.Q4KGEMMMode(); got != Q4KGEMMModeScalar {
		t.Fatalf("P=32 graph mode=%v after a refused mode-2 request, want scalar", got)
	}
	// An explicit scalar request stays the accepted no-op, and the encode still returns a result.
	if !g.SetQ4KGEMMMode(Q4KGEMMModeScalar) {
		t.Fatal("P=32 graph refused the scalar mode")
	}
	if _, err := g.EncodeQ4K(w); err != nil {
		t.Fatalf("P=32 scalar encode after refused mode-2 request: %v", err)
	}
}
