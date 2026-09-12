package cacheobs

import (
	"sync"
	"testing"
)

// TestObserveLabeledSourceAtomicContract is the witness for #12886: one observation books
// global depth, per-label depth, per-label SOURCE, and global SOURCE under ONE o.mu
// acquisition, and one combined snapshot operation returns all three coherently. It also
// pins the legacy contract: a depth-only tap books NO source, and an atomic tap with no
// source evidence books explicit UNKNOWN provenance, never a local classification.
//
// Table-driven over the single-observation cases, then a reconciliation pass and a bounded
// concurrent reader/writer subcase.
func TestObserveLabeledSourceAtomicContract(t *testing.T) {
	type sourceCase struct {
		name string
		src  *SourceSplit // nil = legacy / absent source evidence
	}
	cases := []sourceCase{
		{name: "local_hit", src: &SourceSplit{LocalHit: 80}},
		{name: "external_transfer", src: &SourceSplit{ExternalTransfer: 80}},
		{name: "absent_source_legacy", src: nil},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			o := New()
			const prompt, reused, eligible = 100, 80, 100 // one turn; everything eligible
			labels := Labels{Model: "m", Tenant: "acme", Phase: PhasePrefill}
			o.ObserveLabeledSource(labels, prompt, reused, reused, eligible, tc.src)

			cs := o.CombinedSnapshot()

			// Global depth booked exactly once.
			if cs.Depth.Turns != 1 || cs.Depth.PromptTokens != prompt || cs.Depth.ReusedTokens != reused {
				t.Fatalf("global depth = %+v, want one turn prompt=%d reused=%d", cs.Depth, prompt, reused)
			}
			// Per-label depth booked exactly once, reconciling with the global depth.
			if len(cs.Labels) != 1 {
				t.Fatalf("labeled rows = %+v, want exactly 1", cs.Labels)
			}
			row := cs.Labels[0]
			if row.Labels != labels || row.Turns != 1 || row.PromptTokens != prompt || row.ReusedTokens != reused {
				t.Fatalf("label depth row = %+v, want one turn prompt=%d reused=%d", row, prompt, reused)
			}
			// Global source is the sum of the row's source columns (reconciles in ONE read).
			wantRowCompute := row.LocalComputeTokens
			wantRowHit := row.LocalHitTokens
			wantRowExternal := row.ExternalTransferTokens
			wantRowUnknown := row.UnknownSourceTokens
			if cs.Source.LocalComputeTokens != wantRowCompute ||
				cs.Source.LocalHitTokens != wantRowHit ||
				cs.Source.ExternalTransferTokens != wantRowExternal ||
				cs.Source.UnknownTokens != wantRowUnknown {
				t.Fatalf("global source %+v does not reconcile with label row %+v", cs.Source, row)
			}

			switch tc.name {
			case "local_hit":
				if cs.Source.LocalHitTokens != reused || cs.Source.LocalComputeTokens != 0 ||
					cs.Source.ExternalTransferTokens != 0 || cs.Source.UnknownTokens != 0 {
					t.Fatalf("local-hit source = %+v, want hit=%d only", cs.Source, reused)
				}
			case "external_transfer":
				if cs.Source.ExternalTransferTokens != reused || cs.Source.LocalHitTokens != 0 ||
					cs.Source.LocalComputeTokens != 0 || cs.Source.UnknownTokens != 0 {
					t.Fatalf("external source = %+v, want external=%d only", cs.Source, reused)
				}
			case "absent_source_legacy":
				// The load-bearing legacy rule: no evidence is UNKNOWN, never local.
				if cs.Source.UnknownTokens != reused {
					t.Fatalf("absent source = %+v, want unknown=%d", cs.Source, reused)
				}
				if cs.Source.LocalHitTokens != 0 || cs.Source.LocalComputeTokens != 0 {
					t.Fatalf("absent source was mis-classified as local: %+v", cs.Source)
				}
				if row.UnknownSourceTokens != reused || row.LocalHitTokens != 0 || row.LocalComputeTokens != 0 {
					t.Fatalf("absent per-label source was mis-classified local: %+v", row)
				}
			}
		})
	}
}

// TestObserveLabeledSourceAtomicClampAndNoDoubleBooking pins the clamp/no-double-book
// contract on the atomic path: an over-claiming source split is capped at the turn's reuse
// (excess folded to UNKNOWN, never inflating a known bucket above the tokens that exist),
// depth counters are booked exactly once per turn, and a depth-only legacy tap books ZERO
// on the whole source axis.
func TestObserveLabeledSourceAtomicClampAndNoDoubleBooking(t *testing.T) {
	o := New()
	labels := Labels{Model: "m", Tenant: "t", Phase: PhaseDecode}
	// Depth clamps: source over-claims 500 but reuse is 100.
	o.ObserveLabeledSource(labels, 100, 100, 100, 100, &SourceSplit{LocalHit: 400, ExternalTransfer: 100})

	cs := o.CombinedSnapshot()
	if cs.Depth.Turns != 1 || cs.Depth.PromptTokens != 100 || cs.Depth.ReusedTokens != 100 {
		t.Fatalf("depth double-booked or unclamped: %+v", cs.Depth)
	}
	if got := cs.Source.LocalHitTokens + cs.Source.ExternalTransferTokens; got > 100 {
		t.Fatalf("source booked %d tokens against a 100-token turn (double booking)", got)
	}
	if cs.Source.UnknownTokens+cs.Source.LocalHitTokens+cs.Source.ExternalTransferTokens != 100 {
		t.Fatalf("source parts != turn reuse: %+v", cs.Source)
	}
	if cs.Labels[0].Turns != 1 || cs.Labels[0].ReusedTokens != 100 {
		t.Fatalf("label row double-booked: %+v", cs.Labels[0])
	}

	// Legacy depth-only taps must not touch the source axis at all.
	o2 := New()
	o2.Observe(500, 250)
	o2.ObserveLabeled(Labels{Model: "m", Tenant: "t"}, 500, 250, 250, 500)
	if s := o2.SourceSnapshot(); s != (SourceStats{}) {
		t.Fatalf("depth-only taps booked a phantom source spread: %+v", s)
	}
}

// TestObserveLabeledSourceAtomicReconciliation is the same-lock reconciliation witness:
// summing the per-label depth AND source columns across the combined snapshot reconciles
// exactly with the global depth and global source it returned, for mixed labels and mixed
// source evidence (known and absent).
func TestObserveLabeledSourceAtomicReconciliation(t *testing.T) {
	o := New()
	o.ObserveLabeledSource(Labels{Model: "qwen", Tenant: "acme"}, 1000, 900, 800, 1000, &SourceSplit{LocalHit: 500, ExternalTransfer: 300})
	o.ObserveLabeledSource(Labels{Model: "qwen", Tenant: "globex"}, 400, 100, 50, 400, nil) // absent -> unknown
	o.ObserveLabeledSource(Labels{Model: "llama", Tenant: "acme"}, 200, 100, 50, 200, &SourceSplit{LocalCompute: 50})

	cs := o.CombinedSnapshot()
	var depthTurns, depthPrompt, depthReused uint64
	var srcCompute, srcHit, srcExternal, srcUnknown uint64
	for _, r := range cs.Labels {
		depthTurns += r.Turns
		depthPrompt += r.PromptTokens
		depthReused += r.ReusedTokens
		srcCompute += r.LocalComputeTokens
		srcHit += r.LocalHitTokens
		srcExternal += r.ExternalTransferTokens
		srcUnknown += r.UnknownSourceTokens
	}
	if depthTurns != cs.Depth.Turns || depthPrompt != cs.Depth.PromptTokens || depthReused != cs.Depth.ReusedTokens {
		t.Fatalf("label depth sums desync from global depth: turns=%d/%d prompt=%d/%d reused=%d/%d",
			depthTurns, cs.Depth.Turns, depthPrompt, cs.Depth.PromptTokens, depthReused, cs.Depth.ReusedTokens)
	}
	if srcCompute != cs.Source.LocalComputeTokens || srcHit != cs.Source.LocalHitTokens ||
		srcExternal != cs.Source.ExternalTransferTokens || srcUnknown != cs.Source.UnknownTokens {
		t.Fatalf("label source sums desync from global source: %+v vs %+v", cs.Source,
			SourceStats{LocalComputeTokens: srcCompute, LocalHitTokens: srcHit, ExternalTransferTokens: srcExternal, UnknownTokens: srcUnknown})
	}
	// The globex row's absent evidence is UNKNOWN, not local.
	for _, r := range cs.Labels {
		if r.Labels.Tenant == "globex" && (r.UnknownSourceTokens != 50 || r.LocalHitTokens != 0) {
			t.Fatalf("globex absent-source row = %+v, want unknown=50", r)
		}
	}
}

// TestObserveLabeledSourceAtomicConcurrentReader is the bounded reader/writer subcase: real
// atomic observations run while a reader repeatedly takes the combined snapshot, and every
// snapshot it sees must be internally coherent - each axis's own invariants hold AND the
// label sums reconcile with the globals. Coordination is by channels/WaitGroup only, with
// no sleeps and no assumed scheduler order.
func TestObserveLabeledSourceAtomicConcurrentReader(t *testing.T) {
	o := New()
	const writers = 4
	const turnsPerWriter = 500

	done := make(chan struct{})
	var wg sync.WaitGroup

	// Writers: real atomic observations across distinct labels and source evidence.
	for w := 0; w < writers; w++ {
		w := w
		wg.Add(1)
		go func() {
			defer wg.Done()
			labels := Labels{Model: "m", Tenant: "tenant"}
			for i := 0; i < turnsPerWriter; i++ {
				var src *SourceSplit
				switch i % 3 {
				case 0:
					src = &SourceSplit{LocalHit: 4}
				case 1:
					src = &SourceSplit{ExternalTransfer: 4}
				default:
					src = nil
				}
				o.ObserveLabeledSource(labels, 10, 4, 4, 10, src)
			}
			_ = w
		}()
	}

	// One close goroutine so the reader learns completion without racing WaitGroup state.
	go func() {
		wg.Wait()
		close(done)
	}()

	// Reader: no WaitGroup entry, so it never blocks shutdown; it stops on the closed
	// channel, which is only closed after every writer has returned.
	var reads int
	for {
		select {
		case <-done:
			if reads == 0 {
				t.Fatal("reader completed zero coherent snapshots")
			}
			return
		default:
		}
		cs := o.CombinedSnapshot()
		reads++

		// Axis invariants from ONE lock acquisition.
		if cs.Depth.CacheableTokens < cs.Depth.ReusedTokens || cs.Depth.EligibleTokens < cs.Depth.CacheableTokens {
			t.Fatalf("depth invariants broken in a combined snapshot: %+v", cs.Depth)
		}
		if cs.Source.TotalTokens != cs.Source.LocalComputeTokens+cs.Source.LocalHitTokens+
			cs.Source.ExternalTransferTokens+cs.Source.UnknownTokens {
			t.Fatalf("source parts != total in a combined snapshot: %+v", cs.Source)
		}
		var dTurns, dPrompt, dReused, sHit, sExternal, sUnknown uint64
		for _, r := range cs.Labels {
			dTurns += r.Turns
			dPrompt += r.PromptTokens
			dReused += r.ReusedTokens
			sHit += r.LocalHitTokens
			sExternal += r.ExternalTransferTokens
			sUnknown += r.UnknownSourceTokens
		}
		if dTurns != cs.Depth.Turns || dPrompt != cs.Depth.PromptTokens || dReused != cs.Depth.ReusedTokens {
			t.Fatalf("label depth sums desync from global depth in a combined snapshot: %+v", cs)
		}
		if sHit != cs.Source.LocalHitTokens || sExternal != cs.Source.ExternalTransferTokens || sUnknown != cs.Source.UnknownTokens {
			t.Fatalf("label source sums desync from global source in a combined snapshot: %+v", cs)
		}
	}
}
