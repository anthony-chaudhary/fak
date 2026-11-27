package compute

import (
	"errors"
	"math"
	"sync"
	"testing"
)

// legacyGridIDBarrier reproduces the pre-epoch graph-replay failure mode:
// tagging barrier lines with (gridID, splitCount) goes stale when a graph
// node keeps the same gridID across replays.
type legacyGridIDBarrier struct {
	mu    sync.Mutex
	lines []uint64
}

func (b *legacyGridIDBarrier) encode(gridID uint64, arrived uint32) uint64 {
	return (gridID+1)*ReplayDefaultNumBarriers*ReplayDefaultMaxSplits + uint64(arrived)
}

func (b *legacyGridIDBarrier) init(line int, gridID uint64) {
	b.mu.Lock()
	b.lines[line] = b.encode(gridID, 0)
	b.mu.Unlock()
}

func (b *legacyGridIDBarrier) arrive(line int) {
	b.mu.Lock()
	b.lines[line]++
	b.mu.Unlock()
}

func (b *legacyGridIDBarrier) ready(line int, gridID uint64, splits uint32) bool {
	b.mu.Lock()
	v := b.lines[line]
	b.mu.Unlock()
	return v == b.encode(gridID, splits)
}

// fak-test:runtime fast est=10ms lane=default
func TestReplayEpochBarrierTailLayoutAndABIInvariant(t *testing.T) {
	t.Parallel()

	cfg := ReplayBarrierConfig{NumBarriers: 2, NumMBlocks: 3, MaxSplits: 64}
	size, err := cfg.BufferSizeBytes()
	if err != nil {
		t.Fatalf("BufferSizeBytes: %v", err)
	}
	// Host ABI buffer size must equal bare barrier lines (6 * 128) with zero extra bytes.
	if want := 2 * 3 * ReplayBarrierLineBytes; size != want {
		t.Fatalf("buffer size = %d, want %d (ABI unchanged)", size, want)
	}

	buf, err := NewReplayEpochBarrierBuffer(cfg)
	if err != nil {
		t.Fatalf("NewReplayEpochBarrierBuffer: %v", err)
	}
	if buf.BufferBytes() != size {
		t.Fatalf("BufferBytes = %d, want %d", buf.BufferBytes(), size)
	}

	cta0, err := buf.NewCTAContext(101, 7, 0, 2)
	if err != nil {
		t.Fatalf("NewCTAContext: %v", err)
	}
	cta1, err := buf.NewCTAContext(101, 7, 1, 2)
	if err != nil {
		t.Fatalf("NewCTAContext: %v", err)
	}
	cta0.PDLSync()
	cta1.PDLSync()

	if err := cta0.Init(0, 0); err != nil {
		t.Fatalf("Init(0,0): %v", err)
	}
	if err := cta0.Arrive(0, 0); err != nil {
		t.Fatalf("Arrive(0,0): %v", err)
	}

	// Writing to barrier line (0, 0) must not disturb tail metadata at [64..76).
	epoch, exits := buf.InspectTail()
	if epoch != 0 || exits != 0 {
		t.Fatalf("tail corrupted after line 0 barrier write: epoch=%d exits=%d", epoch, exits)
	}

	// Retiring CTA 0 increments exit count without clobbering line (0, 0).
	adv0, ep0, err := cta0.Retire()
	if err != nil || adv0 || ep0 != 0 {
		t.Fatalf("cta0.Retire = (%v, %d, %v), want (false, 0, nil)", adv0, ep0, err)
	}
	epoch, exits = buf.InspectTail()
	if epoch != 0 || exits != 1 {
		t.Fatalf("tail after cta0 retire: epoch=%d exits=%d, want (0, 1)", epoch, exits)
	}
	// Last CTA resets exit count and bumps epoch, leaving line (0, 0) state intact.
	adv1, ep1, err := cta1.Retire()
	if err != nil || !adv1 || ep1 != 1 {
		t.Fatalf("cta1.Retire = (%v, %d, %v), want (true, 1, nil)", adv1, ep1, err)
	}
	epoch, exits = buf.InspectTail()
	if epoch != 1 || exits != 0 {
		t.Fatalf("tail after last retire: epoch=%d exits=%d, want (1, 0)", epoch, exits)
	}
	line0, err := buf.ReadBarrierLine(0, 0)
	if err != nil {
		t.Fatalf("ReadBarrierLine: %v", err)
	}
	if wantLine := buf.EncodeState(0, 1); line0 != wantLine {
		t.Fatalf("line 0 clobbered by tail retire: got %d, want %d", line0, wantLine)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestReplayEpochBarrierPostPDLOrderingEnforced(t *testing.T) {
	t.Parallel()

	buf, err := NewReplayEpochBarrierBuffer(ReplayBarrierConfig{NumMBlocks: 1})
	if err != nil {
		t.Fatalf("NewReplayEpochBarrierBuffer: %v", err)
	}
	cta, err := buf.NewCTAContext(9, 3, 0, 1)
	if err != nil {
		t.Fatalf("NewCTAContext: %v", err)
	}

	if _, err := cta.ReadEpoch(); !errors.Is(err, ErrEpochReadBeforePDLSync) {
		t.Fatalf("ReadEpoch pre-PDL err = %v, want ErrEpochReadBeforePDLSync", err)
	}
	if err := cta.Init(0, 0); !errors.Is(err, ErrEpochReadBeforePDLSync) {
		t.Fatalf("Init pre-PDL err = %v, want ErrEpochReadBeforePDLSync", err)
	}
	if _, err := cta.PollInit(0, 0, 2); !errors.Is(err, ErrEpochReadBeforePDLSync) {
		t.Fatalf("PollInit pre-PDL err = %v, want ErrEpochReadBeforePDLSync", err)
	}
	if err := cta.Arrive(0, 0); !errors.Is(err, ErrEpochReadBeforePDLSync) {
		t.Fatalf("Arrive pre-PDL err = %v, want ErrEpochReadBeforePDLSync", err)
	}
	if _, err := cta.PollWait(0, 0, 2); !errors.Is(err, ErrEpochReadBeforePDLSync) {
		t.Fatalf("PollWait pre-PDL err = %v, want ErrEpochReadBeforePDLSync", err)
	}
	if _, _, err := cta.Retire(); !errors.Is(err, ErrEpochReadBeforePDLSync) {
		t.Fatalf("Retire pre-PDL err = %v, want ErrEpochReadBeforePDLSync", err)
	}

	cta.PDLSync()
	ep, err := cta.ReadEpoch()
	if err != nil || ep != 0 {
		t.Fatalf("ReadEpoch post-PDL = (%d, %v), want (0, nil)", ep, err)
	}
	if _, _, err := cta.Retire(); err != nil {
		t.Fatalf("Retire post-PDL failed: %v", err)
	}
	if _, _, err := cta.Retire(); !errors.Is(err, ErrReplayBarrierRetireOrder) {
		t.Fatalf("duplicate Retire err = %v, want ErrReplayBarrierRetireOrder", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestReplayEpochBarrierTwoReplaysSameNodeOracle(t *testing.T) {
	t.Parallel()

	const (
		fixedNodeID uint64 = 42
		fixedGridID uint64 = 7
		numSplits   uint32 = 4
		numCTAs     uint32 = 5 // 4 split producers + 1 consumer/init CTA
		width              = 8
	)

	// Reference partials per replay (Replay 1 vs Replay 2 carry distinct values).
	replayInputs := [2][4][width]float32{
		{
			{1.25, -2.5, 3.75, 4.0, 0.5, -1.5, 2.25, 8.0},
			{0.5, 1.5, -0.75, 2.0, 1.25, 3.5, -1.25, 4.0},
			{-0.25, 0.75, 1.0, -3.0, 2.5, 0.5, 4.0, -2.0},
			{2.0, -0.5, 1.5, 5.0, -1.0, 2.0, 0.25, 1.0},
		},
		{
			{10.5, 1.25, -4.5, 0.25, 7.0, 2.5, -3.25, 1.5},
			{-2.5, 3.75, 8.5, -1.25, 0.5, 6.0, 1.75, -0.5},
			{4.25, -1.0, 2.0, 3.5, -2.25, 0.75, 5.5, 2.25},
			{1.75, 4.5, -0.5, 6.25, 3.0, -4.25, 1.0, 3.75},
		},
	}

	// Independent eager reference (no graph replay / no shared stale barrier).
	eagerRef := func(inputs [4][width]float32) [width]float32 {
		var out [width]float32
		for s := 0; s < int(numSplits); s++ {
			for i := 0; i < width; i++ {
				out[i] += inputs[s][i]
			}
		}
		return out
	}

	// Negative control: legacy (gridID, splitCount) barrier leaks Replay 1 partials
	// into Replay 2 when the consumer polls before Replay 2 producers overwrite.
	legacy := &legacyGridIDBarrier{lines: make([]uint64, 2)}
	var legacyPartials [4][width]float32
	// Replay 1
	legacy.init(0, fixedGridID)
	for s := 0; s < int(numSplits); s++ {
		legacyPartials[s] = replayInputs[0][s]
		legacy.arrive(0)
	}
	if !legacy.ready(0, fixedGridID, numSplits) {
		t.Fatal("legacy replay 1 barrier should be ready")
	}
	// Replay 2 starts with the SAME gridID; consumer checks before Replay 2 init/arrive:
	if !legacy.ready(0, fixedGridID, numSplits) {
		t.Fatal("expected legacy barrier to appear spuriously ready on replay 2")
	}
	staleRead := eagerRef(legacyPartials)
	wantReplay2 := eagerRef(replayInputs[1])
	if staleRead == wantReplay2 {
		t.Fatal("test fixture failure: replay 1 and replay 2 outputs must differ")
	}

	// Epoch-tagged replay barrier: same buffer, same nodeID & gridID, zero host reset.
	buf, err := NewReplayEpochBarrierBuffer(ReplayBarrierConfig{NumBarriers: 2, NumMBlocks: 1, MaxSplits: 64})
	if err != nil {
		t.Fatalf("NewReplayEpochBarrierBuffer: %v", err)
	}
	partials := make([][width]float32, numSplits)

	runReplay := func(replayIdx int, expectEpoch uint64) [width]float32 {
		t.Helper()

		ctas := make([]*ReplayBarrierCTAContext, numCTAs)
		for id := uint32(0); id < numCTAs; id++ {
			c, err := buf.NewCTAContext(fixedNodeID, fixedGridID, id, numCTAs)
			if err != nil {
				t.Fatalf("replay %d NewCTAContext(%d): %v", replayIdx, id, err)
			}
			ctas[id] = c
		}

		// Consumer CTA completes PDL sync and checks stale barrier BEFORE init.
		consumer := ctas[0]
		consumer.PDLSync()
		gotEpoch, err := consumer.ReadEpoch()
		if err != nil {
			t.Fatalf("replay %d consumer.ReadEpoch: %v", replayIdx, err)
		}
		if gotEpoch != expectEpoch {
			t.Fatalf("replay %d consumer epoch = %d, want %d", replayIdx, gotEpoch, expectEpoch)
		}
		// Crucial replay invariant: consumer must NOT mistake the previous replay's
		// completed barrier line for the current replay!
		readyEarly, err := consumer.PollWait(0, 0, numSplits)
		if err != nil {
			t.Fatalf("replay %d PollWait early: %v", replayIdx, err)
		}
		if readyEarly {
			t.Fatalf("replay %d consumer saw stale barrier ready before replay %d init/arrive", replayIdx, replayIdx)
		}

		// Producer CTA completes PDL sync and checks PollInit BEFORE consumer runs Init.
		prod0 := ctas[1]
		prod0.PDLSync()
		initedEarly, err := prod0.PollInit(0, 0, numSplits)
		if err != nil {
			t.Fatalf("replay %d PollInit early: %v", replayIdx, err)
		}
		if initedEarly {
			t.Fatalf("replay %d producer saw stale previous-epoch barrier as initialized", replayIdx)
		}

		// Consumer initializes the barrier line for the current epoch.
		if err := consumer.Init(0, 0); err != nil {
			t.Fatalf("replay %d consumer.Init: %v", replayIdx, err)
		}

		// Launch all producer split CTAs and consumer CTA concurrently.
		var wg sync.WaitGroup
		var gotOut [width]float32

		for s := uint32(0); s < numSplits; s++ {
			splitIdx := s
			prod := ctas[1+splitIdx]
			wg.Add(1)
			go func() {
				defer wg.Done()
				prod.PDLSync()
				ep, err := prod.ReadEpoch()
				if err != nil || ep != expectEpoch {
					t.Errorf("replay %d split %d epoch=(%d,%v), want %d", replayIdx, splitIdx, ep, err, expectEpoch)
					return
				}
				for {
					ok, err := prod.PollInit(0, 0, numSplits)
					if err != nil {
						t.Errorf("replay %d split %d PollInit: %v", replayIdx, splitIdx, err)
						return
					}
					if ok {
						break
					}
				}
				partials[splitIdx] = replayInputs[replayIdx][splitIdx]
				if err := prod.Arrive(0, 0); err != nil {
					t.Errorf("replay %d split %d Arrive: %v", replayIdx, splitIdx, err)
					return
				}
				if _, _, err := prod.Retire(); err != nil {
					t.Errorf("replay %d split %d Retire: %v", replayIdx, splitIdx, err)
				}
			}()
		}

		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				ready, err := consumer.PollWait(0, 0, numSplits)
				if err != nil {
					t.Errorf("replay %d consumer PollWait: %v", replayIdx, err)
					return
				}
				if ready {
					break
				}
			}
			for s := 0; s < int(numSplits); s++ {
				for i := 0; i < width; i++ {
					gotOut[i] += partials[s][i]
				}
			}
			if _, _, err := consumer.Retire(); err != nil {
				t.Errorf("replay %d consumer Retire: %v", replayIdx, err)
			}
		}()

		wg.Wait()
		return gotOut
	}

	out1 := runReplay(0, 0)
	want1 := eagerRef(replayInputs[0])
	for i := range want1 {
		if math.Float32bits(out1[i]) != math.Float32bits(want1[i]) {
			t.Fatalf("replay 1[%d] bits mismatch: got %g, want %g", i, out1[i], want1[i])
		}
	}
	epAfter1, exitsAfter1 := buf.InspectTail()
	if epAfter1 != 1 || exitsAfter1 != 0 {
		t.Fatalf("after replay 1 tail=(%d,%d), want (1,0)", epAfter1, exitsAfter1)
	}

	out2 := runReplay(1, 1)
	for i := range wantReplay2 {
		if math.Float32bits(out2[i]) != math.Float32bits(wantReplay2[i]) {
			t.Fatalf("replay 2[%d] bits mismatch: got %g, want %g (stale replay 1 leaked?)", i, out2[i], wantReplay2[i])
		}
	}
	epAfter2, exitsAfter2 := buf.InspectTail()
	if epAfter2 != 2 || exitsAfter2 != 0 {
		t.Fatalf("after replay 2 tail=(%d,%d), want (2,0)", epAfter2, exitsAfter2)
	}
}
