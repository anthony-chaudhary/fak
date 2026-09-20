package modelengine

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func testModelConfig() model.Config {
	return model.Config{
		NumLayers:  2,
		NumKVHeads: 2,
		HeadDim:    64,
		VocabSize:  128,
	}
}

func TestElasticBlockAllocatorInitAndInvariant(t *testing.T) {
	cfg := ElasticBlockAllocatorConfig{
		InitialBlocks: 32,
		ModelCfg:      testModelConfig(),
		BlockTokens:   16,
	}
	alloc, err := NewElasticBlockAllocator(cfg)
	if err != nil {
		t.Fatalf("NewElasticBlockAllocator failed: %v", err)
	}

	if got := alloc.State(); got != StateStable {
		t.Fatalf("expected state %v, got %v", StateStable, got)
	}

	stats := alloc.Stats()
	if stats.CurrentTotalBlocks != 32 || stats.FreeBlocks != 32 || stats.AllocatedBlocks != 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("invariant violated on init: %v", err)
	}
}

func TestElasticBlockAllocatorExpansion(t *testing.T) {
	alloc, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks: 10,
		ModelCfg:      testModelConfig(),
		BlockTokens:   16,
	})
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	if err := alloc.Expand(20); err != nil {
		t.Fatalf("expand failed: %v", err)
	}

	if got := alloc.State(); got != StateStable {
		t.Fatalf("expected StateStable, got %v", got)
	}

	stats := alloc.Stats()
	if stats.CurrentTotalBlocks != 20 || stats.FreeBlocks != 20 || stats.AllocatedBlocks != 0 {
		t.Fatalf("expected 20 total/free blocks, got %+v", stats)
	}
	if stats.TotalExpansions != 1 {
		t.Fatalf("expected 1 expansion, got %d", stats.TotalExpansions)
	}

	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("invariant violated after expansion: %v", err)
	}

	// Invalid expansion
	if err := alloc.Expand(15); err == nil {
		t.Fatal("expected error expanding to smaller or equal target, got nil")
	}
}

func TestElasticBlockAllocatorDrainFreeBlocks(t *testing.T) {
	alloc, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks: 20,
		ModelCfg:      testModelConfig(),
		BlockTokens:   16,
	})
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	// Drain down to 10 when all blocks are free
	if err := alloc.RequestDrain(10); err != nil {
		t.Fatalf("request drain failed: %v", err)
	}

	// Since all blocks were free, drain should have completed immediately
	if got := alloc.State(); got != StateStable {
		t.Fatalf("expected StateStable after draining free blocks, got %v", got)
	}

	stats := alloc.Stats()
	if stats.CurrentTotalBlocks != 10 || stats.FreeBlocks != 10 || stats.AllocatedBlocks != 0 {
		t.Fatalf("expected 10 blocks, got %+v", stats)
	}

	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("invariant violated: %v", err)
	}
}

func TestElasticBlockAllocatorProgressiveDrainWithAllocations(t *testing.T) {
	alloc, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks: 10,
		ModelCfg:      testModelConfig(),
		BlockTokens:   16,
	})
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	_ = alloc.RegisterSequence("seq-1", 1)
	_ = alloc.RegisterSequence("seq-2", 2)

	// Allocate 4 blocks to seq-1 and 4 blocks to seq-2 (total 8 allocated, 2 free)
	seq1Blocks := make([]int, 4)
	for i := 0; i < 4; i++ {
		b, err := alloc.AllocBlock("seq-1")
		if err != nil {
			t.Fatalf("alloc block failed: %v", err)
		}
		seq1Blocks[i] = b
	}

	seq2Blocks := make([]int, 4)
	for i := 0; i < 4; i++ {
		b, err := alloc.AllocBlock("seq-2")
		if err != nil {
			t.Fatalf("alloc block failed: %v", err)
		}
		seq2Blocks[i] = b
	}

	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("invariant violated after allocations: %v", err)
	}

	// Request drain down to 6 blocks. Current total is 10 (8 allocated, 2 free).
	// Initial request drain should reclaim the 2 free blocks -> current total = 8.
	// State should be StateDraining because we haven't reached target 6 yet.
	if err := alloc.RequestDrain(6); err != nil {
		t.Fatalf("request drain failed: %v", err)
	}

	if got := alloc.State(); got != StateDraining {
		t.Fatalf("expected StateDraining, got %v", got)
	}
	stats := alloc.Stats()
	if stats.CurrentTotalBlocks != 8 || stats.AllocatedBlocks != 8 || stats.FreeBlocks != 0 {
		t.Fatalf("expected 8 current total (8 allocated, 0 free), got %+v", stats)
	}

	// Free 2 blocks from seq-1 manually. The allocator should retire them instead of adding to free list.
	if err := alloc.FreeBlock("seq-1", seq1Blocks[0]); err != nil {
		t.Fatalf("free block failed: %v", err)
	}
	if err := alloc.FreeBlock("seq-1", seq1Blocks[1]); err != nil {
		t.Fatalf("free block failed: %v", err)
	}

	// Now current total should be 6, and drain should be complete -> StateStable
	if got := alloc.State(); got != StateStable {
		t.Fatalf("expected StateStable after freeing blocks, got %v", got)
	}
	stats = alloc.Stats()
	if stats.CurrentTotalBlocks != 6 || stats.AllocatedBlocks != 6 || stats.FreeBlocks != 0 {
		t.Fatalf("expected 6 current total (6 allocated, 0 free), got %+v", stats)
	}

	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("invariant violated after manual drain completion: %v", err)
	}
}

func TestElasticBlockAllocatorProgressiveDrainWithPrefixEvictorAndVictim(t *testing.T) {
	var prefixEvictions int
	var victimPreemptions int

	prefixCallback := func(neededBlocks int) (int, error) {
		prefixEvictions++
		// pretend we freed 1 block
		return 1, nil
	}

	victimCallback := func(active []ActiveSequenceInfo, needed int) (string, error) {
		victimPreemptions++
		if len(active) == 0 {
			return "", nil
		}
		// Pick sequence with lowest priority
		lowestPri := active[0].Priority
		victimID := active[0].SequenceID
		for _, a := range active {
			if a.Priority < lowestPri {
				lowestPri = a.Priority
				victimID = a.SequenceID
			}
		}
		return victimID, nil
	}

	alloc, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks:   8,
		ModelCfg:        testModelConfig(),
		BlockTokens:     16,
		PrefixReclaimer: prefixCallback,
		VictimSelector:  victimCallback,
	})
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	_ = alloc.RegisterSequence("low-pri", 1)
	_ = alloc.RegisterSequence("high-pri", 10)

	for i := 0; i < 4; i++ {
		_, _ = alloc.AllocBlock("low-pri")
		_, _ = alloc.AllocBlock("high-pri")
	}

	// Now 8 blocks allocated, 0 free. Request drain to 4.
	if err := alloc.RequestDrain(4); err != nil {
		t.Fatalf("request drain failed: %v", err)
	}
	if alloc.State() != StateDraining {
		t.Fatalf("expected StateDraining, got %v", alloc.State())
	}

	// Call ProgressDrain()
	done, err := alloc.ProgressDrain()
	if err != nil {
		t.Fatalf("ProgressDrain failed: %v", err)
	}
	if !done {
		t.Fatalf("expected ProgressDrain to complete, but returned false")
	}

	if alloc.State() != StateStable {
		t.Fatalf("expected StateStable, got %v", alloc.State())
	}

	stats := alloc.Stats()
	if stats.CurrentTotalBlocks != 4 {
		t.Fatalf("expected 4 total blocks, got %d", stats.CurrentTotalBlocks)
	}
	if stats.PreemptedSequences != 1 {
		t.Fatalf("expected 1 preempted sequence, got %d", stats.PreemptedSequences)
	}
	if prefixEvictions == 0 {
		t.Fatal("expected prefix callback to be called")
	}
	if victimPreemptions == 0 {
		t.Fatal("expected victim callback to be called")
	}

	// Verify low-pri was evicted and high-pri survives
	if _, err := alloc.SequenceAllocatedBlocks("low-pri"); err != ErrSequenceNotFound {
		t.Fatalf("expected low-pri to be removed, got err: %v", err)
	}
	if blks, err := alloc.SequenceAllocatedBlocks("high-pri"); err != nil || blks != 4 {
		t.Fatalf("expected high-pri to have 4 blocks, got %d (err: %v)", blks, err)
	}

	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("invariant violated: %v", err)
	}
}

func TestElasticBlockAllocatorConcurrentExpandAndDrain(t *testing.T) {
	alloc, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks: 50,
		ModelCfg:      testModelConfig(),
		BlockTokens:   16,
	})
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	var wg sync.WaitGroup
	var stop int32

	// Register worker sequences
	const numSeqs = 5
	for i := 0; i < numSeqs; i++ {
		_ = alloc.RegisterSequence(fmt.Sprintf("worker-%d", i), i)
	}

	// Worker goroutines allocating and freeing blocks
	for i := 0; i < numSeqs; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			seqID := fmt.Sprintf("worker-%d", id)
			var held []int
			for atomic.LoadInt32(&stop) == 0 {
				blk, err := alloc.AllocBlock(seqID)
				if err == nil {
					held = append(held, blk)
				}
				if len(held) > 3 {
					toFree := held[0]
					held = held[1:]
					_ = alloc.FreeBlock(seqID, toFree)
				}
				_ = alloc.TouchSequence(seqID)
				time.Sleep(100 * time.Microsecond)
			}
			// Clean up remaining held
			for _, b := range held {
				_ = alloc.FreeBlock(seqID, b)
			}
		}(i)
	}

	// Controller goroutine cycling expand and drain
	wg.Add(1)
	go func() {
		defer wg.Done()
		for cycle := 0; cycle < 10; cycle++ {
			_ = alloc.Expand(70)
			_ = alloc.VerifyInvariant()
			time.Sleep(500 * time.Microsecond)

			_ = alloc.RequestDrain(30)
			_, _ = alloc.ProgressDrain()
			_ = alloc.VerifyInvariant()
			time.Sleep(500 * time.Microsecond)
		}
		atomic.StoreInt32(&stop, 1)
	}()

	wg.Wait()

	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("final invariant check failed: %v", err)
	}
}

func TestElasticBlockAllocator_NonCachedLIFO(t *testing.T) {
	alloc, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks: 6,
		ModelCfg:      testModelConfig(),
		BlockTokens:   16,
	})
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := alloc.RegisterSequence("seq", 1); err != nil {
		t.Fatalf("register sequence: %v", err)
	}

	blocks := make([]int, 3)
	for i := range blocks {
		blocks[i], err = alloc.AllocBlock("seq")
		if err != nil {
			t.Fatalf("allocate block %d: %v", i, err)
		}
	}
	if err := alloc.FreeBlock("seq", blocks[0]); err != nil {
		t.Fatalf("free oldest released block: %v", err)
	}
	if err := alloc.FreeBlock("seq", blocks[2]); err != nil {
		t.Fatalf("free newest released block: %v", err)
	}

	for i, want := range []int{blocks[2], blocks[0]} {
		got, err := alloc.AllocBlock("seq")
		if err != nil {
			t.Fatalf("reallocate released block %d: %v", i, err)
		}
		if got != want {
			t.Fatalf("released block %d: got %d, want LIFO block %d", i, got, want)
		}
	}
	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("invariant after LIFO reuse: %v", err)
	}
}

func TestElasticBlockAllocator_FreshDeterministic(t *testing.T) {
	alloc, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks: 3,
		ModelCfg:      testModelConfig(),
		BlockTokens:   16,
	})
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}
	if err := alloc.RegisterSequence("seq", 1); err != nil {
		t.Fatalf("register sequence: %v", err)
	}
	if err := alloc.Expand(6); err != nil {
		t.Fatalf("expand: %v", err)
	}

	for want := 0; want < 6; want++ {
		got, err := alloc.AllocBlock("seq")
		if err != nil {
			t.Fatalf("allocate fresh block %d: %v", want, err)
		}
		if got != want {
			t.Fatalf("fresh allocation %d: got %d, want lowest free ID %d", want, got, want)
		}
	}
	if err := alloc.VerifyInvariant(); err != nil {
		t.Fatalf("invariant after deterministic fresh allocation: %v", err)
	}
}

func TestElasticBlockAllocator_LIFOInvariantTransitions(t *testing.T) {
	alloc, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
		InitialBlocks: 8,
		ModelCfg:      testModelConfig(),
		BlockTokens:   16,
		VictimSelector: func(_ []ActiveSequenceInfo, _ int) (string, error) {
			return "victim", nil
		},
	})
	if err != nil {
		t.Fatalf("init failed: %v", err)
	}

	peakTotal := 0
	check := func(stage string) {
		t.Helper()
		if err := alloc.VerifyInvariant(); err != nil {
			t.Fatalf("%s: invariant failed: %v", stage, err)
		}
		stats := alloc.Stats()
		if stats.AllocatedBlocks+stats.FreeBlocks != stats.CurrentTotalBlocks {
			t.Fatalf("%s: allocated %d + free %d != current total %d",
				stage, stats.AllocatedBlocks, stats.FreeBlocks, stats.CurrentTotalBlocks)
		}
		if stats.CurrentTotalBlocks > peakTotal {
			peakTotal = stats.CurrentTotalBlocks
		}
		if got := len(alloc.recycledBlocks); got > stats.CurrentTotalBlocks {
			t.Fatalf("%s: recycled metadata length %d exceeds current total %d",
				stage, got, stats.CurrentTotalBlocks)
		}
		if got := cap(alloc.recycledBlocks); got > peakTotal {
			t.Fatalf("%s: recycled metadata capacity %d exceeds peak total %d",
				stage, got, peakTotal)
		}
	}

	if err := alloc.RegisterSequence("main", 10); err != nil {
		t.Fatalf("register main: %v", err)
	}
	mainBlocks := make([]int, 8)
	for i := range mainBlocks {
		mainBlocks[i], err = alloc.AllocBlock("main")
		if err != nil {
			t.Fatalf("allocate main block %d: %v", i, err)
		}
	}
	check("initial allocation")

	if err := alloc.FreeBlock("main", mainBlocks[6]); err != nil {
		t.Fatalf("free block %d before drain: %v", mainBlocks[6], err)
	}
	if err := alloc.FreeBlock("main", mainBlocks[7]); err != nil {
		t.Fatalf("free block %d before drain: %v", mainBlocks[7], err)
	}
	check("released before drain")
	if err := alloc.RequestDrain(6); err != nil {
		t.Fatalf("drain released blocks: %v", err)
	}
	check("drain retired released blocks")
	if got, err := alloc.AllocBlock("main"); err != ErrAllocCapacityExceeded || got != -1 {
		t.Fatalf("allocate drain-retired block: got (%d, %v), want (-1, %v)",
			got, err, ErrAllocCapacityExceeded)
	}
	if err := alloc.Expand(8); err != nil {
		t.Fatalf("expand after retired releases: %v", err)
	}
	check("expand after retired releases")
	for i, want := range mainBlocks[6:] {
		got, err := alloc.AllocBlock("main")
		if err != nil {
			t.Fatalf("allocate fresh expanded block %d: %v", i, err)
		}
		if got != want {
			t.Fatalf("fresh expanded block %d: got %d, want lowest ID %d; stale recycle order survived drain",
				i, got, want)
		}
	}
	check("fresh allocation after retired releases")

	if err := alloc.FreeBlock("main", mainBlocks[2]); err != nil {
		t.Fatalf("free block for unregister sequence: %v", err)
	}
	if err := alloc.FreeBlock("main", mainBlocks[3]); err != nil {
		t.Fatalf("free second block for unregister sequence: %v", err)
	}
	if err := alloc.RegisterSequence("unregister", 1); err != nil {
		t.Fatalf("register unregister sequence: %v", err)
	}
	for i := 0; i < 2; i++ {
		if _, err := alloc.AllocBlock("unregister"); err != nil {
			t.Fatalf("allocate unregister block %d: %v", i, err)
		}
	}
	released, err := alloc.UnregisterSequence("unregister")
	if err != nil {
		t.Fatalf("unregister sequence: %v", err)
	}
	if len(released) != 2 {
		t.Fatalf("unregister released %d blocks, want 2", len(released))
	}
	for i := len(released) - 1; i >= 0; i-- {
		got, err := alloc.AllocBlock("main")
		if err != nil {
			t.Fatalf("reuse unregister block %d: %v", i, err)
		}
		if got != released[i] {
			t.Fatalf("reuse unregister block %d: got %d, want reverse release order %d",
				i, got, released[i])
		}
	}
	check("unregister LIFO reuse")

	if err := alloc.FreeBlock("main", mainBlocks[4]); err != nil {
		t.Fatalf("free first victim block: %v", err)
	}
	if err := alloc.FreeBlock("main", mainBlocks[5]); err != nil {
		t.Fatalf("free second victim block: %v", err)
	}
	if err := alloc.RegisterSequence("victim", 0); err != nil {
		t.Fatalf("register victim: %v", err)
	}
	for i := 0; i < 2; i++ {
		if _, err := alloc.AllocBlock("victim"); err != nil {
			t.Fatalf("allocate victim block %d: %v", i, err)
		}
	}
	if err := alloc.RequestDrain(6); err != nil {
		t.Fatalf("request preemption drain: %v", err)
	}
	done, err := alloc.ProgressDrain()
	if err != nil {
		t.Fatalf("progress preemption drain: %v", err)
	}
	if !done {
		t.Fatal("preemption drain did not reach target")
	}
	check("preemption retired victim blocks")
	if stats := alloc.Stats(); stats.PreemptedSequences != 1 {
		t.Fatalf("preempted sequences: got %d, want 1", stats.PreemptedSequences)
	}
	if got, err := alloc.AllocBlock("main"); err != ErrAllocCapacityExceeded || got != -1 {
		t.Fatalf("allocate preemption-retired block: got (%d, %v), want (-1, %v)",
			got, err, ErrAllocCapacityExceeded)
	}

	if err := alloc.Expand(8); err != nil {
		t.Fatalf("expand after preemption: %v", err)
	}
	var churnBlock int
	for i, want := range mainBlocks[4:6] {
		got, err := alloc.AllocBlock("main")
		if err != nil {
			t.Fatalf("allocate after preemption %d: %v", i, err)
		}
		if got != want {
			t.Fatalf("fresh allocation after preemption %d: got %d, want %d", i, got, want)
		}
		if i == 0 {
			churnBlock = got
		}
	}
	check("expand after preemption")

	const churnCycles = 32
	for cycle := 0; cycle < churnCycles; cycle++ {
		if err := alloc.FreeBlock("main", churnBlock); err != nil {
			t.Fatalf("churn %d free block %d: %v", cycle, churnBlock, err)
		}
		check(fmt.Sprintf("churn %d release", cycle))
		if err := alloc.RequestDrain(7); err != nil {
			t.Fatalf("churn %d drain: %v", cycle, err)
		}
		check(fmt.Sprintf("churn %d drain", cycle))
		if err := alloc.Expand(8); err != nil {
			t.Fatalf("churn %d expand: %v", cycle, err)
		}
		check(fmt.Sprintf("churn %d expand", cycle))
		got, err := alloc.AllocBlock("main")
		if err != nil {
			t.Fatalf("churn %d allocate: %v", cycle, err)
		}
		if got != churnBlock {
			t.Fatalf("churn %d allocation: got %d, want only fresh ID %d", cycle, got, churnBlock)
		}
		churnBlock = got
		check(fmt.Sprintf("churn %d allocation", cycle))
	}

	t.Run("preemption surplus", func(t *testing.T) {
		surplus, err := NewElasticBlockAllocator(ElasticBlockAllocatorConfig{
			InitialBlocks: 6,
			ModelCfg:      testModelConfig(),
			BlockTokens:   16,
			VictimSelector: func(_ []ActiveSequenceInfo, _ int) (string, error) {
				return "victim", nil
			},
		})
		if err != nil {
			t.Fatalf("init surplus allocator: %v", err)
		}

		peakTotal := 0
		checkSurplus := func(stage string) {
			t.Helper()
			if err := surplus.VerifyInvariant(); err != nil {
				t.Fatalf("%s: invariant failed: %v", stage, err)
			}
			stats := surplus.Stats()
			if stats.AllocatedBlocks+stats.FreeBlocks != stats.CurrentTotalBlocks {
				t.Fatalf("%s: allocated %d + free %d != current total %d",
					stage, stats.AllocatedBlocks, stats.FreeBlocks, stats.CurrentTotalBlocks)
			}
			if stats.CurrentTotalBlocks > peakTotal {
				peakTotal = stats.CurrentTotalBlocks
			}
			if got := len(surplus.recycledBlocks); got > stats.CurrentTotalBlocks {
				t.Fatalf("%s: recycled metadata length %d exceeds current total %d",
					stage, got, stats.CurrentTotalBlocks)
			}
			if got := cap(surplus.recycledBlocks); got > peakTotal {
				t.Fatalf("%s: recycled metadata capacity %d exceeds peak total %d",
					stage, got, peakTotal)
			}
		}

		if err := surplus.RegisterSequence("main", 10); err != nil {
			t.Fatalf("register surplus main: %v", err)
		}
		if err := surplus.RegisterSequence("victim", 0); err != nil {
			t.Fatalf("register surplus victim: %v", err)
		}
		mainBlocks := make([]int, 3)
		victimBlocks := make(map[int]bool, 3)
		for i := range mainBlocks {
			mainBlocks[i], err = surplus.AllocBlock("main")
			if err != nil {
				t.Fatalf("allocate surplus main block %d: %v", i, err)
			}
		}
		for i := 0; i < 3; i++ {
			block, err := surplus.AllocBlock("victim")
			if err != nil {
				t.Fatalf("allocate surplus victim block %d: %v", i, err)
			}
			victimBlocks[block] = true
		}
		checkSurplus("surplus initial allocation")

		if err := surplus.RequestDrain(5); err != nil {
			t.Fatalf("request surplus drain: %v", err)
		}
		checkSurplus("surplus drain requested")
		done, err := surplus.ProgressDrain()
		if err != nil {
			t.Fatalf("progress surplus drain: %v", err)
		}
		if !done {
			t.Fatal("surplus preemption drain did not reach target")
		}
		checkSurplus("surplus preemption completed")
		if stats := surplus.Stats(); stats.CurrentTotalBlocks != 5 ||
			stats.AllocatedBlocks != 3 || stats.FreeBlocks != 2 ||
			stats.PreemptedSequences != 1 {
			t.Fatalf("unexpected surplus preemption stats: %+v", stats)
		}

		if err := surplus.FreeBlock("main", mainBlocks[0]); err != nil {
			t.Fatalf("release known main block after surplus preemption: %v", err)
		}
		checkSurplus("known main release after surplus preemption")
		got, err := surplus.AllocBlock("main")
		if err != nil {
			t.Fatalf("reuse known main block after surplus preemption: %v", err)
		}
		if got != mainBlocks[0] {
			t.Fatalf("newest known release: got %d, want %d before preemption surplus",
				got, mainBlocks[0])
		}
		checkSurplus("known main reuse before preemption surplus")

		seen := make(map[int]bool, 2)
		for i := 0; i < 2; i++ {
			got, err := surplus.AllocBlock("main")
			if err != nil {
				t.Fatalf("allocate preemption surplus block %d: %v", i, err)
			}
			if !victimBlocks[got] {
				t.Fatalf("preemption surplus block %d: got %d outside victim block set", i, got)
			}
			if seen[got] {
				t.Fatalf("preemption surplus block %d allocated twice", got)
			}
			seen[got] = true
			checkSurplus(fmt.Sprintf("preemption surplus allocation %d", i))
		}
	})
}
