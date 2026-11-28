package grammar

import (
	"errors"
	"math"
	"slices"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestSchemaCompileCacheHitMissAndEviction(t *testing.T) {
	dec := jsonSchemaTokenDecoder{
		[]byte(`{"a":`),
		[]byte(`"x"`),
		[]byte(`}`),
		nil, // EOS
	}
	schemaA := []byte(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)
	schemaB := []byte(`{"type":"object","properties":{"b":{"type":"integer"}},"required":["b"],"additionalProperties":false}`)
	schemaC := []byte(`{"type":"object","properties":{"c":{"type":"boolean"}},"required":["c"],"additionalProperties":false}`)

	cache := NewSchemaCompileCache(2)
	m1, hit, err := cache.GetOrCompile(schemaA, dec, CompileOptions{EOS: 3})
	if err != nil || hit || m1 == nil {
		t.Fatalf("first compile = (%v, hit=%v, err=%v); want miss with non-nil mask", m1, hit, err)
	}

	m2, hit, err := cache.GetOrCompile(schemaA, dec, CompileOptions{EOS: 3})
	if err != nil || !hit || m2 != m1 {
		t.Fatalf("second compile = (%p, hit=%v, err=%v); want cache hit returning %p", m2, hit, err, m1)
	}

	// Different EOS must miss.
	m3, hit, err := cache.GetOrCompile(schemaA, dec, CompileOptions{EOS: 2})
	if err != nil || hit || m3 == m1 {
		t.Fatalf("different EOS compile = (%p, hit=%v, err=%v); want miss", m3, hit, err)
	}

	// Adding a third distinct entry evicts the oldest (schemaA with EOS=3).
	if _, hit, err := cache.GetOrCompile(schemaB, dec, CompileOptions{EOS: 3}); err != nil || hit {
		t.Fatalf("schemaB compile hit=%v err=%v; want miss", hit, err)
	}
	if cache.Len() != 2 || cache.Evictions() != 1 {
		t.Fatalf("cache len=%d evictions=%d; want len=2 evictions=1", cache.Len(), cache.Evictions())
	}

	m4, hit, err := cache.GetOrCompile(schemaA, dec, CompileOptions{EOS: 3})
	if err != nil || hit || m4 == m1 {
		t.Fatalf("evicted schemaA recompile hit=%v same=%v err=%v; want miss", hit, m4 == m1, err)
	}

	// Explicit vocabKey path also hits deterministically.
	if _, hit, err := cache.GetOrCompileWithKey("qwen3.8-7b", schemaC, dec, CompileOptions{EOS: 3}); err != nil || hit {
		t.Fatalf("explicit key first compile hit=%v err=%v; want miss", hit, err)
	}
	if _, hit, err := cache.GetOrCompileWithKey("qwen3.8-7b", schemaC, dec, CompileOptions{EOS: 3}); err != nil || !hit {
		t.Fatalf("explicit key second compile hit=%v err=%v; want hit", hit, err)
	}

	if cache.Hits() != 2 || cache.Misses() != 5 {
		t.Fatalf("hits=%d misses=%d; want hits=2 misses=5", cache.Hits(), cache.Misses())
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBatchBitmaskFSMMatchesSingleSequenceMaskLogits(t *testing.T) {
	const eos = 0
	dec := jsonSchemaTokenDecoder{
		nil,            // 0: EOS
		[]byte(`{`),    // 1
		[]byte(`"a":`), // 2
		[]byte(`"x"`),  // 3
		[]byte(`}`),    // 4
		[]byte(`"b":`), // 5
		[]byte(`42`),   // 6
		[]byte(`bad`),  // 7
	}
	schemaA := []byte(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)
	schemaB := []byte(`{"type":"object","properties":{"b":{"type":"integer"}},"required":["b"],"additionalProperties":false}`)

	maskA, err := CompileJSONSchema(schemaA, dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema(A): %v", err)
	}
	maskB, err := CompileJSONSchema(schemaB, dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema(B): %v", err)
	}

	// Row 0: schemaA, Row 1: schemaB, Row 2: unconstrained (nil).
	fsm, err := NewBatchBitmaskFSM([]*JSONSchemaMask{maskA, maskB, nil}, len(dec))
	if err != nil {
		t.Fatalf("NewBatchBitmaskFSM: %v", err)
	}

	hist0 := []int{}
	hist1 := []int{}
	assertMatchesSingle := func() {
		t.Helper()
		flatLogits := make([]float32, 3*len(dec))
		if err := fsm.ApplyFlatLogits(flatLogits); err != nil {
			t.Fatalf("ApplyFlatLogits: %v", err)
		}
		ref0 := make([]float32, len(dec))
		maskA.MaskLogits(hist0, ref0)
		ref1 := make([]float32, len(dec))
		maskB.MaskLogits(hist1, ref1)

		for id := range dec {
			if math.IsInf(float64(flatLogits[id]), -1) != math.IsInf(float64(ref0[id]), -1) {
				t.Fatalf("row 0 token %d mask mismatch: batch=%v single=%v", id, flatLogits[id], ref0[id])
			}
			if math.IsInf(float64(flatLogits[len(dec)+id]), -1) != math.IsInf(float64(ref1[id]), -1) {
				t.Fatalf("row 1 token %d mask mismatch: batch=%v single=%v", id, flatLogits[len(dec)+id], ref1[id])
			}
			if math.IsInf(float64(flatLogits[2*len(dec)+id]), -1) {
				t.Fatalf("row 2 unconstrained token %d unexpectedly masked", id)
			}
		}
	}

	assertMatchesSingle()

	// Step all 3 rows in lockstep: row 0 -> '{', row 1 -> '{', row 2 -> 'bad'.
	if err := fsm.StepBatch([]int{1, 1, 7}); err != nil {
		t.Fatalf("StepBatch 1: %v", err)
	}
	hist0 = append(hist0, 1)
	hist1 = append(hist1, 1)
	assertMatchesSingle()

	// Advance row 0 to completion and EOS while row 1 stays live.
	for _, tok := range []int{2, 3, 4} {
		if err := fsm.StepRow(0, tok); err != nil {
			t.Fatalf("StepRow(0, %d): %v", tok, err)
		}
		hist0 = append(hist0, tok)
	}
	assertMatchesSingle()

	if err := fsm.StepRow(0, eos); err != nil {
		t.Fatalf("StepRow(0, eos): %v", err)
	}
	fin0, _ := fsm.IsFinished(0)
	fin1, _ := fsm.IsFinished(1)
	if !fin0 || fin1 {
		t.Fatalf("finished = (%v, %v); want (true, false)", fin0, fin1)
	}

	// Finished row 0 bitmask must allow ONLY EOS; row 1 remains live on '"b":'.
	batchBits := make([]uint64, fsm.BatchSize()*fsm.BitmaskWords())
	if err := fsm.FillBatchBitmask(batchBits); err != nil {
		t.Fatalf("FillBatchBitmask: %v", err)
	}
	if batchBits[0] != (1 << eos) {
		t.Fatalf("finished row 0 bitmask = %b; want only EOS bit %d", batchBits[0], eos)
	}
	if batchBits[1]&(1<<5) == 0 {
		t.Fatalf("live row 1 bitmask = %b; want token 5 allowed", batchBits[1])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBatchBitmaskFSMSpeculativeRollbackBatchEquivalence(t *testing.T) {
	const eos = 0
	dec := jsonSchemaTokenDecoder{
		nil,            // 0: EOS
		[]byte(`{`),    // 1
		[]byte(`"a":`), // 2
		[]byte(`"x"`),  // 3
		[]byte(`}`),    // 4
		[]byte(`"y"`),  // 5
		[]byte(`"b":`), // 6
		[]byte(`true`), // 7
	}
	schemaA := []byte(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)
	schemaB := []byte(`{"type":"object","properties":{"b":{"type":"boolean"}},"required":["b"],"additionalProperties":false}`)

	maskA, err := CompileJSONSchema(schemaA, dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema(A): %v", err)
	}
	maskB, err := CompileJSONSchema(schemaB, dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema(B): %v", err)
	}

	specFSM, err := NewBatchBitmaskFSM([]*JSONSchemaMask{maskA, maskB}, len(dec))
	if err != nil {
		t.Fatalf("NewBatchBitmaskFSM(spec): %v", err)
	}
	refFSM, err := NewBatchBitmaskFSM([]*JSONSchemaMask{maskA, maskB}, len(dec))
	if err != nil {
		t.Fatalf("NewBatchBitmaskFSM(ref): %v", err)
	}

	type hookEvent struct {
		row, fromStep, toStep int
	}
	var events []hookEvent
	specFSM.SetRollbackHook(func(row, fromStep, toStep int) {
		events = append(events, hookEvent{row: row, fromStep: fromStep, toStep: toStep})
	})

	// Both batches advance row 0 -> '{' and row 1 -> '{'.
	if err := specFSM.StepBatch([]int{1, 1}); err != nil {
		t.Fatalf("spec StepBatch: %v", err)
	}
	if err := refFSM.StepBatch([]int{1, 1}); err != nil {
		t.Fatalf("ref StepBatch: %v", err)
	}

	// Row 1 advances independently to '"b":'.
	if err := specFSM.StepRow(1, 6); err != nil {
		t.Fatalf("spec StepRow(1, 6): %v", err)
	}
	if err := refFSM.StepRow(1, 6); err != nil {
		t.Fatalf("ref StepRow(1, 6): %v", err)
	}

	// Speculative draft on row 0: draft 3 tokens [2 ('"a":'), 3 ('"x"'), 4 ('}')],
	// then accept only the first 1 token (2) and roll back the rejected suffix [3, 4].
	if err := specFSM.StepRow(0, 2); err != nil {
		t.Fatalf("spec draft 2: %v", err)
	}
	cpAfterAccepted, err := specFSM.CheckpointRow(0)
	if err != nil {
		t.Fatalf("CheckpointRow: %v", err)
	}
	if err := specFSM.StepRow(0, 3); err != nil {
		t.Fatalf("spec draft 3: %v", err)
	}
	if err := specFSM.StepRow(0, 4); err != nil {
		t.Fatalf("spec draft 4: %v", err)
	}
	if err := specFSM.StepRow(0, eos); err != nil {
		t.Fatalf("spec draft eos: %v", err)
	}
	if err := specFSM.RollbackRow(0, cpAfterAccepted); err != nil {
		t.Fatalf("RollbackRow: %v", err)
	}

	// Reference batch only ever stepped the accepted token 2 on row 0.
	if err := refFSM.StepRow(0, 2); err != nil {
		t.Fatalf("ref StepRow(0, 2): %v", err)
	}

	if len(events) != 1 || events[0] != (hookEvent{row: 0, fromStep: 5, toStep: 2}) {
		t.Fatalf("rollback hook events = %+v; want [{row:0 fromStep:5 toStep:2}]", events)
	}

	assertBatchBitmasksEqual := func(context string) {
		t.Helper()
		specBits := make([]uint64, specFSM.BatchSize()*specFSM.BitmaskWords())
		refBits := make([]uint64, refFSM.BatchSize()*refFSM.BitmaskWords())
		if err := specFSM.FillBatchBitmask(specBits); err != nil {
			t.Fatalf("%s spec FillBatchBitmask: %v", context, err)
		}
		if err := refFSM.FillBatchBitmask(refBits); err != nil {
			t.Fatalf("%s ref FillBatchBitmask: %v", context, err)
		}
		if !slices.Equal(specBits, refBits) {
			t.Fatalf("%s bitmask mismatch: spec=%v ref=%v", context, specBits, refBits)
		}
	}

	assertBatchBitmasksEqual("post-rollback")

	// Continue decoding both batches with a different token ('"y"') on row 0 and 'true' on row 1.
	if err := specFSM.StepBatch([]int{5, 7}); err != nil {
		t.Fatalf("spec StepBatch post-rollback: %v", err)
	}
	if err := refFSM.StepBatch([]int{5, 7}); err != nil {
		t.Fatalf("ref StepBatch post-rollback: %v", err)
	}
	assertBatchBitmasksEqual("post-continue")

	// Full-batch snapshot and rollback preserves equivalence too.
	snap := specFSM.Snapshot()
	if err := specFSM.StepBatch([]int{4, 4}); err != nil {
		t.Fatalf("spec StepBatch before full rollback: %v", err)
	}
	if err := specFSM.Rollback(snap); err != nil {
		t.Fatalf("Rollback(snap): %v", err)
	}
	assertBatchBitmasksEqual("post-full-batch-rollback")
}

// fak-test:runtime fast est=10ms lane=default
func TestBatchBitmaskFSMRefusalsAreTransactional(t *testing.T) {
	const eos = 0
	dec := jsonSchemaTokenDecoder{
		nil,            // 0: EOS
		[]byte(`{`),    // 1
		[]byte(`"a":`), // 2
		[]byte(`"x"`),  // 3
		[]byte(`}`),    // 4
		[]byte(`bad`),  // 5
	}
	schema := []byte(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`)
	mask, err := CompileJSONSchema(schema, dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema: %v", err)
	}
	fsm, err := NewUniformBatchBitmaskFSM(mask, 2, len(dec))
	if err != nil {
		t.Fatalf("NewUniformBatchBitmaskFSM: %v", err)
	}

	if err := fsm.StepRow(0, 1); err != nil {
		t.Fatalf("StepRow(0, 1): %v", err)
	}
	// Row 0 is at step 1, row 1 is at step 0. Batch step where row 1 fails must leave row 0 at step 1.
	if err := fsm.StepBatch([]int{2, 5}); !errors.Is(err, ErrBatchFSMRefusal) {
		t.Fatalf("StepBatch invalid = %v; want ErrBatchFSMRefusal", err)
	}
	step0, _ := fsm.RowStep(0)
	step1, _ := fsm.RowStep(1)
	if step0 != 1 || step1 != 0 {
		t.Fatalf("steps after failed StepBatch = (%d, %d); want (1, 0)", step0, step1)
	}

	// EOS before complete must be refused.
	if err := fsm.StepRow(0, eos); !errors.Is(err, ErrBatchFSMRefusal) {
		t.Fatalf("premature EOS = %v; want ErrBatchFSMRefusal", err)
	}

	// Out-of-range row / token and wrong-row checkpoint must be refused.
	if err := fsm.StepRow(2, 1); !errors.Is(err, ErrBatchFSMRefusal) {
		t.Fatalf("out-of-range row = %v; want ErrBatchFSMRefusal", err)
	}
	if err := fsm.StepRow(0, len(dec)); !errors.Is(err, ErrBatchFSMRefusal) {
		t.Fatalf("out-of-range token = %v; want ErrBatchFSMRefusal", err)
	}
	cp0, err := fsm.CheckpointRow(0)
	if err != nil {
		t.Fatalf("CheckpointRow(0): %v", err)
	}
	if err := fsm.RollbackRow(1, cp0); !errors.Is(err, ErrBatchFSMRefusal) {
		t.Fatalf("cross-row rollback = %v; want ErrBatchFSMRefusal", err)
	}
}
