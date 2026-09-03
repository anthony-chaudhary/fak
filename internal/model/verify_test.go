package model

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// cfgV is a small plain-PreNorm GQA synthetic config (the regime VerifyForward batches),
// matching the shape internal/spec and cmd/polymodelbench use.
func cfgV(hidden, layers, nHeads, nKV, headDim, inter int) Config {
	return Config{
		HiddenSize:        hidden,
		NumLayers:         layers,
		NumHeads:          nHeads,
		NumKVHeads:        nKV,
		HeadDim:           headDim,
		IntermediateSize:  inter,
		VocabSize:         256,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

func argmaxV(v []float32) int {
	bi, bv := 0, float32(0)
	for i, x := range v {
		if i == 0 || x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

func qwen38HybridMTPEnabledSyntheticModel(t *testing.T) *Model {
	t.Helper()
	cfg := qwen35HybridTestCfg()
	cfg.Name = "Qwen3.8 hybrid MTP synthetic"
	cfg.ModelType = "qwen3_5_text"
	cfg.MTPNumHiddenLayers = 1
	m := NewSynthetic(cfg)
	shapes, err := qwen35MTPExpectedShapes(cfg)
	if err != nil {
		t.Fatalf("Qwen3.8 hybrid MTP shapes: %v", err)
	}
	for tensorIndex, name := range qwen35MTPRequiredTensors {
		shape := shapes[name]
		elements := 1
		for _, dim := range shape {
			elements *= dim
		}
		start := len(m.raw)
		for i := 0; i < elements; i++ {
			value := float32(tensorIndex+1)/100 + float32(i)/100000
			var bits [4]byte
			binary.LittleEndian.PutUint32(bits[:], math.Float32bits(value))
			m.raw = append(m.raw, bits[:]...)
		}
		m.manifest[name] = tensorMeta{
			Dtype:  "F32",
			Shape:  append([]int(nil), shape...),
			Offset: start,
			Nbytes: elements * 4,
		}
	}
	mode, err := m.Qwen35MTPMode(false)
	if err != nil {
		t.Fatalf("Qwen3.8 hybrid MTP mode: %v", err)
	}
	if !mode.Enabled || mode.Engine != targetVerificationEngine {
		t.Fatalf("Qwen3.8 hybrid MTP mode = %+v, want enabled fak-native", mode)
	}
	return m
}

// TestVerifyForwardChainMatchesSerial proves the single-pass batched verify (the chain
// case: nil pos, nil allow) is BIT-IDENTICAL to P sequential Session.Step calls — same
// per-position logits AND the same full KV-cache state (K/Kraw/V/pos in every layer). This
// is the losslessness contract that lets internal/spec.SpeculativeGreedy swap its sequential
// verify loop for one VerifyForward call: a batched verify that built a byte-different cache
// would silently corrupt the promote/squash offsets, so byte-equality is the gate.
func TestVerifyForwardChainMatchesSerial(t *testing.T) {
	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	drafts := []int{13, 17, 4, 99, 200, 5, 42} // arbitrary, incl. repeats

	ref := m.NewSession()
	ref.Prefill(prompt)
	refLogits := make([][]float32, len(drafts))
	for j, d := range drafts {
		refLogits[j] = ref.Step(d)
	}

	ver := m.NewSession()
	ver.Prefill(prompt)
	got := ver.VerifyForward(drafts, nil, nil)

	if len(got) != len(refLogits) {
		t.Fatalf("VerifyForward returned %d logit vecs, want %d", len(got), len(refLogits))
	}
	for j := range refLogits {
		a, b := refLogits[j], got[j]
		if len(a) != len(b) {
			t.Fatalf("pos %d logit width %d != %d", j, len(a), len(b))
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				t.Fatalf("pos %d logit[%d]: serial %v != verify %v (NOT bit-identical)", j, i, a[i], b[i])
			}
		}
	}

	if ver.Cache.Len() != ref.Cache.Len() {
		t.Fatalf("cache len: verify %d != serial %d", ver.Cache.Len(), ref.Cache.Len())
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for name, pair := range map[string][2][]float32{
			"K":    {ref.Cache.K[l], ver.Cache.K[l]},
			"Kraw": {ref.Cache.Kraw[l], ver.Cache.Kraw[l]},
			"V":    {ref.Cache.V[l], ver.Cache.V[l]},
		} {
			a, b := pair[0], pair[1]
			if len(a) != len(b) {
				t.Fatalf("layer %d %s len %d != %d", l, name, len(a), len(b))
			}
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
					t.Fatalf("layer %d %s[%d]: serial %v != verify %v (cache not bit-identical)", l, name, i, a[i], b[i])
				}
			}
		}
	}
	for i := range ref.Cache.pos {
		if ref.Cache.pos[i] != ver.Cache.pos[i] {
			t.Fatalf("pos[%d]: serial %d != verify %d", i, ref.Cache.pos[i], ver.Cache.pos[i])
		}
	}
}

// TestVerifyForwardEmpty is the trivial edge: no candidates ⇒ nil, no cache mutation.
func TestVerifyForwardEmpty(t *testing.T) {
	m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
	s := m.NewSession()
	s.Prefill([]int{1, 2, 3})
	before := s.Cache.Len()
	if got := s.VerifyForward(nil, nil, nil); got != nil {
		t.Fatalf("empty VerifyForward = %v, want nil", got)
	}
	if s.Cache.Len() != before {
		t.Fatalf("empty VerifyForward mutated cache: %d != %d", s.Cache.Len(), before)
	}
}

func TestQwen38VerifyForwardOneOperationMatchesNativeDecode(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 3
	if !q8PrefillNeedsTokenLoop(m.Cfg) {
		t.Fatal("Qwen3.8 hybrid precondition did not select the legacy token-loop verifier")
	}

	reference := m.NewSession()
	reference.captureTargetHidden = true
	t.Cleanup(reference.Close)
	logits := reference.Prefill(prompt)
	draft := make([]int, depth)
	wantRows := make([][]float32, depth)
	for i := range draft {
		draft[i] = argmaxF32(logits)
		logits = reference.Step(draft[i])
		wantRows[i] = append([]float32(nil), logits...)
	}

	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill(prompt)
	wantPrefix := m.NewSession()
	wantPrefix.captureTargetHidden = true
	t.Cleanup(wantPrefix.Close)
	wantPrefix.Prefill(prompt)

	operations := 0
	rows, receipt, err := target.verifyForwardOneOperation(draft, boundary, func(m *Model, tokens []int) *Activations {
		operations++
		return m.Forward(tokens)
	})
	if err != nil {
		t.Fatalf("Qwen3.8 one-operation verification: %v", err)
	}
	if operations != 1 {
		t.Fatalf("Qwen3.8 target operations = %d, want exactly 1", operations)
	}
	if receipt.Engine != targetVerificationEngine || receipt.Path != targetVerificationQwen38Path ||
		receipt.TargetVerificationOperations != 1 || receipt.TargetDecodeSteps != 0 || !receipt.OneOperation {
		t.Fatalf("Qwen3.8 verification receipt = %+v, want one fak-native target operation", receipt)
	}
	if receipt.EndToEndMeasured() {
		t.Fatal("target-only verification receipt was incorrectly treated as end-to-end performance evidence")
	}
	for _, missing := range []string{"drafting", "rejection", "memory", "recovery"} {
		if !containsString(receipt.MissingCostCategories(), missing) {
			t.Fatalf("partial receipt missing categories = %v, want %q explicit", receipt.MissingCostCategories(), missing)
		}
	}
	if len(rows) != len(wantRows) {
		t.Fatalf("Qwen3.8 verification rows = %d, want %d", len(rows), len(wantRows))
	}
	for i := range rows {
		if got, want := argmaxF32(rows[i]), argmaxF32(wantRows[i]); got != want {
			t.Fatalf("Qwen3.8 row %d argmax = %d, want ordinary native decode %d", i, got, want)
		}
		if delta := maxAbsF32Delta(rows[i], wantRows[i]); delta > qwen38VerifyBoundaryTolerance {
			t.Fatalf("Qwen3.8 row %d max delta = %g, want <= %g", i, delta, qwen38VerifyBoundaryTolerance)
		}
	}
	assertQwen35MTPTargetStateEqual(t, target, wantPrefix)
}

func TestQwen38VerifyForwardOneOperationTypedDowngradeUsesOrdinaryNativeDecode(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	draft := []int{2, 3, 4}
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	boundary := target.Prefill(prompt)
	wantPrefix := m.NewSession()
	wantPrefix.captureTargetHidden = true
	t.Cleanup(wantPrefix.Close)
	wantPrefix.Prefill(prompt)

	// F16 is outside the witnessed one-operation envelope. The ordinary native
	// Step path remains f32, so it is a safe way to exercise the explicit
	// downgrade without introducing an external runtime or unprepared weights.
	target.F16 = true
	operations := 0
	rows, receipt, err := target.verifyForwardOneOperation(draft, boundary, func(m *Model, tokens []int) *Activations {
		operations++
		return m.Forward(tokens)
	})
	var downgrade *TargetVerificationDowngradeError
	if !errors.As(err, &downgrade) || !errors.Is(err, ErrTargetVerificationDowngrade) {
		t.Fatalf("one-operation refusal = %v, want typed target-decode downgrade", err)
	}
	if rows != nil || operations != 0 || receipt.OneOperation || receipt.TargetVerificationOperations != 0 {
		t.Fatalf("pre-operation downgrade rows=%v operations=%d receipt=%+v", rows, operations, receipt)
	}
	assertQwen35MTPTargetStateEqual(t, target, wantPrefix)
	normalizeSnapshotForTest(t, wantPrefix)

	tx, err := beginQwen35MTPTargetTransaction(target, boundary)
	if err != nil {
		t.Fatal(err)
	}
	rows, err = tx.Verify(draft)
	if err != nil {
		t.Fatalf("explicit ordinary native target downgrade: %v", err)
	}
	if len(rows) != len(draft) {
		t.Fatalf("ordinary native target rows = %d, want %d", len(rows), len(draft))
	}
	receipt = tx.VerificationReceipt()
	if receipt.Engine != targetVerificationEngine || receipt.Path != targetVerificationDecodePath ||
		receipt.TargetVerificationOperations != 0 || receipt.TargetDecodeSteps != len(draft) ||
		receipt.OneOperation || receipt.DowngradeReason == "" {
		t.Fatalf("ordinary target-decode receipt = %+v", receipt)
	}
	if err := tx.Abort(); err != nil {
		t.Fatal(err)
	}
	assertQwen35MTPTargetStateEqual(t, target, wantPrefix)
}

func normalizeSnapshotForTest(t *testing.T, session *Session) {
	t.Helper()
	snapshot, err := session.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if err := snapshot.Restore(session); err != nil {
		snapshot.Close()
		t.Fatal(err)
	}
	snapshot.Close()
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

// TestVerifyForwardTreeMaskIsolatesBranches proves the tree-attention mask is correct: a
// node verified with the ancestor mask gets EXACTLY the greedy context (its argmax matches
// a sequential Step down its branch), and a sibling branch never contaminates it (adding a
// sibling leaves the node's logits bit-identical — the mask excludes non-ancestors).
//
// Tree (BFS): A=g0 (depth1), B=distractor (depth1), C=child-of-A=g1 (depth2). The accepted
// path A->C is the greedy continuation; B is a rejected sibling that must not affect A or C.
func TestVerifyForwardTreeMaskIsolatesBranches(t *testing.T) {
	m := NewSynthetic(cfgV(48, 3, 4, 2, 16, 96))
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Greedy reference: the true continuation down the accepted branch.
	greedy := m.NewSession()
	tl := greedy.Prefill(prompt)
	g0 := argmaxV(tl)
	g1 := argmaxV(greedy.Step(g0))
	g2 := argmaxV(greedy.Step(g1))

	base := greedy.Cache.Len() // == len(prompt); positions are 0-indexed contiguous
	// Panel (BFS): A=g0 (depth1), B=a distractor != g0 (depth1), C=g1 (depth2, child of A).
	distractor := (g0 + 7) % 256
	if distractor == g0 {
		distractor = (g0 + 13) % 256
	}
	tokens := []int{g0, distractor, g1}
	parent := []int{-1, -1, 0} // C's parent is A (panel index 0)
	// pos[i] = base + depth(i) - 1; siblings (same depth) share a position. depth walks the
	// parent chain to the root.
	pos := make([]int, len(tokens))
	for i := range pos {
		depth := 1
		for p := parent[i]; p >= 0; p = parent[p] {
			depth++
		}
		pos[i] = base + depth - 1
	}

	// Ancestor matrix within the panel: anc[q][k] iff k is q, or k is an ancestor of q.
	anc := make([][]bool, len(tokens))
	for q := range anc {
		anc[q] = make([]bool, len(tokens))
		anc[q][q] = true
		for p := parent[q]; p >= 0; p = parent[p] {
			anc[q][p] = true
		}
	}
	allow := func(q, k int) bool { return anc[q][k] }

	// Verify the whole tree in one pass on a fresh session.
	tree := m.NewSession()
	tree.Prefill(prompt)
	logits := tree.VerifyForward(tokens, pos, allow)
	if logits == nil {
		t.Fatal("tree VerifyForward returned nil for a supported regime")
	}
	// A (panel 0, token g0) predicts the token after g0 given prefix+g0 = the greedy g1.
	if got := argmaxV(logits[0]); got != g1 {
		t.Errorf("tree node A argmax = %d, want greedy g1=%d (ancestor mask gave wrong context)", got, g1)
	}
	// C (panel 2, token g1) predicts the token after g1 given prefix+g0+g1 = the greedy g2.
	if got := argmaxV(logits[2]); got != g2 {
		t.Errorf("tree node C argmax = %d, want greedy g2=%d (ancestor mask gave wrong context)", got, g2)
	}

	// Isolation: A's logits must be BIT-IDENTICAL to a solo verify of just [A] (no sibling B,
	// no niece C) — proving B's presence never reached A through the mask. The solo panel
	// attends to prefix + itself, exactly A's key set in the full tree.
	solo := m.NewSession()
	solo.Prefill(prompt)
	soloA := solo.VerifyForward([]int{g0}, []int{base}, func(q, k int) bool { return true })
	for i := range logits[0] {
		if math.Float32bits(logits[0][i]) != math.Float32bits(soloA[0][i]) {
			t.Fatalf("node A logit[%d]: tree %v != solo %v (sibling branch leaked through the mask)", i, logits[0][i], soloA[0][i])
		}
	}

	// Sanity: B (the distractor) is NOT the greedy token, so it differs from g0.
	if tokens[1] == g0 {
		t.Fatalf("distractor == g0; fix the test")
	}
}

func TestQwen38MTPBatchedTargetVerificationBlockAccounting(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	target := m.NewSession()
	target.captureTargetHidden = true
	t.Cleanup(target.Close)
	before := target.Prefill(prompt)

	draft := []int{2, 3, 4}
	tx, err := beginQwen35MTPTargetTransaction(target, before)
	if err != nil {
		t.Fatalf("begin target transaction: %v", err)
	}

	rows, err := tx.Verify(draft)
	if err != nil {
		t.Fatalf("target transaction verify: %v", err)
	}
	if len(rows) != len(draft) {
		t.Fatalf("verify rows = %d, want %d", len(rows), len(draft))
	}

	receipt := tx.VerificationReceipt()
	if receipt.Engine != targetVerificationEngine {
		t.Fatalf("receipt engine = %q, want %q", receipt.Engine, targetVerificationEngine)
	}
	if !receipt.OneOperation || receipt.TargetVerificationOperations != 1 {
		t.Fatalf("receipt operations = %d (oneOp=%v), want exactly 1 operation per block", receipt.TargetVerificationOperations, receipt.OneOperation)
	}
	if receipt.TargetDecodeSteps != 0 {
		t.Fatalf("target decode steps = %d, want 0 on batched verify", receipt.TargetDecodeSteps)
	}
	if !receipt.Accounting.Setup.Measured || !receipt.Accounting.TargetVerification.Measured {
		t.Fatalf("accounting setup/target_verification unmeasured: %+v", receipt.Accounting)
	}

	accepted := 2
	committedLogits, err := tx.Commit(accepted)
	if err != nil {
		t.Fatalf("commit accepted prefix: %v", err)
	}
	if len(committedLogits) == 0 {
		t.Fatal("committed logits empty")
	}

	finalReceipt := tx.VerificationReceipt()
	if finalReceipt.AcceptedTokens != accepted || finalReceipt.RejectedTokens != len(draft)-accepted {
		t.Fatalf("final receipt tokens accepted=%d rejected=%d, want accepted=%d rejected=%d",
			finalReceipt.AcceptedTokens, finalReceipt.RejectedTokens, accepted, len(draft)-accepted)
	}
	if !finalReceipt.Accounting.Rollback.Measured || !finalReceipt.Accounting.Synchronization.Measured {
		t.Fatalf("accounting rollback/sync unmeasured: %+v", finalReceipt.Accounting)
	}
}

func BenchmarkQwen38MTPBatchedTargetVerification(b *testing.B) {
	m := qwen38HybridMTPEnabledSyntheticModelBench(b)
	prompt := []int{0, 1}
	draft := []int{2, 3, 4}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target := m.NewSession()
		target.captureTargetHidden = true
		before := target.Prefill(prompt)

		tx, err := beginQwen35MTPTargetTransaction(target, before)
		if err != nil {
			b.Fatal(err)
		}

		rows, err := tx.Verify(draft)
		if err != nil || len(rows) != len(draft) {
			b.Fatalf("Verify failed: err=%v rows=%d", err, len(rows))
		}

		receipt := tx.VerificationReceipt()
		if receipt.TargetVerificationOperations != 1 || !receipt.OneOperation {
			b.Fatalf("expected 1 verification operation per block, got %d", receipt.TargetVerificationOperations)
		}

		_, err = tx.Commit(2)
		if err != nil {
			b.Fatal(err)
		}
		target.Close()
	}
}

func qwen38HybridMTPEnabledSyntheticModelBench(b *testing.B) *Model {
	b.Helper()
	cfg := qwen35HybridTestCfg()
	cfg.Name = "Qwen3.8 hybrid MTP synthetic bench"
	cfg.ModelType = "qwen3_5_text"
	cfg.MTPNumHiddenLayers = 1
	m := NewSynthetic(cfg)
	shapes, err := qwen35MTPExpectedShapes(cfg)
	if err != nil {
		b.Fatalf("Qwen3.8 hybrid MTP shapes: %v", err)
	}
	for tensorIndex, name := range qwen35MTPRequiredTensors {
		shape := shapes[name]
		elements := 1
		for _, dim := range shape {
			elements *= dim
		}
		start := len(m.raw)
		for i := 0; i < elements; i++ {
			value := float32(tensorIndex+1)/100 + float32(i)/100000
			var bits [4]byte
			binary.LittleEndian.PutUint32(bits[:], math.Float32bits(value))
			m.raw = append(m.raw, bits[:]...)
		}
		m.manifest[name] = tensorMeta{
			Dtype:  "F32",
			Shape:  append([]int(nil), shape...),
			Offset: start,
			Nbytes: elements * 4,
		}
	}
	return m
}
