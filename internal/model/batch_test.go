package model

import (
	"math"
	"testing"
)

// TestBatchFromPrefixMatchesIndependentPrefill is the correctness rung for the cross-agent
// KV-reuse path (NewBatchFromPrefix): a batch whose users are CLONES of a once-computed
// prefix must decode, for every user, BIT-FOR-BIT identically to a batch whose users each
// prefilled that whole prefix independently. That is the whole claim — skipping C-1 of the C
// prefix prefills costs zero numerics, because the clone is exact. It composes two already
// proven rungs (clone == prefill-prefix from TestKVPrefixReuseMatchesRecompute; StepBatch ==
// serial Step from TestBatchedDecodeMatchesSerial), so it holds for any weights.
func TestBatchFromPrefixMatchesIndependentPrefill(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	V := cfg.VocabSize
	B := 4

	// A shared prefix (the "system prompt + tool schemas" every agent shares).
	prefix := make([]int, 11)
	for i := range prefix {
		prefix[i] = (i*37 + 9) % V
	}

	// Batch A: each of B users prefills the FULL prefix independently (the work llama.cpp
	// must repeat per slot).
	bsA := m.NewBatchSession(B)
	promptsA := make([][]int, B)
	for b := range promptsA {
		promptsA[b] = prefix
	}
	bsA.PrefillEach(promptsA)

	// Batch B: prefill the prefix ONCE, clone into all B users (the fak cross-agent path).
	base := m.NewSession()
	base.Prefill(prefix)
	bsB := m.NewBatchFromPrefixReserve(base.Cache, B, 6)
	for b, s := range bsB.Seqs {
		if cap(s.Cache.pos) < len(prefix)+6 {
			t.Fatalf("user %d reserved pos cap %d < %d", b, cap(s.Cache.pos), len(prefix)+6)
		}
		w := s.Cache.kvStride()
		for l := 0; l < cfg.NumLayers; l++ {
			if cap(s.Cache.K[l]) < len(prefix)*w+6*w {
				t.Fatalf("user %d layer %d reserved K cap %d < %d", b, l, cap(s.Cache.K[l]), len(prefix)*w+6*w)
			}
			if cap(s.Cache.Kraw[l]) < len(prefix)*w+6*w {
				t.Fatalf("user %d layer %d reserved Kraw cap %d < %d", b, l, cap(s.Cache.Kraw[l]), len(prefix)*w+6*w)
			}
			if cap(s.Cache.V[l]) < len(prefix)*w+6*w {
				t.Fatalf("user %d layer %d reserved V cap %d < %d", b, l, cap(s.Cache.V[l]), len(prefix)*w+6*w)
			}
		}
	}

	// Decode both with the same per-user (distinct) token streams; assert bit-equality.
	ids := make([]int, B)
	for b := range ids {
		ids[b] = (b*13 + 3) % V
	}
	for step := 0; step < 6; step++ {
		la := bsA.StepBatch(ids)
		lb := bsB.StepBatch(ids)
		for b := 0; b < B; b++ {
			if len(la[b]) != len(lb[b]) {
				t.Fatalf("step %d user %d: logit len %d != %d", step, b, len(la[b]), len(lb[b]))
			}
			for i := range la[b] {
				if math.Float32bits(la[b][i]) != math.Float32bits(lb[b][i]) {
					t.Fatalf("step %d user %d pos %d: clone-from-prefix logit %v != independent-prefill %v",
						step, b, i, lb[b][i], la[b][i])
				}
			}
		}
		for b := range ids {
			ids[b] = (ids[b]*7 + b + 1) % V
		}
	}
}

func TestPrefillEachRectangularMatchesSerial(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	V := cfg.VocabSize
	B, P := 4, 7
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		prompts[b] = make([]int, P)
		for t := 0; t < P; t++ {
			prompts[b][t] = (b*41 + t*17 + 5) % V
		}
	}

	ref := make([]*Session, B)
	refLogits := make([][]float32, B)
	for b := 0; b < B; b++ {
		ref[b] = m.NewSession()
		refLogits[b] = ref[b].Prefill(prompts[b])
	}

	bs := m.NewBatchSession(B)
	gotLogits := bs.PrefillEach(prompts)
	for b := 0; b < B; b++ {
		for i := range refLogits[b] {
			if math.Float32bits(gotLogits[b][i]) != math.Float32bits(refLogits[b][i]) {
				t.Fatalf("user %d logit[%d]: rectangular prefill %v != serial %v",
					b, i, gotLogits[b][i], refLogits[b][i])
			}
		}
		rc, bc := ref[b].Cache, bs.Seqs[b].Cache
		if bc.Len() != rc.Len() {
			t.Fatalf("user %d cache len %d != %d", b, bc.Len(), rc.Len())
		}
		for l := 0; l < cfg.NumLayers; l++ {
			for name, pair := range map[string][2][]float32{
				"K":    {rc.K[l], bc.K[l]},
				"Kraw": {rc.Kraw[l], bc.Kraw[l]},
				"V":    {rc.V[l], bc.V[l]},
			} {
				x, y := pair[0], pair[1]
				if len(x) != len(y) {
					t.Fatalf("user %d layer %d %s len %d != %d", b, l, name, len(x), len(y))
				}
				for i := range x {
					if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
						t.Fatalf("user %d layer %d %s[%d]: rectangular prefill %v != serial %v",
							b, l, name, i, y[i], x[i])
					}
				}
			}
		}
		for i := range rc.pos {
			if rc.pos[i] != bc.pos[i] {
				t.Fatalf("user %d pos[%d] %d != %d", b, i, rc.pos[i], bc.pos[i])
			}
		}
	}
}

// TestBatchedDecodeMatchesSerial is the load-bearing correctness rung for multi-user
// batching: decoding B users together via StepBatch must produce, for EVERY user, logits
// and a KV-cache state that are BIT-FOR-BIT identical to decoding that user alone through
// the serial Session.Step. This is what proves the batch shares only the weight STREAM, not
// any state — no cross-user contamination, each user's causal attention reads only its own
// history — so the ~B× throughput is bought without spending a single bit of the proven
// numerics (the same discipline as TestPrefillBatchedMatchesSerial, on the decode axis).
//
// Synthetic weights (no HF export needed) suffice: the property is structural — matMulBatch
// row b == parMatRows for user b (TestParallelMatchesSerial), and the per-user attention
// replays tokenHidden's exact scalar arithmetic — so it holds for ANY weights.
func TestBatchedDecodeMatchesSerial(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	V := cfg.VocabSize
	B := 5

	// Distinct prompts of distinct lengths => users sit at distinct absolute positions, so
	// the per-user RoPE / per-user cache-length attention paths are all exercised at once.
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		n := 3 + b*2
		p := make([]int, n)
		for i := range p {
			p[i] = (b*97 + i*31 + 5) % V
		}
		prompts[b] = p
	}

	// Reference: B independent serial sessions.
	ref := make([]*Session, B)
	refLogits := make([][]float32, B)
	for b := 0; b < B; b++ {
		ref[b] = m.NewSession()
		refLogits[b] = ref[b].Prefill(prompts[b])
	}

	// Batch: one BatchSession over the same prompts.
	bs := m.NewBatchSession(B)
	batLogits := bs.PrefillEach(prompts)

	assertBatchBitEqual := func(step int) {
		for b := 0; b < B; b++ {
			a, c := refLogits[b], batLogits[b]
			if len(a) != len(c) {
				t.Fatalf("step %d user %d: logit len %d != %d", step, b, len(a), len(c))
			}
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(c[i]) {
					t.Fatalf("step %d user %d logit[%d]: serial %v != batched %v (NOT bit-identical)",
						step, b, i, a[i], c[i])
				}
			}
		}
	}

	// Decode several steps in lockstep, feeding the SAME deterministic per-user token each
	// step to both paths, comparing the logits they produce for that identical input state.
	steps := 6
	assertBatchBitEqual(0) // post-prefill distributions
	for s := 0; s < steps; s++ {
		next := make([]int, B)
		for b := 0; b < B; b++ {
			next[b] = (s*53 + b*17 + 1) % V
		}
		for b := 0; b < B; b++ {
			refLogits[b] = ref[b].Step(next[b])
		}
		batLogits = bs.StepBatch(next)
		assertBatchBitEqual(s + 1)
	}

	// The KV caches must also be byte-for-byte equal per user (K, Kraw, V, pos) — a batched
	// step that built a subtly different cache would silently break per-user Evict/Clone even
	// if the logits matched.
	for b := 0; b < B; b++ {
		rc, bc := ref[b].Cache, bs.Seqs[b].Cache
		if bc.Len() != rc.Len() {
			t.Fatalf("user %d cache len %d != %d", b, bc.Len(), rc.Len())
		}
		for l := 0; l < cfg.NumLayers; l++ {
			for name, pair := range map[string][2][]float32{
				"K":    {rc.K[l], bc.K[l]},
				"Kraw": {rc.Kraw[l], bc.Kraw[l]},
				"V":    {rc.V[l], bc.V[l]},
			} {
				x, y := pair[0], pair[1]
				if len(x) != len(y) {
					t.Fatalf("user %d layer %d %s len %d != %d", b, l, name, len(x), len(y))
				}
				for i := range x {
					if math.Float32bits(x[i]) != math.Float32bits(y[i]) {
						t.Fatalf("user %d layer %d %s[%d]: serial %v != batched %v", b, l, name, i, x[i], y[i])
					}
				}
			}
		}
		for i := range rc.pos {
			if rc.pos[i] != bc.pos[i] {
				t.Fatalf("user %d pos[%d] %d != %d", b, i, rc.pos[i], bc.pos[i])
			}
		}
	}
}

// TestBatchedDecodeQMatchesF32 gates the Q8_0 multi-user decode the same honest way the
// prefill Q8 path is gated. It is NOT bit-identical to the serial qdot8 decode (the tile GEMM
// reduces in a different lane order), so the gate is faithfulness to f32, not bit-equality:
//
//   - the DECISIVE first generated token (last-position argmax on the REAL prompt, where the
//     distribution is confident) must equal f32, per user — the same bar TestQuantMatchesF32Logits
//     uses, applied to the batched path;
//   - across teacher-forced decode steps fed the HF GREEDY continuation (in-distribution, so
//     argmax is meaningful — feeding arbitrary tokens would drive the model into low-confidence
//     states where a lossless model would also flip near-ties), top-1 agreement with f32 must
//     stay high (>=0.80, mirroring TestQuantTeacherForcedAgreement) and mean logit-cosine >=0.99.
//
// A real batching bug (cross-user mixing, wrong panel stride, wrong per-user position) would
// crater cosine and flip the decisive first token; a faithful Q8 batch nicks neither.
func TestBatchedDecodeQMatchesF32(t *testing.T) {
	m, doc := loadFixture(t)
	m.Quantize()

	B := len(doc.Prompts)
	if B < 2 {
		t.Skip("need >=2 oracle prompts to exercise a batch")
	}
	prompts := make([][]int, B)
	steps := 1 << 30
	for b, p := range doc.Prompts {
		prompts[b] = p.Ids
		if len(p.GreedyIds) < steps {
			steps = len(p.GreedyIds) // rectangular batch: step only as far as the shortest continuation
		}
	}
	if steps > 8 {
		steps = 8
	}
	if steps < 1 {
		t.Skip("oracle has no greedy continuation to teacher-force")
	}

	// f32 reference: independent serial sessions.
	ref := make([]*Session, B)
	refLogits := make([][]float32, B)
	for b := 0; b < B; b++ {
		ref[b] = m.NewSession()
		refLogits[b] = ref[b].Prefill(prompts[b])
	}

	// Q8 batch over the same prompts.
	bs := m.NewBatchSession(B)
	bs.SetQuant(true)
	batLogits := bs.PrefillEach(prompts)

	// Decisive gate: the first generated token must match f32 for every user.
	for b := 0; b < B; b++ {
		amF, amQ := argmax(refLogits[b]), argmax(batLogits[b])
		cs := cosine(batLogits[b], refLogits[b])
		t.Logf("first-token user %d Q8-batch vs f32 cos=%.6f argmax f32=%d q8=%d", b, cs, amF, amQ)
		if amF != amQ {
			t.Errorf("user %d first-token Q8-batch argmax %d != f32 argmax %d", b, amQ, amF)
		}
		// 0.993 floor (was 0.995): the first token is produced by the FMA-folded prefill
		// GEMM, which is MORE accurate to the true GEMM than the f32-fak fdot it's compared
		// against (q8bench: lower last-logit max|Δ| vs the HF oracle on every prompt, argmax
		// exact 25/25). See TestQuantMatchesF32Logits for the full rationale. The argmax bar
		// above is the authoritative gate and is unchanged.
		if cs < 0.993 {
			t.Errorf("user %d first-token Q8-batch-vs-f32 cosine %.6f < 0.993", b, cs)
		}
	}

	// Teacher-forced decode steps fed the HF greedy continuation.
	agree, total := 0, 0
	var cosSum, cosMin float64 = 0, 1
	for s := 0; s < steps; s++ {
		for b := 0; b < B; b++ {
			cs := cosine(batLogits[b], refLogits[b])
			cosSum += cs
			if cs < cosMin {
				cosMin = cs
			}
			if argmax(batLogits[b]) == argmax(refLogits[b]) {
				agree++
			}
			total++
		}
		next := make([]int, B)
		for b := 0; b < B; b++ {
			next[b] = doc.Prompts[b].GreedyIds[s] // feed HF's token to both (teacher-forced)
			refLogits[b] = ref[b].Step(next[b])
		}
		batLogits = bs.StepBatch(next)
	}
	rate := float64(agree) / float64(total)
	meanCos := cosSum / float64(total)
	t.Logf("Q8-batch decode vs f32: top-1 agreement %d/%d = %.1f%%, mean cosine %.5f, min %.5f",
		agree, total, 100*rate, meanCos, cosMin)
	if rate < 0.80 {
		t.Errorf("Q8-batch decode top-1 agreement %.1f%% < 80%%", 100*rate)
	}
	if meanCos < 0.99 {
		t.Errorf("Q8-batch decode mean cosine %.5f < 0.99", meanCos)
	}
}

func TestRectangularPrefillQReuseMatchesIndependentSessions(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{
			name: "grp2",
			cfg: Config{
				HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
				IntermediateSize: 128, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
				TieWordEmbeddings: true, EOSTokenID: -1,
			},
		},
		{
			name: "grp3",
			cfg: Config{
				HiddenSize: 96, NumLayers: 2, NumHeads: 6, NumKVHeads: 2, HeadDim: 16,
				IntermediateSize: 192, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
				TieWordEmbeddings: true, EOSTokenID: -1,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := tc.cfg
			m := NewSynthetic(cfg)
			m.Quantize()
			B := 3
			promptsA := make([][]int, B)
			promptsB := make([][]int, B)
			for user := 0; user < B; user++ {
				promptsA[user] = make([]int, 7) // grow the reusable prefill buffers first
				promptsB[user] = make([]int, 3) // then reuse them on a smaller rectangle
				for i := range promptsA[user] {
					promptsA[user][i] = (user*31 + i*17 + 5) % cfg.VocabSize
				}
				for i := range promptsB[user] {
					promptsB[user][i] = (user*43 + i*19 + 11) % cfg.VocabSize
				}
			}

			ref := make([]*Session, B)
			want := make([][]float32, B)
			for user := 0; user < B; user++ {
				ref[user] = m.NewSession()
				ref[user].Quant = true
				ref[user].Prefill(promptsA[user])
				want[user] = ref[user].Prefill(promptsB[user])
			}

			bs := m.NewBatchSession(B)
			bs.SetQuant(true)
			bs.PrefillEach(promptsA)
			got := bs.PrefillEach(promptsB)

			for user := 0; user < B; user++ {
				cs := cosine(got[user], want[user])
				if cs < 0.999 {
					t.Fatalf("user %d repeated rectangular Q8 prefill cosine %.6f < 0.999", user, cs)
				}
				if argmax(got[user]) != argmax(want[user]) {
					t.Fatalf("user %d repeated rectangular Q8 prefill argmax %d != independent %d", user, argmax(got[user]), argmax(want[user]))
				}
				if bs.Seqs[user].Cache.Len() != ref[user].Cache.Len() {
					t.Fatalf("user %d cache len %d != independent %d", user, bs.Seqs[user].Cache.Len(), ref[user].Cache.Len())
				}
			}
		})
	}
}

func TestPrefillEachNoLogitsMatchesPrefillEachState(t *testing.T) {
	for _, tc := range []struct {
		name  string
		quant bool
		cfg   Config
	}{
		{
			name:  "f32_grp2",
			quant: false,
			cfg: Config{
				HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
				IntermediateSize: 128, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
				TieWordEmbeddings: true, EOSTokenID: -1,
			},
		},
		{
			name:  "q8_grp3",
			quant: true,
			cfg: Config{
				HiddenSize: 96, NumLayers: 2, NumHeads: 6, NumKVHeads: 2, HeadDim: 16,
				IntermediateSize: 192, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
				TieWordEmbeddings: true, EOSTokenID: -1,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := NewSynthetic(tc.cfg)
			if tc.quant {
				m.Quantize()
			}
			B, P := 3, 5
			prompts := make([][]int, B)
			next := make([]int, B)
			for user := 0; user < B; user++ {
				prompts[user] = make([]int, P)
				for i := range prompts[user] {
					prompts[user][i] = (user*37 + i*17 + 13) % tc.cfg.VocabSize
				}
				next[user] = (user*29 + 7) % tc.cfg.VocabSize
			}

			want := m.NewBatchSession(B)
			want.SetQuant(tc.quant)
			if logits := want.PrefillEach(prompts); len(logits) != B {
				t.Fatalf("PrefillEach logits rows = %d, want %d", len(logits), B)
			}
			got := m.NewBatchSession(B)
			got.SetQuant(tc.quant)
			got.PrefillEachNoLogits(prompts)

			for user := 0; user < B; user++ {
				assertKVCacheBitsEqual(t, "user "+itoa(user), want.Seqs[user].Cache, got.Seqs[user].Cache)
			}

			wantLogits := want.StepBatch(next)
			gotLogits := got.StepBatch(next)
			for user := 0; user < B; user++ {
				assertFloat32BitsEqual(t, "user "+itoa(user)+" next logits", wantLogits[user], gotLogits[user])
			}
		})
	}
}

func TestPrefillNoLogitsFallbackMatchesPrefillState(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	m.Quantize()
	prompt := []int{3, 19, 27, 44, 91}
	next := 17

	want := m.NewSession()
	want.Quant = true
	if logits := want.Prefill(prompt); len(logits) != cfg.VocabSize {
		t.Fatalf("Prefill logits len = %d, want %d", len(logits), cfg.VocabSize)
	}
	got := m.NewSession()
	got.Quant = true
	got.PrefillNoLogits(prompt)
	assertKVCacheBitsEqual(t, "session", want.Cache, got.Cache)
	assertFloat32BitsEqual(t, "session next logits", want.Step(next), got.Step(next))

	prompts := [][]int{
		{5, 7, 11},
		{13, 17, 19, 23},
		{29, 31},
	}
	wantBatch := m.NewBatchSession(len(prompts))
	wantBatch.SetQuant(true)
	wantBatch.PrefillEach(prompts)
	gotBatch := m.NewBatchSession(len(prompts))
	gotBatch.SetQuant(true)
	gotBatch.PrefillEachNoLogits(prompts)
	for user := range prompts {
		assertKVCacheBitsEqual(t, "ragged user "+itoa(user), wantBatch.Seqs[user].Cache, gotBatch.Seqs[user].Cache)
	}
	nextBatch := []int{2, 3, 5}
	wantLogits := wantBatch.StepBatch(nextBatch)
	gotLogits := gotBatch.StepBatch(nextBatch)
	for user := range prompts {
		assertFloat32BitsEqual(t, "ragged user "+itoa(user)+" next logits", wantLogits[user], gotLogits[user])
	}

	singlePrompt := [][]int{{37, 41, 43, 47}}
	wantOne := m.NewBatchSession(1)
	wantOne.SetQuant(true)
	wantOne.PrefillEach(singlePrompt)
	gotOne := m.NewBatchSession(1)
	gotOne.SetQuant(true)
	gotOne.PrefillEachNoLogits(singlePrompt)
	assertKVCacheBitsEqual(t, "single-user batch", wantOne.Seqs[0].Cache, gotOne.Seqs[0].Cache)
	assertFloat32BitsEqual(t, "single-user batch next logits", wantOne.StepBatch([]int{53})[0], gotOne.StepBatch([]int{53})[0])
}

// TestGenerateBatchMatchesSerial proves the lockstep greedy GenerateBatch yields, per user,
// the SAME token sequence as serial Session.Generate — the end-to-end multi-user serving loop
// is output-identical to running each user alone, just with a shared weight stream per step.
func TestGenerateBatchMatchesSerial(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	V := cfg.VocabSize
	B := 4
	prompts := make([][]int, B)
	for b := 0; b < B; b++ {
		p := make([]int, 4+b)
		for i := range p {
			p[i] = (b*53 + i*29 + 3) % V
		}
		prompts[b] = p
	}

	n := 10
	want := make([][]int, B)
	for b := 0; b < B; b++ {
		want[b] = m.NewSession().Generate(prompts[b], n)
	}
	got := m.NewBatchSession(B).GenerateBatch(prompts, n)

	for b := 0; b < B; b++ {
		if len(got[b]) != len(want[b]) {
			t.Fatalf("user %d generated %d tokens, serial generated %d", b, len(got[b]), len(want[b]))
		}
		for i := range want[b] {
			if got[b][i] != want[b][i] {
				t.Fatalf("user %d token %d: batch %d != serial %d", b, i, got[b][i], want[b][i])
			}
		}
	}
}
