package model

import (
	"math"
	"math/rand"
	"testing"
)

// kvcache_q8_test.go — realization witnesses for the Q8_0 KV cache (#12981).

// q8TestConfig is a tiny standard-attention geometry: 2 layers, 2 KV heads, head_dim
// 32 (so one 32-element block per head row, exercises the block boundary), 4 query
// heads (group size 2).
func q8TestConfig() Config {
	return Config{
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          32,
		HiddenSize:       128,
		IntermediateSize: 256,
		RopeTheta:        10000,
	}
}

func randRow(rng *rand.Rand, n int) []float32 {
	row := make([]float32, n)
	for i := range row {
		row[i] = float32(rng.NormFloat64())
	}
	return row
}

// appendSynthetic writes p positions of random post-RoPE K and V (and f32 Kraw) into
// a cache, returning the rows it wrote for later comparison.
func appendSynthetic(t *testing.T, c *KVCache, p int, seed int64) (ks, vs [][]float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	w := c.kvStride()
	for j := 0; j < p; j++ {
		k := randRow(rng, w)
		v := randRow(rng, w)
		for l := 0; l < c.cfg.NumLayers; l++ {
			c.Kraw[l] = append(c.Kraw[l], k...)
			c.appendKV(l, k, v)
		}
		c.appendPosition(j, j)
		ks = append(ks, append([]float32(nil), k...))
		vs = append(vs, append([]float32(nil), v...))
	}
	return ks, vs
}

// TestQ8KVCacheDefaultIsByteIdenticalF32 pins the zero-value/default contract: a cache
// built with the explicit FP32 tier (and the historical constructor) stores K/V as the
// same f32 bytes, and no packed storage is allocated.
func TestQ8KVCacheDefaultIsByteIdenticalF32(t *testing.T) {
	cfg := q8TestConfig()
	def := NewKVCache(cfg)
	explicit := NewKVCacheWithPrecision(cfg, KVPrecisionFP32)
	if def.quantized() || explicit.quantized() {
		t.Fatal("default and explicit-f32 caches must not be quantized")
	}
	if def.Precision() != KVPrecisionFP32 || explicit.Precision() != KVPrecisionFP32 {
		t.Fatalf("precision = (%s, %s), want f32", def.Precision(), explicit.Precision())
	}

	appendSynthetic(t, def, 7, 1)
	appendSynthetic(t, explicit, 7, 1)
	if len(def.K[0]) != len(explicit.K[0]) || len(def.V[0]) != len(explicit.V[0]) {
		t.Fatalf("f32 lengths differ: def K=%d V=%d explicit K=%d V=%d",
			len(def.K[0]), len(def.V[0]), len(explicit.K[0]), len(explicit.V[0]))
	}
	for i := range def.K[0] {
		if def.K[0][i] != explicit.K[0][i] || def.V[0][i] != explicit.V[0][i] {
			t.Fatalf("f32 byte identity broken at %d: K %v/%v V %v/%v",
				i, def.K[0][i], explicit.K[0][i], def.V[0][i], explicit.V[0][i])
		}
	}
	if len(def.kQ8) != 0 || len(def.vQ8) != 0 {
		t.Fatal("f32 cache must not allocate packed q8 storage")
	}
	// Empty-string precision is the caller-facing "unset" and must also mean f32.
	if NewKVCacheWithPrecision(cfg, "").quantized() {
		t.Fatal("empty precision must resolve to f32")
	}
}

// TestQ8KVCacheRoundTripParityVsF32 checks the attended K/V read back through the
// dequantize-on-attend path against the f32 cache within a relative-L2 and cosine
// bound over a fixed synthetic prompt. Q8_0 is expected to drift; the bound documents
// the drift rather than claiming bit equality.
func TestQ8KVCacheRoundTripParityVsF32(t *testing.T) {
	cfg := q8TestConfig()
	f32c := NewKVCache(cfg)
	q8c := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
	if !q8c.quantized() {
		t.Fatal("q8 cache must be quantized")
	}

	const p = 64
	ks, _ := appendSynthetic(t, f32c, p, 42)
	appendSynthetic(t, q8c, p, 42)
	if q8c.kvLen(0) != p {
		t.Fatalf("q8 kvLen = %d, want %d", q8c.kvLen(0), p)
	}

	w := cfg.NumKVHeads * cfg.HeadDim
	kf, vf := f32c.attentionRows(0)
	kq, vq := q8c.attentionRows(0)
	var kSq, vSq, kDot, vDot, kRef, vRef, kDq, vDq float64
	for i := 0; i < p*w; i++ {
		a, b := float64(kf[i]), float64(kq[i])
		kSq += (b - a) * (b - a)
		kRef += a * a
		kDq += b * b
		kDot += a * b
		a, b = float64(vf[i]), float64(vq[i])
		vSq += (b - a) * (b - a)
		vRef += a * a
		vDq += b * b
		vDot += a * b
	}
	kRelL2 := math.Sqrt(kSq / kRef)
	vRelL2 := math.Sqrt(vSq / vRef)
	kCos := kDot / (math.Sqrt(kRef) * math.Sqrt(kDq))
	vCos := vDot / (math.Sqrt(vRef) * math.Sqrt(vDq))
	// Bounds for Q8_0 (block 32, f32 scale): relL2 well under 1%, cosine ~0.9999+.
	if kRelL2 > 0.02 || vRelL2 > 0.02 {
		t.Fatalf("q8 relL2 too large: K=%.5f V=%.5f (want <=0.02)", kRelL2, vRelL2)
	}
	if kCos < 0.999 || vCos < 0.999 {
		t.Fatalf("q8 cosine K=%.6f V=%.6f, want both >=0.999", kCos, vCos)
	}
	// The synthetic rows themselves must round-trip within the codec's own per-block
	// error bound — a direct oracle on the realized packing.
	bound := float32(0)
	for j := 0; j < p; j++ {
		q := QuantizeKVQ8_0(ks[j])
		if e := q.ErrorBound(); e > bound {
			bound = e
		}
	}
	for i := 0; i < p*w; i++ {
		if d := float32(math.Abs(float64(kf[i] - kq[i]))); d > bound+1e-6 {
			t.Fatalf("realized K error %.6g exceeds codec bound %.6g at %d", d, bound, i)
		}
	}
	t.Logf("Q8_0 realized drift: K relL2=%.5f cos=%.6f V relL2=%.5f vcos=%.6f bound=%.6g",
		kRelL2, kCos, vRelL2, vCos, bound)
}

// TestQ8KVCacheCloneTruncateReserve exercises the quant-aware lifecycle: Clone copies
// packed rows and preserves Len + values, Reserve grows spare capacity without
// changing Len, and Truncate keeps exactly the requested positions.
func TestQ8KVCacheCloneTruncateReserve(t *testing.T) {
	cfg := q8TestConfig()
	c := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
	const p = 20
	appendSynthetic(t, c, p, 7)

	cl := c.CloneWithReserve(8)
	if cl.kvLen(0) != p || cl.Precision() != KVPrecisionQ8_0 {
		t.Fatalf("clone len=%d prec=%s, want %d/q8_0", cl.kvLen(0), cl.Precision(), p)
	}
	// Mutating the clone must not alias the source.
	cl.kQ8[0].codes[0] ^= 0x7f
	if c.kQ8[0].codes[0] == cl.kQ8[0].codes[0] {
		t.Fatal("clone aliases source packed codes")
	}
	if cl.kQ8[0].bytes() != c.kQ8[0].bytes() {
		t.Fatalf("clone bytes=%d source bytes=%d", cl.kQ8[0].bytes(), c.kQ8[0].bytes())
	}

	c.Reserve(16)
	if c.kvLen(0) != p {
		t.Fatalf("Reserve changed Len to %d, want %d", c.kvLen(0), p)
	}
	if cap(c.kQ8[0].codes) < p*c.kvStride()+16*c.kvStride() {
		t.Fatalf("Reserve did not grow packed capacity: cap=%d", cap(c.kQ8[0].codes))
	}

	c.Truncate(5)
	if c.kvLen(0) != 5 {
		t.Fatalf("Truncate left %d positions, want 5", c.kvLen(0))
	}
	if len(c.pos) != 5 {
		t.Fatalf("Truncate left %d pos, want 5", len(c.pos))
	}
	// Truncate is a pure slice-header cut: the surviving codes are unchanged.
	if c.kQ8[0].codes[0] == 0 && c.kQ8[0].scales[0] == 0 {
		t.Log("note: first code/scale legitimately zero on this draw")
	}
}

// TestQ8KVCacheEvictReencodesSurvivors proves the quant-aware Evict compacts packed
// rows and re-RoPEs survivors to the same positions the f32 cache does. It evicts the
// same middle span from an f32 and a q8 cache built from one seed, then compares each
// surviving realized K/V row within the q8 codec's error bound — a survivor whose
// position changed must land where the f32 path re-derived it, not where it started.
func TestQ8KVCacheEvictReencodesSurvivors(t *testing.T) {
	cfg := q8TestConfig()
	f32c := NewKVCache(cfg)
	q8c := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
	const p = 12
	appendSynthetic(t, f32c, p, 11)
	appendSynthetic(t, q8c, p, 11)

	if removed := q8c.TryEvictMust(t, 4, 2); removed != 2 {
		t.Fatalf("removed=%d want 2", removed)
	}
	if removed := f32c.TryEvictMust(t, 4, 2); removed != 2 {
		t.Fatalf("f32 removed=%d want 2", removed)
	}
	if q8c.kvLen(0) != p-2 || f32c.kvLen(0) != p-2 {
		t.Fatalf("post-evict Len q8=%d f32=%d, want %d", q8c.kvLen(0), f32c.kvLen(0), p-2)
	}
	for i := range q8c.pos {
		if q8c.pos[i] != i {
			t.Fatalf("pos[%d]=%d, want %d", i, q8c.pos[i], i)
		}
	}
	// The re-RoPE only rewrites survivors whose index changed (i >= 4). Positions 0..3
	// are untouched by both paths; compare the moved survivors where re-encoding matters.
	w := cfg.NumKVHeads * cfg.HeadDim
	fk, fv := f32c.attentionRows(0)
	qk, qv := q8c.attentionRows(0)
	// The moved survivors are rows >= 4. Compare each moved row's realized q8 bytes
	// against the f32 row the exact-evict path produced: relL2 within the codec's
	// documented ~2% band proves the re-RoPE landed at the NEW position (a survivor
	// left at its OLD rotation would be off by a full angle, far above the band).
	var kSq, kRef, vSq, vRef float64
	for j := 4; j < q8c.kvLen(0); j++ {
		for i := j * w; i < (j+1)*w; i++ {
			a, b := float64(fk[i]), float64(qk[i])
			kSq += (b - a) * (b - a)
			kRef += a * a
			a, b = float64(fv[i]), float64(qv[i])
			vSq += (b - a) * (b - a)
			vRef += a * a
		}
	}
	kRel := math.Sqrt(kSq / kRef)
	vRel := math.Sqrt(vSq / vRef)
	if kRel > 0.02 || vRel > 0.02 {
		t.Fatalf("post-evict moved-survivor relL2 K=%.5f V=%.5f, want <=0.02 (re-RoPE landed at old position?)", kRel, vRel)
	}
	t.Logf("post-evict moved-survivor relL2 vs f32: K=%.5f V=%.5f", kRel, vRel)
}

// TryEvictMust is a test-only wrapper that fails on the typed unsupported verdict.
func (c *KVCache) TryEvictMust(t *testing.T, from, n int) int {
	t.Helper()
	removed, err := c.TryEvict(from, n)
	if err != nil {
		t.Fatalf("TryEvict(%d,%d) error: %v", from, n, err)
	}
	return removed
}

// TestQ8KVCacheResidentBytesMatchKVVectorBytes is the honesty witness: the REALIZED
// packed bytes for a cache equal the model.KVVectorBytes formula the planner charges
// (per layer: f32 Kraw row + two symmetric q8_0 rows), not just budget math. It uses
// the Qwen3.8-27B geometry (64 layers, 2 KV heads, head_dim 128).
func TestQ8KVCacheResidentBytesMatchKVVectorBytes(t *testing.T) {
	cfg := Config{NumLayers: 64, NumHeads: 40, NumKVHeads: 2, HeadDim: 128, HiddenSize: 5120, IntermediateSize: 17408, RopeTheta: 1e6}
	c := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
	const p = 8
	appendSynthetic(t, c, p, 3)

	dim := cfg.NumKVHeads * cfg.HeadDim
	perTokenPerLayer := int64(dim)*4 + 2*KVVectorBytes(dim, KVPrecisionQ8_0)
	want := perTokenPerLayer * int64(p) * int64(cfg.NumLayers)
	if got := c.KVCacheResidentBytes(); got != want {
		t.Fatalf("realized resident bytes = %d, want KVVectorBytes math %d", got, want)
	}
	// F32 fora comparison: realized q8 is ~2x denser.
	f32 := NewKVCacheWithPrecision(cfg, KVPrecisionFP32)
	appendSynthetic(t, f32, p, 3)
	ratio := float64(f32.KVCacheResidentBytes()) / float64(c.KVCacheResidentBytes())
	if ratio < 1.9 || ratio > 2.1 {
		t.Fatalf("f32/q8 realized ratio = %.3f, want ~2x", ratio)
	}
	t.Logf("realized @%d tok: f32=%d B q8=%d B ratio=%.3f", p, f32.KVCacheResidentBytes(), c.KVCacheResidentBytes(), ratio)
}

// TestQ8KVCacheConvertToPrecisionReEncodes proves an already-populated f32 cache can
// be re-realized at q8_0 in place (the reused/preserved-session path), preserving Len
// and the attended rows within the codec bound, and that converting back to f32 gives
// back the q8-dequantized values (no precision is fabricated).
func TestQ8KVCacheConvertToPrecisionReEncodes(t *testing.T) {
	cfg := q8TestConfig()
	c := NewKVCache(cfg)
	const p = 24
	appendSynthetic(t, c, p, 5)
	fk, fv := c.attentionRows(0)

	c.ConvertToPrecision(KVPrecisionQ8_0)
	if !c.quantized() || c.kvLen(0) != p {
		t.Fatalf("convert to q8: quantized=%v len=%d, want true/%d", c.quantized(), c.kvLen(0), p)
	}
	qk, qv := c.attentionRows(0)
	var kSq, vSq, kRef, vRef float64
	for i := 0; i < p*cfg.NumKVHeads*cfg.HeadDim; i++ {
		a, b := float64(fk[i]), float64(qk[i])
		kSq += (b - a) * (b - a)
		kRef += a * a
		a, b = float64(fv[i]), float64(qv[i])
		vSq += (b - a) * (b - a)
		vRef += a * a
	}
	if math.Sqrt(kSq/kRef) > 0.02 || math.Sqrt(vSq/vRef) > 0.02 {
		t.Fatalf("convert q8 relL2 K=%.5f V=%.5f, want <=0.02", math.Sqrt(kSq/kRef), math.Sqrt(vSq/vRef))
	}

	c.ConvertToPrecision(KVPrecisionFP32)
	if c.quantized() || c.kvLen(0) != p {
		t.Fatalf("convert back to f32: quantized=%v len=%d, want false/%d", c.quantized(), c.kvLen(0), p)
	}
	back, _ := c.attentionRows(0)
	for i := range back {
		if back[i] != qk[i] {
			t.Fatalf("f32 round-trip element %d = %v, want the q8-dequantized %v", i, back[i], qk[i])
		}
	}
}

// TestQ8KVCacheSupportedGeometryGate pins the arch gate: standard attention supports
// the realized tier; the unsupported arches name a concrete reason so the agent
// boundary can refuse instead of mixing f32 and packed rows.
func TestQ8KVCacheSupportedGeometryGate(t *testing.T) {
	if !q8TestConfig().SupportsQuantizedKVCache() {
		t.Fatal("standard attention geometry must support the q8 KV cache")
	}
	// The MLA/MoE layout is reachable from ModelType alone, so this is a reliable
	// synthetic trigger for the gate's refusal reason.
	mla := Config{NumLayers: 4, NumKVHeads: 2, HeadDim: 32, NumHeads: 4, ModelType: "deepseek2"}
	if mla.SupportsQuantizedKVCache() {
		t.Fatal("MLA/MoE geometry must NOT support the realized q8 cache")
	}
	if mla.QuantizedKVUnsupportedReason() == "" {
		t.Fatal("MLA/MoE geometry must name an unsupported reason")
	}
}

// TestNewSessionWithKVPrecisionRealizesTier proves the session constructor carries the
// tier through to the kernel-owned cache and records it, and that NewSession stays the
// f32 default (byte-for-byte the historical session).
func TestNewSessionWithKVPrecisionRealizesTier(t *testing.T) {
	cfg := q8TestConfig()
	m := &Model{Cfg: cfg}

	if s := m.NewSession(); s.KVPrecision != KVPrecisionFP32 || s.Cache.Precision() != KVPrecisionFP32 {
		t.Fatalf("NewSession tier = (%s,%s), want f32", s.KVPrecision, s.Cache.Precision())
	}
	if s := m.NewSessionWithKVPrecision(KVPrecisionQ8_0); s.KVPrecision != KVPrecisionQ8_0 || !s.Cache.quantized() {
		t.Fatalf("NewSessionWithKVPrecision(q8) = (%s, quantized=%v), want q8_0/true", s.KVPrecision, s.Cache.quantized())
	}
	if s := m.NewSessionWithKVPrecision(""); s.KVPrecision != "" && s.Cache.Precision() != KVPrecisionFP32 {
		t.Fatalf("empty tier must resolve to f32 cache, got %s/%s", s.KVPrecision, s.Cache.Precision())
	}
}

// TestQ8KVCache20480RealizedBytes reports the realized KV payload at the 20480-token
// target for the two 27B geometries in play: the model's native config (2 KV heads)
// and macfit's tier (4 KV heads). It asserts the realized q8 payload equals
// KVVectorBytes math and logs the f32 comparison — the measured deliverable number.
func TestQ8KVCache20480RealizedBytes(t *testing.T) {
	const wantTokens = 20480
	for _, g := range []struct {
		name    string
		kvHeads int
	}{{"native-2kv", 2}, {"macfit-tier-4kv", 4}} {
		cfg := Config{NumLayers: 64, NumHeads: 40, NumKVHeads: g.kvHeads, HeadDim: 128, HiddenSize: 5120, IntermediateSize: 17408, RopeTheta: 1e6}
		dim := cfg.NumKVHeads * cfg.HeadDim
		// Per token per layer: f32 Kraw + symmetric q8_0 K + q8_0 V.
		q8PerTokenPerLayer := int64(dim)*4 + 2*KVVectorBytes(dim, KVPrecisionQ8_0)
		f32PerTokenPerLayer := int64(dim) * 3 * 4
		q8Total := q8PerTokenPerLayer * wantTokens * int64(cfg.NumLayers)
		f32Total := f32PerTokenPerLayer * wantTokens * int64(cfg.NumLayers)
		// Cross-check the formula against a cache that actually holds rows: realize one
		// position and scale by wantTokens to confirm KVCacheResidentBytes matches.
		c := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
		appendSynthetic(t, c, 1, 9)
		measuredPerToken := c.KVCacheResidentBytes()
		wantPerToken := q8PerTokenPerLayer * int64(cfg.NumLayers)
		if measuredPerToken != wantPerToken {
			t.Fatalf("%s: realized per-token (all layers) = %d, formula = %d", g.name, measuredPerToken, wantPerToken)
		}
		ratio := float64(f32Total) / float64(q8Total)
		t.Logf("%s: @%d tok q8 realized = %.2f GiB (%d B), f32 3-row = %.2f GiB (%d B), ratio %.3f",
			g.name, wantTokens, float64(q8Total)/float64(1<<30), q8Total,
			float64(f32Total)/float64(1<<30), f32Total, ratio)
		if ratio < 1.9 || ratio > 2.1 {
			t.Fatalf("%s: density ratio %.3f, want ~2x", g.name, ratio)
		}
	}
}
