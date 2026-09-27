package compute

import (
	"math"
	"reflect"
	"testing"
)

// v4FlashCompressedKVLayers is the DeepSeek-V4 Flash compress_ratios schedule the fak#13555
// charge mirrors, built independently of the model package: 43 decoder layers, a 128-row window
// on every layer, ratio 0 on layers 0-1, ratio 4 on even layers >= 2 (21 layers, the overlapping
// compressor + indexer regime) and ratio 128 on odd layers >= 3 (20 layers, no indexer). Rows are
// the native f32 K=V latent of head_dim 512 (2048 B); ratio-4 layers add one 128-wide f32 indexer
// key (512 B) per compressed row.
func v4FlashCompressedKVLayers() []CompressedKVLayer {
	layers := make([]CompressedKVLayer, 43)
	for l := range layers {
		layer := CompressedKVLayer{WindowRows: 128, RowBytes: 2048}
		switch {
		case l < 2:
			layer.Ratio = 0
		case l%2 == 0:
			layer.Ratio = 4
			layer.IndexRowBytes = 512
		default:
			layer.Ratio = 128
		}
		layers[l] = layer
	}
	return layers
}

// v4FlashCompressorInflightBytes is the compressor in-flight SessionState the spec pins:
// 21 ratio-4 layers x 7 rows (2r-1, overlapping) + 20 ratio-128 layers x 127 rows (r-1), each row
// hidden_size (4096) + head_dim (512) f32 wide = 2,709,504 + 46,817,280 [SW-VERIFIED by hand].
const v4FlashCompressorInflightBytes = int64(49526784)

// v4FlashCompressedContextSizeConfig is a compressed ContextSizeConfig over the V4 Flash
// schedule. KV keeps the uniform allocation geometry (43 layers x 1 KV head x 512), which is what
// the uniform 3-row charge would size (264,192 B/token); the scratch geometry is illustrative —
// every scratch assertion compares against EstimateHALTransientMemoryPlan of the same config.
func v4FlashCompressedContextSizeConfig() ContextSizeConfig {
	return ContextSizeConfig{
		KV: KVConfig{NumLayers: 43, NumKVHeads: 1, HeadDim: 512, RopeTheta: 10000},
		SessionState: MemoryPlan{{
			Class: MemoryKVCache, Bytes: v4FlashCompressorInflightBytes,
			Detail: "deepseek-v4-compressor-inflight-rows", DType: "f32",
		}},
		Scratch: TransformerScratchConfig{
			HiddenSize: 4096, IntermediateSize: 2048, VocabSize: 129280,
			NumLayers: 43, NumHeads: 64, NumKVHeads: 1, HeadDim: 512, IncludeLogits: true,
		},
		MaxContext:   1048576,
		CompressedKV: v4FlashCompressedKVLayers(),
	}
}

// v4FlashGoldenKVBytes is the spec's golden table (window + compressed + indexer rows, f32),
// re-derived independently: bytes(t) = 43·min(128,t)·2048 + 21·ceil(t/4)·2560 + 20·ceil(t/128)·2048.
// 32,769 pins ceil (a floor would under-charge the partial group) [SW-VERIFIED].
var v4FlashGoldenKVBytes = []struct {
	tokens int
	want   int64
}{
	{1, 182784},
	{127, 12945408},
	{128, 13033472},
	{129, 13128192},
	{8192, 123994112},
	{32768, 462159872},
	{32769, 462254592},
	{131072, 1814822912},
	{1048576, 14439677952},
}

const (
	v4FlashBoundFixed    = int64(11366912) // 43·128·2048 window + 21·2560 + 20·2048 ceil slack
	v4FlashBoundPerToken = int64(13760)    // 21·(2048+512)/4 + 20·2048/128
)

func TestCompressedKVBytesV4FlashGoldens(t *testing.T) {
	layers := v4FlashCompressedKVLayers()
	for _, tc := range v4FlashGoldenKVBytes {
		if got := CompressedKVBytes(layers, tc.tokens); got != tc.want {
			t.Errorf("CompressedKVBytes(v4 flash, %d) = %d, want %d", tc.tokens, got, tc.want)
		}
	}
	for _, tok := range []int{0, -1, -4096} {
		if got := CompressedKVBytes(layers, tok); got != 0 {
			t.Errorf("CompressedKVBytes(v4 flash, %d) = %d, want 0 for a non-positive token count", tok, got)
		}
	}
	// The schedule is a charge far below the uniform 3-row layout (264,192 B/token) and above the
	// window-only undercount (43·128·6144 = 33,816,576 once t >= 128).
	if got := CompressedKVBytes(layers, 8192); got >= 43*8192*512*3*4 || got <= 33816576 {
		t.Errorf("CompressedKVBytes(v4 flash, 8192) = %d, want strictly between window-only 33816576 and uniform %d", got, int64(43*8192*512*3*4))
	}
}

func TestCompressedKVBoundV4FlashEnvelope(t *testing.T) {
	layers := v4FlashCompressedKVLayers()
	fixed, perToken := CompressedKVBound(layers)
	if fixed != v4FlashBoundFixed || perToken != v4FlashBoundPerToken {
		t.Fatalf("CompressedKVBound(v4 flash) = (%d, %d), want (%d, %d)", fixed, perToken, v4FlashBoundFixed, v4FlashBoundPerToken)
	}
	check := func(tok int) {
		if got, env := CompressedKVBytes(layers, tok), fixed+perToken*int64(tok); got > env {
			t.Errorf("bound broken at t=%d: bytes %d > fixed+perToken·t %d", tok, got, env)
		}
	}
	for tok := 0; tok <= 4096; tok++ {
		check(tok)
	}
	for _, tc := range v4FlashGoldenKVBytes {
		check(tc.tokens)
	}
}

// mixedCompressedKVLayers covers every layer kind the formulas distinguish, with small numbers so
// each expected value is derivable by hand:
//
//	A full attention, no ratio, no index        bytes 100·t                      slope 100  fixed 0
//	B full attention, indexed non-compressing   bytes (40+8)·t                   slope 48   fixed 0
//	C ratio 5 with no window (full base rows)   bytes 30·t + (30+10)·ceil(t/5)   slope 38   fixed 40
//	D window 10, indexed non-compressing        bytes (50+6)·min(10,t)           slope 0    fixed 560
//	E window 4 + ratio 7 + index                bytes 20·min(4,t) + 32·ceil(t/7) slope ceil(32/7)=5 fixed 80+32
//
// bytes(t) = 178t + 40·ceil(t/5) + 56·min(10,t) + 20·min(4,t) + 32·ceil(t/7); bound (712, 191).
func mixedCompressedKVLayers() []CompressedKVLayer {
	return []CompressedKVLayer{
		{WindowRows: 0, Ratio: 0, RowBytes: 100, IndexRowBytes: 0},
		{WindowRows: 0, Ratio: 0, RowBytes: 40, IndexRowBytes: 8},
		{WindowRows: -1, Ratio: 5, RowBytes: 30, IndexRowBytes: 10},
		{WindowRows: 10, Ratio: 0, RowBytes: 50, IndexRowBytes: 6},
		{WindowRows: 4, Ratio: 7, RowBytes: 20, IndexRowBytes: 12},
	}
}

func TestCompressedKVMixedLayerKinds(t *testing.T) {
	layers := mixedCompressedKVLayers()
	for _, tc := range []struct {
		tokens int
		want   int64
	}{
		{0, 0},
		{1, 326},   // 178 + 40 + 56 + 20 + 32
		{7, 1830},  // 1246 + 40·2 + 56·7 + 20·4 + 32·1
		{8, 2096},  // 1424 + 40·2 + 56·8 + 20·4 + 32·2
		{35, 7310}, // 6230 + 40·7 + 56·10 + 20·4 + 32·5
		{36, 7560}, // 6408 + 40·8 + 56·10 + 20·4 + 32·6
	} {
		if got := CompressedKVBytes(layers, tc.tokens); got != tc.want {
			t.Errorf("CompressedKVBytes(mixed, %d) = %d, want %d", tc.tokens, got, tc.want)
		}
	}
	fixed, perToken := CompressedKVBound(layers)
	if fixed != 712 || perToken != 191 {
		t.Fatalf("CompressedKVBound(mixed) = (%d, %d), want (712, 191)", fixed, perToken)
	}
	for tok := 0; tok <= 5000; tok++ {
		if got, env := CompressedKVBytes(layers, tok), fixed+perToken*int64(tok); got > env {
			t.Fatalf("mixed bound broken at t=%d: bytes %d > %d", tok, got, env)
		}
	}
}

func TestCompressedKVEmptyAndSaturating(t *testing.T) {
	if got := CompressedKVBytes(nil, 4096); got != 0 {
		t.Errorf("CompressedKVBytes(nil, 4096) = %d, want 0 (empty sum)", got)
	}
	if fixed, perToken := CompressedKVBound(nil); fixed != 0 || perToken != 0 {
		t.Errorf("CompressedKVBound(nil) = (%d, %d), want (0, 0)", fixed, perToken)
	}
	huge := []CompressedKVLayer{{WindowRows: 0, Ratio: 0, RowBytes: math.MaxInt64 / 2}}
	if got := CompressedKVBytes(huge, 4); got != math.MaxInt64 {
		t.Errorf("CompressedKVBytes overflow = %d, want saturation at %d", got, int64(math.MaxInt64))
	}
}

// With CompressedKV nil the new accessors are the uniform numbers verbatim, and a non-empty
// SessionState (a Qwen3.5 recurrent mixer) stays OUT of the per-stream fixed term.
func TestContextSizeKVAccessorsUncompressedUnchanged(t *testing.T) {
	plain := tinyContextSizeConfig()
	windowed := tinyContextSizeConfig()
	windowed.KV.WindowPerLayer = []int{4, 0}
	q8 := tinyContextSizeConfig()
	q8.KV.Precision = KVPrecisionQ8
	recurrent := MemoryPlan{{Class: MemoryKVCache, Bytes: 4096, Detail: "test-recurrent-state", DType: "f32"}}

	for _, tc := range []struct {
		name string
		cfg  ContextSizeConfig
	}{
		{"plain", plain},
		{"window-per-layer", windowed},
		{"q8", q8},
	} {
		for _, ss := range []MemoryPlan{nil, recurrent} {
			cfg := tc.cfg
			cfg.SessionState = ss
			for _, tok := range []int{0, 1, 3, 4, 5, 10, 4096} {
				if got, want := cfg.KVBytes(tok), EstimateKVStoreBytes(cfg.KV, tok); got != want {
					t.Errorf("%s: KVBytes(%d) = %d, want EstimateKVStoreBytes %d", tc.name, tok, got, want)
				}
			}
			if got, want := cfg.KVBytesPerToken(), EstimateKVStoreBytes(cfg.KV, 1); got != want {
				t.Errorf("%s: KVBytesPerToken = %d, want EstimateKVStoreBytes(KV, 1) %d", tc.name, got, want)
			}
			if got := cfg.FixedKVBytesPerStream(); got != 0 {
				t.Errorf("%s (session state %d B): FixedKVBytesPerStream = %d, want 0", tc.name, ss.Total(), got)
			}
			for _, tok := range []int{0, 5, 4096} {
				var want MemoryPlan
				if tok > 0 {
					want = append(want, EstimateKVStoreMemoryPlan(cfg.KV, tok)...)
				}
				want = append(want, ss...)
				want = append(want, EstimateHALTransientMemoryPlan(cfg.Scratch)...)
				if got := cfg.PerContextMemoryPlan(tok); !reflect.DeepEqual(got, want) {
					t.Errorf("%s: PerContextMemoryPlan(%d) = %#v, want %#v", tc.name, tok, got, want)
				}
			}
		}
	}
	// Independent anchor for the plain f32 tier: 2 layers x 2 heads x 8 dims x 3 rows x 4 B.
	if got := plain.KVBytesPerToken(); got != 384 {
		t.Errorf("plain KVBytesPerToken = %d, want 384", got)
	}
}

func TestContextSizeCompressedKVAccessors(t *testing.T) {
	cfg := v4FlashCompressedContextSizeConfig()
	for _, tc := range v4FlashGoldenKVBytes {
		if got := cfg.KVBytes(tc.tokens); got != tc.want {
			t.Errorf("compressed KVBytes(%d) = %d, want %d", tc.tokens, got, tc.want)
		}
		if got, want := cfg.KVBytes(tc.tokens), CompressedKVBytes(cfg.CompressedKV, tc.tokens); got != want {
			t.Errorf("compressed KVBytes(%d) = %d, want CompressedKVBytes %d", tc.tokens, got, want)
		}
	}
	if got := cfg.KVBytes(0); got != 0 {
		t.Errorf("compressed KVBytes(0) = %d, want 0", got)
	}
	if got := cfg.KVBytesPerToken(); got != v4FlashBoundPerToken {
		t.Errorf("compressed KVBytesPerToken = %d, want %d", got, v4FlashBoundPerToken)
	}
	if got := cfg.FixedKVBytesPerStream(); got != v4FlashBoundFixed+cfg.SessionState.Total() || got != 60893696 {
		t.Errorf("compressed FixedKVBytesPerStream = %d, want %d + %d = 60893696", got, v4FlashBoundFixed, cfg.SessionState.Total())
	}
	// The per-stream envelope covers KV plus the in-flight SessionState at every size.
	for tok := 0; tok <= 4096; tok++ {
		if lhs, rhs := cfg.KVBytes(tok)+cfg.SessionState.Total(), cfg.FixedKVBytesPerStream()+cfg.KVBytesPerToken()*int64(tok); lhs > rhs {
			t.Fatalf("per-stream envelope broken at t=%d: %d > %d", tok, lhs, rhs)
		}
	}
}

func TestContextSizeCompressedPerContextMemoryPlan(t *testing.T) {
	cfg := v4FlashCompressedContextSizeConfig()
	scratch := EstimateHALTransientMemoryPlan(cfg.Scratch)
	if len(scratch) == 0 {
		t.Fatalf("fixture scratch geometry must yield HAL demands")
	}

	got := cfg.PerContextMemoryPlan(8192)
	want := MemoryPlan{{Class: MemoryKVCache, Bytes: 123994112, Detail: "compressed-kv-rows", DType: "f32"}}
	want = append(want, cfg.SessionState...)
	want = append(want, scratch...)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("compressed PerContextMemoryPlan(8192) = %#v, want %#v", got, want)
	}
	rows := 0
	for _, d := range got {
		if d.Detail == "hal-kv-store" {
			t.Errorf("compressed plan must not carry the uniform hal-kv-store demand: %#v", d)
		}
		if d.Class == MemoryKVCache && d.Bytes == 123994112 {
			rows++
		}
	}
	if rows != 1 {
		t.Errorf("compressed plan carries %d KV row demands of 123994112 B, want exactly 1", rows)
	}

	// tokens <= 0 omits the KV demand; SessionState and scratch still apply.
	zeroWant := append(append(MemoryPlan(nil), cfg.SessionState...), scratch...)
	if got := cfg.PerContextMemoryPlan(0); !reflect.DeepEqual(got, zeroWant) {
		t.Fatalf("compressed PerContextMemoryPlan(0) = %#v, want %#v", got, zeroWant)
	}
}

// v4FlashFitInputs returns the compressed config, an 80 GiB device weight plan, the HAL scratch
// total, and the non-KV bytes an avail must cover before any KV row fits.
func v4FlashFitInputs() (cfg ContextSizeConfig, weights MemoryPlan, scratch, nonKV int64) {
	cfg = v4FlashCompressedContextSizeConfig()
	weights = MemoryPlan{{Class: MemoryWeights, Bytes: int64(80) << 30, Scope: MemoryScopeDevice}}
	scratch = EstimateHALTransientMemoryPlan(cfg.Scratch).Total()
	nonKV = weights.DeviceTotal() + cfg.SessionState.DeviceTotal() + scratch
	return cfg, weights, scratch, nonKV
}

// kvBytesAt100000 is the V4 Flash schedule at 100,000 tokens: 11,272,192 window + 21·25,000·2,560
// + 20·782·2,048 = 1,387,302,912 [SW-VERIFIED by hand]. 100,000 ≡ 0 (mod 4), so the next token
// opens a fresh ratio-4 group and costs 21·2,560 = 53,760 B more.
const kvBytesAt100000 = int64(1387302912)

func TestLargestFittingCompressedInteriorAnswer(t *testing.T) {
	cfg, weights, _, nonKV := v4FlashFitInputs()
	if got := cfg.KVBytes(100000); got != kvBytesAt100000 {
		t.Fatalf("KVBytes(100000) = %d, want %d", got, kvBytesAt100000)
	}
	for _, tc := range []struct {
		name  string
		slack int64
		want  int
	}{
		{"exact", 0, 100000},
		{"one byte short of the next group", 53759, 100000},
		// The next group's bytes buy four tokens: 100,001..100,004 share ceil(t/4) = 25,001.
		{"next group", 53760, 100004},
	} {
		avail := nonKV + kvBytesAt100000 + tc.slack
		got := LargestFittingContextTokens(cfg, weights, avail)
		if got != tc.want {
			t.Errorf("%s: LargestFittingContextTokens = %d, want %d", tc.name, got, tc.want)
		}
		if cfg.KVBytes(got)+nonKV > avail {
			t.Errorf("%s: t=%d does not fit: %d + %d > %d", tc.name, got, cfg.KVBytes(got), nonKV, avail)
		}
		if cfg.KVBytes(got+1)+nonKV <= avail {
			t.Errorf("%s: t=%d is not the largest: t+1 also fits (%d + %d <= %d)", tc.name, got, cfg.KVBytes(got+1), nonKV, avail)
		}
		tokens, plan := AutoSizeContextPlan(cfg, weights, avail, -1)
		if tokens != got {
			t.Errorf("%s: AutoSizeContextPlan tokens = %d, want LargestFittingContextTokens %d", tc.name, tokens, got)
		}
		if want := cfg.PerContextMemoryPlan(tokens); !reflect.DeepEqual(plan, want) {
			t.Errorf("%s: AutoSizeContextPlan plan = %#v, want %#v", tc.name, plan, want)
		}
	}

	// An explicit override above the fitted bound clamps down to it; one below is verbatim.
	avail := nonKV + kvBytesAt100000
	if tokens, _ := AutoSizeContextPlan(cfg, weights, avail, 1<<20); tokens != 100000 {
		t.Errorf("override 1<<20: tokens = %d, want clamp to 100000", tokens)
	}
	if tokens, _ := AutoSizeContextPlan(cfg, weights, avail, 4096); tokens != 4096 {
		t.Errorf("override 4096: tokens = %d, want verbatim 4096", tokens)
	}
}

func TestAutoSizeCompressedFloorCeilingAndUnknown(t *testing.T) {
	cfg, weights, _, nonKV := v4FlashFitInputs()
	fullKV := int64(14439677952) // golden at 1,048,576

	for _, tc := range []struct {
		name  string
		avail int64
		want  int
	}{
		{"cannot even hold the non-KV demands", weights.DeviceTotal() + 1, MinAutoContextTokens},
		{"room for ~100 tokens clamps to the floor", nonKV + 10191360, MinAutoContextTokens},
		{"exactly the full window", nonKV + fullKV, 1048576},
		// 1,048,573..1,048,576 share every ceil, so one byte short drops the whole group.
		{"one byte short of the full window", nonKV + fullKV - 1, 1048572},
		{"huge", int64(1) << 62, 1048576},
	} {
		if got := LargestFittingContextTokens(cfg, weights, tc.avail); got != tc.want {
			t.Errorf("%s: LargestFittingContextTokens = %d, want %d", tc.name, got, tc.want)
		}
		if tokens, _ := AutoSizeContextPlan(cfg, weights, tc.avail, -1); tokens != tc.want {
			t.Errorf("%s: AutoSizeContextPlan tokens = %d, want %d", tc.name, tokens, tc.want)
		}
	}

	for _, avail := range []int64{0, FreeUnknown} {
		if got := LargestFittingContextTokens(cfg, weights, avail); got != cfg.MaxContext {
			t.Errorf("avail %d: LargestFittingContextTokens = %d, want MaxContext %d", avail, got, cfg.MaxContext)
		}
		tokens, plan := AutoSizeContextPlan(cfg, weights, avail, -1)
		if tokens != cfg.MaxContext {
			t.Errorf("avail %d: AutoSizeContextPlan tokens = %d, want MaxContext %d", avail, tokens, cfg.MaxContext)
		}
		if len(plan) == 0 || plan[0].Bytes != fullKV || plan[0].Detail != "compressed-kv-rows" {
			t.Errorf("avail %d: full-window plan head = %#v, want compressed-kv-rows of %d B", avail, plan, fullKV)
		}
	}

	// A declared window below the floor caps the floor, exactly as on the uniform branch.
	small := cfg
	small.MaxContext = 256
	if got := LargestFittingContextTokens(small, weights, weights.DeviceTotal()+1); got != 256 {
		t.Errorf("MaxContext 256 below floor: LargestFittingContextTokens = %d, want 256", got)
	}
}

// A compressed geometry that sizes to zero bytes cannot be inverted: fail open to the full window.
func TestLargestFittingCompressedZeroBytesFailsOpen(t *testing.T) {
	cfg, weights, _, _ := v4FlashFitInputs()
	cfg.CompressedKV = []CompressedKVLayer{{WindowRows: 128, Ratio: 4}}
	if got := cfg.KVBytes(cfg.MaxContext); got != 0 {
		t.Fatalf("zero-row geometry KVBytes(MaxContext) = %d, want 0", got)
	}
	if got := LargestFittingContextTokens(cfg, weights, 1); got != cfg.MaxContext {
		t.Errorf("zero-row geometry: LargestFittingContextTokens = %d, want fail-open MaxContext %d", got, cfg.MaxContext)
	}
}

// Host-scoped weights compete with compressed KV only on a shared (APU) pool, as on the uniform
// branch. 688,015,360 = KVBytes(100,000) - KVBytes(50,000) (699,287,552), and 50,000 ≡ 0 (mod 4).
func TestLargestFittingCompressedPoolScope(t *testing.T) {
	cfg, weights, _, nonKV := v4FlashFitInputs()
	const hostBytes = int64(688015360)
	withHost := append(append(MemoryPlan(nil), weights...), MemoryDemand{Class: MemoryWeights, Bytes: hostBytes, Scope: MemoryScopeHost})
	avail := nonKV + kvBytesAt100000

	if got := LargestFittingContextTokens(cfg, withHost, avail); got != 100000 {
		t.Errorf("discrete pool: LargestFittingContextTokens = %d, want 100000 (host weights not charged)", got)
	}
	cfg.PoolSharedWithHost = true
	if got := LargestFittingContextTokens(cfg, withHost, avail); got != 50000 {
		t.Errorf("shared pool: LargestFittingContextTokens = %d, want 50000 (host weights charged)", got)
	}
}

// The uncompressed derivation is untouched: it stays the closed form
// (avail - weights - SessionState - scratch) / perToken, clamped to [floor, MaxContext].
func TestLargestFittingUncompressedClosedFormUnchanged(t *testing.T) {
	qwen := qwen36_27BContextSizeConfig()
	qwen.SessionState = MemoryPlan{{Class: MemoryKVCache, Bytes: 12345, Detail: "test-recurrent-state", DType: "f32"}}
	const qwenPerToken = int64(786432)
	qwenWeights := MemoryPlan{{Class: MemoryWeights, Bytes: int64(20) << 30, Scope: MemoryScopeDevice}}
	qwenScratch := EstimateHALTransientMemoryPlan(qwen.Scratch).Total()

	// The same V4 Flash inputs WITHOUT the schedule: uniform 43·512·3·4 = 264,192 B/token, so the
	// 1,387,302,912 B of KV budget buys floor(1,387,302,912 / 264,192) = 5,251 tokens.
	v4, v4Weights, v4Scratch, _ := v4FlashFitInputs()
	v4.CompressedKV = nil

	for _, tc := range []struct {
		name     string
		cfg      ContextSizeConfig
		weights  MemoryPlan
		scratch  int64
		perToken int64
		kvBudget int64
		want     int
	}{
		{"qwen36-27b remainder", qwen, qwenWeights, qwenScratch, qwenPerToken, 12345*qwenPerToken + 777, 12345},
		{"v4 flash uniform", v4, v4Weights, v4Scratch, 264192, kvBytesAt100000, 5251},
	} {
		if got := EstimateKVStoreBytes(tc.cfg.KV, 1); got != tc.perToken {
			t.Fatalf("%s: kv/token = %d, want %d", tc.name, got, tc.perToken)
		}
		avail := tc.weights.DeviceTotal() + tc.cfg.SessionState.DeviceTotal() + tc.scratch + tc.kvBudget
		closed := int((avail - tc.weights.DeviceTotal() - tc.cfg.SessionState.DeviceTotal() - tc.scratch) / tc.perToken)
		if closed != tc.want {
			t.Fatalf("%s: closed form = %d, want %d", tc.name, closed, tc.want)
		}
		if got := LargestFittingContextTokens(tc.cfg, tc.weights, avail); got != tc.want {
			t.Errorf("%s: LargestFittingContextTokens = %d, want closed form %d", tc.name, got, tc.want)
		}
		if tokens, _ := AutoSizeContextPlan(tc.cfg, tc.weights, avail, -1); tokens != tc.want {
			t.Errorf("%s: AutoSizeContextPlan tokens = %d, want closed form %d", tc.name, tokens, tc.want)
		}
	}
}
