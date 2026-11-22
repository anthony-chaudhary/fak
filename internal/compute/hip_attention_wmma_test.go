package compute

// Host-side witness for the gfx1151 WMMA attention port. The fixture
// generator, the case shapes, the comparison metric and the FP64 oracle are
// ported from gufo-org/gufo tests/models/qwen/hip/attention/
// wmma_prefill_ops_test.cpp at commit
// d0f7c3a023a11af81161264eeb24e9a97aa7498a (MIT License, Copyright (c) 2026
// gufo contributors). Upstream runs them against the device kernel; here they
// run against the CPU mirror of that kernel, so they prove the tiling and
// masking logic, not the silicon.

import (
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

const (
	wmmaAttnWidth = HIPWMMAQueryHeads * HIPWMMAHeadDim
	wmmaKVWidth   = HIPWMMAKVHeads * HIPWMMAHeadDim
	// Upstream's bound for FP16-operand kernels against double precision.
	wmmaTolerance = 2e-3
)

// wmmaRng is upstream's fixture LCG.
type wmmaRng struct{ state uint32 }

func (r *wmmaRng) uniform(lo, hi float32) float32 {
	r.state = r.state*1664525 + 1013904223
	u := float32((r.state>>8)&0xFFFF) / 65535.0
	return lo + u*(hi-lo)
}

type wmmaCase struct {
	start, batch int
	lse          bool
	keyBegin     int
}

func (c wmmaCase) String() string {
	return fmt.Sprintf("start=%d,batch=%d,lse=%v,key_begin=%d", c.start, c.batch, c.lse, c.keyBegin)
}

// wmmaFutureRows of populated cache sit past every chunk, so a kernel that
// failed to mask beyond the chunk end would read real values, not padding.
const wmmaFutureRows = 16

func buildWMMACase(c wmmaCase) HIPAttentionWMMAInput {
	rng := &wmmaRng{state: 0xA11CE ^ uint32(c.start)*131 ^ uint32(c.batch)}
	fill := func(n int, lo, hi float32) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.uniform(lo, hi)
		}
		return s
	}
	q := fill(c.batch*wmmaAttnWidth, -1, 1)
	gate := fill(c.batch*wmmaAttnWidth, -2, 2)
	chunkK := fill(c.batch*wmmaKVWidth, -1, 1)
	chunkV := fill(c.batch*wmmaKVWidth, -1, 1)
	prefixK := fill(c.start*wmmaKVWidth, -1, 1)
	prefixV := fill(c.start*wmmaKVWidth, -1, 1)
	futureK := fill(wmmaFutureRows*wmmaKVWidth, -1, 1)
	futureV := fill(wmmaFutureRows*wmmaKVWidth, -1, 1)
	return HIPAttentionWMMAInput{
		Q: q, Gate: gate,
		K:        append(append(prefixK, chunkK...), futureK...),
		V:        append(append(prefixV, chunkV...), futureV...),
		StartPos: c.start, BatchSize: c.batch, KeyBegin: c.keyBegin, WantLSE: c.lse,
	}
}

// wmmaOracleFP64 is upstream's independent FP64 attention over the original
// FP32 inputs, before any operand is rounded to FP16, widened from upstream's
// two sampled rows of head zero to every row of every head.
func wmmaOracleFP64(in HIPAttentionWMMAInput) (out, lse []float32) {
	gqa := HIPWMMAQueryHeads / HIPWMMAKVHeads
	out = make([]float32, in.BatchSize*wmmaAttnWidth)
	lse = make([]float32, HIPWMMAQueryHeads*in.BatchSize)
	for row := 0; row < in.BatchSize; row++ {
		end := in.StartPos + row + 1
		for head := 0; head < HIPWMMAQueryHeads; head++ {
			kvOff := (head / gqa) * HIPWMMAHeadDim
			qBase := row*wmmaAttnWidth + head*HIPWMMAHeadDim
			scores := make([]float64, end)
			maximum := math.Inf(-1)
			for key := in.KeyBegin; key < end; key++ {
				dot := 0.0
				for dim := 0; dim < HIPWMMAHeadDim; dim++ {
					dot += float64(in.Q[qBase+dim]) * float64(in.K[key*wmmaKVWidth+kvOff+dim])
				}
				scores[key] = dot / math.Sqrt(float64(HIPWMMAHeadDim))
				maximum = math.Max(maximum, scores[key])
			}
			sum := 0.0
			for key := in.KeyBegin; key < end; key++ {
				scores[key] = math.Exp(scores[key] - maximum)
				sum += scores[key]
			}
			for dim := 0; dim < HIPWMMAHeadDim; dim++ {
				value := 0.0
				for key := in.KeyBegin; key < end; key++ {
					value += scores[key] * float64(in.V[key*wmmaKVWidth+kvOff+dim])
				}
				if sum > 0 {
					value /= sum
				} else {
					value = 0
				}
				if !in.WantLSE {
					value /= 1 + math.Exp(-float64(in.Gate[qBase+dim]))
				}
				out[qBase+dim] = float32(value)
			}
			stat := math.Inf(-1)
			if sum > 0 {
				stat = maximum + math.Log(sum)
			}
			lse[head*in.BatchSize+row] = float32(stat)
		}
	}
	return out, lse
}

// wmmaCompare is upstream's Compare: worst absolute error relative to the
// largest reference magnitude, with any non-finite candidate a failure. A
// reference of -Inf (an empty attention window) must be matched exactly.
func wmmaCompare(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d elements, want %d", label, len(got), len(want))
	}
	maxAbs, magnitude := 0.0, 0.0
	for i := range want {
		if math.IsInf(float64(want[i]), -1) {
			if !math.IsInf(float64(got[i]), -1) {
				t.Fatalf("%s: element %d = %v, want -Inf", label, i, got[i])
			}
			continue
		}
		if math.IsInf(float64(got[i]), 0) || math.IsNaN(float64(got[i])) {
			t.Fatalf("%s: element %d is non-finite (%v)", label, i, got[i])
		}
		maxAbs = math.Max(maxAbs, math.Abs(float64(got[i])-float64(want[i])))
		magnitude = math.Max(magnitude, math.Abs(float64(want[i])))
	}
	rel := maxAbs
	if magnitude > 0 {
		rel = maxAbs / magnitude
	}
	if rel > wmmaTolerance {
		t.Fatalf("%s: max_abs=%g magnitude=%g rel=%g exceeds %g", label, maxAbs, magnitude, rel, wmmaTolerance)
	}
}

func wmmaSameBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d elements, want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: row=%d offset=%d got=%x want=%x", label, i/wmmaAttnWidth, i%wmmaAttnWidth,
				math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func wmmaRun(t *testing.T, in HIPAttentionWMMAInput) (out, lse []float32) {
	t.Helper()
	out, lse, err := HIPAttentionWMMAReference(in)
	if err != nil {
		t.Fatal(err)
	}
	return out, lse
}

func readWMMASource(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(name)
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return strings.ReplaceAll(string(b), "\r\n", "\n")
}

func TestHIPAttentionWMMAContract(t *testing.T) {
	hip := readWMMASource(t, "rocm_kernels.hip")

	t.Run("kernel carries the upstream notice", func(t *testing.T) {
		for _, want := range []string{
			"gufo-org/gufo, src/models/qwen/hip/kernels/attention_wmma.hip",
			"d0f7c3a023a11af81161264eeb24e9a97aa7498a",
			"MIT License",
			"Copyright (c) 2026 gufo contributors",
			"The above copyright notice and this permission notice shall be included in all",
		} {
			if !strings.Contains(hip, want) {
				t.Errorf("rocm_kernels.hip lost %q", want)
			}
		}
	})

	t.Run("kernel constants equal the mirror constants", func(t *testing.T) {
		for name, want := range map[string]int{
			"kHeadDim":       HIPWMMAHeadDim,
			"kQueryHeads":    HIPWMMAQueryHeads,
			"kKvHeads":       HIPWMMAKVHeads,
			"kWmmaHeads":     HIPWMMAHeadsPerBlock,
			"kWmmaQueryRows": HIPWMMAQueryRows,
			"kWmmaKeys":      HIPWMMAKeyTile,
		} {
			m := regexp.MustCompile(`constexpr std::uint32_t `+name+` =\s*(\d+);`).FindAllStringSubmatch(hip, -1)
			if len(m) != 1 {
				t.Errorf("%s: %d definitions in rocm_kernels.hip, want 1", name, len(m))
				continue
			}
			if got, _ := strconv.Atoi(m[0][1]); got != want {
				t.Errorf("%s = %d in the kernel, %d in the mirror", name, got, want)
			}
		}
		if !strings.Contains(hip, "constexpr float attention_scale = 1.0F / 16.0F;") || hipWMMAScale != 0.0625 {
			t.Error("kernel and mirror disagree on the folded 1/16 attention scale")
		}
		if HIPWMMAHeadDim != 256 || HIPWMMAQueryHeads != 24 || HIPWMMAKVHeads != 4 || HIPWMMAArch != "gfx1151" {
			t.Error("the pinned geometry is exactly gfx1151, 24 query heads, 4 KV heads, head size 256")
		}
	})

	t.Run("device guard implements the selection contract", func(t *testing.T) {
		start := strings.Index(hip, "bool wmma_attention_selected(")
		if start < 0 {
			t.Fatal("rocm_kernels.hip has no wmma_attention_selected")
		}
		guard := hip[start:]
		guard = guard[:strings.Index(guard, "\n}\n")]
		for _, want := range []string{
			`getenv("` + HIPWMMAOptInEnv + `")`,
			`strcmp(opt_in, "1") != 0) return false;`,
			`strncmp(active_gfx, "` + HIPWMMAArch + `", 7) != 0`,
			"q_heads == static_cast<int>(kQueryHeads)",
			"kv_heads == static_cast<int>(kKvHeads)",
			"head_dim == static_cast<int>(kHeadDim)",
			"group_size == static_cast<int>(kGqa)",
			"positions >= 1",
			"scale == 1.0F / 16.0F",
		} {
			if !strings.Contains(guard, want) {
				t.Errorf("device guard lost clause %q", want)
			}
		}
		if !strings.Contains(hip, "strncpy(active_gfx, prop.gcnArchName, sizeof(active_gfx) - 1);") {
			t.Error("frocm_init no longer records the device arch the guard reads")
		}
	})

	t.Run("runtime call path reaches the kernel only behind the guard", func(t *testing.T) {
		backend := readWMMASource(t, "rocm_backend.go")
		method := backend[strings.Index(backend, "func (r *rocmBackend) Attention("):]
		method = method[:strings.Index(method, "\n}\n")]
		if !strings.Contains(method, "C.frocm_attention_f32(") {
			t.Fatal("rocmBackend.Attention no longer enters frocm_attention_f32; the WMMA route is unreachable")
		}

		entry := hip[strings.Index(hip, `extern "C" int frocm_attention_f32(`):]
		entry = entry[:strings.Index(entry, "\n}\n")]
		order := []string{
			`return failf(-1, "attention_f32: invalid pointer or geometry");`,
			"#if defined(" + HIPWMMABuildMacro + ")",
			"if (wmma_attention_selected(q_heads, kv_heads, positions, head_dim,",
			"WmmaCausalAttention<kWmmaQueryRows, kWmmaKeys><<<grid, 256, 0, stream>>>(",
			"q, nullptr, k, v, out, static_cast<std::uint32_t>(positions - 1),",
			"batch_size, nullptr, 0);",
			"#endif",
			"attention<<<(q_heads + 63) / 64, 64, 0, stream>>>(",
		}
		at := 0
		for _, want := range order {
			i := strings.Index(entry[at:], want)
			if i < 0 {
				t.Fatalf("frocm_attention_f32 lost %q after offset %d; guard, WMMA launch and generic fallback must stay in this order", want, at)
			}
			at += i + len(want)
		}
		if n := strings.Count(hip, "WmmaCausalAttention<kWmmaQueryRows"); n != 1 {
			t.Errorf("%d WMMA launch sites, want exactly the guarded one", n)
		}
		if strings.Count(hip, "#if defined("+HIPWMMABuildMacro+")") != 2 ||
			!strings.Contains(hip, "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32") {
			t.Error("the kernel and its launch must both sit behind the build macro")
		}
	})

	t.Run("FP16 operand rounding is nearest-even", func(t *testing.T) {
		pow := func(e int) float32 { return float32(math.Ldexp(1, e)) }
		for _, tc := range []struct{ in, want float32 }{
			{1, 1},
			{1 + pow(-11), 1},                       // tie, even mantissa stays
			{1 + 3*pow(-11), 1 + 4*pow(-11)},        // tie, odd mantissa rounds up
			{1 + pow(-11) + pow(-20), 1 + pow(-10)}, // above the tie
			{0.1, 0.0999755859375},                  // 0x2E66
			{-0.1, -0.0999755859375},
			{65504, 65504}, // largest finite half
			{65519.996, 65504},
			{65520, float32(math.Inf(1))},
			{pow(-24), pow(-24)}, // smallest subnormal
			{pow(-25), 0},        // tie to even zero
			{1.5 * pow(-25), pow(-24)},
			{pow(-14) - pow(-25), pow(-14)}, // subnormal carry into the normal range
		} {
			if got := hipWMMAHalf(tc.in); got != tc.want {
				t.Errorf("hipWMMAHalf(%g) = %g, want %g", tc.in, got, tc.want)
			}
		}
		if got := hipWMMAHalf(float32(math.Copysign(0, -1))); got != 0 || !math.Signbit(float64(got)) {
			t.Errorf("hipWMMAHalf(-0) = %g, want -0", got)
		}
	})

	// Upstream's RunCase list, scaled to depths a CPU mirror covers in seconds
	// while keeping every tiling boundary it names.
	cases := []wmmaCase{
		{start: 0, batch: 100},                          // ragged: not a multiple of the 32-row block
		{start: 0, batch: 16},                           // exactly one key tile, prefetch never fires
		{start: 0, batch: 32},                           // exactly two
		{start: 0, batch: 64, lse: true},                // LSE at depth zero
		{start: 48, batch: 40},                          // prefix and diagonal in one pass
		{start: 37, batch: 50},                          // depth not a multiple of the 16-key tile
		{start: 48, batch: 40, lse: true},               // LSE, which also suppresses the gate
		{start: 64, batch: 40, lse: true, keyBegin: 16}, // aligned key window
		{start: 64, batch: 40, lse: true, keyBegin: 17}, // unaligned key window
		{start: 64, batch: 40, keyBegin: 80},            // window opens inside the chunk
	}
	for _, c := range cases {
		c := c
		t.Run(c.String(), func(t *testing.T) {
			in := buildWMMACase(c)
			got, gotLSE := wmmaRun(t, in)
			want, wantLSE := wmmaOracleFP64(in)
			wmmaCompare(t, "FP64 original-input oracle", got, want)
			if c.lse {
				wmmaCompare(t, "log-sum-exp", gotLSE, wantLSE)
			} else if gotLSE != nil {
				t.Fatal("LSE returned without being requested")
			}

			// Rows whose window is empty must be exact zeros, not a 0/0.
			for row := 0; row < c.batch && c.start+row < c.keyBegin; row++ {
				for _, v := range got[row*wmmaAttnWidth : (row+1)*wmmaAttnWidth] {
					if v != 0 {
						t.Fatalf("row %d has an empty window but a non-zero context %v", row, v)
					}
				}
			}

			again, againLSE := wmmaRun(t, in)
			wmmaSameBits(t, "replay", again, got)
			if c.lse {
				wmmaSameBits(t, "replay log-sum-exp", againLSE, gotLSE)
			}

			if c.lse || c.batch <= 1 {
				return
			}
			// The whole chunk is already in the cache, so a sub-chunk sees its
			// successors as populated, masked future keys. That must have
			// exactly the effect of absent padding.
			for _, budget := range []int{31, 512} {
				chunked := make([]float32, len(got))
				for offset := 0; offset < c.batch; offset += budget {
					count := budget
					if rest := c.batch - offset; rest < count {
						count = rest
					}
					part, _ := wmmaRun(t, HIPAttentionWMMAInput{
						Q:    in.Q[offset*wmmaAttnWidth : (offset+count)*wmmaAttnWidth],
						Gate: in.Gate[offset*wmmaAttnWidth : (offset+count)*wmmaAttnWidth],
						K:    in.K, V: in.V,
						StartPos: c.start + offset, BatchSize: count, KeyBegin: c.keyBegin,
					})
					copy(chunked[offset*wmmaAttnWidth:], part)
				}
				wmmaSameBits(t, fmt.Sprintf("chunk budget %d", budget), chunked, got)
			}
		})
	}

	t.Run("masked future keys are never read as values", func(t *testing.T) {
		in := buildWMMACase(wmmaCase{start: 5, batch: 20})
		want, _ := wmmaRun(t, in)
		poisoned := in
		poisoned.K = append([]float32(nil), in.K...)
		poisoned.V = append([]float32(nil), in.V...)
		for i := (in.StartPos + in.BatchSize) * wmmaKVWidth; i < len(poisoned.K); i++ {
			poisoned.K[i] = float32(math.NaN())
			poisoned.V[i] = float32(math.Inf(1))
		}
		got, _ := wmmaRun(t, poisoned)
		wmmaSameBits(t, "poisoned future cache", got, want)
	})

	t.Run("inputs off the pinned geometry are refused", func(t *testing.T) {
		in := buildWMMACase(wmmaCase{start: 2, batch: 3})
		for name, mutate := range map[string]func(*HIPAttentionWMMAInput){
			"short q":        func(i *HIPAttentionWMMAInput) { i.Q = i.Q[:len(i.Q)-1] },
			"short gate":     func(i *HIPAttentionWMMAInput) { i.Gate = i.Gate[:1] },
			"short k cache":  func(i *HIPAttentionWMMAInput) { i.K = i.K[:(i.StartPos+i.BatchSize)*wmmaKVWidth-1] },
			"short v cache":  func(i *HIPAttentionWMMAInput) { i.V = i.V[:wmmaKVWidth] },
			"empty batch":    func(i *HIPAttentionWMMAInput) { i.BatchSize, i.Q, i.Gate = 0, nil, nil },
			"negative start": func(i *HIPAttentionWMMAInput) { i.StartPos = -1 },
			"negative begin": func(i *HIPAttentionWMMAInput) { i.KeyBegin = -1 },
		} {
			bad := in
			mutate(&bad)
			if _, _, err := HIPAttentionWMMAReference(bad); !errors.Is(err, ErrHIPWMMAShape) {
				t.Errorf("%s: err = %v, want ErrHIPWMMAShape", name, err)
			}
		}
	})
}

// wmmaGenericAttentionFP64 is the `attention` kernel of rocm_kernels.hip, the
// fallback every unselected call keeps, transcribed in FP64.
func wmmaGenericAttentionFP64(q, k, v []float32, qHeads, kvHeads, positions, headDim, group int, scale float64) []float32 {
	out := make([]float32, qHeads*headDim)
	stride := kvHeads * headDim
	for qh := 0; qh < qHeads; qh++ {
		kvh := qh / group
		acc := make([]float64, headDim)
		maxScore, denom := math.Inf(-1), 0.0
		for p := 0; p < positions; p++ {
			dot := 0.0
			for d := 0; d < headDim; d++ {
				dot += float64(q[qh*headDim+d]) * float64(k[p*stride+kvh*headDim+d])
			}
			score := dot * scale
			nextMax := math.Max(maxScore, score)
			correction := math.Exp(maxScore - nextMax)
			probability := math.Exp(score - nextMax)
			denom = denom*correction + probability
			for d := 0; d < headDim; d++ {
				acc[d] = acc[d]*correction + probability*float64(v[p*stride+kvh*headDim+d])
			}
			maxScore = nextMax
		}
		for d := 0; d < headDim; d++ {
			if denom > 0 {
				out[qh*headDim+d] = float32(acc[d] / denom)
			}
		}
	}
	return out
}

func TestHIPAttentionWMMAFallback(t *testing.T) {
	pinned := HIPAttentionWMMACall{
		Arch: "gfx1151", QueryHeads: 24, KVHeads: 4, HeadDim: 256, GroupSize: 6, Positions: 1, Scale: 0.0625,
	}

	t.Run("default selection is the generic kernel", func(t *testing.T) {
		if v, set := os.LookupEnv(HIPWMMAOptInEnv); set {
			t.Setenv(HIPWMMAOptInEnv, v)
			os.Unsetenv(HIPWMMAOptInEnv)
		}
		if HIPAttentionWMMAOptIn() {
			t.Fatal("the WMMA route is opted in with the environment unset")
		}
		if got := SelectHIPAttentionRoute(HIPAttentionWMMAOptIn(), pinned); got != HIPAttentionRouteGeneric {
			t.Fatalf("route = %q for the pinned geometry without opt-in, want generic", got)
		}
	})

	t.Run("only the literal 1 opts in", func(t *testing.T) {
		for value, want := range map[string]bool{"": false, "0": false, "true": false, "yes": false, " 1": false, "1": true} {
			t.Setenv(HIPWMMAOptInEnv, value)
			if got := HIPAttentionWMMAOptIn(); got != want {
				t.Errorf("%s=%q: opt-in = %v, want %v", HIPWMMAOptInEnv, value, got, want)
			}
		}
	})

	t.Run("opt-in selects WMMA only at the pinned geometry", func(t *testing.T) {
		if got := SelectHIPAttentionRoute(true, pinned); got != HIPAttentionRouteWMMA {
			t.Fatalf("route = %q for the opted-in pinned geometry, want wmma", got)
		}
		suffixed, deep := pinned, pinned
		suffixed.Arch = "gfx1151:sramecc-:xnack-"
		deep.Positions = 32768
		for name, c := range map[string]HIPAttentionWMMACall{"feature suffix": suffixed, "deep context": deep} {
			if got := SelectHIPAttentionRoute(true, c); got != HIPAttentionRouteWMMA {
				t.Errorf("%s: route = %q, want wmma", name, got)
			}
		}
		for name, mutate := range map[string]func(*HIPAttentionWMMACall){
			"gfx1100 (wave32 WMMA, wrong part)": func(c *HIPAttentionWMMACall) { c.Arch = "gfx1100" },
			"gfx942 (wave64)":                   func(c *HIPAttentionWMMACall) { c.Arch = "gfx942" },
			"gfx11510":                          func(c *HIPAttentionWMMACall) { c.Arch = "gfx11510" },
			"upper-case arch":                   func(c *HIPAttentionWMMACall) { c.Arch = "GFX1151" },
			"empty arch":                        func(c *HIPAttentionWMMACall) { c.Arch = "" },
			"25 query heads":                    func(c *HIPAttentionWMMACall) { c.QueryHeads = 25 },
			"16 query heads, group 4":           func(c *HIPAttentionWMMACall) { c.QueryHeads, c.GroupSize = 16, 4 },
			"8 KV heads, group 3":               func(c *HIPAttentionWMMACall) { c.KVHeads, c.GroupSize = 8, 3 },
			"head size 128":                     func(c *HIPAttentionWMMACall) { c.HeadDim = 128 },
			"head size 128 with its own scale":  func(c *HIPAttentionWMMACall) { c.HeadDim, c.Scale = 128, float32(1/math.Sqrt(128)) },
			"group 4":                           func(c *HIPAttentionWMMACall) { c.GroupSize = 4 },
			"empty cache":                       func(c *HIPAttentionWMMACall) { c.Positions = 0 },
			"scale 1/8":                         func(c *HIPAttentionWMMACall) { c.Scale = 0.125 },
			"scale one ulp off":                 func(c *HIPAttentionWMMACall) { c.Scale = math.Nextafter32(0.0625, 1) },
		} {
			c := pinned
			mutate(&c)
			if got := SelectHIPAttentionRoute(true, c); got != HIPAttentionRouteGeneric {
				t.Errorf("%s: route = %q, want generic", name, got)
			}
			if got := SelectHIPAttentionRoute(false, c); got != HIPAttentionRouteGeneric {
				t.Errorf("%s without opt-in: route = %q, want generic", name, got)
			}
		}
	})

	// The reached seam is one decode row over the resident cache. On the WMMA
	// route that row must agree with the generic kernel it replaces, and must
	// be the same row the batched prefill form would have produced.
	t.Run("decode row agrees with the generic kernel and the prefill row", func(t *testing.T) {
		const depth = 34 // two full key tiles and a partial third
		in := buildWMMACase(wmmaCase{start: 0, batch: depth})
		in.Gate = nil
		prefill, _ := wmmaRun(t, in)
		for _, positions := range []int{1, 15, 16, 17, 32, 33, 34} {
			row := positions - 1
			q := in.Q[row*wmmaAttnWidth : (row+1)*wmmaAttnWidth]
			if got := SelectHIPAttentionRoute(true, HIPAttentionWMMACall{
				Arch: "gfx1151", QueryHeads: 24, KVHeads: 4, HeadDim: 256, GroupSize: 6, Positions: positions, Scale: 0.0625,
			}); got != HIPAttentionRouteWMMA {
				t.Fatalf("positions=%d: route = %q, want wmma", positions, got)
			}
			// The resident cache holds exactly `positions` rows on this path.
			got, err := HIPAttentionWMMADecodeReference(q, in.K[:positions*wmmaKVWidth], in.V[:positions*wmmaKVWidth], positions)
			if err != nil {
				t.Fatal(err)
			}
			generic := wmmaGenericAttentionFP64(q, in.K, in.V, 24, 4, positions, 256, 6, 0.0625)
			wmmaCompare(t, fmt.Sprintf("positions=%d vs generic fallback", positions), got, generic)
			wmmaSameBits(t, fmt.Sprintf("positions=%d vs prefill row", positions), got, prefill[row*wmmaAttnWidth:(row+1)*wmmaAttnWidth])
		}
		if _, err := HIPAttentionWMMADecodeReference(in.Q[:wmmaAttnWidth], in.K, in.V, 0); !errors.Is(err, ErrHIPWMMAShape) {
			t.Fatalf("positions=0: err = %v, want ErrHIPWMMAShape", err)
		}
	})
}
