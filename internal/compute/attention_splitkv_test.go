package compute

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
)

// attention_splitkv_test.go — [SW-VERIFIED] CPU/in-memory witness for the
// split-KV chunked-prefill panel borrowed from vLLM merge_attn_states
// (vllm-project/vllm csrc/attention/merge_attn_states.cu:18 at
// ae065918640a198a56f6f17eb45e251c3229fe41, Apache-2.0; ADAPT, notice at
// third_party/vllm-merge-attn-states/NOTICE). The device kernels live in
// cuda/attention.cuh beside k_qwen38_causal_attention_panel_hd256:
// k_qwen38_causal_attention_panel_hd256_splitkv_partial (one CTA per
// (row, head, partition) writing the unnormalized (m, l, O) partial) and
// k_qwen38_splitkv_lse_merge_hd256 (the LSE fold into the row output). The Go
// mirrors below re-express that exact contract — floor-divided partition
// boundaries, first-key seeding of each partition's (m, l, acc), the sentinel
// partial for an empty partition, and the LSE fold — and are pinned against
// the single-pass panel oracle; the final subtest audits that the fenced CUDA
// source carries the same markers. No GPU execution, no throughput claim.

const (
	splitKVPrefixThreshold = 1024 // == QWEN38_SPLITKV_PREFIX_THRESHOLD
	splitKVMaxPartitions   = 4    // == QWEN38_SPLITKV_MAX_PARTITIONS
	// The Qwen3.8 prompt-panel geometry the device entry point admits (its
	// validator rejects anything else); mirrored here because the home file
	// for the production constants is cuda-tagged.
	splitKVHeads   = 24
	splitKVKVHeads = 4
	splitKVHeadDim = 256
)

// splitKVNegInf is the float32 negative infinity the sentinel partial writes.
var splitKVNegInf = float32(math.Inf(-1))

// errSplitKVInvalid is the closed refusal for out-of-domain geometry; the C
// launcher refuses the same domain with -1 before launch.
var errSplitKVInvalid = errors.New("splitkv: invalid geometry")

// splitKVPartitions mirrors qwen38_splitkv_partitions exactly.
func splitKVPartitions(prefix, tokens int) (int, error) {
	if prefix < 0 || tokens <= 0 {
		return 0, errSplitKVInvalid
	}
	if prefix < splitKVPrefixThreshold {
		return 1, nil
	}
	p := 1 + prefix/splitKVPrefixThreshold
	if p > splitKVMaxPartitions {
		p = splitKVMaxPartitions
	}
	return p, nil
}

// splitKVBoundaries mirrors the kernel's floor-divided partition boundaries
// b[i] = i * nPos / nsplits, so unaligned splits and empty leading partitions
// fall out of the integer arithmetic.
func splitKVBoundaries(nPos, nsplits int) []int {
	b := make([]int, nsplits+1)
	for i := 0; i <= nsplits; i++ {
		b[i] = int(int64(i) * int64(nPos) / int64(nsplits))
	}
	return b
}

type splitKVPartial struct {
	m float32
	l float32
	o []float32
}

// splitKVExp is the mirror's expf stand-in: deterministic f32-range exp. Bit
// identity to CUDA expf is NOT claimed; the recorded envelope in
// checkSplitEnvelope absorbs that drift class.
func splitKVExp(x float32) float32 { return float32(math.Exp(float64(x))) }

// fmax32 mirrors fmaxf on the finite panel scores.
func fmax32(a, b float32) float32 {
	if a >= b {
		return a
	}
	return b
}

// panelSinglePassCPU mirrors k_qwen38_causal_attention_panel_hd256: one row
// per (token, head), the online-softmax recurrence over the full causal
// window, seeded at j == 0. The serial dot order is the mirror's own
// reference, not the device tree reduction.
func panelSinglePassCPU(q, k, v []float32, tokens, prefix, heads, kvHeads, headDim int, scale float32) []float32 {
	width := kvHeads * headDim
	out := make([]float32, 0, tokens*heads*headDim)
	for token := 0; token < tokens; token++ {
		nPos := prefix + token + 1
		for h := 0; h < heads; h++ {
			kvh := h / (heads / kvHeads)
			qh := q[token*heads*headDim+h*headDim : token*heads*headDim+(h+1)*headDim]
			var m, l float32
			acc := make([]float32, headDim)
			for j := 0; j < nPos; j++ {
				dot := float32(0)
				kj := k[j*width+kvh*headDim : j*width+(kvh+1)*headDim]
				for d := 0; d < headDim; d++ {
					dot += qh[d] * kj[d]
				}
				score := dot * scale
				vj := v[j*width+kvh*headDim : j*width+(kvh+1)*headDim]
				if j == 0 {
					m = score
					l = 1
					for d := 0; d < headDim; d++ {
						acc[d] = vj[d]
					}
					continue
				}
				nextM := fmax32(m, score)
				correction := splitKVExp(m - nextM)
				probability := splitKVExp(score - nextM)
				l = l*correction + probability
				for d := 0; d < headDim; d++ {
					acc[d] = acc[d]*correction + probability*vj[d]
				}
				m = nextM
			}
			invL := float32(0)
			if l > 0 {
				invL = 1 / l
			}
			for d := 0; d < headDim; d++ {
				out = append(out, acc[d]*invL)
			}
		}
	}
	return out
}

// panelSplitKVCPURow mirrors k_qwen38_causal_attention_panel_hd256_splitkv_partial
// for one (row, head): the per-partition unnormalized (m, l, O) partials, with
// the sentinel partial for an empty partition.
func panelSplitKVCPURow(q, k, v []float32, token, prefix, h, heads, kvHeads, headDim int, scale float32, nsplits int) []splitKVPartial {
	width := kvHeads * headDim
	nPos := prefix + token + 1
	bounds := splitKVBoundaries(nPos, nsplits)
	kvh := h / (heads / kvHeads)
	qh := q[token*heads*headDim+h*headDim : token*heads*headDim+(h+1)*headDim]
	partials := make([]splitKVPartial, nsplits)
	for p := 0; p < nsplits; p++ {
		j0, j1 := bounds[p], bounds[p+1]
		if j0 >= j1 {
			partials[p] = splitKVPartial{m: splitKVNegInf, l: 0, o: make([]float32, headDim)}
			continue
		}
		var m, l float32
		acc := make([]float32, headDim)
		for j := j0; j < j1; j++ {
			dot := float32(0)
			kj := k[j*width+kvh*headDim : j*width+(kvh+1)*headDim]
			for d := 0; d < headDim; d++ {
				dot += qh[d] * kj[d]
			}
			score := dot * scale
			vj := v[j*width+kvh*headDim : j*width+(kvh+1)*headDim]
			if j == j0 {
				m = score
				l = 1
				for d := 0; d < headDim; d++ {
					acc[d] = vj[d]
				}
				continue
			}
			nextM := fmax32(m, score)
			correction := splitKVExp(m - nextM)
			probability := splitKVExp(score - nextM)
			l = l*correction + probability
			for d := 0; d < headDim; d++ {
				acc[d] = acc[d]*correction + probability*vj[d]
			}
			m = nextM
		}
		partials[p] = splitKVPartial{m: m, l: l, o: acc}
	}
	return partials
}

// mergeSplitKVCPURow mirrors k_qwen38_splitkv_lse_merge_hd256: M = max m_p,
// L = sum l_p exp(m_p - M), O = sum O_p exp(m_p - M), out = O/L; an all-empty
// row writes zeros instead of dividing by zero.
func mergeSplitKVCPURow(partials []splitKVPartial, headDim int) []float32 {
	out := make([]float32, headDim)
	mMax := splitKVNegInf
	for _, p := range partials {
		mMax = fmax32(mMax, p.m)
	}
	if mMax == splitKVNegInf {
		return out
	}
	var lSum float32
	acc := make([]float32, headDim)
	for _, p := range partials {
		weight := splitKVExp(p.m - mMax)
		lSum += p.l * weight
		for d := 0; d < headDim; d++ {
			acc[d] += p.o[d] * weight
		}
	}
	invL := float32(0)
	if lSum > 0 {
		invL = 1 / lSum
	}
	for d := 0; d < headDim; d++ {
		out[d] = acc[d] * invL
	}
	return out
}

// splitKVLCG is the deterministic data source (the sibling panel witnesses'
// LCG, re-declared here because its home file is cuda-tagged).
type splitKVLCG uint64

func (r *splitKVLCG) values(n int, scale float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		*r = *r*6364136223846793005 + 1442695040888963407
		out[i] = (float32(uint32(*r>>32))/float32(uint64(1)<<32) - 0.5) * scale
	}
	return out
}

// Recorded f32 reassociation envelope for split-vs-single-pass: the split path
// groups the same terms differently (per-partition sums folded by the LSE
// weights), so bit identity is not the contract — the closed contract is this
// fixed envelope, measured on the mirror and held with margin.
const (
	splitKVCosineMin   = 0.99999
	splitKVMaxDeltaMax = 1e-5
)

// checkSplitEnvelope holds a split result to the recorded reassociation
// envelope against the single-pass oracle (fixed floors, not prose).
func checkSplitEnvelope(t *testing.T, name string, want, got []float32) {
	t.Helper()
	c := cosine(want, got)
	d := maxAbsDelta(want, got)
	if math.IsNaN(c) || c < splitKVCosineMin {
		t.Fatalf("%s cosine %.8f < %.5f (max abs delta %.3e)", name, c, splitKVCosineMin, d)
	}
	if d > splitKVMaxDeltaMax {
		t.Fatalf("%s max abs delta %.3e > %.0e", name, d, splitKVMaxDeltaMax)
	}
	t.Logf("%s cosine=%.8f max_abs_delta=%.3e [SW-VERIFIED]", name, c, d)
}

// TestSplitKVPrefillMergeMatchesPanelOracle pins the split-KV prefill borrow:
// (a) a prefix=0 chunk equals the existing panel bitwise, (b) a 2-way split at
// an unaligned boundary equals single-pass within the recorded envelope, and
// (c) a 4-way split with an empty leading partition preserves (m, l) seeding.
// The final subtest audits that the fenced CUDA source carries the same
// contract markers. [SW-VERIFIED] CPU/in-memory witness; no GPU or throughput
// claim.
//
// fak-test:runtime medium est=1s lane=default
func TestSplitKVPrefillMergeMatchesPanelOracle(t *testing.T) {
	const (
		heads   = splitKVHeads
		kvHeads = splitKVKVHeads
		headDim = splitKVHeadDim
	)
	scale := float32(1 / math.Sqrt(headDim))

	t.Run("plan-closed-contract", func(t *testing.T) {
		for _, c := range []struct{ prefix, tokens, want int }{
			{0, 1, 1},
			{1023, 64, 1},
			{1024, 1, 2},
			{1500, 8, 2},
			{2048, 2, 3},
			{3072, 1, 4},
			{1 << 20, 4, 4},
		} {
			got, err := splitKVPartitions(c.prefix, c.tokens)
			if err != nil {
				t.Fatalf("splitKVPartitions(%d, %d): %v", c.prefix, c.tokens, err)
			}
			if got != c.want {
				t.Fatalf("splitKVPartitions(%d, %d) = %d, want %d", c.prefix, c.tokens, got, c.want)
			}
		}
		if _, err := splitKVPartitions(-1, 1); !errors.Is(err, errSplitKVInvalid) {
			t.Fatalf("negative prefix must refuse with errSplitKVInvalid, got %v", err)
		}
		if _, err := splitKVPartitions(0, 0); !errors.Is(err, errSplitKVInvalid) {
			t.Fatalf("zero tokens must refuse with errSplitKVInvalid, got %v", err)
		}
	})

	t.Run("prefix-zero-equals-panel-bitwise", func(t *testing.T) {
		// Pin (a): a prefix=0 chunk keeps the single-pass panel bitwise.
		rng := splitKVLCG(0x53701174)
		tokens := 3
		q := rng.values(tokens*heads*headDim, 0.5)
		k := rng.values(tokens*kvHeads*headDim, 0.5)
		v := rng.values(tokens*kvHeads*headDim, 0.5)
		nsplits, err := splitKVPartitions(0, tokens)
		if err != nil || nsplits != 1 {
			t.Fatalf("prefix=0 must plan exactly 1 partition, got %d (%v)", nsplits, err)
		}
		want := panelSinglePassCPU(q, k, v, tokens, 0, heads, kvHeads, headDim, scale)
		got := make([]float32, 0, len(want))
		for token := 0; token < tokens; token++ {
			for h := 0; h < heads; h++ {
				got = append(got, mergeSplitKVCPURow(
					panelSplitKVCPURow(q, k, v, token, 0, h, heads, kvHeads, headDim, scale, 1), headDim)...)
			}
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("prefix=0 partition[%d] = %v, want panel %v (bitwise)", i, got[i], want[i])
			}
		}
	})

	t.Run("two-way-unaligned-equals-single-pass", func(t *testing.T) {
		// Pin (b): a 2-way split at an unaligned boundary equals single-pass.
		prefix, tokens := 1024, 5
		rng := splitKVLCG(0x53701174)
		q := rng.values(tokens*heads*headDim, 0.5)
		k := rng.values((prefix+tokens)*kvHeads*headDim, 0.5)
		v := rng.values((prefix+tokens)*kvHeads*headDim, 0.5)
		nsplits, err := splitKVPartitions(prefix, tokens)
		if err != nil || nsplits != 2 {
			t.Fatalf("prefix=%d must plan exactly 2 partitions, got %d (%v)", prefix, nsplits, err)
		}
		row0 := prefix + 1
		bounds := splitKVBoundaries(row0, 2)
		if row0%2 == 0 || bounds[1] == row0-bounds[1] {
			t.Fatalf("row 0 window %d must split into unequal halves at an unaligned boundary", row0)
		}
		want := panelSinglePassCPU(q, k, v, tokens, prefix, heads, kvHeads, headDim, scale)
		got := make([]float32, 0, len(want))
		for token := 0; token < tokens; token++ {
			for h := 0; h < heads; h++ {
				got = append(got, mergeSplitKVCPURow(
					panelSplitKVCPURow(q, k, v, token, prefix, h, heads, kvHeads, headDim, scale, nsplits), headDim)...)
			}
		}
		checkSplitEnvelope(t, "two-way-unaligned", want, got)
	})

	t.Run("four-way-empty-leading-preserves-seeding", func(t *testing.T) {
		// Pin (c): a 4-way split with an empty leading partition preserves
		// (m, l) seeding. The default plan never emits an empty partition
		// (every row's window is >= threshold+1 > 4 keys), so this pins the
		// defensive contract the kernels must hold whenever more partitions
		// than keys are forced: the empty partition writes the sentinel
		// partial and the merge drops it exactly, while the surviving
		// partitions seed (m, l) at their own j0.
		nsplits, err := splitKVPartitions(3072, 1)
		if err != nil || nsplits != 4 {
			t.Fatalf("prefix=3072 must plan exactly 4 partitions, got %d (%v)", nsplits, err)
		}
		const nPos = 3 // floor boundaries [0, 0, 1, 2, 3]: partition 0 is empty
		rng := splitKVLCG(0x53701174)
		q := rng.values(heads*headDim, 0.5)
		k := rng.values(nPos*kvHeads*headDim, 0.5)
		v := rng.values(nPos*kvHeads*headDim, 0.5)
		want := panelSinglePassCPU(q, k, v, 1, nPos-1, heads, kvHeads, headDim, scale)
		for h := 0; h < heads; h++ {
			partials := panelSplitKVCPURow(q, k, v, 0, nPos-1, h, heads, kvHeads, headDim, scale, nsplits)
			empty := partials[0]
			if empty.m != splitKVNegInf || empty.l != 0 {
				t.Fatalf("head %d: empty leading partition must write the sentinel (m=-Inf, l=0), got (m=%v, l=%v)", h, empty.m, empty.l)
			}
			for i, x := range empty.o {
				if x != 0 {
					t.Fatalf("head %d: empty leading partition O[%d] = %v, want 0", h, i, x)
				}
			}
			// Seeding preserved: partition 1 starts at key 0, so its (m, l)
			// equal the single-pass seed exactly.
			kvh := h / (heads / kvHeads)
			dot := float32(0)
			for d := 0; d < headDim; d++ {
				dot += q[h*headDim+d] * k[kvh*headDim+d]
			}
			if partials[1].m != dot*scale || partials[1].l != 1 {
				t.Fatalf("head %d: partition 1 must seed (m, l) at its j0: got (m=%v, l=%v), want (m=%v, l=1)",
					h, partials[1].m, partials[1].l, dot*scale)
			}
			merged4 := mergeSplitKVCPURow(partials, headDim)
			merged3 := mergeSplitKVCPURow(partials[1:], headDim)
			for i := range merged4 {
				if merged4[i] != merged3[i] {
					t.Fatalf("head %d: empty leading partition must drop exactly: merged4[%d]=%v, merged3=%v", h, i, merged4[i], merged3[i])
				}
			}
			checkSplitEnvelope(t, fmt.Sprintf("four-way-empty-leading-h%d", h), want[h*headDim:(h+1)*headDim], merged4)
		}
	})

	t.Run("device-source-carries-same-contract", func(t *testing.T) {
		read := func(path string) string {
			t.Helper()
			b, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			return string(b)
		}
		src := read("cuda/attention.cuh")
		for _, want := range []string{
			"k_qwen38_causal_attention_panel_hd256_splitkv_partial",
			"k_qwen38_splitkv_lse_merge_hd256",
			"#define QWEN38_SPLITKV_PREFIX_THRESHOLD 1024",
			"#define QWEN38_SPLITKV_MAX_PARTITIONS 4",
			"qwen38_splitkv_partitions(prefix, tokens)",
			"(long long)partition * nPos / nsplits",
			"(long long)(partition + 1) * nPos / nsplits",
			"slot[0] = -INFINITY;",
			"slot[1] = 0.f;",
			"if (j == j0) {",
			"expf(slot[0] - mMax)",
			"mMax == -INFINITY",
			"ae065918640a198a56f6f17eb45e251c3229fe41",
			"csrc/attention/merge_attn_states.cu",
			// The mirror's 24/4/256 geometry is the one the device entry point
			// admits; pin it where the validator enforces it.
			"nH != 24 || nKV != 4 ||",
			"#define QWEN38_PROMPT_HEAD_DIM 256",
			// Short prefixes keep the unchanged single-pass panel bitwise, and
			// the P-05 gate semantics stay put.
			"FCUDA_TS(\"k_qwen38_causal_attention_panel_hd256\", tokens * nH,",
			"#define QWEN38_PREFILL_ATTN_MMA_DEFAULT 0",
			"FAK_CUDA_QWEN38_PREFILL_ATTN",
		} {
			if !strings.Contains(src, want) {
				t.Errorf("missing split-KV contract marker %q in cuda/attention.cuh", want)
			}
		}
		// The tensor-core launcher keeps its own seam: complementary, not
		// replaced.
		if !strings.Contains(read("cuda/fattn_mma_hd256.cuh"), "static int fak_qwen38_fattn_mma_hd256_launch(") {
			t.Error("fattn_mma_hd256.cuh lost the tensor-core launcher seam")
		}
	})
}
