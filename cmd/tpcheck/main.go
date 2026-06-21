// Command tpcheck runs fak's NATIVE-engine TENSOR-parallel decomposition end to end over
// SIMULATED ranks (single box, in-memory, no GPU/NCCL/checkpoint) — the runnable form of
// the intra-layer sharding contract (internal/model/tensor_parallel.go).
// Pipeline parallelism (cmd/pipelinegen) splits the layer STACK across workers; tensor
// parallelism splits a SINGLE layer's matmuls across workers and crosses a collective
// (AllGather / AllReduce) inside the layer. A real 8xA100 plan composes both. This is the
// "tensor parallelism within a layer" lever GLM-5.2-NATIVE-ENGINE-GAP names as the
// remaining multi-A100 step.
//
// It runs on hardware that exists — no GPU, no checkpoint — by sharding deterministic
// in-memory weights across simulated ranks and witnessing the two Megatron identities:
//
//   - COLUMN-parallel matmul (shard output features, AllGather) is BIT-EXACT (max|Δ|=0)
//     vs the single-rank result — sharding the output features adds zero numeric drift.
//   - ROW-parallel matmul (shard the contraction, AllReduce) and the composed Megatron
//     FFN match within the AllReduce's documented reassociation round-off (~1e-6/1e-5),
//     never garbage.
//
// The single-rank TP result is the monolithic reference here; the unit test
// (tensor_parallel_test.go) pins that 1-rank result bit-for-bit to matRows/the SwiGLU FFN,
// so this command's "vs rank-1" comparison is a faithful "vs monolith" witness.
//
// Usage:
//
//	go run ./cmd/tpcheck
//	go run ./cmd/tpcheck -selfcheck            # explicit; same as default
//	go run ./cmd/tpcheck -hidden 1024 -inter 4096 -ranks 1,2,4,8 -seed 7
//
// Exit status is non-zero if any identity fails, so it doubles as a CI witness.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func main() {
	hidden := flag.Int("hidden", 512, "hidden size H (matmul contraction / FFN input)")
	inter := flag.Int("inter", 1536, "intermediate size I (FFN); also the output dim of the column-parallel probe")
	ranksCSV := flag.String("ranks", "1,2,4,8", "comma-separated tensor-parallel rank counts to witness")
	seed := flag.Int("seed", 1, "deterministic weight seed")
	flag.Bool("selfcheck", true, "run the in-memory witness (default; no GPU or checkpoint needed)")
	wired := flag.Bool("wired", true, "also witness the WIRED forward path (ForwardTP vs Forward over a synthetic model)")
	flag.Parse()

	ranks, err := parseInts(*ranksCSV)
	if err != nil {
		fail("parse -ranks: %v", err)
	}
	if len(ranks) == 0 {
		fail("no ranks given")
	}
	H, I := *hidden, *inter
	if H <= 0 || I <= 0 {
		fail("hidden and inter must be > 0 (got H=%d I=%d)", H, I)
	}

	// Deterministic weights: a [I,H] projection (column-parallel probe over I and the FFN
	// gate/up), a [H,I] down projection (row-parallel probe), and an [H] input.
	gateW := lcg(I*H, uint64(*seed)*131+1)
	upW := lcg(I*H, uint64(*seed)*137+2)
	downW := lcg(H*I, uint64(*seed)*149+3)
	x := lcg(H, uint64(*seed)*151+4)

	fmt.Printf("tpcheck: H=%d I=%d ranks=%v seed=%d\n", H, I, ranks, *seed)
	fmt.Printf("  column-parallel probe: y = gateW[%d,%d] . x   (shard output dim I, AllGather)\n", I, H)
	fmt.Printf("  row-parallel probe:    y = downW[%d,%d] . a   (shard contraction I, AllReduce)\n", H, I)
	fmt.Printf("  composed FFN:          down( silu(gate.x)*up.x )  (one AllReduce per block)\n\n")

	coll := model.LocalCollective{}

	// Rank-1 references (the unit test pins these bit-for-bit to the monolith).
	planRef, err := model.NewTPPlan(I, 1)
	if err != nil {
		fail("NewTPPlan(I,1): %v", err)
	}
	colRef, err := model.ColumnParallelMatMul(gateW, x, I, H, planRef, coll)
	if err != nil {
		fail("column ref: %v", err)
	}
	// Activation a = silu(gate.x)*up.x, reconstructed via two rank-1 column-parallel matmuls
	// and an exported elementwise SwiGLU step is not available; instead use the FFN itself at
	// rank 1 for the FFN reference, and a rank-1 row-parallel matmul over `colRef` for the
	// row reference (colRef stands in for the activation — any vector works as a probe).
	planRowRef, err := model.NewTPPlan(I, 1)
	if err != nil {
		fail("NewTPPlan(I,1) row: %v", err)
	}
	rowRef, err := model.RowParallelMatMul(downW, colRef, H, I, planRowRef, coll)
	if err != nil {
		fail("row ref: %v", err)
	}
	ffnRef, err := model.TensorParallelFFN(gateW, upW, downW, x, H, I, planRef, coll)
	if err != nil {
		fail("ffn ref: %v", err)
	}

	fmt.Printf("  %-6s | %-22s | %-22s | %-22s\n", "ranks", "col-parallel vs r1", "row-parallel vs r1", "FFN vs r1")
	fmt.Printf("  %s\n", strings.Repeat("-", 78))

	failed := false
	for _, r := range ranks {
		if r < 1 || r > I {
			fmt.Printf("  %-6d | skipped (rank count out of [1,%d])\n", r, I)
			continue
		}
		planCol, err := model.NewTPPlan(I, r)
		if err != nil {
			fail("NewTPPlan(I,%d): %v", r, err)
		}
		col, err := model.ColumnParallelMatMul(gateW, x, I, H, planCol, coll)
		if err != nil {
			fail("col r=%d: %v", r, err)
		}
		row, err := model.RowParallelMatMul(downW, colRef, H, I, planCol, coll)
		if err != nil {
			fail("row r=%d: %v", r, err)
		}
		shardRefRow, err := model.RowParallelReference(downW, colRef, H, I, planCol)
		if err != nil {
			fail("row ref r=%d: %v", r, err)
		}
		ffn, err := model.TensorParallelFFN(gateW, upW, downW, x, H, I, planCol, coll)
		if err != nil {
			fail("ffn r=%d: %v", r, err)
		}

		colExact := bitExact(col, colRef)
		rowVsShard := bitExact(row, shardRefRow) // must be exact: same reassociation
		rowRel := maxRel(row, rowRef)
		ffnRel := maxRel(ffn, ffnRef)

		colMsg := "max|Δ|=0 (bit-exact)"
		if !colExact {
			colMsg = "DRIFT (BUG)"
			failed = true
		}
		rowMsg := fmt.Sprintf("rel %.1e", rowRel)
		if !rowVsShard {
			rowMsg = "!= shard ref (BUG)"
			failed = true
		} else if rowRel > 1e-3 {
			rowMsg = fmt.Sprintf("rel %.1e (HIGH)", rowRel)
			failed = true
		}
		ffnMsg := fmt.Sprintf("rel %.1e", ffnRel)
		if ffnRel > 1e-3 {
			ffnMsg = fmt.Sprintf("rel %.1e (HIGH)", ffnRel)
			failed = true
		}
		fmt.Printf("  %-6d | %-22s | %-22s | %-22s\n", r, colMsg, rowMsg, ffnMsg)
	}

	fmt.Println()
	if failed {
		fail("one or more tensor-parallel identities failed")
	}
	fmt.Println("tpcheck: OK — column-parallel bit-exact; row-parallel and FFN within AllReduce round-off.")
	fmt.Println("  (LocalCollective is the single-box stand-in; a real NCCL/RDMA collective is a 2-method swap.)")

	if *wired {
		if err := runWired(ranks, coll); err != nil {
			fail("%v", err)
		}
	}
}

// runWired witnesses the WIRED forward path: the SAME Megatron decomposition driven through
// the LIVE attention + dense-SwiGLU FFN of a real (synthetic) model — every feature the live
// path applies (RoPE, the per-head softmax, the residual stream, the LM head) re-entered —
// sharded across simulated ranks. It builds a tiny dense Llama-style model with no GPU and no
// checkpoint, runs the monolithic Forward as the reference, and compares ForwardTP across rank
// counts: BIT-EXACT logits at ranks=1 (the transitive HF-oracle link), and within the
// accumulated AllReduce round-off — with the SAME argmax at every position — at ranks>1.
func runWired(ranks []int, coll model.Collective) error {
	cfg := model.Config{
		HiddenSize: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 4, HeadDim: 8,
		IntermediateSize: 256, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "llama", EOSTokenID: -1,
	}
	m := model.NewSynthetic(cfg)
	ids := []int{1, 2, 3, 4, 5, 6}
	full := m.Forward(ids)
	last := len(ids) - 1

	fmt.Printf("\n  wired forward: ForwardTP vs monolithic Forward  (model H=%d L=%d nKV=%d I=%d, %d-token prompt)\n",
		cfg.HiddenSize, cfg.NumLayers, cfg.NumKVHeads, cfg.IntermediateSize, len(ids))
	fmt.Printf("  attention sharded over nKV=%d kv-head groups; FFN over I=%d  (ranks clamped to each dim)\n", cfg.NumKVHeads, cfg.IntermediateSize)
	fmt.Printf("  %-6s | %-26s | %-18s\n", "ranks", "last-logits vs Forward", "argmax agreement")
	fmt.Printf("  %s\n", strings.Repeat("-", 60))

	wiredFailed := false
	for _, r := range ranks {
		if r < 1 {
			continue
		}
		act, err := m.ForwardTP(ids, model.TPConfig{AttnRanks: r, FFNRanks: r, Coll: coll})
		if err != nil {
			return fmt.Errorf("ForwardTP ranks=%d: %v", r, err)
		}
		agree := 0
		for pos := range full.Logits {
			if argmaxIdx(act.Logits[pos]) == argmaxIdx(full.Logits[pos]) {
				agree++
			}
		}
		var msg string
		if r == 1 {
			if bitExact(act.Logits[last], full.Logits[last]) {
				msg = "max|Δ|=0 (bit-exact)"
			} else {
				msg = "DRIFT @ rank-1 (BUG)"
				wiredFailed = true
			}
		} else {
			d := maxAbs(act.Logits[last], full.Logits[last])
			msg = fmt.Sprintf("max|Δ|=%.2e", d)
			if d > 5e-3 {
				msg += " (HIGH)"
				wiredFailed = true
			}
		}
		agreeMsg := fmt.Sprintf("%d/%d positions", agree, len(full.Logits))
		if agree != len(full.Logits) {
			agreeMsg += " (MISMATCH)"
			wiredFailed = true
		}
		fmt.Printf("  %-6d | %-26s | %-18s\n", r, msg, agreeMsg)
	}
	fmt.Println()
	if wiredFailed {
		return fmt.Errorf("wired forward witness failed")
	}
	fmt.Println("tpcheck: OK — wired ForwardTP bit-exact vs Forward at rank-1; same argmax within AllReduce round-off across ranks.")
	fmt.Println("  (Forward is gated bit-exact against the HF oracle in internal/model/oracle_test.go, so ForwardTP inherits it.)")
	return nil
}

// argmaxIdx returns the index of the maximum element (first on ties).
func argmaxIdx(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

// maxAbs is the max absolute element-wise difference.
func maxAbs(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

// lcg is the package-wide deterministic generator (same LCG as the unit tests), so the
// command reproduces exactly and needs no rng dependency.
func lcg(n int, seed uint64) []float32 {
	v := make([]float32, n)
	s := seed
	for i := range v {
		s = s*6364136223846793005 + 1442695040888963407
		v[i] = float32(int64(s>>40))/float32(1<<23) - 0.5
	}
	return v
}

func bitExact(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func maxRel(got, ref []float32) float64 {
	var m float64
	for i := range got {
		den := math.Abs(float64(ref[i]))
		if den > 1e-6 {
			if r := math.Abs(float64(got[i]-ref[i])) / den; r > m {
				m = r
			}
		}
	}
	return m
}

func parseInts(csv string) ([]int, error) {
	parts := strings.Split(csv, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "tpcheck: "+format+"\n", a...)
	os.Exit(1)
}
