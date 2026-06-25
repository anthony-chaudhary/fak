package model

import (
	"fmt"
	"math"
	"testing"
)

// distTPCollective is a test-only Collective adapter that routes each ForwardTP
// collective through DistComm. ForwardTP still presents all rank parts at the
// model seam, but each DistComm rank receives only parts[rank] over the loopback
// process-group wire before the adapter returns rank 0's consensus result.
type distTPCollective struct {
	t *testing.T
}

func (d distTPCollective) AllReduceSum(parts [][]float32) ([]float32, error) {
	if len(parts) == 0 {
		return nil, fmt.Errorf("model: distTPCollective AllReduceSum has no parts")
	}
	results, errs := runGroup(d.t, len(parts), func(g *DistComm) ([]float32, error) {
		return g.AllReduceSum(parts[g.Rank()])
	})
	return distTPConsensus("AllReduceSum", results, errs)
}

func (d distTPCollective) AllGather(parts [][]float32, plan TPPlan) ([]float32, error) {
	if len(parts) == 0 {
		return nil, fmt.Errorf("model: distTPCollective AllGather has no parts")
	}
	results, errs := runGroup(d.t, len(parts), func(g *DistComm) ([]float32, error) {
		return g.AllGather(parts[g.Rank()], plan)
	})
	return distTPConsensus("AllGather", results, errs)
}

func distTPConsensus(op string, results [][]float32, errs []error) ([]float32, error) {
	for r, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("model: DistComm %s rank %d: %w", op, r, err)
		}
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("model: DistComm %s produced no rank results", op)
	}
	out := results[0]
	for r := 1; r < len(results); r++ {
		if len(results[r]) != len(out) {
			return nil, fmt.Errorf("model: DistComm %s rank %d len = %d, want %d", op, r, len(results[r]), len(out))
		}
		for i := range out {
			if math.Float32bits(results[r][i]) != math.Float32bits(out[i]) {
				return nil, fmt.Errorf("model: DistComm %s rank %d[%d] = %v, rank0 has %v", op, r, i, results[r][i], out[i])
			}
		}
	}
	return out, nil
}

// TestForwardTPViaDistCommCollective is the host-landable #295 witness: a full
// ForwardTP cacheless prefill runs its tensor-parallel row reductions over the
// real DistComm loopback wire and reproduces LocalCollective bit-for-bit. This
// is still host-float32 plumbing, not NCCL/RCCL or a multi-GPU claim.
func TestForwardTPViaDistCommCollective(t *testing.T) {
	m := NewSynthetic(tpFwdBaseCfg())
	ids := []int{1, 2, 3, 4, 5, 6}
	dist := distTPCollective{t: t}
	for _, c := range []struct{ attn, ffn int }{{2, 2}, {4, 3}} {
		ref, err := m.ForwardTP(ids, TPConfig{AttnRanks: c.attn, FFNRanks: c.ffn, Coll: LocalCollective{}})
		if err != nil {
			t.Fatalf("ForwardTP via Local (a=%d f=%d): %v", c.attn, c.ffn, err)
		}
		got, err := m.ForwardTP(ids, TPConfig{AttnRanks: c.attn, FFNRanks: c.ffn, Coll: dist})
		if err != nil {
			t.Fatalf("ForwardTP via DistComm (a=%d f=%d): %v", c.attn, c.ffn, err)
		}
		exact, mx := bitExactRows(got.Logits, ref.Logits)
		if !exact {
			t.Fatalf("ForwardTP via DistComm != LocalCollective (a=%d f=%d): max|delta|=%.3e, want bit-exact 0", c.attn, c.ffn, mx)
		}
	}
}

// TestForwardTPDistCommRanksMatchLocalForwardTP is the stricter #295 host
// witness: every DistComm rank owns an independent model instance, computes only
// its rank's tensor-parallel attention/FFN shard, and receives full hidden rows
// only through the rank-local DistComm AllReduceSum. The exact oracle is a
// two-rank LocalCollective ForwardTP, not monolithic Forward, because TP row
// reductions intentionally reassociate arithmetic.
func TestForwardTPDistCommRanksMatchLocalForwardTP(t *testing.T) {
	const size = 2
	ids := []int{1, 2, 3, 4, 5, 6}
	refModel := NewSynthetic(tpFwdBaseCfg())
	ref, err := refModel.ForwardTP(ids, TPConfig{AttnRanks: size, FFNRanks: size, Coll: LocalCollective{}})
	if err != nil {
		t.Fatalf("Local ForwardTP reference: %v", err)
	}
	want := flatten(ref.Logits)

	results, errs := runGroup(t, size, func(g *DistComm) ([]float32, error) {
		m := NewSynthetic(tpFwdBaseCfg())
		act, err := m.forwardTPDistRank(ids, g)
		if err != nil {
			return nil, err
		}
		return flatten(act.Logits), nil
	})
	for r := 0; r < size; r++ {
		if errs[r] != nil {
			t.Fatalf("rank %d ForwardTP over DistComm: %v", r, errs[r])
		}
		distAssertBitsEqual(t, fmt.Sprintf("rank %d DistComm ForwardTP logits", r), results[r], want)
	}
}

func (m *Model) forwardTPDistRank(ids []int, g *DistComm) (*Activations, error) {
	if err := m.forwardTPSupported(); err != nil {
		return nil, err
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)
	attnPlan, err := NewTPPlan(cfg.NumKVHeads, g.Size())
	if err != nil {
		return nil, fmt.Errorf("model: ForwardTP DistComm attention plan: %w", err)
	}
	ffnPlan, err := NewTPPlan(cfg.IntermediateSize, g.Size())
	if err != nil {
		return nil, fmt.Errorf("model: ForwardTP DistComm ffn plan: %w", err)
	}

	embed := m.embedRows()
	x := make([][]float32, seq)
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		scaleEmbedInPlace(x[t], cfg)
	}
	act := &Activations{Seq: seq, Hidden: [][]float32{flatten(x)}}

	var subErr error
	for l := 0; l < cfg.NumLayers; l++ {
		rp := newRopeForLayer(cfg, l, seq)
		attnSub := func(xn [][]float32) [][]float32 {
			if subErr != nil {
				return zeroRows(seq, H)
			}
			out, err := m.tpAttnLayerDistRank(l, xn, rp, attnPlan, g)
			if err != nil {
				subErr = err
				return zeroRows(seq, H)
			}
			return out
		}
		mlpSub := func(xn [][]float32) [][]float32 {
			if subErr != nil {
				return zeroRows(seq, H)
			}
			out, err := m.tpFFNLayerDistRank(l, xn, ffnPlan, g)
			if err != nil {
				subErr = err
				return zeroRows(seq, H)
			}
			return out
		}
		composeSeqSublayer(cfg.BlockTopology, x, m.attentionNorms(l), eps, cfg, attnSub)
		composeSeqSublayer(cfg.BlockTopology, x, m.mlpNorms(l), eps, cfg, mlpSub)
		if subErr != nil {
			return nil, subErr
		}
		act.Hidden = append(act.Hidden, flatten(x))
	}

	mat := residentKernel{m}
	act.Logits = make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xf := m.finalNorm(x[t])
		logits := mat.mul(m.headName(), mat.prep(xf), cfg.VocabSize, H)
		logitScaleInPlace(logits, cfg)
		act.Logits[t] = logits
	}
	return act, nil
}

func (m *Model) tpAttnLayerDistRank(l int, xn [][]float32, rp rope, plan TPPlan, g *DistComm) ([][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if g.Rank() >= len(plan.Shards) {
		return nil, fmt.Errorf("model: DistComm rank %d has no attention shard in %d-shard plan", g.Rank(), len(plan.Shards))
	}
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	qHeadDim := cfg.NumHeads * hd
	band := attnBandForShard(plan.Shards[g.Rank()], cfg.GroupSize())
	attnOut, err := m.tpAttnBandPreProj(l, xn, rp, band)
	if err != nil {
		return nil, err
	}
	oName := layerName(l, "self_attn.o_proj.weight")
	if !m.has(oName) {
		return nil, fmt.Errorf("model: ForwardTP requires f32-resident %s (quant-aware TP sharding is a later lever)", oName)
	}
	oCols := shardWeightColumns(m.tensor(oName), H, qHeadDim, band.QLo*hd, band.QHi*hd)
	bandW := (band.QHi - band.QLo) * hd
	part := make([][]float32, len(xn))
	for t := range xn {
		part[t] = matRows(oCols, attnOut[t], H, bandW)
	}
	return m.tpDistReduceWithBias(part, g, layerName(l, "self_attn.o_proj.bias"))
}

func (m *Model) tpFFNLayerDistRank(l int, xn [][]float32, plan TPPlan, g *DistComm) ([][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if g.Rank() >= len(plan.Shards) {
		return nil, fmt.Errorf("model: DistComm rank %d has no ffn shard in %d-shard plan", g.Rank(), len(plan.Shards))
	}
	cfg := m.Cfg
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	if plan.Dim != I {
		return nil, fmt.Errorf("model: tpFFNLayer plan.Dim = %d, want intermediate = %d", plan.Dim, I)
	}
	p := func(s string) string { return layerName(l, s) }
	for _, name := range []string{"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"} {
		if !m.has(p(name)) {
			return nil, fmt.Errorf("model: ForwardTP requires f32-resident %s (dense SwiGLU TP only; quant/MoE TP are later levers)", p(name))
		}
	}
	s := plan.Shards[g.Rank()]
	lo, w := s.Lo, s.Width()
	gateW := shardWeightRows(m.tensor(p("mlp.gate_proj.weight")), H, s.Lo, s.Hi)
	upW := shardWeightRows(m.tensor(p("mlp.up_proj.weight")), H, s.Lo, s.Hi)
	downW := shardWeightColumns(m.tensor(p("mlp.down_proj.weight")), H, I, s.Lo, s.Hi)
	gBias := m.tensorOptional(p("mlp.gate_proj.bias"))
	uBias := m.tensorOptional(p("mlp.up_proj.bias"))

	part := make([][]float32, len(xn))
	for t := range xn {
		gate := matRows(gateW, xn[t], w, H)
		up := matRows(upW, xn[t], w, H)
		if gBias != nil {
			for i := 0; i < w; i++ {
				gate[i] += gBias[lo+i]
			}
		}
		if uBias != nil {
			for i := 0; i < w; i++ {
				up[i] += uBias[lo+i]
			}
		}
		for i := 0; i < w; i++ {
			gate[i] = act(gate[i], cfg) * up[i]
		}
		part[t] = matRows(downW, gate, H, w)
	}
	return m.tpDistReduceWithBias(part, g, layerName(l, "mlp.down_proj.bias"))
}

func (m *Model) tpDistReduceWithBias(part [][]float32, g *DistComm, biasName string) ([][]float32, error) {
	out := make([][]float32, len(part))
	for t := range part {
		reduced, err := g.AllReduceSum(part[t])
		if err != nil {
			return nil, fmt.Errorf("model: DistComm tp AllReduce at position %d: %w", t, err)
		}
		m.addBiasIfPresent(reduced, biasName)
		out[t] = reduced
	}
	return out, nil
}

func TestDistTPCollectiveFailsClosedLikeLocal(t *testing.T) {
	dist := distTPCollective{t: t}
	if _, err := dist.AllReduceSum([][]float32{{1, 2}, {3}}); err == nil {
		t.Fatal("DistComm AllReduceSum should reject ragged rank parts")
	}
	plan, err := NewTPPlan(4, 2)
	if err != nil {
		t.Fatalf("NewTPPlan: %v", err)
	}
	if _, err := dist.AllGather([][]float32{{1, 2}, {3}}, plan); err == nil {
		t.Fatal("DistComm AllGather should reject a part whose width disagrees with the plan")
	}
}
