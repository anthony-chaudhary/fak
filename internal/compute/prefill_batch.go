package compute

import (
	"fmt"
	"math"
)

// BatchedPrefillBackend is the capability interface a compute.Backend implements
// to execute batched prompt prefill across a sequence panel (P x D) in 1 pass,
// eliminating the serial single-token loop (#11036).
type BatchedPrefillBackend interface {
	Backend
	PrefillBatch(args PrefillBatchArgs) (PrefillBatchResult, error)
}

// PrefillBatchBackend is an alias for BatchedPrefillBackend.
type PrefillBatchBackend = BatchedPrefillBackend

// PrefillBatchArgs contains the parameters for a batched prompt prefill operation.
type PrefillBatchArgs struct {
	X          Tensor  // [P, D] input prompt activation panel
	Wq         Tensor  // [nH*hd, D] query projection weight
	Wk         Tensor  // [nKV*hd, D] key projection weight
	Wv         Tensor  // [nKV*hd, D] value projection weight
	Wo         Tensor  // [D, nH*hd] output projection weight (optional)
	KV         KVStore // KV store to record keys/values (optional)
	Layer      int     // layer index in KVStore
	StartPos   int     // starting absolute sequence position (e.g. 0)
	NumHeads   int     // number of query heads (nH)
	NumKVHeads int     // number of key/value heads (nKV)
	HeadDim    int     // dimension per head (hd)
	Tokens     int     // optional token count P; if <= 0, derived from X
	RopeTheta  float64 // rotary embedding base theta (e.g. 10000.0)
	Scale      float32 // attention scaling factor; 0 => 1/sqrt(HeadDim)
}

// PrefillBatchResult returns the outputs of a batched prompt prefill operation.
type PrefillBatchResult struct {
	Output  Tensor // [P, D] if Wo is provided; otherwise [P, nH*hd]
	Context Tensor // [P, nH*hd] attention context before output projection
	Tokens  int    // P
}

func validatePrefillBatchArgs(args *PrefillBatchArgs) (int, int, error) {
	if args == nil {
		return 0, 0, fmt.Errorf("compute: PrefillBatch nil args")
	}
	if args.X.buf == nil || args.X.Numel() <= 0 {
		return 0, 0, fmt.Errorf("compute: PrefillBatch unallocated or empty X tensor")
	}
	if args.Wq.buf == nil || args.Wk.buf == nil || args.Wv.buf == nil {
		return 0, 0, fmt.Errorf("compute: PrefillBatch missing Q/K/V projection weight")
	}
	if args.NumHeads <= 0 || args.NumKVHeads <= 0 || args.HeadDim <= 0 {
		return 0, 0, fmt.Errorf("compute: PrefillBatch invalid heads nH=%d nKV=%d hd=%d", args.NumHeads, args.NumKVHeads, args.HeadDim)
	}
	if args.NumHeads%args.NumKVHeads != 0 {
		return 0, 0, fmt.Errorf("compute: PrefillBatch NumHeads (%d) not divisible by NumKVHeads (%d)", args.NumHeads, args.NumKVHeads)
	}
	if args.StartPos < 0 {
		return 0, 0, fmt.Errorf("compute: PrefillBatch negative StartPos=%d", args.StartPos)
	}
	P := args.Tokens
	if P <= 0 {
		if len(args.X.Shape) >= 2 {
			P = args.X.Shape[0]
		} else {
			return 0, 0, fmt.Errorf("compute: PrefillBatch cannot infer token count P from 1-D X without Tokens field")
		}
	}
	if P <= 0 || args.X.Numel()%P != 0 {
		return 0, 0, fmt.Errorf("compute: PrefillBatch invalid token count P=%d for X numel=%d", P, args.X.Numel())
	}
	D := args.X.Numel() / P
	if D <= 0 {
		return 0, 0, fmt.Errorf("compute: PrefillBatch invalid hidden dimension D=%d", D)
	}
	qOut := args.NumHeads * args.HeadDim
	kvOut := args.NumKVHeads * args.HeadDim
	if args.Wq.Numel() != qOut*D {
		return 0, 0, fmt.Errorf("compute: PrefillBatch Wq numel %d != expected %d", args.Wq.Numel(), qOut*D)
	}
	if args.Wk.Numel() != kvOut*D {
		return 0, 0, fmt.Errorf("compute: PrefillBatch Wk numel %d != expected %d", args.Wk.Numel(), kvOut*D)
	}
	if args.Wv.Numel() != kvOut*D {
		return 0, 0, fmt.Errorf("compute: PrefillBatch Wv numel %d != expected %d", args.Wv.Numel(), kvOut*D)
	}
	if args.Wo.buf != nil && args.Wo.Numel() != D*qOut {
		return 0, 0, fmt.Errorf("compute: PrefillBatch Wo numel %d != expected %d", args.Wo.Numel(), D*qOut)
	}
	if args.RopeTheta <= 0 {
		args.RopeTheta = 10000.0
	}
	if args.Scale <= 0 {
		args.Scale = float32(1.0 / math.Sqrt(float64(args.HeadDim)))
	}
	return P, D, nil
}

// prefillBatchCausalAttention computes scaled dot-product causal attention across the prompt panel
// for all P query tokens against cached keys and values.
func prefillBatchCausalAttention(qRoped, allK, allV []float32, P, startPos, totalPositions, nH, nKV, hd int, scale float32) []float32 {
	grp := nH / nKV
	qOut := nH * hd
	strideKV := nKV * hd
	context := make([]float32, P*qOut)
	scores := make([]float32, totalPositions)

	for t := 0; t < P; t++ {
		pos := startPos + t
		attendLen := pos + 1
		if attendLen > totalPositions {
			attendLen = totalPositions
		}

		for h := 0; h < nH; h++ {
			kvh := h / grp
			qh := qRoped[t*qOut+h*hd : t*qOut+(h+1)*hd]

			for j := 0; j < attendLen; j++ {
				kh := allK[j*strideKV+kvh*hd : j*strideKV+(kvh+1)*hd]
				scores[j] = dot(qh, kh) * scale
			}

			softmaxInPlace(scores[:attendLen])

			oh := context[t*qOut+h*hd : t*qOut+(h+1)*hd]
			for d := 0; d < hd; d++ {
				oh[d] = 0
			}
			for j := 0; j < attendLen; j++ {
				vh := allV[j*strideKV+kvh*hd : j*strideKV+(kvh+1)*hd]
				wj := scores[j]
				for d := 0; d < hd; d++ {
					oh[d] += wj * vh[d]
				}
			}
		}
	}
	return context
}

// PrefillBatch executes batched prompt prefill across a sequence panel (P x D) in 1 pass.
// If the backend owning X implements BatchedPrefillBackend, it dispatches to that backend;
// otherwise it falls back to the Default backend.
func PrefillBatch(be Backend, args PrefillBatchArgs) (PrefillBatchResult, error) {
	if be == nil {
		be = args.X.Backend()
	}
	if be == nil {
		be = Default()
	}
	if bpb, ok := be.(BatchedPrefillBackend); ok {
		return bpb.PrefillBatch(args)
	}
	ref, ok := Default().(BatchedPrefillBackend)
	if !ok {
		return PrefillBatchResult{}, fmt.Errorf("compute: Default backend does not implement BatchedPrefillBackend")
	}
	return ref.PrefillBatch(args)
}

// PrefillBatch on cpuBackend is the bit-exact CPU reference implementation (#11036).
func (c *cpuBackend) PrefillBatch(args PrefillBatchArgs) (PrefillBatchResult, error) {
	P, _, err := validatePrefillBatchArgs(&args)
	if err != nil {
		return PrefillBatchResult{}, err
	}
	nH := args.NumHeads
	nKV := args.NumKVHeads
	hd := args.HeadDim
	qOut := nH * hd
	kvOut := nKV * hd
	startPos := args.StartPos

	// 1. Batched projections: panelize P tokens into a single matrix multiplication
	// across sequence (P x D) for Q, K, V.
	qTen := c.BatchedMatMul(args.Wq, args.X, P)
	kTen := c.BatchedMatMul(args.Wk, args.X, P)
	vTen := c.BatchedMatMul(args.Wv, args.X, P)

	qf := c.f32(qTen)
	kf := c.f32(kTen)
	vf := c.f32(vTen)

	// 2. RoPE q and k per token position
	kRawAll := append([]float32(nil), kf...)
	qRoped := make([]float32, len(qf))
	kRoped := make([]float32, len(kf))

	for t := 0; t < P; t++ {
		pos := startPos + t
		qRow := c.result([]int{qOut}, append([]float32(nil), qf[t*qOut:(t+1)*qOut]...))
		qR := c.RoPE(qRow, pos, nH, hd, args.RopeTheta)
		copy(qRoped[t*qOut:(t+1)*qOut], c.f32(qR))

		kRow := c.result([]int{kvOut}, append([]float32(nil), kf[t*kvOut:(t+1)*kvOut]...))
		kR := c.RoPE(kRow, pos, nKV, hd, args.RopeTheta)
		copy(kRoped[t*kvOut:(t+1)*kvOut], c.f32(kR))
	}

	// 2. Append to KVStore if provided
	if args.KV != nil {
		for t := 0; t < P; t++ {
			pos := startPos + t
			rawRow := kRawAll[t*kvOut : (t+1)*kvOut]
			ropeRowSlice := kRoped[t*kvOut : (t+1)*kvOut]
			vRow := vf[t*kvOut : (t+1)*kvOut]

			rawTen := c.result([]int{kvOut}, append([]float32(nil), rawRow...))
			ropeTen := c.result([]int{kvOut}, append([]float32(nil), ropeRowSlice...))
			vRowTen := c.result([]int{kvOut}, append([]float32(nil), vRow...))

			args.KV.AppendKV(args.Layer, rawTen, ropeTen, vRowTen, pos)
		}
	}

	var allK, allV []float32
	var totalPositions int
	strideKV := kvOut

	if args.KV != nil {
		allK = c.f32(args.KV.KeysView(args.Layer))
		allV = c.f32(args.KV.ValuesView(args.Layer))
		totalPositions = len(allK) / strideKV
	} else {
		allK = kRoped
		allV = vf
		totalPositions = P
		startPos = 0
	}

	// 3. Causal attention across the prompt panel in 1 batched pass
	context := prefillBatchCausalAttention(qRoped, allK, allV, P, startPos, totalPositions, nH, nKV, hd, args.Scale)
	ctxTen := c.result([]int{P, qOut}, context)

	// 4. Output projection if Wo is provided
	var outTen Tensor
	if args.Wo.buf != nil {
		outTen = c.BatchedMatMul(args.Wo, ctxTen, P)
	} else {
		outTen = ctxTen
	}

	return PrefillBatchResult{
		Output:  outTen,
		Context: ctxTen,
		Tokens:  P,
	}, nil
}

var _ BatchedPrefillBackend = (*cpuBackend)(nil)
