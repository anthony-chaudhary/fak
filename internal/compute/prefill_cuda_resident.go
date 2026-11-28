package compute

import "fmt"

// Ink2PDenseMMQKind returns the native fcuda_ink2p_dense_mmq format selector
// (8=Q8_0, 4=Q4_K, 5=Q5_K, 6=Q6_K) when a prefill projection is eligible for the
// Inkling2 dense MMQ tensor-core fast path. Returns (0, false) when the
// fallback fak-native launcher should run instead.
func Ink2PDenseMMQKind(dt Dtype, quant *QuantSpec, tokens, out, in int, denseFast bool) (int, bool) {
	if !denseFast || tokens < 16 || out <= 0 || in <= 0 {
		return 0, false
	}
	switch dt {
	case Q8_0:
		if quant != nil && quant.Block == 32 && in%32 == 0 {
			return 8, true
		}
	case Q4_K:
		if in%q4kSuper == 0 {
			return 4, true
		}
	case Q5_K:
		if in%q4kSuper == 0 {
			return 5, true
		}
	case Q6_K:
		if in%q4kSuper == 0 {
			return 6, true
		}
	}
	return 0, false
}

// Ink2PMatMulRoute describes the selected fak-native prefill GEMM dispatch route.
func Ink2PMatMulRoute(dt Dtype, quant *QuantSpec, tokens, out, in int, denseFast bool) string {
	if kind, ok := Ink2PDenseMMQKind(dt, quant, tokens, out, in, denseFast); ok {
		return fmt.Sprintf("ink2p_dense_mmq:%s:kind=%d", dt.String(), kind)
	}
	return "ink2p_native:" + dt.String()
}

// ResidentPrefillHooks exposes device-resident panel primitives so batched
// prefill executes Q/K/V projections, RoPE, KV append, causal attention, and
// optional Wo projection on device without per-token or intermediate host
// round-trips.
type ResidentPrefillHooks interface {
	DeviceArch() (major, minor int, ok bool)
	IsHostTensor(t Tensor) bool
	UploadHostProjectedQKV(x Tensor, wq, wk, wv Tensor, P, qOut, kvOut int) (q, k, v Tensor, uploads int, err error)
	ProjectQKVDevice(wq, wk, wv, x Tensor, P int) (q, k, v Tensor, weightUploads int, err error)
	RoPEPanelDevice(x Tensor, tokens, startPos, nHeads, headDim int, theta float64) (Tensor, error)
	AppendKVPanelDevice(kv KVStore, layer int, kRaw, kRoPE, v Tensor, startPos, tokens int) error
	AttentionPanelDevice(q, kFallback, vFallback Tensor, kv KVStore, layer, tokens, startPos, nH, nKV, hd int, scale float32, backend CUDAAttentionKernel) (Tensor, error)
	ProjectOutputDevice(wo, ctx Tensor, P int) (Tensor, int, error)
}

// ResidentPrefillReport summarizes device-resident dispatch behavior and host
// transfer counts for one PrefillBatch execution.
type ResidentPrefillReport struct {
	SelectedBackend       CUDAAttentionKernel
	AttentionSelection    CUDAAttentionSelection
	InputWasHost          bool
	HostReadCount         int
	HostUploadCount       int
	PerTokenHostTransfers int
	RoPEDispatches        int
	KVAppendDispatches    int
	AttentionDispatches   int
}

// prefillResidentBatch executes batched prefill using ResidentPrefillHooks and
// SelectCUDAAttentionKernel(CUDARegimePrefill), keeping Q, K, V, RoPE, KV, and
// attention outputs device-resident throughout.
func prefillResidentBatch(hooks ResidentPrefillHooks, args PrefillBatchArgs, kvPrec KVPrecision, paged bool) (PrefillBatchResult, ResidentPrefillReport, error) {
	P, _, err := validatePrefillBatchArgs(&args)
	if err != nil {
		return PrefillBatchResult{}, ResidentPrefillReport{}, err
	}
	if hooks == nil {
		return PrefillBatchResult{}, ResidentPrefillReport{}, fmt.Errorf("compute: nil resident prefill hooks")
	}

	major, minor, archOK := hooks.DeviceArch()
	if !archOK {
		major, minor = 8, 0
	}
	sel := ResolveCUDAAttentionSelection(major, minor, CUDARegimePrefill, CUDAAttentionOptions{
		HeadDim:     args.HeadDim,
		Tokens:      P,
		KVPrecision: kvPrec,
		Paged:       paged,
	})
	if !sel.Backend.GPUResident() {
		return PrefillBatchResult{}, ResidentPrefillReport{
			SelectedBackend:    sel.Backend,
			AttentionSelection: sel,
		}, fmt.Errorf("compute: cuda attention backend %s unsupported for %s", sel.Backend, sel.Arch)
	}

	qOut := args.NumHeads * args.HeadDim
	kvOut := args.NumKVHeads * args.HeadDim
	report := ResidentPrefillReport{
		SelectedBackend:    sel.Backend,
		AttentionSelection: sel,
	}

	var qTen, kTen, vTen Tensor
	if hooks.IsHostTensor(args.X) {
		report.InputWasHost = true
		var uploads int
		qTen, kTen, vTen, uploads, err = hooks.UploadHostProjectedQKV(args.X, args.Wq, args.Wk, args.Wv, P, qOut, kvOut)
		if err != nil {
			return PrefillBatchResult{}, report, err
		}
		report.HostUploadCount += uploads
	} else {
		var uploads int
		qTen, kTen, vTen, uploads, err = hooks.ProjectQKVDevice(args.Wq, args.Wk, args.Wv, args.X, P)
		if err != nil {
			return PrefillBatchResult{}, report, err
		}
		report.HostUploadCount += uploads
	}

	startPos := args.StartPos

	qRopedDev, err := hooks.RoPEPanelDevice(qTen, P, startPos, args.NumHeads, args.HeadDim, args.RopeTheta)
	if err != nil {
		return PrefillBatchResult{}, report, err
	}
	kRopedDev, err := hooks.RoPEPanelDevice(kTen, P, startPos, args.NumKVHeads, args.HeadDim, args.RopeTheta)
	if err != nil {
		return PrefillBatchResult{}, report, err
	}
	report.RoPEDispatches = 2

	if args.KV != nil {
		if err := hooks.AppendKVPanelDevice(args.KV, args.Layer, kTen, kRopedDev, vTen, startPos, P); err != nil {
			return PrefillBatchResult{}, report, err
		}
		report.KVAppendDispatches = 1
	}

	ctxDev, err := hooks.AttentionPanelDevice(qRopedDev, kRopedDev, vTen, args.KV, args.Layer, P, startPos, args.NumHeads, args.NumKVHeads, args.HeadDim, args.Scale, sel.Backend)
	if err != nil {
		return PrefillBatchResult{}, report, err
	}
	report.AttentionDispatches = 1

	outTen := ctxDev
	if args.Wo.buf != nil {
		var woUploads int
		outTen, woUploads, err = hooks.ProjectOutputDevice(args.Wo, ctxDev, P)
		if err != nil {
			return PrefillBatchResult{}, report, err
		}
		report.HostUploadCount += woUploads
	}

	return PrefillBatchResult{
		Output:  outTen,
		Context: ctxDev,
		Tokens:  P,
	}, report, nil
}

// ropePanelF32 computes batched HF non-interleaved RoPE over row-major [tokens, nHeads*headDim]
// on slices using cpuRefRopeTables and applyRope so simulated device-hook execution is bit-exact
// to cpuBackend.RoPE.
func ropePanelF32(src []float32, tokens, startPos, nHeads, headDim int, theta float64) []float32 {
	out := append([]float32(nil), src...)
	stride := nHeads * headDim
	for t := 0; t < tokens; t++ {
		cos, sin := cpuRefRopeTables.row(theta, headDim, startPos+t)
		rowBase := t * stride
		for h := 0; h < nHeads; h++ {
			hBase := rowBase + h*headDim
			applyRope(out[hBase:hBase+headDim], cos, sin)
		}
	}
	return out
}
