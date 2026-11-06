package compute

import (
	"errors"
	"math"
)

var errKVFP8QwenIngress = errors.New("compute: FP8 Qwen producer ingress refused")

// Inputs are unnormalized Q/K, V, and exact F32 head gains. Norm/rotary results
// are produced here, never trusted from caller-supplied producer metadata.
type kvFP8QwenInput struct {
	backend       *kvFP8BackendIdentity
	device, heads int
	operands      [5]kvFP8Allocation // Q, K, V, Q gain, K gain
	root          any
}

type kvFP8QwenStage struct {
	input         kvFP8QwenInput
	work          [4]kvFP8Allocation // normalized Q/K, rotated Q/K
	status        kvFP8Allocation
	tokens, first int
	contract      kvFP8RotaryContract
	epsilon       float32
}

type kvFP8QwenRoots struct {
	prior any
	input kvFP8QwenInput
	work  [4]kvFP8Allocation
}

type kvFP8QwenOps struct {
	admit                func() int // query-only, no submission/allocation; must refuse timestamp profiling
	validate, norm, rope func(kvFP8QwenStage) int
	attention            func(kvFP8AppendLayer, kvFP8Allocation, kvFP8Allocation, int, float32) int
}

// The existing append transaction already owns ForwardExecutionOwner/cudaMu.
// This eager correctness leaf binds the original block-RMSNorm and panel
// partial-RoPE arithmetic. It is not the alternative fused vLLM QK kernel.
// The sole output is the checked callback-scoped attention view. No public
// Tensor, reusable scratch, model entry or graph selection is created.
// peakBudget additionally includes all four temporary planes and attention
// output. Source tensors, host status readbacks and use allocations are separate.
func kvFP8QwenIngressEager(t *kvFP8AppendTransaction, layer int, in kvFP8QwenInput,
	epsilon, scale float32, peakBudget int64, ops kvFP8QwenOps,
	use func(kvFP8AttentionOutput) int) (err error) {
	if err := t.enter(layer, false); err != nil {
		return err
	}
	normalReturn := false
	var roots *kvFP8QwenRoots
	defer func() {
		t.busy = false
		if !normalReturn {
			t.uncertain(errKVFP8QwenIngress, 0)
		}
	}()
	err = func() error {
		qbytes, shapeErr := kvFP8AttentionBytes(t.tokens, in.heads, t.owner.heads, t.owner.headDim)
		if shapeErr != nil {
			return t.uncertain(shapeErr, 0)
		}
		c := t.owner.rotary[layer]
		if in.heads != 24 || in.root == nil || in.backend != t.owner.backend ||
			in.device != t.owner.device || !(epsilon > 0) || !kvFP8Finite(epsilon) ||
			!(scale > 0) || !kvFP8Finite(scale) || ops.admit == nil || ops.validate == nil ||
			ops.norm == nil || ops.rope == nil || ops.attention == nil || use == nil ||
			c.producer != kvFP8QwenPartialPanelV1 || c.headDim != 256 || c.rotaryDim != 64 ||
			(c.mainLayer < 0 || c.mainLayer/4 != layer || c.mainLayer%4 != 3) || !(math.Float64frombits(c.thetaBits) > 0) ||
			math.IsInf(math.Float64frombits(c.thetaBits), 0) {
			return t.uncertain(errKVFP8QwenIngress, 0)
		}
		sizes, _, shapeErr := t.owner.planeBytes(t.tokens)
		if shapeErr != nil {
			return t.uncertain(shapeErr, 0)
		}
		want := [5]int{qbytes, sizes[kvFP8Raw], sizes[kvFP8Raw], 256 * 4, 256 * 4}
		for i, a := range in.operands {
			if a.ptr == nil || a.bytes != want[i] || uintptr(a.bytes) > ^uintptr(0)-uintptr(a.ptr) {
				return t.uncertain(ErrKVFP8Geometry, 0)
			}
		}
		temporary := 2*int64(qbytes) + 2*int64(sizes[kvFP8Raw])
		current := t.owner.allocatedBytes()
		if peakBudget < current || temporary > peakBudget-current ||
			int64(qbytes) > peakBudget-current-temporary {
			return t.uncertain(errKVFP8Budget, 0)
		}
		if code := ops.admit(); code != 0 {
			return t.uncertain(&kvFP8ReceiptError{cause: errKVFP8QwenIngress, status: code}, code)
		}
		if t.failure != nil || t.owner.uncertain || t.owner.backend.stopped.Load() {
			return t.uncertain(errKVFP8QwenIngress, 0)
		}
		roots = &kvFP8QwenRoots{prior: t.inputs[layer].root, input: in}
		t.inputs[layer].root = roots // before allocator, launch or abnormal unwind
		for i, n := range [4]int{qbytes, sizes[kvFP8Raw], qbytes, sizes[kvFP8Raw]} {
			roots.work[i].bytes = n
			if e := t.ops.eager.owner.allocate(&roots.work[i], n, -1, i+2); e != nil {
				return t.uncertain(e, 0)
			}
			if roots.work[i].ptr == nil || roots.work[i].bytes != n {
				return t.uncertain(errKVFP8QwenIngress, 0)
			}
		}
		stage := kvFP8QwenStage{input: in, work: roots.work, status: t.stage.workspace[0],
			tokens: t.tokens, first: t.before, contract: c, epsilon: epsilon}
		for _, step := range []func(kvFP8QwenStage) int{ops.validate, ops.norm, ops.rope} {
			if code := step(stage); code != 0 {
				return t.uncertain(&kvFP8ReceiptError{cause: errKVFP8QwenIngress, status: code}, code)
			}
			if e := t.checked(); e != nil {
				return e
			}
		}
		prepared := kvFP8AppendInput{backend: in.backend, device: in.device,
			first: t.before, tokens: t.tokens, contract: c,
			raw: roots.work[1], key: roots.work[3], value: in.operands[2], root: roots}
		t.busy = false // no callback/unlock gap before nested operation re-enters
		appendErr := t.append(layer, prepared)
		t.busy = true
		if appendErr != nil {
			return appendErr
		}
		query := kvFP8AttentionQuery{backend: in.backend, device: in.device, storage: roots.work[2], root: roots}
		t.busy = false
		// allocatedBytes does not count these separately retained temporary planes.
		attentionErr := kvFP8AttendEager(t, layer, query, in.heads, scale, peakBudget-temporary, ops.attention, use)
		t.busy = true
		return attentionErr
	}()
	// Known completion permits retirement even on numerical rejection. Unknown
	// completion, contract/native failure and non-normal exits retain all roots.
	if roots != nil && !t.owner.uncertain && !t.owner.backend.stopped.Load() {
		for i := len(roots.work) - 1; i >= 0; i-- {
			if roots.work[i].ptr == nil {
				continue
			}
			disposition := t.ops.eager.owner.release(roots.work[i])
			if disposition.retired {
				roots.work[i] = kvFP8Allocation{}
			}
			if disposition.status != 0 || !disposition.retired {
				err = t.uncertain(&kvFP8ReceiptError{cause: errKVFP8Retirement, status: disposition.status}, disposition.status)
				break
			}
		}
	}
	normalReturn = true
	return err
}
