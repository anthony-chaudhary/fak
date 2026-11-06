package compute

import (
	"errors"
	"math"
)

var errKVFP8Attention = errors.New("compute: staged FP8 attention refused")

type kvFP8AttentionQuery struct {
	backend *kvFP8BackendIdentity
	device  int
	storage kvFP8Allocation
	root    any
}

// This ephemeral checked F32 view is valid only inside use. status is the
// cumulative device numerical word for every dependent producer in that scope.
type kvFP8AttentionOutput struct {
	storage, status        kvFP8Allocation
	tokens, heads, headDim int
}

// The transaction retains this holder BEFORE allocation/submission. prior keeps
// its original raw/K/V producer owners. Neither the Q owner nor the output can
// disappear if a callback panics and its caller catches that panic.
type kvFP8AttentionRoots struct {
	prior, query any
	output       kvFP8Allocation
}

func kvFP8AttentionBytes(tokens, heads, kvheads, hd int) (int, error) {
	if tokens <= 0 || heads <= 0 || kvheads <= 0 || heads%kvheads != 0 ||
		hd != 256 || kvheads != 4 || heads > math.MaxInt32/hd || tokens > math.MaxInt32/(heads*hd) {
		return 0, ErrKVFP8Geometry
	}
	n := tokens * heads * hd
	if n > int(^uint(0)>>1)/4 {
		return 0, ErrKVFP8Geometry
	}
	return n * 4, nil
}

// kvFP8AttendEager supplies the missing Q/output lifetime around one append
// transaction's checked read window. The output is fresh unpublished F32
// workspace, never a Tensor/Buffer or an allocation transferred to the caller.
// use may enqueue dependent work only; checked completion/numerical status must
// cover every read before this helper retires output. use must not save pointers,
// Free/Recycle owners, recurse into this transaction, or acquire cudaMu.
//
// peakBudget includes this owner's current old+stage+workspace plus this output.
// Q, unrelated sessions and allocations made inside use need separate admission.
// All callers already hold ForwardExecutionOwner and cudaMu. No public route
// calls this helper; it remains an eager correctness baseline, not a speed claim.
func kvFP8AttendEager(t *kvFP8AppendTransaction, layer int, q kvFP8AttentionQuery, heads int, scale float32,
	peakBudget int64, launch func(kvFP8AppendLayer, kvFP8Allocation, kvFP8Allocation, int, float32) int,
	use func(kvFP8AttentionOutput) int) (err error) {
	if err := t.enter(layer, true); err != nil {
		return err
	}
	normalReturn := false
	defer func() {
		t.busy = false
		if !normalReturn {
			t.uncertain(errKVFP8Attention, 0)
		}
	}()
	err = func() error {
		bytes, shapeErr := kvFP8AttentionBytes(t.tokens, heads, t.owner.heads, t.owner.headDim)
		if shapeErr != nil {
			return t.uncertain(shapeErr, 0)
		}
		if launch == nil || use == nil || q.root == nil || q.backend != t.owner.backend ||
			q.device != t.owner.device || q.storage.ptr == nil || q.storage.bytes != bytes ||
			!(scale > 0) || !kvFP8Finite(scale) {
			return t.uncertain(errKVFP8Attention, 0)
		}
		current := t.owner.allocatedBytes()
		if peakBudget < current || int64(bytes) > peakBudget-current {
			return t.uncertain(errKVFP8Budget, 0)
		}
		root := &kvFP8AttentionRoots{prior: t.inputs[layer].root, query: q.root,
			output: kvFP8Allocation{bytes: bytes}}
		t.inputs[layer].root = root
		if allocErr := t.ops.eager.owner.allocate(&root.output, bytes, -1, 1); allocErr != nil {
			return t.uncertain(allocErr, 0)
		}
		if root.output.ptr == nil || root.output.bytes != bytes {
			return t.uncertain(errKVFP8Attention, 0)
		}
		// No callback or unlock occurs between these two state transitions.
		t.busy = false
		readErr := t.consume(layer, func(k kvFP8AppendLayer) int {
			return launch(k, q.storage, root.output, heads, scale)
		})
		t.busy = true
		if readErr == nil {
			code := use(kvFP8AttentionOutput{root.output, t.stage.workspace[0], t.tokens, heads, t.owner.headDim})
			if t.failure != nil {
				readErr = t.failure
			} else if code != 0 {
				readErr = t.uncertain(&kvFP8ReceiptError{cause: errKVFP8Attention, status: code}, code)
			} else {
				readErr = t.checked()
			}
		}
		// A checked numerical failure is safe to retire, but the output was
		// never made observable to use if attention itself was rejected.
		if t.owner.uncertain || t.owner.backend.stopped.Load() {
			return readErr
		}
		disposition := t.ops.eager.owner.release(root.output)
		if disposition.retired {
			root.output = kvFP8Allocation{}
		}
		if disposition.status != 0 || !disposition.retired {
			return t.uncertain(&kvFP8ReceiptError{cause: errKVFP8Retirement, status: disposition.status}, disposition.status)
		}
		return readErr
	}()
	normalReturn = true
	return err
}
