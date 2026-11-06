package compute

import "errors"

var errKVFP8Append = errors.New("compute: FP8 append transaction refused")

const kvFP8AppendFailure = -76201

// kvFP8AppendInput retains the actual source owners, not only native addresses.
// Its rotary identity attests which normalized-K producer made the rows.
// Only a reviewed producer binding may construct it; matching geometry alone
// cannot prove normalized K or the position/theta used for post-RoPE K.
type kvFP8AppendInput struct {
	backend               *kvFP8BackendIdentity
	device, first, tokens int
	contract              kvFP8RotaryContract
	raw, key, value       kvFP8Allocation
	root                  any
}

type kvFP8AppendOps struct {
	eager    kvFP8EagerOps
	validate func(kvFP8AppendInput, kvFP8Allocation) int
	ingest   func(dst [kvFP8Planes]kvFP8Allocation, prefix int, input kvFP8AppendInput, status kvFP8Allocation) int
	stop     func(int) // nonpanicking/non-Goexit latch closure; never reset or submit work
}

func (ops kvFP8AppendOps) complete() bool {
	return ops.eager.owner.allocate != nil && ops.eager.owner.release != nil &&
		ops.eager.begin != nil && ops.eager.copy != nil && ops.eager.finish != nil &&
		ops.validate != nil && ops.ingest != nil && ops.stop != nil
}

// kvFP8AppendLayer is a short-lived device-read capability, not a Tensor/Buffer.
// consume must enqueue only the admitted attention read and write its numerical
// result into status. It must not save these pointers, free/recycle any owner,
// acquire cudaMu recursively, or read an output before consume returns success.
type kvFP8AppendLayer struct {
	planes                                [kvFP8Planes]kvFP8Allocation
	status                                kvFP8Allocation
	layer, prefix, tokens, heads, headDim int
}

type kvFP8AppendTransaction struct {
	owner                         *kvFP8Owner
	stage                         *kvFP8Image
	ops                           kvFP8AppendOps
	inputs                        []kvFP8AppendInput
	before, tokens, next          int
	stamp                         uint64
	active, busy, ready, retained bool
	result                        kvFP8Completion
	failure                       error
	retainNext                    *kvFP8AppendTransaction
}

// Protected by cudaMu. Keep input Tensor/cudaBuf owners as well as destination
// planes after abnormal control transfer or an uncertain asynchronous outcome.
var kvFP8AppendRetained *kvFP8AppendTransaction

func (t *kvFP8AppendTransaction) retain() {
	if !t.retained {
		t.retained = true
		t.retainNext, kvFP8AppendRetained = kvFP8AppendRetained, t
	}
}

func (t *kvFP8AppendTransaction) uncertain(reason error, code int) error {
	if t.failure == nil {
		t.failure = reason
	} else if t.owner.cleanupFailure == nil {
		t.owner.cleanupFailure = reason
	}
	t.owner.primaryFailure = t.failure
	t.owner.retainUncertain()
	t.retain()
	if code == 0 {
		code = kvFP8AppendFailure
	}
	t.result = kvFP8Completion{status: code}
	t.ops.stop(code)
	return t.failure
}

func (t *kvFP8AppendTransaction) enter(layer int, ready bool) error {
	// An escaped handle cannot poison a later, unrelated transaction.
	if t == nil || !t.active {
		return errKVFP8Append
	}
	if t.failure != nil {
		return t.failure
	}
	if t.busy || t.ready != ready || layer != t.next || layer < 0 || layer >= t.owner.layers ||
		t.owner.stage != t.stage || t.owner.stamp.Load() != t.stamp ||
		t.owner.backend.stopped.Load() {
		return t.uncertain(errKVFP8Append, 0)
	}
	t.busy = true
	return nil
}

func (t *kvFP8AppendTransaction) checked() error {
	if t.owner.uncertain || t.owner.backend.stopped.Load() {
		return t.uncertain(errKVFP8Completion, 0)
	}
	t.result = t.ops.eager.finish(&t.stage.workspace[0])
	r := t.result
	if r.completed && r.numericalChecked && r.numerical != 0 {
		t.owner.numericalFailure = r.numerical
		t.owner.primaryFailure = ErrKVFP8NonFinite
		t.failure = ErrKVFP8NonFinite
	}
	if r.status != 0 || !r.completed || !r.numericalChecked {
		cause := &kvFP8ReceiptError{cause: errKVFP8Completion, status: r.status}
		if t.failure != nil {
			cause.cause = errors.Join(t.failure, errKVFP8Completion)
		}
		return t.uncertain(cause, r.status)
	}
	if t.failure != nil {
		return t.failure
	}
	return nil
}

// append validates ALL three F32 sources before copying or quantizing any tail.
// validate's checked completion is mandatory even though each quantizer also
// validates its own input: exact Kraw is independently retained for eviction.
func (t *kvFP8AppendTransaction) append(layer int, input kvFP8AppendInput) error {
	if err := t.enter(layer, false); err != nil {
		return err
	}
	// Seal this operation before a caller can recover a callback panic and
	// retry. A whole-run defer is too late when recovery happens inside run.
	// No recover call: original panic (including panic(nil)) and Goexit propagate.
	normalReturn := false
	defer func() {
		t.busy = false
		if !normalReturn {
			t.uncertain(errKVFP8Append, 0)
		}
	}()
	err := func() error {
		sizes, _, err := t.owner.planeBytes(t.tokens)
		if err != nil {
			return t.uncertain(err, 0)
		}
		if input.root == nil || input.backend != t.owner.backend || input.device != t.owner.device ||
			input.first != t.before || input.tokens != t.tokens || input.contract != t.owner.rotary[layer] {
			return t.uncertain(errKVFP8Append, 0)
		}
		for _, a := range [3]kvFP8Allocation{input.raw, input.key, input.value} {
			start := uintptr(a.ptr)
			if a.ptr == nil || a.bytes != sizes[kvFP8Raw] || uintptr(a.bytes) > ^uintptr(0)-start {
				return t.uncertain(ErrKVFP8Geometry, 0)
			}
		}
		t.inputs[layer] = input // strong roots installed BEFORE any native submission
		if code := t.ops.validate(input, t.stage.workspace[0]); code != 0 {
			return t.uncertain(&kvFP8ReceiptError{cause: errKVFP8Append, status: code}, code)
		}
		if err := t.checked(); err != nil {
			return err
		}
		if code := t.ops.ingest(t.stage.layers[layer], t.before, input, t.stage.workspace[0]); code != 0 {
			return t.uncertain(&kvFP8ReceiptError{cause: errKVFP8Append, status: code}, code)
		}
		if err := t.checked(); err != nil {
			return err
		}
		t.ready = true
		return nil
	}()
	normalReturn = true
	return err
}

// consume fences the attention read BEFORE the next layer may overwrite/recycle
// its producer inputs. The callback must retain its Q/output owner separately
// through completion or quarantine; this transaction owns only its KV/input set.
func (t *kvFP8AppendTransaction) consume(layer int, read func(kvFP8AppendLayer) int) error {
	if err := t.enter(layer, true); err != nil {
		return err
	}
	// Seal this operation before a caller can recover a callback panic and
	// retry. A whole-run defer is too late when recovery happens inside run.
	// No recover call: original panic (including panic(nil)) and Goexit propagate.
	normalReturn := false
	defer func() {
		t.busy = false
		if !normalReturn {
			t.uncertain(errKVFP8Append, 0)
		}
	}()
	err := func() error {
		if read == nil {
			return t.uncertain(errKVFP8Append, 0)
		}
		code := read(kvFP8AppendLayer{t.stage.layers[layer], t.stage.workspace[0],
			layer, t.before, t.tokens, t.owner.heads, t.owner.headDim})
		if t.failure != nil {
			return t.failure
		} // recursive misuse may have stopped admission
		if code != 0 {
			return t.uncertain(&kvFP8ReceiptError{cause: errKVFP8Append, status: code}, code)
		}
		if err := t.checked(); err != nil {
			return err
		}
		t.ready = false
		t.next++
		return nil
	}()
	normalReturn = true
	return err
}

// kvFP8AppendEager reserves a complete replacement before entering run. run
// executes layers serially under the caller's existing ForwardExecutionOwner
// and cudaMu domain. It may call append/consume only for the next attention
// layer. No generic view, snapshot, clone, eviction or graph can observe stage.
//
// Canonical positions are mandatory. Every admitted abort poisons this owner,
// even known OOM, so a partially advanced hybrid model can never reuse old KV.
// Hybrid GDN/graph dependents MUST be invalidated by the future model binding.
// The whole-image correctness baseline is deliberately not a capacity-growth,
// in-place decode, graph, zero-D2H, or performance-qualified implementation.
func kvFP8AppendEager(o *kvFP8Owner, first, tokens int, peakBudget int64, ops kvFP8AppendOps,
	run func(*kvFP8AppendTransaction) error) (err error) {
	if !ops.complete() || run == nil {
		return errKVFP8Append
	}
	if err := o.usable(); err != nil {
		return err
	}
	if first != o.image.positions || tokens <= 0 || first < 0 || tokens > int(^uint(0)>>1)-first {
		return ErrKVFP8Geometry
	}
	if _, _, err := o.planeBytes(first + tokens); err != nil {
		return err
	}
	for _, c := range o.rotary {
		if c.producer != kvFP8QwenPartialPanelV1 || c.headDim != 256 || c.rotaryDim != 64 ||
			o.heads != 4 || o.headDim != 256 {
			return errKVFP8Rotary
		}
	}
	var transaction *kvFP8AppendTransaction
	normalReturn := false
	defer func() {
		if transaction != nil {
			transaction.active = false
			if !normalReturn || o.uncertain {
				o.retainUncertain()
				transaction.retain()
				// A Go panic/Goexit after enqueue must also stop native Free/
				// Recycle entry points, not merely lock-free Go Ready checks.
				if !normalReturn {
					ops.stop(kvFP8AppendFailure)
				}
			} else {
				transaction.inputs = nil
			}
		}
	}()
	err = o.replace(first+tokens, []int{4}, peakBudget, ops.eager.owner,
		func(dst, src *kvFP8Image) kvFP8Completion {
			t := &kvFP8AppendTransaction{owner: o, stage: dst, ops: ops,
				inputs: make([]kvFP8AppendInput, o.layers), before: first, tokens: tokens,
				stamp: o.stamp.Load(), active: true}
			transaction = t
			defer func() { t.active = false }()
			if code := ops.eager.begin(dst.workspace[0]); code != 0 {
				t.uncertain(&kvFP8ReceiptError{cause: errKVFP8Append, status: code}, code)
				return t.result
			}
			for layer := range src.layers {
				for plane, a := range src.layers[layer] {
					if a.bytes == 0 {
						continue
					}
					if code := ops.eager.copy(dst.layers[layer][plane], 0, a, 0, a.bytes); code != 0 {
						t.uncertain(&kvFP8ReceiptError{cause: errKVFP8Append, status: code}, code)
						return t.result
					}
				}
			}
			if t.checked() != nil {
				return t.result
			} // prefix and status complete
			runErr := run(t)
			if t.failure != nil {
				// run may have queued work after observing a numerical error.
				// Cover that tail before any known-complete discard. Never
				// enqueue a new fence after uncertain/native terminal work.
				if !o.uncertain {
					t.checked()
				}
				return t.result
			}
			if runErr != nil || t.busy || t.ready || t.next != o.layers {
				if runErr == nil {
					runErr = errKVFP8Append
				}
				t.uncertain(runErr, 0)
				return t.result
			}
			t.checked() // includes any final stream work issued by run
			t.active = false
			return t.result
		}, true)
	normalReturn = true
	if transaction != nil && transaction.failure != nil && err != nil && !errors.Is(err, transaction.failure) {
		return errors.Join(transaction.failure, err)
	}
	return err
}
