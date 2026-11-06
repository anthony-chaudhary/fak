package compute

import "errors"

// kvFP8EagerOps separates actual ordering from the native ABI. All callbacks
// require the same cudaMu domain as the owner; they must not recursively lock.
// Status zero on a submission means accepted, never completion. finish(nil)
// is restricted to a pure bit-copy clone of a previously validated image.
type kvFP8EagerOps struct {
	owner    kvFP8OwnerOps
	begin    func(status kvFP8Allocation) int
	copy     func(dst kvFP8Allocation, dstOffset int, src kvFP8Allocation, srcOffset, bytes int) int
	rotary   func(dst, raw kvFP8Allocation, before, from, end, heads int, contract kvFP8RotaryContract, status kvFP8Allocation) int
	quantize func(codes kvFP8Allocation, codeOffset int, scales kvFP8Allocation, scaleOffset int,
		src kvFP8Allocation, rows, headDim int, status kvFP8Allocation) int
	finish func(status *kvFP8Allocation) kvFP8Completion
}

func (ops kvFP8EagerOps) complete() bool {
	return ops.owner.allocate != nil && ops.owner.release != nil &&
		ops.begin != nil && ops.copy != nil && ops.rotary != nil && ops.quantize != nil && ops.finish != nil
}

// kvFP8EvictEager is a complete-image, eager correctness baseline. It does not
// mutate the old image, compact in place, or expose an intermediate layer.
// Physical peak includes old+new planes, one device status and one F32 suffix.
// Binding/geometry and budget refusal happen before any callback.
func kvFP8EvictEager(o *kvFP8Owner, from, count int, peakBudget int64, ops kvFP8EagerOps) (int, error) {
	if !ops.complete() {
		return 0, errKVFP8Owner
	}
	plan, err := o.planEvict(from, count)
	if err != nil || plan == nil {
		return 0, err
	}
	return kvFP8ApplyEvictionEager(plan, peakBudget, ops)
}

func kvFP8ApplyEvictionEager(plan *kvFP8EvictionPlan, peakBudget int64, ops kvFP8EagerOps) (int, error) {
	if plan == nil || plan.owner == nil || !ops.complete() {
		return 0, errKVFP8Owner
	}
	o := plan.owner
	if err := o.usable(); err != nil {
		return 0, err
	}
	if o.stamp.Load() != plan.stamp || o.image.positions != plan.before {
		return 0, errKVFP8Owner
	}
	units, _, err := o.planeBytes(1)
	if err != nil {
		return 0, err
	}
	tail := plan.before - plan.end
	workspace := []int{4}
	if tail > 0 {
		sizes, _, shapeErr := o.planeBytes(tail)
		if shapeErr != nil {
			return 0, shapeErr
		}
		workspace = append(workspace, sizes[kvFP8Raw])
	}
	// A numerical failure can coexist with failed host-status retirement.
	// Preserve it in diagnostics and the returned error, not just a native code.
	var primary error
	check := func(r kvFP8Completion) bool {
		if r.completed && r.numericalChecked && r.numerical != 0 {
			primary = ErrKVFP8NonFinite
			o.primaryFailure = primary
			if r.status != 0 {
				o.cleanupFailure = &kvFP8ReceiptError{cause: errKVFP8Completion, status: r.status}
			}
		}
		return r.status == 0 && r.completed && r.numericalChecked && r.numerical == 0
	}
	err = plan.apply(workspace, peakBudget, ops.owner, func(dst, src *kvFP8Image, p *kvFP8EvictionPlan) kvFP8Completion {
		status := dst.workspace[0]
		if code := ops.begin(status); code != 0 {
			return kvFP8Completion{status: code}
		}
		var last kvFP8Completion
		for layer := range src.layers {
			copySpan := func(plane, srcRow, dstRow, rows int) int {
				if rows == 0 {
					return 0
				}
				return ops.copy(dst.layers[layer][plane], dstRow*units[plane],
					src.layers[layer][plane], srcRow*units[plane], rows*units[plane])
			}
			// Kraw and V/scales are exact bit-copies; no widening/requantization.
			for _, plane := range []int{kvFP8Raw, kvFP8Value, kvFP8ValueScale} {
				if code := copySpan(plane, 0, 0, p.from); code != 0 {
					return kvFP8Completion{status: code}
				}
				if code := copySpan(plane, p.end, p.from, tail); code != 0 {
					return kvFP8Completion{status: code}
				}
			}
			for _, plane := range []int{kvFP8Key, kvFP8KeyScale} {
				if code := copySpan(plane, 0, 0, p.from); code != 0 {
					return kvFP8Completion{status: code}
				}
			}
			if tail == 0 {
				continue
			}
			scratch := dst.workspace[1]
			if code := ops.rotary(scratch, src.layers[layer][kvFP8Raw], p.before, p.from, p.end, o.heads, p.rotary[layer], status); code != 0 {
				return kvFP8Completion{status: code}
			}
			last = ops.finish(&status)
			if !check(last) {
				return last
			} // NO quantizer read before this check
			if code := ops.quantize(dst.layers[layer][kvFP8Key], p.from*units[kvFP8Key],
				dst.layers[layer][kvFP8KeyScale], p.from*units[kvFP8KeyScale],
				scratch, tail*o.heads, o.headDim, status); code != 0 {
				return kvFP8Completion{status: code}
			}
			last = ops.finish(&status)
			if !check(last) {
				return last
			} // NO scratch overwrite/next layer before this check
		}
		if tail == 0 {
			last = ops.finish(&status)
			check(last)
		}
		return last
	})
	if err != nil {
		if primary != nil && !errors.Is(err, primary) {
			return 0, errors.Join(primary, err)
		}
		return 0, err
	}
	return plan.end - plan.from, nil
}

// kvFP8CloneEager bit-copies all five planes and fences source reads before
// publication. No numerical kernel or status allocation is used in this arm.
func kvFP8CloneEager(o *kvFP8Owner, peakBudget int64, ops kvFP8EagerOps) (*kvFP8Owner, error) {
	if ops.owner.allocate == nil || ops.owner.release == nil || ops.copy == nil || ops.finish == nil {
		return nil, errKVFP8Owner
	}
	return o.clone(peakBudget, ops.owner, func(dst, src *kvFP8Image) kvFP8Completion {
		for layer := range src.layers {
			for plane, a := range src.layers[layer] {
				if a.bytes == 0 {
					continue
				}
				if code := ops.copy(dst.layers[layer][plane], 0, a, 0, a.bytes); code != 0 {
					return kvFP8Completion{status: code}
				}
			}
		}
		return ops.finish(nil)
	})
}
