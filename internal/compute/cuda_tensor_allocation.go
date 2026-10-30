//go:build cuda && cgo

package compute

import "sync/atomic"

func (c *cudaBackend) dalloc(nbytes int) *cudaBuf {
	return c.dallocClass(nbytes, MemoryUnknown, "dalloc")
}

// dallocWeight places an explicit weight buffer device-local while under the residency budget,
// else managed (deliberately, in upload order). budgetBytes==0 means unbounded -> always attempt
// device-local first. Caller holds cudaMu.
func (c *cudaBackend) dallocWeight(nbytes int) *cudaBuf {
	if c.budgetBytes > 0 && c.dlUsed+int64(nbytes) > c.budgetBytes {
		buf := c.dallocManagedClass(nbytes, MemoryOffload, "dallocManaged")
		c.accountWeightPlacement(buf, nbytes)
		return buf
	}
	buf := c.dallocClass(nbytes, MemoryWeights, "dallocWeight")
	c.accountWeightPlacement(buf, nbytes)
	return buf
}

func (c *cudaBackend) dev(shape []int, dt Dtype) (Tensor, *cudaBuf) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	// dev is currently used only by the ordinary F32 weight-upload path.
	// Preserve that allocation class on cudaBuf so strict consumers do not have
	// to infer mutability/lifetime from shape or cache membership.
	buf := c.dallocClass(n*dt.Bytes(), MemoryWeights, "f32-weight")
	return makeTensor(c, dt, RowMajor, append([]int(nil), shape...), nil, buf), buf
}

// devTr is dev() for an op OUTPUT: the buffer is registered as transient so Recycle() can
// return it to the pool at the next token boundary. Weights (Upload) deliberately use dev,
// not devTr, so they are never recycled out from under the resident-weight cache.
func (c *cudaBackend) devTr(shape []int, dt Dtype) (Tensor, *cudaBuf) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	b := c.dallocClass(n*dt.Bytes(), MemoryScratchpad, "transient")
	t := makeTensor(c, dt, RowMajor, append([]int(nil), shape...), nil, b)
	// Mark async: this output is enqueued on g_stream in the current fence generation, so it
	// reports Ready()==false until the next Read/Argmax drains the stream (#482).
	b.be = c
	b.bornGen = atomic.LoadUint64(&c.fenceGen)
	c.transient = append(c.transient, b)
	return t, b
}

// devTrDeviceOnly is the strict GDN scratch/output counterpart of devTr. It
// performs checked byte arithmetic and never substitutes cudaMallocManaged.
// Caller holds cudaMu.
func (c *cudaBackend) devTrDeviceOnly(shape []int, dt Dtype, site string) (Tensor, *cudaBuf, error) {
	nbytes, ok := qwen35GDNShapeBytes(shape, dt.Bytes())
	if !ok {
		return Tensor{}, nil, &Qwen35GDNGeometryError{Operand: site, Reason: "shape byte size overflows host allocation capacity"}
	}
	b, err := c.dallocDeviceOnlyClass(nbytes, MemoryScratchpad, site)
	if err != nil {
		return Tensor{}, nil, err
	}
	t := makeTensor(c, dt, RowMajor, append([]int(nil), shape...), nil, b)
	b.be = c
	b.bornGen = atomic.LoadUint64(&c.fenceGen)
	c.transient = append(c.transient, b)
	return t, b, nil
}

// devF16 is dev() for an F16-resident weight: an out*in*2-byte VRAM buffer carrying the
// requested Layout (so MatMul can read w.Layout to pick the HGEMM op). Weights use dev/devF16,
// never devTr, so the resident-weight cache is never recycled out from under them.
func (c *cudaBackend) devF16(shape []int, layout Layout) (Tensor, *cudaBuf) {
	n := 1
	for _, d := range shape {
		n *= d
	}
	buf := c.dallocWeight(n * F16.Bytes())
	return makeTensor(c, F16, layout, append([]int(nil), shape...), nil, buf), buf
}

// devQ4K allocates a resident Q4_K weight: a single nbytes-long uint8 buffer holding the raw GGUF
// super-block bytes (d/dmin/scales/codes all packed; no scale side-channel). nbytes is the size of
// the host byte slice (= (out*in/256)*144). The QuantSpec records the 256-elem super-block.
func (c *cudaBackend) devRawKQuant(dt Dtype, shape []int, nbytes int) (Tensor, *cudaBuf) {
	return c.devRawKQuantWithAllocator(dt, shape, nbytes, c.dallocWeight)
}

// devRawKQuantWithAllocator keeps the packed tensor descriptor identical for
// immutable weights and independently owned offload allocations.
func (c *cudaBackend) devRawKQuantWithAllocator(dt Dtype, shape []int, nbytes int, allocate func(int) *cudaBuf) (Tensor, *cudaBuf) {
	buf := allocate(nbytes)
	bits := 4
	switch dt {
	case Q2_K, IQ2_XXS, IQ2_XS:
		bits = 2
	case Q3_K:
		bits = 3
	case IQ1_M:
		bits = 1
	case Q5_K:
		bits = 5
	case Q6_K:
		bits = 6
	default:
		// callers pass packed k-quants only; the unlisted q4 family keeps the 4-bit default
	}
	q := &QuantSpec{Block: 256, Axis: 2, Bits: uint8(bits), Symmetric: false}
	return makeTensor(c, dt, RowMajor, append([]int(nil), shape...), q, buf), buf
}
