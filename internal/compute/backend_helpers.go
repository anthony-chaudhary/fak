package compute

import (
	"crypto/sha256"
	"sync"
	"unsafe"
)

func normalizeUploadClass(class MemoryClass, site, sitePrefix string) (MemoryClass, string) {
	if class == "" {
		class = MemoryUnknown
	}
	if site == "" {
		site = sitePrefix + string(class)
	}
	return class, site
}

func makeF32TensorLike(be Backend, source Tensor, buf Buffer) Tensor {
	return makeTensor(be, F32, RowMajor, append([]int(nil), source.Shape...), nil, buf)
}

func finishF32Upload(out Tensor, values []float32, copyToDevice func([]float32)) Tensor {
	if len(values) > 0 {
		copyToDevice(values)
	}
	return out
}

func readF32Tensor(t Tensor, copyFromDevice func(Buffer, []float32)) []float32 {
	if hb, ok := t.buf.(*hostBuf); ok {
		return hb.f32
	}
	out := make([]float32, t.Numel())
	copyFromDevice(t.buf, out)
	return out
}

func clearDeviceSliceMetadata(length, capacity *int) {
	*length = 0
	*capacity = 0
}

func releaseDeviceSlice(pointer *unsafe.Pointer, length, capacity *int, release func(unsafe.Pointer)) {
	if *pointer != nil {
		release(*pointer)
		*pointer = nil
	}
	clearDeviceSliceMetadata(length, capacity)
}

func releaseKVDeviceSlices[D any](keys, rawKeys, values []D, positions *[]int, release func(*D)) {
	for layer := range keys {
		release(&keys[layer])
		release(&rawKeys[layer])
		release(&values[layer])
	}
	*positions = nil
}

// Compute kernel source-hash cache (INSPIRE-ONLY clean-room adaptation of the
// mlx Metal kernel cache, which keys compiled pipelines on the kernel source
// so identical sources compile once: ml-explore/mlx
// mlx/backend/common/metal_kernel.cpp @f624f8ec47e8049253c93360868b2bfea8a5b061,
// MIT; no upstream code copied). Backends re-emit equivalent programs per
// call site; keying the compiled-program cache on a content hash of the
// source lets byte-identical sources share one entry while edited sources
// recompile under their own.

// kernelSourceHash identifies a compute kernel's source text by content:
// byte-identical sources share one hash regardless of call site.
type kernelSourceHash [sha256.Size]byte

func hashKernelSource(source string) kernelSourceHash {
	return sha256.Sum256([]byte(source))
}

// kernelProgramCache[P] maps kernel source hashes to compiled programs P.
// Compile holds the mutex across lookup and compilation, so each distinct
// source compiles exactly once per cache and every later lookup for that
// source serves the cached program without invoking compile. A changed
// source hashes differently and recompiles under its own entry; a failed
// compilation stores nothing, so it never shadows a retry.
type kernelProgramCache[P any] struct {
	mu       sync.Mutex
	programs map[kernelSourceHash]P
}

func newKernelProgramCache[P any]() *kernelProgramCache[P] {
	return &kernelProgramCache[P]{programs: make(map[kernelSourceHash]P)}
}

// Compile returns the compiled program for source. An exact source-hash hit
// returns the cached program; a miss invokes compile(source), stores its
// result under the source's hash, and returns it.
func (c *kernelProgramCache[P]) Compile(source string, compile func(string) (P, error)) (P, error) {
	var zero P
	h := hashKernelSource(source)
	c.mu.Lock()
	defer c.mu.Unlock()
	if program, ok := c.programs[h]; ok {
		return program, nil
	}
	program, err := compile(source)
	if err != nil {
		return zero, err
	}
	c.programs[h] = program
	return program, nil
}

// Len reports the number of distinct compiled programs held.
func (c *kernelProgramCache[P]) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.programs)
}
