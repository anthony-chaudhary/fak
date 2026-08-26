package compute

import "unsafe"

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
