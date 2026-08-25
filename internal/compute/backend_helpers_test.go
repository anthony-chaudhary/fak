package compute

import (
	"testing"
	"unsafe"
)

func TestNormalizeUploadClass(t *testing.T) {
	class, site := normalizeUploadClass("", "", "cuda-upload-")
	if class != MemoryUnknown || site != "cuda-upload-unknown" {
		t.Fatalf("empty class/site normalized to %q, %q", class, site)
	}
	class, site = normalizeUploadClass(MemoryKVCache, "caller-site", "ignored-")
	if class != MemoryKVCache || site != "caller-site" {
		t.Fatalf("explicit class/site changed to %q, %q", class, site)
	}
}

func TestBackendTensorHelpersPreserveShapeAndTransferSemantics(t *testing.T) {
	source := Tensor{Shape: []int{2, 3}}
	out := makeF32TensorLike(fakeDevice{}, source, devBuf{ready: true})
	source.Shape[0] = 9
	if out.Dtype != F32 || out.Layout != RowMajor || out.Shape[0] != 2 || out.Backend().Name() != "fake-dev" {
		t.Fatalf("makeF32TensorLike = %#v", out)
	}

	copies := 0
	if got := finishF32Upload(out, []float32{1}, func([]float32) { copies++ }); got.buf != out.buf || copies != 1 {
		t.Fatalf("non-empty upload returned %#v with %d copies", got, copies)
	}
	finishF32Upload(out, nil, func([]float32) { copies++ })
	if copies != 1 {
		t.Fatalf("empty upload invoked copy; copies=%d", copies)
	}
}

func TestReadF32TensorPreservesHostFastPathAndDeviceValidation(t *testing.T) {
	host := NewF32(Default(), []int{2}, []float32{1, 2})
	called := false
	if got := readF32Tensor(host, func(Buffer, []float32) { called = true }); len(got) != 2 || got[0] != 1 || got[1] != 2 || called {
		t.Fatalf("host read = %v, device callback called=%v", got, called)
	}

	device := makeTensor(fakeDevice{}, F32, RowMajor, []int{2}, nil, devBuf{ready: true})
	got := readF32Tensor(device, func(buf Buffer, dst []float32) {
		if _, ok := buf.(devBuf); !ok {
			t.Fatalf("device buffer type = %T", buf)
		}
		dst[0], dst[1] = 3, 4
	})
	if len(got) != 2 || got[0] != 3 || got[1] != 4 {
		t.Fatalf("device read = %v", got)
	}
}

func TestClearDeviceSliceMetadata(t *testing.T) {
	length, capacity := 3, 5
	clearDeviceSliceMetadata(&length, &capacity)
	if length != 0 || capacity != 0 {
		t.Fatalf("metadata = len %d cap %d", length, capacity)
	}
}

func TestReleaseDeviceSlice(t *testing.T) {
	value := 1
	pointer := unsafe.Pointer(&value)
	want := pointer
	length, capacity := 3, 5
	var released unsafe.Pointer
	releaseDeviceSlice(&pointer, &length, &capacity, func(got unsafe.Pointer) { released = got })
	if released != want || pointer != nil || length != 0 || capacity != 0 {
		t.Fatalf("released=%p pointer=%p len=%d cap=%d", released, pointer, length, capacity)
	}
}
