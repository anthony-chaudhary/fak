package compute

import (
	"encoding/hex"
	"errors"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"
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

func TestReleaseKVDeviceSlicesPreservesLayerOrderAndClearsPositions(t *testing.T) {
	type deviceSlice struct {
		name     string
		released bool
	}
	keys := []deviceSlice{{name: "k0"}, {name: "k1"}}
	rawKeys := []deviceSlice{{name: "raw0"}, {name: "raw1"}}
	values := []deviceSlice{{name: "v0"}, {name: "v1"}}
	positions := []int{4, 5}
	var released []string

	releaseKVDeviceSlices(keys, rawKeys, values, &positions, func(slice *deviceSlice) {
		released = append(released, slice.name)
		slice.released = true
	})

	wantOrder := []string{"k0", "raw0", "v0", "k1", "raw1", "v1"}
	if !slices.Equal(released, wantOrder) {
		t.Fatalf("release order = %v, want %v", released, wantOrder)
	}
	for _, group := range [][]deviceSlice{keys, rawKeys, values} {
		for _, slice := range group {
			if !slice.released {
				t.Fatalf("slice %q was not released", slice.name)
			}
		}
	}
	if positions != nil {
		t.Fatalf("positions = %v, want nil", positions)
	}
}

// TestComputeKernelSourceHashCache pins the kernel source-hash cache identity
// semantics: byte-identical sources share one entry and compile once; an
// edited source rehashes and recompiles under its own entry; no stale hit
// ever serves a program compiled from different source text.
func TestComputeKernelSourceHashCache(t *testing.T) {
	const srcV1 = "float sum(float a, float b) { return a + b; }"
	const srcV2 = "float sum(float a, float b) { return a - b; }"

	t.Run("HashIdentityIsExactAndContentKeyed", func(t *testing.T) {
		h := hashKernelSource(srcV1)
		got := hex.EncodeToString(h[:])
		if got != "30cf4a7fd66256e50ff7cd470459fb7917da2199e015ec74329cd858796bb155" {
			t.Fatalf("sha256(%q) = %s, want pinned vector", srcV1, got)
		}
		if hashKernelSource(srcV1) != hashKernelSource(srcV1) {
			t.Fatal("equal sources hashed to different values")
		}
		if hashKernelSource(srcV1) == hashKernelSource(srcV2) {
			t.Fatal("distinct sources shared one hash")
		}
	})

	t.Run("SameSourceServesFromCacheWithoutRecompiling", func(t *testing.T) {
		cache := newKernelProgramCache[string]()
		compiles := 0
		compile := func(source string) (string, error) {
			compiles++
			return "compiled:" + source, nil
		}

		first, err := cache.Compile(srcV1, compile)
		if err != nil {
			t.Fatalf("first Compile failed: %v", err)
		}
		second, err := cache.Compile(srcV1, compile)
		if err != nil {
			t.Fatalf("second Compile failed: %v", err)
		}
		if first != "compiled:"+srcV1 || second != first {
			t.Fatalf("Compile results = %q, %q; want both %q", first, second, "compiled:"+srcV1)
		}
		if compiles != 1 {
			t.Fatalf("identical source compiled %d times, want 1", compiles)
		}
		if cache.Len() != 1 {
			t.Fatalf("cache holds %d entries for one source, want 1", cache.Len())
		}
	})

	t.Run("ChangedSourceRecompilesAndNoStaleHit", func(t *testing.T) {
		cache := newKernelProgramCache[string]()
		compiles := 0
		compile := func(source string) (string, error) {
			compiles++
			return "compiled:" + source, nil
		}

		v1, err := cache.Compile(srcV1, compile)
		if err != nil {
			t.Fatalf("Compile(v1) failed: %v", err)
		}
		v2, err := cache.Compile(srcV2, compile)
		if err != nil {
			t.Fatalf("Compile(v2) failed: %v", err)
		}
		if v2 != "compiled:"+srcV2 {
			t.Fatalf("edited source served %q, want %q (stale hit)", v2, "compiled:"+srcV2)
		}
		again, err := cache.Compile(srcV1, compile)
		if err != nil {
			t.Fatalf("re-Compile(v1) failed: %v", err)
		}
		if again != v1 {
			t.Fatalf("original source served %q after edit, want unchanged %q", again, v1)
		}
		if compiles != 2 {
			t.Fatalf("two distinct sources compiled %d times, want 2", compiles)
		}
		if cache.Len() != 2 {
			t.Fatalf("cache holds %d entries for two sources, want 2", cache.Len())
		}
	})

	t.Run("FailedCompileStoresNothing", func(t *testing.T) {
		var errRefused = errors.New("kernel compile refused")
		cache := newKernelProgramCache[string]()
		refuse := true
		compile := func(source string) (string, error) {
			if refuse {
				return "", errRefused
			}
			return "compiled:" + source, nil
		}

		if _, err := cache.Compile(srcV1, compile); !errors.Is(err, errRefused) {
			t.Fatalf("failed Compile returned %v, want sentinel errRefused", err)
		}
		if cache.Len() != 0 {
			t.Fatalf("failed compile stored %d entries, want 0", cache.Len())
		}

		refuse = false
		got, err := cache.Compile(srcV1, compile)
		if err != nil {
			t.Fatalf("retry Compile failed: %v", err)
		}
		if got != "compiled:"+srcV1 || cache.Len() != 1 {
			t.Fatalf("retry stored %q with %d entries, want %q with 1", got, cache.Len(), "compiled:"+srcV1)
		}
	})

	t.Run("ConcurrentSameSourceCompilesOnce", func(t *testing.T) {
		cache := newKernelProgramCache[string]()
		var compiles atomic.Int32
		compile := func(source string) (string, error) {
			compiles.Add(1)
			time.Sleep(time.Millisecond) // widen the window so unserialized designs double-compile
			return "compiled:" + source, nil
		}

		const workers = 8
		results := make([]string, workers)
		var wg sync.WaitGroup
		for i := range workers {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				got, err := cache.Compile(srcV1, compile)
				if err != nil {
					t.Errorf("goroutine %d Compile failed: %v", i, err)
					return
				}
				results[i] = got
			}(i)
		}
		wg.Wait()

		if compiles.Load() != 1 {
			t.Fatalf("contended identical source compiled %d times, want 1", compiles.Load())
		}
		for i, got := range results {
			if got != "compiled:"+srcV1 {
				t.Fatalf("goroutine %d result = %q, want %q", i, got, "compiled:"+srcV1)
			}
		}
		if cache.Len() != 1 {
			t.Fatalf("cache holds %d entries after contended same-source compiles, want 1", cache.Len())
		}
	})
}
