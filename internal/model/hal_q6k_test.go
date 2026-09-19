package model

import (
	"encoding/binary"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type denseQ6KRecordingBackend struct {
	compute.Backend
	deviceMemory, uploadDtype, q6k bool
	q6Uploads                      [][]byte
	matMulDtypes, batchDtypes      []compute.Dtype
}

func (b *denseQ6KRecordingBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = b.deviceMemory
	c.UploadDtype = b.uploadDtype
	return c
}
func (b *denseQ6KRecordingBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.Q6_K && b.q6k
}
func (b *denseQ6KRecordingBackend) Upload(src compute.Tensor, as compute.Dtype) compute.Tensor {
	if as == compute.Q6_K {
		host, ok := src.Buf().(compute.HostBuffer)
		if !ok {
			panic("Q6_K upload source is not host-addressable")
		}
		in := host.I8()
		raw := make([]byte, len(in))
		for i := range in {
			raw[i] = byte(in[i])
		}
		b.q6Uploads = append(b.q6Uploads, raw)
	}
	return b.Backend.Upload(src, as)
}
func (b *denseQ6KRecordingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matMulDtypes = append(b.matMulDtypes, w.Dtype)
	return b.Backend.MatMul(w, x)
}
func (b *denseQ6KRecordingBackend) BatchedMatMul(w, x compute.Tensor, p int) compute.Tensor {
	b.batchDtypes = append(b.batchDtypes, w.Dtype)
	return b.Backend.BatchedMatMul(w, x, p)
}

type optionalDenseQ6KBackend struct {
	*denseQ6KRecordingBackend
	native bool
}

func (b *optionalDenseQ6KBackend) SupportsQ6KMatMul() bool { return b.native }

func denseQ6KHALFixture() (string, []byte, *Model) {
	const out, in = 4, qkK
	name := "model.layers.0.mlp.down_proj.weight"
	raw := make([]byte, out*(in/qkK)*q6kBlockBytes)
	for block := 0; block < out*(in/qkK); block++ {
		binary.LittleEndian.PutUint16(raw[(block+1)*q6kBlockBytes-2:], 0x3c00)
	}
	fallback := make([]byte, out*in*4)
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ6K)
	meta := tensorMeta{Dtype: "F32", Shape: []int{out, in}, Nbytes: len(fallback)}
	m := &Model{manifest: map[string]tensorMeta{name: meta, "lm_head.weight": meta}, raw: fallback, kqw: map[string]*kQuantTensor{name: qt, "lm_head.weight": qt}}
	return name, raw, m
}

func TestHALDenseQ6KAdmitsNativePacked(t *testing.T) {
	name, raw, m := denseQ6KHALFixture()
	rec := &denseQ6KRecordingBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q6k: true}
	s := &Session{M: m, Backend: &optionalDenseQ6KBackend{denseQ6KRecordingBackend: rec, native: true}, halW: map[string]compute.Tensor{}}
	got := s.matWeightHAL(name)
	if got.Dtype != compute.Q6_K {
		t.Fatalf("matWeightHAL dtype = %s, want Q6_K", got.Dtype)
	}
	if len(rec.q6Uploads) != 1 {
		t.Fatalf("Q6_K uploads = %d, want 1", len(rec.q6Uploads))
	}
	if len(rec.q6Uploads[0]) != 4*q6kBlockBytes {
		t.Fatalf("Q6_K upload bytes = %d, want four %d-byte blocks", len(rec.q6Uploads[0]), q6kBlockBytes)
	}
	if q6kBlockBytes != 210 {
		t.Fatalf("q6kBlockBytes = %d, want GGUF Q6_K block size 210", q6kBlockBytes)
	}
	for i := range raw {
		if rec.q6Uploads[0][i] != raw[i] {
			t.Fatalf("Q6_K upload changed checkpoint byte %d", i)
		}
	}
	_ = s.matWeightHAL(name)
	if len(rec.q6Uploads) != 1 {
		t.Fatalf("cached Q6_K weight uploaded %d times, want once", len(rec.q6Uploads))
	}
	if head := s.lmHeadMatHAL(); head.Dtype != compute.Q6_K {
		t.Fatalf("lmHeadMatHAL dtype = %s, want Q6_K", head.Dtype)
	}
	if len(rec.q6Uploads) != 2 {
		t.Fatalf("dense projection plus lm head Q6_K uploads = %d, want 2", len(rec.q6Uploads))
	}
}

func TestHALDenseQ6KDeclinesUnsupported(t *testing.T) {
	tests := []struct {
		name                                          string
		deviceMemory, uploadDtype, q6k, optionalProbe bool
	}{
		{name: "cpu-ref-dtype-probe", uploadDtype: true, q6k: true},
		{name: "no-quantized-upload", deviceMemory: true, q6k: true},
		{name: "no-q6k-device-kernel", deviceMemory: true, uploadDtype: true},
		{name: "optional-bundle-missing", deviceMemory: true, uploadDtype: true, q6k: true, optionalProbe: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			name, _, m := denseQ6KHALFixture()
			rec := &denseQ6KRecordingBackend{Backend: compute.Default(), deviceMemory: tc.deviceMemory, uploadDtype: tc.uploadDtype, q6k: tc.q6k}
			var backend compute.Backend = rec
			if tc.optionalProbe {
				backend = &optionalDenseQ6KBackend{denseQ6KRecordingBackend: rec}
			}
			s := &Session{M: m, Backend: backend, halW: map[string]compute.Tensor{}}
			if got := s.matWeightHAL(name); got.Dtype != compute.F32 {
				t.Fatalf("declined dense Q6_K dtype = %s, want F32 fallback", got.Dtype)
			}
			if head := s.lmHeadMatHAL(); head.Dtype != compute.F32 {
				t.Fatalf("declined Q6_K lm head dtype = %s, want F32 fallback", head.Dtype)
			}
			if len(rec.q6Uploads) != 0 {
				t.Fatalf("declined dense Q6_K performed %d packed uploads, want zero", len(rec.q6Uploads))
			}
		})
	}
}

func TestHALDenseQ6KDecodeAndPrefill(t *testing.T) {
	name, _, m := denseQ6KHALFixture()
	// No optional Q6_K interface: this is the CUDA/ROCm-style common dtype contract.
	rec := &denseQ6KRecordingBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q6k: true}
	s := &Session{M: m, Backend: rec, halW: map[string]compute.Tensor{}}
	w := s.matWeightHAL(name)
	x := compute.NewF32(compute.Default(), []int{qkK}, make([]float32, qkK))
	decode := s.Backend.MatMul(w, x)
	defer s.Backend.Free(decode)
	panel := compute.NewF32(compute.Default(), []int{2, qkK}, make([]float32, 2*qkK))
	prefill := s.Backend.BatchedMatMul(w, panel, 2)
	defer s.Backend.Free(prefill)
	if len(rec.matMulDtypes) != 1 || rec.matMulDtypes[0] != compute.Q6_K {
		t.Fatalf("decode MatMul weight dtypes = %v, want [Q6_K]", rec.matMulDtypes)
	}
	if len(rec.batchDtypes) != 1 || rec.batchDtypes[0] != compute.Q6_K {
		t.Fatalf("prefill BatchedMatMul weight dtypes = %v, want [Q6_K]", rec.batchDtypes)
	}
}
