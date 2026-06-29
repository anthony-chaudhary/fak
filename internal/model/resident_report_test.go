package model

import "testing"

// TestResidentReportByteMath pins the resident-inventory tallies: a model with known q4kw,
// q8w, kqw, and f32-manifest entries must report their exact byte/param counts and the
// correct decode-bytes-per-token (= q4k + q8 + raw expert-quant matmul bytes). This is the
// cheap observability witness for the loader's resident split, runnable in ms without any
// 27B load.
func TestResidentReportByteMath(t *testing.T) {
	m := &Model{
		manifest: map[string]tensorMeta{},
		q4w:      nil,
		q8w:      map[string]*q8Tensor{},
		q4kw:     map[string]*q4kTensor{},
		kqw:      map[string]*kQuantTensor{},
	}
	// One Q4_K tensor: out=2, in=256 (1 super-block), raw = 2*1*144 = 288 bytes, 512 params.
	raw := make([]byte, 2*1*q4kBlockBytes)
	m.q4kw["a"] = quantizeQ4KFromRaw(raw, 2, 256)
	// One Q8_0 tensor: out=4, in=32 (1 block), codes = 4*32 = 128 int8, scales = 4 f32.
	q8 := &q8Tensor{out: 4, in: 32, nblk: 1, q: make([]int8, 4*32), d: make([]float32, 4)}
	m.q8w["b"] = q8
	// One raw-resident expert Q8_0 tensor: out=3, in=32, raw = 3*(2+32) bytes, 96 params.
	kq := quantizeKQuantFromRaw(make([]byte, 3*q8_0BlockBytes), 3, 32, kindQ8_0)
	m.kqw["c"] = kq
	// Two f32 manifest tensors.
	m.manifest["n1"] = tensorMeta{Nbytes: 400}
	m.manifest["n2"] = tensorMeta{Nbytes: 80}

	r := m.ResidentReport()
	if r.Q4KTensors != 1 || r.Q4KBytes != 288 || r.Q4KParams != 512 {
		t.Errorf("q4k: tensors=%d bytes=%d params=%d want 1/288/512", r.Q4KTensors, r.Q4KBytes, r.Q4KParams)
	}
	wantQ8Bytes := int64(len(q8.q)) + int64(len(q8.d))*4 // 128 + 16
	if r.Q8Tensors != 1 || r.Q8Bytes != wantQ8Bytes || r.Q8Params != 128 {
		t.Errorf("q8: tensors=%d bytes=%d params=%d want 1/%d/128", r.Q8Tensors, r.Q8Bytes, r.Q8Params, wantQ8Bytes)
	}
	if r.KQuantTensors != 1 || r.KQuantBytes != int64(len(kq.raw)) || r.KQuantParams != 96 {
		t.Errorf("kquant: tensors=%d bytes=%d params=%d want 1/%d/96", r.KQuantTensors, r.KQuantBytes, r.KQuantParams, len(kq.raw))
	}
	if r.F32Tensors != 2 || r.F32Bytes != 480 {
		t.Errorf("f32: tensors=%d bytes=%d want 2/480", r.F32Tensors, r.F32Bytes)
	}
	if r.TotalResidentBytes != 288+wantQ8Bytes+int64(len(kq.raw))+480 {
		t.Errorf("total=%d want %d", r.TotalResidentBytes, 288+wantQ8Bytes+int64(len(kq.raw))+480)
	}
	if r.DecodeBytesPerToken != 288+wantQ8Bytes+int64(len(kq.raw)) {
		t.Errorf("decode bytes/token=%d want %d (q4k+q8+kquant)", r.DecodeBytesPerToken, 288+wantQ8Bytes+int64(len(kq.raw)))
	}
	// Ceiling at a notional 100 GB/s: tok/s = 100 / (GiB/tok * 1.0737). Sanity: a 288+128+...
	// stream is tiny so the ceiling must be enormous; just check it's positive and finite.
	if ceil := r.DecodeTokSCeiling(100); ceil <= 0 {
		t.Errorf("ceiling = %v, want positive", ceil)
	}
	if FormatResidentReport(r) == "" {
		t.Error("FormatResidentReport returned empty")
	}
}
