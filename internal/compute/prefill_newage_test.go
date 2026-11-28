package compute

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"math"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=20ms lane=default
func TestCUDAAttentionBackendSelectionAndBlackwell(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		major     int
		minor     int
		regime    CUDAAttentionRegime
		opts      CUDAAttentionOptions
		want      CUDAAttentionKernel
		blackwell bool
		resident  bool
	}{
		{
			name:   "unsupported_volta_sm70",
			major:  7,
			minor:  0,
			regime: CUDARegimePrefill,
			opts:   CUDAAttentionOptions{HeadDim: 128, Tokens: 256},
			want:   CUDAAttentionUnsupported,
		},
		{
			name:      "ampere_sm80_prefill_panel",
			major:     8,
			minor:     0,
			regime:    CUDARegimePrefill,
			opts:      CUDAAttentionOptions{HeadDim: 128, Tokens: 256},
			want:      CUDAAttentionFlashPrefillPanel,
			blackwell: false,
			resident:  true,
		},
		{
			name:      "ada_sm89_hd256_mma_prefill",
			major:     8,
			minor:     9,
			regime:    CUDARegimePrefill,
			opts:      CUDAAttentionOptions{HeadDim: 256, Tokens: 512},
			want:      CUDAAttentionMMA256Prefill,
			blackwell: false,
			resident:  true,
		},
		{
			name:      "hopper_sm90_decode",
			major:     9,
			minor:     0,
			regime:    CUDARegimeDecode,
			opts:      CUDAAttentionOptions{HeadDim: 128, Tokens: 1},
			want:      CUDAAttentionFlashDecode,
			blackwell: false,
			resident:  true,
		},
		{
			name:      "blackwell_sm100_prefill_panel",
			major:     10,
			minor:     0,
			regime:    CUDARegimePrefill,
			opts:      CUDAAttentionOptions{HeadDim: 128, Tokens: 512},
			want:      CUDAAttentionFlashPrefillPanel,
			blackwell: true,
			resident:  true,
		},
		{
			name:      "blackwell_sm100_hd256_mma_prefill",
			major:     10,
			minor:     0,
			regime:    CUDARegimePrefill,
			opts:      CUDAAttentionOptions{HeadDim: 256, Tokens: 512},
			want:      CUDAAttentionMMA256Prefill,
			blackwell: true,
			resident:  true,
		},
		{
			name:      "blackwell_sm103_prefill_typed_kv",
			major:     10,
			minor:     3,
			regime:    CUDARegimePrefill,
			opts:      CUDAAttentionOptions{HeadDim: 128, Tokens: 256, KVPrecision: KVPrecisionBF16},
			want:      CUDAAttentionTypedKV,
			blackwell: true,
			resident:  true,
		},
		{
			name:      "blackwell_sm103_decode",
			major:     10,
			minor:     3,
			regime:    CUDARegimeDecode,
			opts:      CUDAAttentionOptions{HeadDim: 128, Tokens: 1},
			want:      CUDAAttentionFlashDecode,
			blackwell: true,
			resident:  true,
		},
		{
			name:      "blackwell_sm120_paged_fp8_prefill",
			major:     12,
			minor:     0,
			regime:    CUDARegimePrefill,
			opts:      CUDAAttentionOptions{HeadDim: 128, Tokens: 512, KVDtype: FP8, Paged: true},
			want:      CUDAAttentionPagedFP8Prefill,
			blackwell: true,
			resident:  true,
		},
		{
			name:      "blackwell_sm120_panel_prefill",
			major:     12,
			minor:     0,
			regime:    CUDARegimePrefill,
			opts:      CUDAAttentionOptions{HeadDim: 128, Tokens: 1024},
			want:      CUDAAttentionFlashPrefillPanel,
			blackwell: true,
			resident:  true,
		},
		{
			name:      "blackwell_sm120_decode_typed_kv",
			major:     12,
			minor:     0,
			regime:    CUDARegimeDecode,
			opts:      CUDAAttentionOptions{HeadDim: 128, Tokens: 1, KVPrecision: KVPrecisionF16},
			want:      CUDAAttentionTypedKV,
			blackwell: true,
			resident:  true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := SelectCUDAAttentionKernel(tc.major, tc.minor, tc.regime, tc.opts)
			if got != tc.want {
				t.Fatalf("SelectCUDAAttentionKernel(%d,%d,%s) = %s, want %s", tc.major, tc.minor, tc.regime, got, tc.want)
			}
			sel := ResolveCUDAAttentionSelection(tc.major, tc.minor, tc.regime, tc.opts)
			if sel.Blackwell != tc.blackwell {
				t.Fatalf("Blackwell = %v, want %v (arch=%s)", sel.Blackwell, tc.blackwell, sel.Arch)
			}
			if sel.GPUResident != tc.resident {
				t.Fatalf("GPUResident = %v, want %v", sel.GPUResident, tc.resident)
			}
		})
	}

	tierCases := []struct {
		tier    string
		wantMaj int
		wantMin int
		wantOK  bool
	}{
		{"sm_100", 10, 0, true},
		{"sm_103", 10, 3, true},
		{"sm_120", 12, 0, true},
		{"CUDA sm120", 12, 0, true},
		{"sm89", 8, 9, true},
		{"cpu", 0, 0, false},
	}
	for _, tc := range tierCases {
		maj, min, ok := parseCUDATierMajorMinor(tc.tier)
		if ok != tc.wantOK || maj != tc.wantMaj || min != tc.wantMin {
			t.Fatalf("parseCUDATierMajorMinor(%q) = (%d,%d,%v), want (%d,%d,%v)", tc.tier, maj, min, ok, tc.wantMaj, tc.wantMin, tc.wantOK)
		}
	}

	// Verify real non-test callers reference SelectCUDAAttentionKernel / ResolveCUDAAttentionSelection.
	nonTestCallers := 0
	for _, file := range []string{"cuda.go", "prefill_cuda_resident.go", "prefill_midmarket.go"} {
		f, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", file, err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			if id, ok := call.Fun.(*ast.Ident); ok {
				if id.Name == "SelectCUDAAttentionKernel" || id.Name == "ResolveCUDAAttentionSelection" {
					nonTestCallers++
				}
			}
			return true
		})
	}
	if nonTestCallers < 3 {
		t.Fatalf("expected >= 3 non-test callers of CUDA attention selector, got %d", nonTestCallers)
	}
}

type simDeviceBuf struct {
	data []float32
}

func (b *simDeviceBuf) Ready() bool { return true }

type simDeviceKV struct {
	cfg   KVConfig
	seq   int
	pos   []int
	kRaw  [][]float32
	kRope [][]float32
	v     [][]float32
}

func (s *simDeviceKV) Len() int { return s.seq }
func (s *simDeviceKV) AppendKV(layer int, kRaw, kRoPE, v Tensor, pos int) {
	if s.seq != pos {
		panic(fmt.Sprintf("simDeviceKV expected pos=%d, got %d", s.seq, pos))
	}
	rowKRaw := kRaw.buf.(*simDeviceBuf).data
	rowKRope := kRoPE.buf.(*simDeviceBuf).data
	rowV := v.buf.(*simDeviceBuf).data
	s.kRaw[layer] = append(s.kRaw[layer], rowKRaw...)
	s.kRope[layer] = append(s.kRope[layer], rowKRope...)
	s.v[layer] = append(s.v[layer], rowV...)
	if layer == s.cfg.NumLayers-1 || s.cfg.NumLayers == 1 {
		s.seq++
		s.pos = append(s.pos, pos)
	}
}
func (s *simDeviceKV) KeysView(layer int) Tensor {
	return Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{s.seq, s.cfg.NumKVHeads * s.cfg.HeadDim}, buf: &simDeviceBuf{data: append([]float32(nil), s.kRope[layer]...)}}
}
func (s *simDeviceKV) ValuesView(layer int) Tensor {
	return Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{s.seq, s.cfg.NumKVHeads * s.cfg.HeadDim}, buf: &simDeviceBuf{data: append([]float32(nil), s.v[layer]...)}}
}
func (s *simDeviceKV) Pos() []int { return append([]int(nil), s.pos...) }
func (s *simDeviceKV) Evict(from, n int) int {
	return 0
}
func (s *simDeviceKV) Clone() KVStore {
	cp := &simDeviceKV{
		cfg:   s.cfg,
		seq:   s.seq,
		pos:   append([]int(nil), s.pos...),
		kRaw:  make([][]float32, len(s.kRaw)),
		kRope: make([][]float32, len(s.kRope)),
		v:     make([][]float32, len(s.v)),
	}
	for l := range s.kRaw {
		cp.kRaw[l] = append([]float32(nil), s.kRaw[l]...)
		cp.kRope[l] = append([]float32(nil), s.kRope[l]...)
		cp.v[l] = append([]float32(nil), s.v[l]...)
	}
	return cp
}
func (s *simDeviceKV) ResidentBytes() int64 {
	var n int64
	for l := range s.kRaw {
		n += int64(len(s.kRaw[l])+len(s.kRope[l])+len(s.v[l])) * 4
	}
	return n
}
func (s *simDeviceKV) Free() {}

// simDeviceBackend simulates a CUDA-semantics GPU backend in pure Go and records
// every dispatch, submit, and host Read/Upload call to enforce zero host round-trips
// inside PrefillBatch.
type simDeviceBackend struct {
	tier       string
	cpu        *cpuBackend
	failOnRead atomic.Bool
	reads      atomic.Uint64
	uploads    atomic.Uint64
	dispatches atomic.Uint64
	submits    atomic.Uint64
	lastReport ResidentPrefillReport
}

func newSimDeviceBackend(tier string) *simDeviceBackend {
	return &simDeviceBackend{tier: tier, cpu: &cpuBackend{}}
}

func (b *simDeviceBackend) Name() string            { return "cuda-sim" }
func (b *simDeviceBackend) Tier() string            { return b.tier }
func (b *simDeviceBackend) Class() CorrectnessClass { return Reference }
func (b *simDeviceBackend) Caps() Caps {
	return Caps{Async: true, DeviceMemory: true, FusedAttn: true, BatchedPrefill: true}
}

func (b *simDeviceBackend) Upload(t Tensor, as Dtype) Tensor {
	b.uploads.Add(1)
	if hb, ok := t.buf.(HostBuffer); ok {
		cp := append([]float32(nil), hb.F32()...)
		return makeTensor(b, F32, t.Layout, append([]int(nil), t.Shape...), nil, &simDeviceBuf{data: cp})
	}
	if db, ok := t.buf.(*simDeviceBuf); ok {
		cp := append([]float32(nil), db.data...)
		return makeTensor(b, F32, t.Layout, append([]int(nil), t.Shape...), nil, &simDeviceBuf{data: cp})
	}
	panic("simDeviceBackend.Upload: unknown buffer")
}

func (b *simDeviceBackend) Host(t Tensor) ([]float32, bool) {
	if hb, ok := t.buf.(HostBuffer); ok {
		return hb.F32(), true
	}
	return nil, false
}

func (b *simDeviceBackend) Read(t Tensor) []float32 {
	b.reads.Add(1)
	if b.failOnRead.Load() {
		panic("simDeviceBackend.Read called while GPU-resident PrefillBatch was active")
	}
	if hb, ok := t.buf.(HostBuffer); ok {
		return append([]float32(nil), hb.F32()...)
	}
	return append([]float32(nil), t.buf.(*simDeviceBuf).data...)
}

func (b *simDeviceBackend) Free(t Tensor) {}
func (b *simDeviceBackend) NewKV(cfg KVConfig) KVStore {
	return &simDeviceKV{
		cfg:   cfg,
		kRaw:  make([][]float32, cfg.NumLayers),
		kRope: make([][]float32, cfg.NumLayers),
		v:     make([][]float32, cfg.NumLayers),
	}
}

func (b *simDeviceBackend) MatMul(w, x Tensor) Tensor {
	b.dispatches.Add(1)
	b.submits.Add(1)
	wt := b.asCPUTensor(w)
	xt := b.asCPUTensor(x)
	out := b.cpu.MatMul(wt, xt)
	return makeTensor(b, F32, RowMajor, out.Shape, nil, &simDeviceBuf{data: out.buf.(HostBuffer).F32()})
}

func (b *simDeviceBackend) BatchedMatMul(w, X Tensor, P int) Tensor {
	b.dispatches.Add(1)
	b.submits.Add(1)
	wt := b.asCPUTensor(w)
	xt := b.asCPUTensor(X)
	out := b.cpu.BatchedMatMul(wt, xt, P)
	return makeTensor(b, F32, RowMajor, out.Shape, nil, &simDeviceBuf{data: out.buf.(HostBuffer).F32()})
}

func (b *simDeviceBackend) RMSNorm(x, w Tensor, eps float32) Tensor {
	b.dispatches.Add(1)
	out := b.cpu.RMSNorm(b.asCPUTensor(x), b.asCPUTensor(w), eps)
	return makeTensor(b, F32, RowMajor, out.Shape, nil, &simDeviceBuf{data: out.buf.(HostBuffer).F32()})
}
func (b *simDeviceBackend) RoPE(x Tensor, pos, nH, hd int, theta float64) Tensor {
	b.dispatches.Add(1)
	out := b.cpu.RoPE(b.asCPUTensor(x), pos, nH, hd, theta)
	return makeTensor(b, F32, RowMajor, out.Shape, nil, &simDeviceBuf{data: out.buf.(HostBuffer).F32()})
}
func (b *simDeviceBackend) SwiGLU(g, u Tensor) Tensor {
	b.dispatches.Add(1)
	out := b.cpu.SwiGLU(b.asCPUTensor(g), b.asCPUTensor(u))
	return makeTensor(b, F32, RowMajor, out.Shape, nil, &simDeviceBuf{data: out.buf.(HostBuffer).F32()})
}
func (b *simDeviceBackend) AddInPlace(dst, src Tensor) {}
func (b *simDeviceBackend) AddBias(dst, bias Tensor)   {}
func (b *simDeviceBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	panic("unused in panel prefill test")
}
func (b *simDeviceBackend) Argmax(logits Tensor) int { return 0 }

func (b *simDeviceBackend) BackendCounters() (BackendCounterSnapshot, error) {
	return BackendCounterSnapshot{
		ComputeDispatches: b.dispatches.Load(),
		DispatchSubmits:   b.submits.Load(),
	}, nil
}
func (b *simDeviceBackend) DeviceSubmits() uint64 { return b.submits.Load() }

func (b *simDeviceBackend) asCPUTensor(t Tensor) Tensor {
	if _, ok := t.buf.(HostBuffer); ok {
		return NewF32(b.cpu, t.Shape, append([]float32(nil), t.buf.(HostBuffer).F32()...))
	}
	return NewF32(b.cpu, t.Shape, append([]float32(nil), t.buf.(*simDeviceBuf).data...))
}

func (b *simDeviceBackend) DeviceArch() (major, minor int, ok bool) {
	return parseCUDATierMajorMinor(b.tier)
}
func (b *simDeviceBackend) IsHostTensor(t Tensor) bool {
	_, ok := b.Host(t)
	return ok
}
func (b *simDeviceBackend) UploadHostProjectedQKV(x Tensor, wq, wk, wv Tensor, P, qOut, kvOut int) (q, k, v Tensor, uploads int, err error) {
	xDev := b.Upload(x, F32)
	uploads = 1
	q, k, v, wu, err := b.ProjectQKVDevice(wq, wk, wv, xDev, P)
	return q, k, v, uploads + wu, err
}
func (b *simDeviceBackend) ProjectQKVDevice(wq, wk, wv, x Tensor, P int) (q, k, v Tensor, weightUploads int, err error) {
	ensureDev := func(t Tensor) Tensor {
		if b.IsHostTensor(t) {
			weightUploads++
			return b.Upload(t, F32)
		}
		return t
	}
	wqDev := ensureDev(wq)
	wkDev := ensureDev(wk)
	wvDev := ensureDev(wv)
	return b.BatchedMatMul(wqDev, x, P), b.BatchedMatMul(wkDev, x, P), b.BatchedMatMul(wvDev, x, P), weightUploads, nil
}
func (b *simDeviceBackend) RoPEPanelDevice(x Tensor, tokens, startPos, nHeads, headDim int, theta float64) (Tensor, error) {
	b.dispatches.Add(1)
	b.submits.Add(1)
	src := x.buf.(*simDeviceBuf).data
	rot := ropePanelF32(src, tokens, startPos, nHeads, headDim, theta)
	return makeTensor(b, F32, RowMajor, []int{tokens, nHeads * headDim}, nil, &simDeviceBuf{data: rot}), nil
}
func (b *simDeviceBackend) AppendKVPanelDevice(kv KVStore, layer int, kRaw, kRoPE, v Tensor, startPos, tokens int) error {
	b.dispatches.Add(1)
	b.submits.Add(1)
	dkv, ok := kv.(*simDeviceKV)
	if !ok {
		return fmt.Errorf("simDeviceBackend expects *simDeviceKV")
	}
	if dkv.seq != startPos {
		return fmt.Errorf("simDeviceKV startPos mismatch: got %d, want %d", startPos, dkv.seq)
	}
	dkv.kRaw[layer] = append(dkv.kRaw[layer], kRaw.buf.(*simDeviceBuf).data...)
	dkv.kRope[layer] = append(dkv.kRope[layer], kRoPE.buf.(*simDeviceBuf).data...)
	dkv.v[layer] = append(dkv.v[layer], v.buf.(*simDeviceBuf).data...)
	dkv.seq += tokens
	return nil
}
func (b *simDeviceBackend) AttentionPanelDevice(q, kFallback, vFallback Tensor, kv KVStore, layer, tokens, startPos, nH, nKV, hd int, scale float32, backend CUDAAttentionKernel) (Tensor, error) {
	b.dispatches.Add(1)
	b.submits.Add(1)
	if !backend.GPUResident() {
		return Tensor{}, fmt.Errorf("expected GPU-resident backend, got %s", backend)
	}
	qData := q.buf.(*simDeviceBuf).data
	var allK, allV []float32
	var totalPositions int
	effStart := 0
	if kv != nil {
		dkv := kv.(*simDeviceKV)
		allK = dkv.kRope[layer]
		allV = dkv.v[layer]
		totalPositions = dkv.seq
		effStart = startPos
	} else {
		allK = kFallback.buf.(*simDeviceBuf).data
		allV = vFallback.buf.(*simDeviceBuf).data
		totalPositions = tokens
	}
	ctx := prefillBatchCausalAttention(qData, allK, allV, tokens, effStart, totalPositions, nH, nKV, hd, scale)
	return makeTensor(b, F32, RowMajor, []int{tokens, nH * hd}, nil, &simDeviceBuf{data: ctx}), nil
}
func (b *simDeviceBackend) ProjectOutputDevice(wo, ctx Tensor, P int) (Tensor, int, error) {
	uploads := 0
	woDev := wo
	if b.IsHostTensor(woDev) {
		uploads++
		woDev = b.Upload(woDev, F32)
	}
	return b.BatchedMatMul(woDev, ctx, P), uploads, nil
}
func (b *simDeviceBackend) PrefillBatch(args PrefillBatchArgs) (PrefillBatchResult, error) {
	b.failOnRead.Store(true)
	defer b.failOnRead.Store(false)
	res, rep, err := prefillResidentBatch(b, args, KVPrecisionF32, false)
	b.lastReport = rep
	return res, err
}

// fak-test:runtime fast est=40ms lane=default
func TestGPUResidentPrefillBatchZeroRoundTripAndParity(t *testing.T) {
	t.Parallel()

	cpu := &cpuBackend{}
	sim := newSimDeviceBackend("sm_120")

	mkData := func(n int, bias float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = bias + 0.03*float32((i*7)%19-9)
		}
		return out
	}

	configs := []struct {
		name     string
		dModel   int
		nHeads   int
		nKVHeads int
		headDim  int
		P        int
		prefix   int
		withKV   bool
		withWo   bool
	}{
		{name: "mha_no_kv", dModel: 64, nHeads: 4, nKVHeads: 4, headDim: 16, P: 8, withKV: false, withWo: true},
		{name: "gqa_kv_fresh", dModel: 128, nHeads: 8, nKVHeads: 2, headDim: 16, P: 16, withKV: true, withWo: true},
		{name: "gqa_kv_continuation", dModel: 128, nHeads: 4, nKVHeads: 1, headDim: 32, P: 12, prefix: 5, withKV: true, withWo: false},
	}

	for _, cfg := range configs {
		cfg := cfg
		t.Run(cfg.name, func(t *testing.T) {
			qOut := cfg.nHeads * cfg.headDim
			kvOut := cfg.nKVHeads * cfg.headDim

			wqData := mkData(qOut*cfg.dModel, 0.02)
			wkData := mkData(kvOut*cfg.dModel, -0.01)
			wvData := mkData(kvOut*cfg.dModel, 0.015)
			woData := mkData(cfg.dModel*qOut, 0.005)
			xData := mkData(cfg.P*cfg.dModel, 0.1)

			var cpuKV, simKV KVStore
			if cfg.withKV {
				kvCfg := KVConfig{NumLayers: 1, NumKVHeads: cfg.nKVHeads, HeadDim: cfg.headDim, Precision: KVPrecisionF32}
				cpuKV = cpu.NewKV(kvCfg)
				simKV = sim.NewKV(kvCfg)
				if cfg.prefix > 0 {
					for p := 0; p < cfg.prefix; p++ {
						raw := mkData(kvOut, float32(p+1)*0.02)
						rot := ropePanelF32(raw, 1, p, cfg.nKVHeads, cfg.headDim, 10000.0)
						val := mkData(kvOut, float32(p+1)*-0.01)
						cpuKV.AppendKV(0, NewF32(cpu, []int{kvOut}, raw), NewF32(cpu, []int{kvOut}, rot), NewF32(cpu, []int{kvOut}, val), p)
						simKV.AppendKV(0, sim.Upload(NewF32(cpu, []int{kvOut}, raw), F32), sim.Upload(NewF32(cpu, []int{kvOut}, rot), F32), sim.Upload(NewF32(cpu, []int{kvOut}, val), F32), p)
					}
				}
			}

			cpuArgs := PrefillBatchArgs{
				X:          NewF32(cpu, []int{cfg.P, cfg.dModel}, xData),
				Wq:         NewF32(cpu, []int{qOut, cfg.dModel}, wqData),
				Wk:         NewF32(cpu, []int{kvOut, cfg.dModel}, wkData),
				Wv:         NewF32(cpu, []int{kvOut, cfg.dModel}, wvData),
				KV:         cpuKV,
				Layer:      0,
				StartPos:   cfg.prefix,
				NumHeads:   cfg.nHeads,
				NumKVHeads: cfg.nKVHeads,
				HeadDim:    cfg.headDim,
				RopeTheta:  10000.0,
				Scale:      float32(1.0 / math.Sqrt(float64(cfg.headDim))),
			}
			if cfg.withWo {
				cpuArgs.Wo = NewF32(cpu, []int{cfg.dModel, qOut}, woData)
			}

			cpuRes, err := PrefillBatch(cpu, cpuArgs)
			if err != nil {
				t.Fatalf("cpu PrefillBatch: %v", err)
			}

			// Upload all tensors onto the simulated GPU device BEFORE PrefillBatch so the
			// GPU-resident device path runs with zero host uploads and zero host reads.
			simArgs := PrefillBatchArgs{
				X:          sim.Upload(NewF32(cpu, []int{cfg.P, cfg.dModel}, xData), F32),
				Wq:         sim.Upload(NewF32(cpu, []int{qOut, cfg.dModel}, wqData), F32),
				Wk:         sim.Upload(NewF32(cpu, []int{kvOut, cfg.dModel}, wkData), F32),
				Wv:         sim.Upload(NewF32(cpu, []int{kvOut, cfg.dModel}, wvData), F32),
				KV:         simKV,
				Layer:      0,
				StartPos:   cfg.prefix,
				NumHeads:   cfg.nHeads,
				NumKVHeads: cfg.nKVHeads,
				HeadDim:    cfg.headDim,
				RopeTheta:  10000.0,
				Scale:      cpuArgs.Scale,
			}
			if cfg.withWo {
				simArgs.Wo = sim.Upload(NewF32(cpu, []int{cfg.dModel, qOut}, woData), F32)
			}

			readsBefore := sim.reads.Load()
			uploadsBefore := sim.uploads.Load()
			simRes, err := PrefillBatch(sim, simArgs)
			if err != nil {
				t.Fatalf("sim PrefillBatch: %v", err)
			}
			if got := sim.reads.Load() - readsBefore; got != 0 {
				t.Fatalf("PrefillBatch issued %d host Read calls, want 0", got)
			}
			if got := sim.uploads.Load() - uploadsBefore; got != 0 {
				t.Fatalf("PrefillBatch issued %d host Upload calls for resident inputs, want 0", got)
			}
			if sim.lastReport.HostReadCount != 0 || sim.lastReport.PerTokenHostTransfers != 0 {
				t.Fatalf("unexpected host transfer report: %+v", sim.lastReport)
			}
			if sim.lastReport.RoPEDispatches != 2 || sim.lastReport.AttentionDispatches != 1 {
				t.Fatalf("unexpected panel dispatch counts: %+v", sim.lastReport)
			}
			if !sim.lastReport.SelectedBackend.GPUResident() {
				t.Fatalf("expected GPU-resident attention backend, got %s", sim.lastReport.SelectedBackend)
			}

			gotOut := sim.Read(simRes.Output)
			wantOut := cpu.Read(cpuRes.Output)
			assertSlicesClose(t, "Output", gotOut, wantOut, 1e-5, 0.99999)

			gotCtx := sim.Read(simRes.Context)
			wantCtx := cpu.Read(cpuRes.Context)
			assertSlicesClose(t, "Context", gotCtx, wantCtx, 1e-5, 0.99999)

			if cfg.withKV {
				assertSlicesClose(t, "KV.KeysView", sim.Read(simKV.KeysView(0)), cpu.Read(cpuKV.KeysView(0)), 1e-5, 0.99999)
				assertSlicesClose(t, "KV.ValuesView", sim.Read(simKV.ValuesView(0)), cpu.Read(cpuKV.ValuesView(0)), 1e-5, 0.99999)
			}
		})
	}
}

func assertSlicesClose(t *testing.T, label string, got, want []float32, maxAbsTol, minCos float64) {
	t.Helper()
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("%s length mismatch: got %d want %d", label, len(got), len(want))
	}
	var maxAbs, dot, ng, nw float64
	for i := range got {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if d > maxAbs {
			maxAbs = d
		}
		dot += float64(got[i]) * float64(want[i])
		ng += float64(got[i]) * float64(got[i])
		nw += float64(want[i]) * float64(want[i])
	}
	cos := 1.0
	if ng > 0 && nw > 0 {
		cos = dot / (math.Sqrt(ng) * math.Sqrt(nw))
	}
	if maxAbs > maxAbsTol || cos < minCos {
		t.Fatalf("%s parity drift: maxAbs=%.6g (tol=%.6g), cosine=%.8f (min=%.8f)", label, maxAbs, maxAbsTol, cos, minCos)
	}
}

// fak-test:runtime fast est=15ms lane=default
func TestCUDAPrefillBatchAndInk2PQ4KSourceContracts(t *testing.T) {
	t.Parallel()

	// 1. Verify Q4_K MMQ dispatch contract in Ink2PDenseMMQKind and Ink2PMatMul.
	kind, ok := Ink2PDenseMMQKind(Q4_K, nil, 32, 4096, 4096, true)
	if !ok || kind != 4 {
		t.Fatalf("Ink2PDenseMMQKind(Q4_K, T=32, in=4096) = (%d, %v), want (4, true)", kind, ok)
	}
	if _, ok := Ink2PDenseMMQKind(Q4_K, nil, 8, 4096, 4096, true); ok {
		t.Fatal("Ink2PDenseMMQKind should reject T<16")
	}
	if _, ok := Ink2PDenseMMQKind(Q4_K, nil, 32, 4096, 300, true); ok {
		t.Fatal("Ink2PDenseMMQKind should reject in not divisible by 256")
	}
	if _, ok := Ink2PDenseMMQKind(Q4_K, nil, 32, 4096, 4096, false); ok {
		t.Fatal("Ink2PDenseMMQKind should respect denseFast=false")
	}
	if route := Ink2PMatMulRoute(Q4_K, nil, 64, 4096, 4096, true); route != "ink2p_dense_mmq:q4_k:kind=4" {
		t.Fatalf("Ink2PMatMulRoute(Q4_K) = %q", route)
	}

	fInk, err := parser.ParseFile(token.NewFileSet(), "cuda_inkling2_prefill.go", nil, 0)
	if err != nil {
		t.Fatalf("parse cuda_inkling2_prefill.go: %v", err)
	}
	var sawMMQHelper, sawNativeMMQCall bool
	for _, d := range fInk.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "Ink2PMatMul" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch fun := call.Fun.(type) {
			case *ast.Ident:
				if fun.Name == "Ink2PDenseMMQKind" {
					sawMMQHelper = true
				}
			case *ast.SelectorExpr:
				if fun.Sel.Name == "fcuda_ink2p_dense_mmq" {
					sawNativeMMQCall = true
				}
			}
			return true
		})
	}
	if !sawMMQHelper || !sawNativeMMQCall {
		t.Fatalf("Ink2PMatMul missing MMQ wiring: helper=%v native=%v", sawMMQHelper, sawNativeMMQCall)
	}

	// 2. Verify cuda.go PrefillBatch wires SelectCUDAAttentionKernel + GPU-resident panel calls.
	fCUDA, err := parser.ParseFile(token.NewFileSet(), filepath.Clean("cuda.go"), nil, 0)
	if err != nil {
		t.Fatalf("parse cuda.go: %v", err)
	}
	seen := map[string]bool{}
	for _, d := range fCUDA.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Name.Name != "PrefillBatch" {
			continue
		}
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			switch fun := call.Fun.(type) {
			case *ast.Ident:
				seen[fun.Name] = true
			case *ast.SelectorExpr:
				seen[fun.Sel.Name] = true
			}
			return true
		})
	}
	for _, req := range []string{"SelectCUDAAttentionKernel", "PromptPanelRoPE", "PromptPanelAppendKV", "PromptPanelAttention", "promptPanelAttentionDirect"} {
		if !seen[req] {
			t.Fatalf("cuda.go PrefillBatch missing GPU-resident call %q", req)
		}
	}
	if seen["RoPE"] {
		t.Fatal("cuda.go PrefillBatch still calls per-token c.RoPE")
	}
}

// fak-test:runtime fast est=15ms lane=default
func TestPrefillCostModelFP8NVFP4Q4KAndMidMarketScope(t *testing.T) {
	t.Parallel()

	g := PrefillGeometry{
		DModel:   4096,
		NHeads:   32,
		NKVHeads: 8,
		HeadDim:  128,
		DFF:      14336,
		NLayers:  32,
		Vocab:    128256,
		P:        512,
	}

	gQ4 := g
	gQ4.WeightDtype = Q4_K
	gFP8 := g
	gFP8.WeightDtype = FP8
	gNV4 := g
	gNV4.WeightDtype = NVFP4

	if got, want := gQ4.weightBytes(4096, 4096), int64(4096*(4096/256)*144); got != want {
		t.Fatalf("Q4_K weightBytes = %d, want %d", got, want)
	}
	if got, want := gFP8.weightBytes(4096, 4096), int64(4096*4096); got != want {
		t.Fatalf("FP8 weightBytes = %d, want %d", got, want)
	}
	if got, want := gNV4.weightBytes(4096, 4096), int64(4096*(4096/2)+4096*(4096/16)+4); got != want {
		t.Fatalf("NVFP4 weightBytes = %d, want %d", got, want)
	}

	for _, bad := range []MidMarketNodeSpec{
		{NumGPUs: 0, VRAMPerGPUBytes: 24 * giBytes},
		{NumGPUs: 9, VRAMPerGPUBytes: 80 * giBytes},
		{NumGPUs: 1, VRAMPerGPUBytes: 8 * giBytes},
		{NumGPUs: 1, VRAMPerGPUBytes: 192 * giBytes},
		{NumGPUs: 8, VRAMPerGPUBytes: 80 * giBytes, CrossNodeRDMA: true},
		{NumGPUs: 8, VRAMPerGPUBytes: 141 * giBytes, MultiNodeNVL72: true},
	} {
		if _, err := SizePrefillMidMarket(gQ4, bad); !errors.Is(err, ErrOutOfScopeCluster) {
			t.Fatalf("SizePrefillMidMarket(%+v) error = %v, want ErrOutOfScopeCluster", bad, err)
		}
	}

	// Verify TP scaling moves a ~15B FP8 checkpoint from non-fitting on 1x12 GiB
	// into fitting comfortably on 2x12 GiB.
	g15B := PrefillGeometry{
		DModel:      5120,
		NHeads:      40,
		NKVHeads:    8,
		HeadDim:     128,
		DFF:         13824,
		NLayers:     48,
		Vocab:       128256,
		P:           512,
		WeightDtype: FP8,
	}
	p1, err := SizePrefillMidMarket(g15B, MidMarketNodeSpec{
		NumGPUs:         1,
		VRAMPerGPUBytes: 12 * giBytes,
		SMMajor:         12,
		SMMinor:         0,
		PrefixTokens:    2048,
		KVDtype:         FP8,
	})
	if err != nil {
		t.Fatalf("1x12GiB sizing: %v", err)
	}
	p2, err := SizePrefillMidMarket(g15B, MidMarketNodeSpec{
		NumGPUs:         2,
		VRAMPerGPUBytes: 12 * giBytes,
		SMMajor:         12,
		SMMinor:         0,
		PrefixTokens:    2048,
		KVDtype:         FP8,
	})
	if err != nil {
		t.Fatalf("2x12GiB sizing: %v", err)
	}
	if p1.Fits {
		t.Fatalf("expected ~15B FP8 model + KV not to fit on 1x12 GiB (used=%d)", p1.TotalRequiredBytesPerGPU)
	}
	if !p2.Fits || p2.WeightsBytesPerGPU >= p1.WeightsBytesPerGPU {
		t.Fatalf("expected 2x12 GiB TP split to fit and halve per-GPU weights: p1=%+v p2=%+v", p1, p2)
	}
}

// fak-test:runtime fast est=80ms lane=default
func TestMidMarketPrefillMatrixAndConversionLedgerHooks(t *testing.T) {
	t.Parallel()

	gpuCounts := []int{1, 2, 4, 8}
	vramGiBs := []int64{12, 16, 24, 32, 48, 80, 96, 141}
	regimes := []struct {
		dtype Dtype
		kv    Dtype
		arch  struct {
			major int
			minor int
		}
	}{
		{dtype: FP8, kv: FP8, arch: struct{ major, minor int }{12, 0}},
		{dtype: NVFP4, kv: F16, arch: struct{ major, minor int }{10, 0}},
		{dtype: Q4_K, kv: F16, arch: struct{ major, minor int }{10, 3}},
	}

	baseGeom := PrefillGeometry{
		DModel:   2048,
		NHeads:   16,
		NKVHeads: 4,
		HeadDim:  128,
		DFF:      5632,
		NLayers:  16,
		Vocab:    32768,
		P:        256,
	}

	ledger := NewConversionLedger()
	var hookCount atomic.Int64
	ledger.RegisterHook(func(m ConversionMetrics) {
		hookCount.Add(1)
		if m.GPUMSPerTok < 0 || m.HostMSPerTok < 0 || m.DispatchesPlusSubmitsPerTok <= 0 || m.EndToEndTokPerSec <= 0 {
			t.Errorf("invalid hook metrics: %+v", m)
		}
	})

	sim := newSimDeviceBackend("sm_120")
	simCPU := &cpuBackend{}
	const (
		panelTokens = 8
		dModel      = 64
		nHeads      = 4
		nKVHeads    = 2
		headDim     = 16
	)
	qOut := nHeads * headDim
	kvOut := nKVHeads * headDim
	fill := func(n int, bias float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = bias + 0.01*float32(i%17-8)
		}
		return out
	}
	resX := sim.Upload(NewF32(simCPU, []int{panelTokens, dModel}, fill(panelTokens*dModel, 0.05)), F32)
	resWq := sim.Upload(NewF32(simCPU, []int{qOut, dModel}, fill(qOut*dModel, 0.01)), F32)
	resWk := sim.Upload(NewF32(simCPU, []int{kvOut, dModel}, fill(kvOut*dModel, 0.02)), F32)
	resWv := sim.Upload(NewF32(simCPU, []int{kvOut, dModel}, fill(kvOut*dModel, 0.03)), F32)
	resWo := sim.Upload(NewF32(simCPU, []int{dModel, qOut}, fill(dModel*qOut, 0.01)), F32)

	matrixCount := 0
	for _, nGPU := range gpuCounts {
		for _, vramGiB := range vramGiBs {
			for _, reg := range regimes {
				matrixCount++
				g := baseGeom
				g.WeightDtype = reg.dtype
				node := MidMarketNodeSpec{
					Name:              fmt.Sprintf("%dx%dGiB-%s", nGPU, vramGiB, reg.dtype),
					NumGPUs:           nGPU,
					VRAMPerGPUBytes:   vramGiB * giBytes,
					SMMajor:           reg.arch.major,
					SMMinor:           reg.arch.minor,
					PrefixTokens:      1024,
					KVDtype:           reg.kv,
					PagedKV:           reg.arch.major == 12,
					FP8Block32Weights: reg.dtype == FP8,
					PeakFLOPSPerGPU:   2e14,
					PeakBWBytesPerGPU: 1e12,
				}
				plan, err := SizePrefillMidMarket(g, node)
				if err != nil {
					t.Fatalf("matrix %s failed: %v", node.Name, err)
				}
				if !plan.AttentionSelection.GPUResident || !plan.AttentionSelection.Blackwell {
					t.Fatalf("matrix %s attention selection = %+v, want Blackwell GPU-resident", node.Name, plan.AttentionSelection)
				}
				if reg.dtype == Q4_K && (!plan.MMQFastPathEligible || !strings.Contains(plan.MMQRoute, "kind=4")) {
					t.Fatalf("matrix %s expected Q4_K MMQ fast path, got route=%q", node.Name, plan.MMQRoute)
				}
				if reg.dtype == Q4_K && vramGiB >= 12 && !plan.Fits {
					t.Fatalf("matrix %s expected compact ~2B Q4_K model to fit on %d GiB", node.Name, vramGiB)
				}
				if plan.RooflineFloorSecondsPerGPU <= 0 {
					t.Fatalf("matrix %s expected positive roofline floor, got %v", node.Name, plan.RooflineFloorSecondsPerGPU)
				}

				kv := sim.NewKV(KVConfig{NumLayers: 1, NumKVHeads: nKVHeads, HeadDim: headDim, Precision: KVPrecisionF32})
				meta := ConversionNodeMeta{
					Label:            node.Name,
					Regime:           plan.Regime,
					Arch:             plan.AttentionSelection.Arch,
					NumGPUs:          nGPU,
					VRAMPerGPUBytes:  node.VRAMPerGPUBytes,
					AttentionBackend: plan.AttentionSelection.Backend.String(),
				}
				_, metrics, err := PrefillBatchObserved(sim, PrefillBatchArgs{
					X:          resX,
					Wq:         resWq,
					Wk:         resWk,
					Wv:         resWv,
					Wo:         resWo,
					KV:         kv,
					Layer:      0,
					NumHeads:   nHeads,
					NumKVHeads: nKVHeads,
					HeadDim:    headDim,
					RopeTheta:  10000.0,
				}, ledger, meta)
				if err != nil {
					t.Fatalf("PrefillBatchObserved(%s): %v", node.Name, err)
				}
				if metrics.Tokens != panelTokens || metrics.Dispatches == 0 || metrics.Submits == 0 {
					t.Fatalf("matrix %s incomplete conversion metrics: %+v", node.Name, metrics)
				}
				if sim.lastReport.HostReadCount != 0 {
					t.Fatalf("matrix %s recorded non-zero host reads: %d", node.Name, sim.lastReport.HostReadCount)
				}
			}
		}
	}

	if got := len(ledger.Entries()); got != matrixCount {
		t.Fatalf("ledger entries = %d, want %d", got, matrixCount)
	}
	if got := hookCount.Load(); got != int64(matrixCount) {
		t.Fatalf("hook count = %d, want %d", got, matrixCount)
	}

	// Verify conversion-ledger mirage detector vs real end-to-end win.
	serialBase, err := BuildConversionMetrics(ConversionSampleInput{
		Meta:         ConversionNodeMeta{Regime: ConversionRegimeQ4K, NumGPUs: 1, VRAMPerGPUBytes: 32 * giBytes},
		Tokens:       256,
		Dispatches:   256 * 20,
		Submits:      256 * 10,
		GPUDuration:  256 * 116 * time.Millisecond,
		HostDuration: 256 * 10 * time.Millisecond,
		WallDuration: 256 * 126 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	mirageCand, err := BuildConversionMetrics(ConversionSampleInput{
		Meta:         ConversionNodeMeta{Regime: ConversionRegimeQ4K, NumGPUs: 1, VRAMPerGPUBytes: 32 * giBytes},
		Tokens:       256,
		Dispatches:   256 * 12,
		Submits:      256 * 8,
		GPUDuration:  256 * 64 * time.Millisecond, // GPU time dropped nearly 2x...
		HostDuration: 256 * 70 * time.Millisecond, // ...but host round-trips filled the gap!
		WallDuration: 256 * 134 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	cmpMirage, err := CompareConversionMetrics(serialBase, mirageCand)
	if err != nil || !cmpMirage.MirageDetected || cmpMirage.EndToEndWin {
		t.Fatalf("expected MirageDetected=true EndToEndWin=false, got %+v err=%v", cmpMirage, err)
	}

	residentWin, err := BuildConversionMetrics(ConversionSampleInput{
		Meta:         ConversionNodeMeta{Regime: ConversionRegimeQ4K, NumGPUs: 1, VRAMPerGPUBytes: 32 * giBytes},
		Tokens:       256,
		Dispatches:   7 * 16,
		Submits:      7 * 16,
		GPUDuration:  256 * 12 * time.Millisecond,
		HostDuration: 256 * 1 * time.Millisecond,
		WallDuration: 256 * 13 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	cmpWin, err := CompareConversionMetrics(serialBase, residentWin)
	if err != nil || cmpWin.MirageDetected || !cmpWin.EndToEndWin {
		t.Fatalf("expected real win MirageDetected=false EndToEndWin=true, got %+v err=%v", cmpWin, err)
	}
}
