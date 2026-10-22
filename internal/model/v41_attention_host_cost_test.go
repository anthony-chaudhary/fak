package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41HostCostBackend admits the selected shared-attention and row-log seams but
// performs no device work: uploads, gathers and the contraction return opaque
// handles, and the output is one shared zero panel. What remains is the
// model-level Go host cost; driver submit, fence and transfer time is absent.
// Like the Vulkan backend it owns a forward execution domain.
type v41HostCostBackend struct {
	compute.Backend
	owner                                *sync.Mutex
	zeros                                []float32
	uploads, uploadRows                  int
	attention, selectedRows              int
	gathers, spans, rowCopies, rowAllocs int
}

func newV41HostCostBackend(heads, dim int) *v41HostCostBackend {
	return &v41HostCostBackend{Backend: compute.Default(), owner: &sync.Mutex{}, zeros: make([]float32, heads*dim)}
}

func (b *v41HostCostBackend) ForwardExecutionOwner() *sync.Mutex { return b.owner }

func (b *v41HostCostBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}

func (b *v41HostCostBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}

func (b *v41HostCostBackend) SupportsV41SharedAttention() bool { return true }

func (b *v41HostCostBackend) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if !strings.HasPrefix(site, "V4.1 shared attention ") && !strings.HasPrefix(site, "V4.1 resident ") {
		return b.Backend.Upload(x, dt)
	}
	b.uploads++
	if strings.HasPrefix(site, "V4.1 resident KV") && len(x.Shape) > 1 {
		b.uploadRows += x.Shape[0]
	}
	return compute.NewF32(b, x.Shape, nil)
}

func (b *v41HostCostBackend) V41SharedAttention(q, kv, sink compute.Tensor, rows, heads, dim int, scale float32, mode compute.V41SharedAttentionMode, hasSink bool) (compute.Tensor, error) {
	b.attention++
	b.selectedRows += rows
	return compute.NewF32(b, []int{heads, dim}, b.zeros), nil
}

func (b *v41HostCostBackend) Read(x compute.Tensor) []float32 {
	if x.Backend() != b {
		return b.Backend.Read(x)
	}
	if h, ok := x.Buf().(compute.HostBuffer); ok {
		return h.F32()
	}
	return nil
}

func (b *v41HostCostBackend) Free(x compute.Tensor) {
	if x.Backend() != b {
		b.Backend.Free(x)
	}
}

func (b *v41HostCostBackend) V41RowLogAvailable() bool { return true }

func (b *v41HostCostBackend) V41RowLogAlloc(rows, width int) (compute.Tensor, error) {
	b.rowAllocs++
	return compute.NewF32(b, []int{rows, width}, nil), nil
}

func (b *v41HostCostBackend) V41RowLogCopy(dst compute.Tensor, dstRow int, src compute.Tensor, srcRow, rows int) error {
	b.rowCopies++
	return nil
}

func (b *v41HostCostBackend) V41RowLogGather(spans []compute.V41RowSpan, width int) (compute.Tensor, error) {
	total := 0
	for _, span := range spans {
		total += span.Rows
	}
	b.gathers++
	b.spans += len(spans)
	return compute.NewF32(b, []int{total, width}, nil), nil
}

func v41HostCostFill(seed *uint64, values []float32) {
	for i := range values {
		*seed ^= *seed << 13
		*seed ^= *seed >> 7
		*seed ^= *seed << 17
		values[i] = float32(int64(*seed%2001)-1000) / 4000
	}
}

func v41HostCostRows(seed *uint64, rows, width int) [][]float32 {
	backing := make([]float32, rows*width)
	v41HostCostFill(seed, backing)
	out := make([][]float32, rows)
	for i := range out {
		out[i] = backing[i*width : (i+1)*width : (i+1)*width]
	}
	return out
}

// Position-sorted top-k over the groups visible to each query, padded with -1
// as v41AttentionIndexList normalizes a short selection.
func v41HostCostIndexList(seq, ratio, groups, topK int) []int32 {
	out := make([]int32, seq*topK)
	for t := 0; t < seq; t++ {
		row := out[t*topK : (t+1)*topK]
		visible := 0
		if t >= ratio-1 {
			visible = min((t-(ratio-1))/ratio+1, groups)
		}
		for i := range row {
			switch {
			case visible <= topK && i < visible:
				row[i] = int32(i)
			case visible > topK:
				row[i] = int32(i * visible / topK)
			default:
				row[i] = -1
			}
		}
	}
	return out
}

type v41HostCostRecord struct {
	Arm                     string  `json:"arm"`
	PrepareRowScanElements  int     `json:"prepare_row_scan_elements"`
	Ratio                   int     `json:"ratio"`
	Positions               int     `json:"positions"`
	Reps                    int     `json:"reps"`
	BankRows                int     `json:"bank_rows"`
	TopK                    int     `json:"top_k"`
	Window                  int     `json:"window"`
	Heads                   int     `json:"heads"`
	HeadDim                 int     `json:"head_dim"`
	ContractNSPerPosition   float64 `json:"contract_ns_per_position"`
	AttentionTimerNSPerCall float64 `json:"attention_timer_ns_per_call"`
	AttentionTimerCalls     int     `json:"attention_timer_calls"`
	MallocsPerPosition      float64 `json:"mallocs_per_position"`
	AllocBytesPerPosition   float64 `json:"alloc_bytes_per_position"`
	GCCycles                uint32  `json:"gc_cycles"`
	DeviceCalls             int     `json:"device_calls"`
	AvgSelectedRows         float64 `json:"avg_selected_rows"`
	AvgSpansPerGather       float64 `json:"avg_spans_per_gather"`
	Uploads                 int     `json:"uploads"`
	ResidentKVUploadRows    int     `json:"resident_kv_upload_rows"`
	RowCopies               int     `json:"row_copies"`
	RowAllocs               int     `json:"row_allocs"`
}

func v41HostCostProfilePath(base string, ratio, ratios int) string {
	if base == "" || ratios == 1 {
		return base
	}
	ext := filepath.Ext(base)
	return fmt.Sprintf("%s-r%d%s", strings.TrimSuffix(base, ext), ratio, ext)
}

// Production per-position path of one cold compressed layer at the published
// V4.1 attention axes (64x512, window 128, top-k 512), through the session's
// selected shared-attention binding and resident KV mirror. The fake backend
// does no device work, so this measures only model-level Go host cost.
// fak-test:justify why=cost when=explicit:FAK_V41_ATTN_HOST_COST
// fak-test:runtime slow est=90s lane=optin
func TestV41ColdAttentionHostCost(t *testing.T) {
	if os.Getenv("FAK_V41_ATTN_HOST_COST") != "1" {
		t.Skip("set FAK_V41_ATTN_HOST_COST=1 (optional _RATIOS=2,1 _POSITIONS=2049 _REPS=3 _PROFILE=path)")
	}
	if compute.Default().Name() != "cpu-ref" {
		t.Fatal("requires cpu-ref")
	}
	envInts := func(name string, fallback []int) []int {
		raw := os.Getenv(name)
		if raw == "" {
			return fallback
		}
		var out []int
		for _, field := range strings.Split(raw, ",") {
			n, err := strconv.Atoi(strings.TrimSpace(field))
			if err != nil || n <= 0 {
				t.Fatalf("%s: positive integers expected, got %q", name, raw)
			}
			out = append(out, n)
		}
		return out
	}
	ratios := envInts("FAK_V41_ATTN_HOST_COST_RATIOS", []int{2, 1})
	seq := envInts("FAK_V41_ATTN_HOST_COST_POSITIONS", []int{2049})[0]
	reps := envInts("FAK_V41_ATTN_HOST_COST_REPS", []int{3})[0]
	base := os.Getenv("FAK_V41_ATTN_HOST_COST_PROFILE")
	for _, ratio := range ratios {
		for _, arm := range []string{"proof", "strict"} {
			t.Run(fmt.Sprintf("ratio_%d/%s", ratio, arm), func(t *testing.T) {
				path := ""
				if arm == "proof" {
					path = v41HostCostProfilePath(base, ratio, len(ratios))
				}
				v41HostCostLayer(t, ratio, seq, reps, arm == "strict", path)
			})
		}
	}
}

// The proof arm is production cold prefill: an owned execution scope whose
// positions consume the composer's validation. The strict arm wraps the same
// binding as an ordinary callback, so the preparation re-scans every row.
func v41HostCostLayer(t *testing.T, ratio, seq, reps int, strict bool, profilePath string) {
	const heads, dim, window, topK, ropeDim = 64, 512, 128, 512, 64
	backend := newV41HostCostBackend(heads, dim)
	m := v41IncrementalPlainModel(t, 1)
	s := v41DenseTestSession(t, m, backend)
	// Widen the admitted fixture to the published attention axes. Only the
	// attention contract below reads them; the envelope is cleared so full
	// geometry is classified by the flat 512-wide head alone.
	d41 := *m.Cfg.DeepSeekV41
	d41.Attention.HeadDim = 0
	d41.CompressRatios = []int{ratio}
	d41.CompressRopeTheta = 160000
	d41.IndexSourceLayerIDs = []int{0}
	m.Cfg.DeepSeekV41 = &d41
	m.Cfg.IndexTopK = topK
	m.Cfg.NumHeads, m.Cfg.HeadDim, m.Cfg.QKRopeHeadDim = heads, dim, ropeDim
	m.Cfg.Window = []int{window}

	binding := s.v41BindSharedAttention()
	if binding == nil {
		t.Fatal("shared attention was not bound")
	}
	seed := uint64(0x9e3779b97f4a7c15) + uint64(ratio)
	groups := seq / ratio
	qkv := v41LayerQKV{qHeads: v41HostCostRows(&seed, seq, heads*dim), kvRows: v41HostCostRows(&seed, seq, dim)}
	plan, err := m.v41AttentionPlan(0)
	if err != nil || plan.Ratio != ratio || plan.topKWidth() != topK {
		t.Fatalf("resolved plan %+v err=%v, want ratio %d top-k %d", plan, err, ratio, topK)
	}
	src := v41LayerSources{plan: plan, sharedKV: v41HostCostRows(&seed, groups, dim)}
	indexList := v41HostCostIndexList(seq, ratio, groups, topK)
	sink := make([]float32, heads)
	v41HostCostFill(&seed, sink)

	m.v41SetExpertFaultPhase(V41PhasePrefill)
	defer m.v41SetExpertFaultPhase(V41PhaseUnknown)
	// Each pass gets a fresh layer owner, so the mirror reseeds exactly as it
	// does once per layer of a cold prefill.
	scans := 0
	v41SharedAttentionPrepareScanProbe = func(elements int) { scans += elements }
	defer func() { v41SharedAttentionPrepareScanProbe = nil }()
	pass := func() {
		scope := s.beginBackendExecution()
		defer scope.finish()
		st := &v41ForwardState{history: make([]int, seq), callbackOwner: s}
		st.bindCanonicalSharedAttention(binding)
		if strict {
			st.setSharedAttention(binding.attentionFunc())
		}
		layerState, err := NewV41AttentionState(dim, 8)
		if err != nil {
			t.Fatal(err)
		}
		st.setLayerState(0, 1, layerState)
		scratch := &v41ProjScratch{}
		scratch.copySharedAttentionFrom(st)
		out, err := m.v41LayerAttentionContract(0, seq, heads, dim, src, qkv, indexList, sink, st, scratch)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != seq || len(out[seq-1]) != heads*dim {
			t.Fatal("attention contract returned the wrong panel")
		}
	}
	pass() // warm the heap and the resident sink
	warm := *backend
	var profile *os.File
	stopProfile := func() {}
	if profilePath != "" {
		var err error
		profile, err = os.OpenFile(profilePath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			t.Fatal(err)
		}
		if err = pprof.StartCPUProfile(profile); err != nil {
			profile.Close()
			t.Fatal(err)
		}
		active := true
		stopProfile = func() {
			if !active {
				return
			}
			active = false
			pprof.StopCPUProfile()
			if closeErr := profile.Close(); closeErr != nil {
				t.Errorf("close CPU profile: %v", closeErr)
			}
		}
		defer stopProfile()
	}
	runtime.GC()
	scans = 0
	before := m.V41ExpertFaultAttribution().Prefill
	var pre, post runtime.MemStats
	runtime.ReadMemStats(&pre)
	started := time.Now()
	for i := 0; i < reps; i++ {
		pass()
	}
	elapsed := time.Since(started)
	runtime.ReadMemStats(&post)
	stopProfile()
	after := m.V41ExpertFaultAttribution().Prefill

	positions := float64(reps * seq)
	calls := backend.attention - warm.attention
	if calls != reps*seq || backend.gathers-warm.gathers != calls {
		t.Fatalf("device route not taken for every position: calls=%d gathers=%d want=%d", calls, backend.gathers-warm.gathers, reps*seq)
	}
	timerCalls := after.AttentionContractionCalls - before.AttentionContractionCalls
	if timerCalls != calls {
		t.Fatalf("attention timer saw %d calls, want %d", timerCalls, calls)
	}
	if strict == (scans == 0) {
		t.Fatalf("strict=%t arm prepared %d row-scan elements", strict, scans)
	}
	arm := "proof"
	if strict {
		arm = "strict"
	}
	record := v41HostCostRecord{
		Arm: arm, PrepareRowScanElements: scans,
		Ratio: ratio, Positions: seq, Reps: reps, BankRows: groups, TopK: topK, Window: window, Heads: heads, HeadDim: dim,
		ContractNSPerPosition:   float64(elapsed.Nanoseconds()) / positions,
		AttentionTimerNSPerCall: float64(after.AttentionContractionNanos-before.AttentionContractionNanos) / float64(timerCalls),
		AttentionTimerCalls:     timerCalls,
		MallocsPerPosition:      float64(post.Mallocs-pre.Mallocs) / positions,
		AllocBytesPerPosition:   float64(post.TotalAlloc-pre.TotalAlloc) / positions,
		GCCycles:                post.NumGC - pre.NumGC,
		DeviceCalls:             calls,
		AvgSelectedRows:         float64(backend.selectedRows-warm.selectedRows) / float64(calls),
		AvgSpansPerGather:       float64(backend.spans-warm.spans) / float64(calls),
		Uploads:                 backend.uploads - warm.uploads,
		ResidentKVUploadRows:    backend.uploadRows - warm.uploadRows,
		RowCopies:               backend.rowCopies - warm.rowCopies,
		RowAllocs:               backend.rowAllocs - warm.rowAllocs,
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	fmt.Printf("V41_ATTN_HOST_COST %s\n", encoded)
}
