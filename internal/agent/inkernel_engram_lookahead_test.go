package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// This producer declares the same retained-read/eligible-source contract as the
// loader bridge. Real verified row reads and public model attachment are used;
// no private model field or synthetic forward result is substituted.
type servingLookAheadReader struct {
	io.ReaderAt
	closing, entered, resume          chan struct{}
	enterOnce, resumeOnce             sync.Once
	holds, releases, heldReads, reads atomic.Int64
	reading, early                    atomic.Bool
	accept, block                     bool
	fault                             error
}

func (r *servingLookAheadReader) V41EngramLookAheadReadPolicy(width int) (int, bool) {
	return 0, width == model.V41EngramPackedRowBytes
}
func (r *servingLookAheadReader) V41EngramHoldSourceReads() (func(), <-chan struct{}, bool) {
	r.holds.Add(1)
	if !r.accept {
		return nil, nil, false
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			if r.reading.Load() {
				r.early.Store(true)
			}
			r.releases.Add(1)
		})
	}, r.closing, true
}
func (r *servingLookAheadReader) unblock() { r.resumeOnce.Do(func() { close(r.resume) }) }
func (r *servingLookAheadReader) ReadAt(dst []byte, off int64) (int, error) {
	r.reads.Add(1)
	if r.holds.Load() > r.releases.Load() {
		r.heldReads.Add(1)
		r.reading.Store(true)
		defer r.reading.Store(false)
		r.enterOnce.Do(func() { close(r.entered) })
		if r.block {
			<-r.resume
		}
		if r.fault != nil {
			return 0, r.fault
		}
	}
	return r.ReaderAt.ReadAt(dst, off)
}

// Arming after emission isolates the actual selected forward entry from prefill.
// Its owner-entry OOM is recovered by the real generation wrapper as InKernelOOMError.
type servingLookAheadRetryBackend struct {
	*v41CompleteBackend
	armed                                   atomic.Bool
	faults, recycled, trimmed, trimmedLarge atomic.Int32
}

func (b *servingLookAheadRetryBackend) ForwardExecutionOwner() *sync.Mutex {
	if b.armed.Swap(false) {
		b.faults.Add(1)
		panic(&compute.DeviceAllocError{Bytes: 4096, Class: compute.MemoryScratchpad, Site: "selected-engram-owner-entry"})
	}
	return b.v41CompleteBackend.ForwardExecutionOwner()
}
func (b *servingLookAheadRetryBackend) Recycle()      { b.recycled.Add(1) }
func (b *servingLookAheadRetryBackend) Trim()         { b.trimmed.Add(1) }
func (b *servingLookAheadRetryBackend) TrimLarge(int) { b.trimmedLarge.Add(1) }

// The existing public Complete fixture recipe is narrowed to one full-geometry
// Engram layer. It is a deterministic software contract, not checkpoint/GPU proof.
func servingLookAheadModel(t *testing.T) (*model.Model, *servingLookAheadReader) {
	t.Helper()
	raw, err := os.ReadFile("../model/testdata/deepseek_v41_flash_config.json")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879" {
		t.Fatal("official config fixture changed")
	}
	var cfg model.Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg.ModelType = "deepseek41"
	cfg.NumLayers, cfg.NumExperts, cfg.NumExpertsPerTok = 1, 384, 6
	cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim = 64, 1, 1, 512
	cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.QLoraRank = 448, 64, 32
	cfg.OLoraRank, cfg.OGroups, cfg.MoEIntermediateSize = 16, 1, 32
	cfg.VocabSize, cfg.NSharedExperts, cfg.RoutedScalingFactor = 320, 1, 1.5
	cfg.EOSTokenID, cfg.EOSTokenIDs = -1, nil
	cfg.RopeScaling, cfg.LongRope, cfg.RopeFactor, cfg.RopeOrigContext, cfg.RopeTheta = "", nil, 0, 0, 10000
	cfg.Window = []int{-1}
	cfg.DeepSeekV41.CompressRatios = []int{0}
	cfg.DeepSeekV41.KVSourceLayerIDs, cfg.DeepSeekV41.IndexSourceLayerIDs, cfg.DeepSeekV41.EngramLayerIDs = nil, nil, nil
	cfg.DeepSeekV41.Attention = model.DeepSeekV41AttentionGeometry{}
	var tensors []model.NamedTensorF32
	seed := uint64(0x9E3779B97F4A7C15)
	add := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			switch {
			case name == "model.norm.weight" || strings.HasSuffix(name, "attn_norm.weight") || strings.HasSuffix(name, "ffn_norm.weight") || strings.HasSuffix(name, "mhc.scale") || strings.HasSuffix(name, "mhc.ffn_scale") || strings.HasSuffix(name, "norm.weight"):
				data[i] = 1
			case strings.HasSuffix(name, "mhc.base") || strings.HasSuffix(name, "mhc.ffn_base"):
			default:
				seed = seed*6364136223846793005 + 1442695040888963407
				x := float32(seed>>40)/float32(1<<24)*2 - 1
				scale := float32(.1)
				if strings.Contains(name, "embed_tokens") {
					scale = .2
				}
				if strings.HasSuffix(name, "attn.sink") {
					scale = .25
				}
				data[i] = x * scale
			}
		}
		// The byte tokenizer covers 0..258. Retain a 320-row head while
		// keeping greedy outputs in its decodable, non-special vocabulary.
		if name == "lm_head.weight" {
			for i := 0; i < n; i++ {
				if i/64 < 3 || i/64 >= 259 {
					data[i] = 0
				}
			}
		}
		tensors = append(tensors, model.NamedTensorF32{Name: name, Shape: shape, Data: data})
	}
	add("model.embed_tokens.weight", 320, 64)
	add("lm_head.weight", 320, 64)
	add("model.norm.weight", 64)
	layer := func(name string, shape ...int) { add("model.layers.0."+name, shape...) }
	layer("attn_norm.weight", 64)
	layer("ffn_norm.weight", 64)
	layer("mhc.mixes.weight", 256, 24)
	layer("mhc.ffn_mixes.weight", 256, 24)
	layer("mhc.ffn_base", 24)
	layer("mhc.ffn_scale", 3)
	layer("mhc.base", 24)
	layer("mhc.scale", 3)
	layer("attn.wq_a.weight", 32, 64)
	layer("attn.wq_a_norm.weight", 32)
	layer("attn.wq_b.weight", 512, 32)
	layer("attn.wkv.weight", 512, 64)
	layer("attn.kv_norm.weight", 512)
	layer("attn.wo_a.weight", 16, 512)
	layer("attn.wo_b.weight", 64, 16)
	layer("attn.sink", 1)
	layer("engram_kv.weight", 24*64, 5*64)
	layer("engram_q_norm.weight", 4*64)
	layer("engram_k_norm.weight", 4*64)
	layer("ffn.gate.weight", 384, 64)
	layer("ffn.gate.e_score_correction_bias", 384)
	layer("ffn.shared_experts.w1.weight", 32, 64)
	layer("ffn.shared_experts.w3.weight", 32, 64)
	layer("ffn.shared_experts.w2.weight", 64, 32)
	for e := 0; e < 384; e++ {
		stem := fmt.Sprintf("ffn.experts.%d.", e)
		layer(stem+"w1.weight", 32, 64)
		layer(stem+"w3.weight", 32, 64)
		layer(stem+"w2.weight", 64, 32)
	}
	m, err := model.NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatalf("public reduced fixture construction: %v", err)
	}
	data := make([]byte, 408*model.V41EngramPackedRowBytes)
	for row := 0; row < 408; row++ {
		for col := 0; col < 256; col++ {
			data[row*model.V41EngramPackedRowBytes+col] = byte((row + col) % 32)
		}
		for col := 256; col < model.V41EngramPackedRowBytes; col++ {
			data[row*model.V41EngramPackedRowBytes+col] = 120
		}
	}
	r := &servingLookAheadReader{ReaderAt: bytes.NewReader(data), closing: make(chan struct{}), entered: make(chan struct{}), resume: make(chan struct{})}
	r.accept = true
	digest, err := model.V41EngramDigestExtent(r, 0, int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	shard := model.V41EngramShard{ID: "caller-owned", ExpectedID: "caller-owned", Size: int64(len(data)), Rows: 408}
	source, err := model.NewV41VerifiedEngramRowSource(r, shard, model.V41EngramArtifactBinding{Shard: shard.ID, Size: shard.Size, Rows: 408, RowBytes: model.V41EngramPackedRowBytes, Digest: digest})
	if err != nil {
		t.Fatal(err)
	}
	layout := model.V41EngramLayout{TokenMap: make([]uint32, 320), CompressedVocab: 320, Rows: []uint32{408}, Multipliers: [][]uint64{{17, 13, 19, 11}}, Primes: [][]uint32{make([]uint32, 24)}, MaxNgramSize: 4, HeadsPerNgram: 8}
	for i := range layout.TokenMap {
		layout.TokenMap[i] = uint32(i)
	}
	for i := range layout.Primes[0] {
		layout.Primes[0][i] = 17
	}
	spec := model.V41EngramAttachSpec{LayerIDs: []int{0}, MaxNgramSize: 4, VocabSize: 320, NHeads: 8, HeadDim: 64, CompressedVocabSize: 320}
	if err := m.AttachV41Engram(spec, layout, []model.V41EngramRowSource{source}, int64(model.V41EngramPackedRowBytes)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	t.Cleanup(r.unblock)
	return m, r
}
func servingLookAheadPlanner(t *testing.T, m *model.Model, enabled bool, be compute.Backend) *InKernelPlanner {
	t.Helper()
	p := NewInKernelPlannerWithConfig(m, loadProbeTok(t), "v41-caller-contract", false, be, false, InKernelPlannerConfig{V41EngramLookAhead: enabled})
	p.quant, p.maxNew, p.tree, p.scopedTree = false, 3, nil, nil
	return p
}
func servingLookAheadFallbacks() uint64 {
	for _, c := range compute.HostFallbackCounts() {
		if c.Stage == "greedy_argmax" {
			return c.Count
		}
	}
	return 0
}
func servingLookAheadLane(t *testing.T, m *model.Model, r *servingLookAheadReader, ctx context.Context, traced bool) *decodeLane {
	t.Helper()
	s := m.NewSession()
	t.Cleanup(s.Close)
	ln := &decodeLane{s: s, ctx: ctx, logits: s.Prefill([]int{1, 3, 2}), rng: rand.New(rand.NewSource(0)), stops: map[int]bool{}, emit: func(int) bool { return false }, maxNew: 2, forwarded: []int{}, engramAttempt: &inKernelEngramAttempt{}}
	if traced {
		ln.measurement = &nativeInferenceMeasurement{inferenceDisabled: true, traceNow: time.Now}
		ln.measurement.startDecodeTrace()
	}
	return ln
}

// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime medium est=12s lane=default
func TestInKernelSerialV41EngramLookAhead(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "off")
	t.Setenv("FAK_INKERNEL_SEED", "0")
	for _, temp := range []float64{0, 0.7} {
		t.Run(fmt.Sprintf("complete-temp-%g", temp), func(t *testing.T) {
			type result struct {
				completion *Completion
				heads      [][]float32
				fallbacks  uint64
			}
			run := func(enabled bool) result {
				m, r := servingLookAheadModel(t)
				be := &v41CompleteBackend{Backend: compute.Default(), available: true}
				p := servingLookAheadPlanner(t, m, enabled, be)
				if p.RuntimeConfig().V41EngramLookAhead != enabled {
					t.Fatal("typed setting not retained")
				}
				before := servingLookAheadFallbacks()
				got, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "a"}}, nil, WithRawPrompt(), WithMaxTokens(3), WithTemperature(&temp))
				if err != nil {
					t.Fatal(err)
				}
				if enabled {
					if r.holds.Load() != 2 || r.releases.Load() != 2 || r.heldReads.Load() == 0 || r.early.Load() {
						t.Fatalf("hold/read/release=%d/%d/%d early=%t", r.holds.Load(), r.heldReads.Load(), r.releases.Load(), r.early.Load())
					}
				} else if r.holds.Load() != 0 {
					t.Fatal("default selected look-ahead")
				}
				return result{got, be.headOutputs, servingLookAheadFallbacks() - before}
			}
			off, on := run(false), run(true)
			if off.completion.Message.Content != on.completion.Message.Content || !reflect.DeepEqual(off.heads, on.heads) {
				t.Fatal("selected caller changed exact text/head logits")
			}
			want := uint64(0)
			if temp == 0 {
				want = 2
			}
			if off.fallbacks != want || on.fallbacks != want {
				t.Fatalf("greedy accounting off/on=%d/%d want%d", off.fallbacks, on.fallbacks, want)
			}
		})
	}
	for _, traced := range []bool{false, true} {
		t.Run(fmt.Sprintf("cancel-effective-context-trace-%t", traced), func(t *testing.T) {
			m, r := servingLookAheadModel(t)
			r.block = true
			laneCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
			ln := servingLookAheadLane(t, m, r, laneCtx, traced)
			outerCtx := context.Background()
			if !traced {
				outerCtx = laneCtx
				ln.ctx = nil
			}
			done := make(chan struct{})
			go func() { defer close(done); inKernelDecodeSerial(outerCtx, ln) }()
			select {
			case <-r.entered:
			case <-time.After(5 * time.Second):
				r.unblock()
				<-done
				t.Fatal("selected read did not enter")
			}
			cancel()
			if r.releases.Load() != 0 {
				t.Error("released blocked read")
			}
			r.unblock()
			<-done
			if !errors.Is(ln.err, context.Canceled) || !ln.done || !ln.engramAttempt.selected || len(ln.forwarded) != 0 {
				t.Fatalf("cancellation state err=%v done=%t selected=%t forwarded=%v", ln.err, ln.done, ln.engramAttempt.selected, ln.forwarded)
			}
			if r.holds.Load() != 1 || r.releases.Load() != 1 || r.early.Load() {
				t.Fatal("read was not joined before release")
			}
			if traced {
				for _, event := range ln.measurement.traceEvents {
					if event.Forward != nil {
						t.Fatal("failed forward published success timing")
					}
				}
			}
		})
	}
	t.Run("hold-refusal-terminal", func(t *testing.T) {
		m, r := servingLookAheadModel(t)
		r.accept = false
		p := servingLookAheadPlanner(t, m, true, nil)
		retried := 0
		_, err := p.generateReusedWithOOMRetry(context.Background(), []int{1, 3, 2}, 2, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(int) bool { return false }, func() { retried++ })
		if !errors.Is(err, model.ErrV41NativeUnsupported) || retried != 0 || r.holds.Load() != 1 || r.releases.Load() != 0 || r.heldReads.Load() != 0 {
			t.Fatalf("refusal err=%v retry=%d hold/release/read=%d/%d/%d", err, retried, r.holds.Load(), r.releases.Load(), r.heldReads.Load())
		}
	})
	t.Run("selected-entry-oom-terminal", func(t *testing.T) {
		m, _ := servingLookAheadModel(t)
		be := &servingLookAheadRetryBackend{v41CompleteBackend: &v41CompleteBackend{Backend: compute.Default(), available: true}}
		p := servingLookAheadPlanner(t, m, true, be)
		emitted, retried := 0, 0
		_, err := p.generateReusedWithOOMRetry(context.Background(), []int{1, 3, 2}, 3, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(int) bool { emitted++; be.armed.Store(true); return false }, func() { retried++ })
		var oom *InKernelOOMError
		if !errors.As(err, &oom) || oom.Site != "selected-engram-owner-entry" {
			t.Fatalf("selected entry did not return actual OOM: %v", err)
		}
		// Session.Close legitimately recycles/trims once. Retry trimming also
		// calls TrimLarge; require exact retirement and no replay.
		if emitted != 1 || retried != 0 || be.faults.Load() != 1 || be.recycled.Load() != 1 || be.trimmed.Load() != 1 || be.trimmedLarge.Load() != 0 || be.kvCreates.Load() != 1 || be.kvCloses.Load() != 1 || len(be.headOutputs) != 1 {
			t.Fatalf("selected OOM replayed or session retirement incomplete: emit=%d retry=%d faults=%d trim=%d/%d/%d kv=%d/%d heads=%d", emitted, retried, be.faults.Load(), be.recycled.Load(), be.trimmed.Load(), be.trimmedLarge.Load(), be.kvCreates.Load(), be.kvCloses.Load(), len(be.headOutputs))
		}
	})
	t.Run("selected-no-retry-preserves-primary", func(t *testing.T) {
		m, r := servingLookAheadModel(t)
		ln := servingLookAheadLane(t, m, r, context.Background(), false)
		if err := ln.step(context.Background(), 5); err != nil {
			t.Fatal(err)
		}
		if !ln.engramAttempt.selected {
			t.Fatal("real greedy forward did not mark selection")
		}
		be := &oomRetryBackend{Backend: compute.Default()}
		p := &InKernelPlanner{backend: be}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		ctx = context.WithValue(ctx, inKernelEngramAttemptContextKey{}, ln.engramAttempt)
		primary := &InKernelOOMError{Bytes: 4096, Class: compute.MemoryScratchpad, Site: "selected-caller"}
		retry, got := p.admitDeviceOOMRetry(ctx, primary, 3, 2)
		if retry || got != primary || be.recycle != 0 || be.trim != 0 {
			t.Fatal("selected failure was replaced, trimmed or replayable")
		}
		before := r.holds.Load()
		ln.engramAttempt = nil
		if err := ln.step(context.Background(), 6); err != nil {
			t.Fatal(err)
		}
		if r.holds.Load() != before {
			t.Fatal("ordinary next forward inherited selection")
		}
	})
	t.Run("pre-selection-retry-unchanged", func(t *testing.T) {
		be := &oomRetryBackend{Backend: compute.Default()}
		p := &InKernelPlanner{backend: be}
		ctx := context.WithValue(context.Background(), inKernelEngramAttemptContextKey{}, &inKernelEngramAttempt{})
		primary := &InKernelOOMError{Bytes: 4096, Class: compute.MemoryScratchpad, Site: "before-selected-forward"}
		retry, got := p.admitDeviceOOMRetry(ctx, primary, 3, 2)
		if !retry || got != nil || be.recycle != 1 || be.trim != 1 {
			t.Fatalf("pre-selection retry changed: %t %v", retry, got)
		}
	})
	for _, mode := range []string{"batch", "no-stage", "expert-parallel", "rank-local", "speculative", "dflash2"} {
		t.Run("refuse-"+mode, func(t *testing.T) {
			m, r := servingLookAheadModel(t)
			be := &v41CompleteBackend{Backend: compute.Default(), available: true}
			p := servingLookAheadPlanner(t, m, true, be)
			switch mode {
			case "batch":
				p.batchDecode = true
			case "no-stage":
				p.m = v41CompleteModel(t)
				t.Cleanup(func() { _ = p.m.CloseWeights() })
			case "expert-parallel":
				m.SetExpertParallelRanks(2)
			case "rank-local":
				m.SetExpertParallelRank(0)
			case "dflash2":
				p.dflash2 = &DFlash2Target{}
			case "speculative":
				p.vulkanMTP = true
			}
			emitted := 0
			reads := r.reads.Load()
			_, err := p.CompleteStream(context.Background(), func(string) error { emitted++; return nil }, []Message{{Role: RoleUser, Content: "a"}}, nil, WithRawPrompt(), WithPerTokenStream(true))
			if !errors.Is(err, model.ErrV41NativeUnsupported) || emitted != 0 || r.holds.Load() != 0 || r.reads.Load() != reads || be.kvCreates.Load() != 0 || len(be.headOutputs) != 0 {
				t.Fatalf("invalid enabled config reached output/forward: %v %d %d", err, emitted, r.holds.Load())
			}
		})
	}
	for _, mode := range []string{"panic", "panic-nil", "goexit"} {
		t.Run("preserve-"+mode, func(t *testing.T) {
			m, _ := servingLookAheadModel(t)
			be := &v41CompleteBackend{Backend: compute.Default(), available: true}
			session := m.NewSession()
			session.Backend = be
			t.Cleanup(session.Close)
			logits := session.Prefill([]int{1, 3, 2})
			ln := &decodeLane{s: session, logits: logits, engramAttempt: &inKernelEngramAttempt{}}
			sentinel := errors.New("unrelated caller panic")
			be.fail = func() {
				switch mode {
				case "goexit":
					runtime.Goexit()
				case "panic-nil":
					panic(nil)
				default:
					panic(sentinel)
				}
			}
			returned := false
			var payload any
			done := make(chan struct{})
			go func() {
				defer close(done)
				defer func() { payload = recover() }()
				_ = ln.step(context.Background(), 5)
				returned = true
			}()
			<-done
			be.fail = nil
			if returned || !ln.engramAttempt.selected {
				t.Fatal("termination converted to normal return")
			}
			switch mode {
			case "goexit":
				if payload != nil {
					t.Fatalf("Goexit changed to panic: %v", payload)
				}
			case "panic-nil":
				if _, ok := payload.(*runtime.PanicNilError); !ok {
					t.Fatalf("panic(nil) changed: %T %v", payload, payload)
				}
			default:
				if payload != sentinel {
					t.Fatalf("primary panic replaced: %v", payload)
				}
			}
		})
	}
}
