package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Host-resident contract fixture: all arithmetic is cpu-ref. DeviceMemory
// exercises dispatch only; this is no physical-provider qualification.
type v41CompleteBackend struct {
	compute.Backend
	owner                           sync.Mutex
	available                       bool
	admissions, kvCreates, kvCloses atomic.Int32
	query                           chan struct{}
	probe                           func()
	fail                            func()
	headRows                        int
	headOutputs                     [][]float32
}

func (b *v41CompleteBackend) ForwardExecutionOwner() *sync.Mutex {
	if b.query != nil {
		select {
		case b.query <- struct{}{}:
		default:
		}
	}
	return &b.owner
}
func (b *v41CompleteBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	return c
}
func (b *v41CompleteBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32
}
func (b *v41CompleteBackend) V41SharedHeadAvailable(dt compute.Dtype, out, in, rows int) bool {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("unowned shared-head query")
	}
	if rows == 4 {
		b.admissions.Add(1)
		if b.probe != nil {
			b.probe()
		}
	}
	return b.available && dt == compute.F32 && out == 320 && in == 64 && rows >= 1 && rows <= 4
}
func (b *v41CompleteBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	b.kvCreates.Add(1)
	return &v41CompleteKV{KVStore: b.Backend.NewKV(cfg), closed: &b.kvCloses}
}

type v41CompleteKV struct {
	compute.KVStore
	closed *atomic.Int32
}

func (k *v41CompleteKV) Free() { k.closed.Add(1); k.KVStore.Free() }
func (b *v41CompleteBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.fail != nil {
		b.fail()
	}
	y := b.Backend.MatMul(w, x)
	b.recordHead(w, y, 1, false)
	return y
}
func (b *v41CompleteBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	if b.fail != nil {
		b.fail()
	}
	y := b.Backend.BatchedMatMul(w, x, rows)
	b.recordHead(w, y, rows, true)
	return y
}
func (b *v41CompleteBackend) recordHead(w, y compute.Tensor, rows int, batch bool) {
	if len(w.Shape) != 2 || w.Shape[0] != 320 || w.Shape[1] != 64 {
		return
	}
	values := b.Backend.Read(y)
	for row := 0; row < rows; row++ {
		b.headOutputs = append(b.headOutputs, append([]float32(nil), values[row*320:(row+1)*320]...))
	}
	if batch && rows > 1 {
		b.headRows += rows
	}
}

// Copy the reduced tensor recipe through the PUBLIC loader, retaining the
// pinned official metadata and canonical GGUF identity. No private Session
// state, fabricated result, operation substitution or synthetic constructor.
func v41CompleteModel(t *testing.T) *model.Model {
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
	cfg.HiddenSize, cfg.NumHeads, cfg.HeadDim = 64, 2, 32
	cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.QLoraRank = 16, 16, 32
	cfg.OLoraRank, cfg.OGroups, cfg.MoEIntermediateSize = 16, 2, 32
	cfg.VocabSize, cfg.NSharedExperts, cfg.RoutedScalingFactor = 320, 1, 1.5
	cfg.EOSTokenID, cfg.EOSTokenIDs = -1, nil
	cfg.RopeScaling, cfg.LongRope, cfg.RopeFactor, cfg.RopeOrigContext, cfg.RopeTheta = "", nil, 0, 0, 10000
	cfg.Window = []int{-1}
	cfg.DeepSeekV41.CompressRatios = []int{0}
	cfg.DeepSeekV41.KVSourceLayerIDs, cfg.DeepSeekV41.IndexSourceLayerIDs, cfg.DeepSeekV41.EngramLayerIDs = nil, nil, nil
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
			case name == "model.norm.weight" || strings.HasSuffix(name, "attn_norm.weight") || strings.HasSuffix(name, "ffn_norm.weight") || strings.HasSuffix(name, "mhc.scale"):
				data[i] = 1
			case strings.HasSuffix(name, "mhc.base"):
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
	layer("mhc.mixes.weight", 24, 64)
	layer("mhc.base", 24)
	layer("mhc.scale", 3)
	layer("attn.wq_a.weight", 32, 64)
	layer("attn.wq_b.weight", 64, 32)
	layer("attn.wkv.weight", 32, 64)
	layer("attn.wo_a.weight", 16, 64)
	layer("attn.wo_b.weight", 64, 32)
	layer("attn.sink", 2)
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
	return m
}
func v41CompletePlanner(t *testing.T, m *model.Model, be *v41CompleteBackend, batch bool) *InKernelPlanner {
	t.Helper()
	p := NewInKernelPlannerWithConfig(m, loadProbeTok(t), "v41-cpu-contract", false, be, false, InKernelPlannerConfig{BatchDecode: batch})
	p.quant, p.maxNew, p.tree, p.scopedTree = false, 3, nil, nil
	return p
}

type v41CompleteCloser func() error

func (f v41CompleteCloser) Close() error { return f() }

// fak-test:runtime fast est=2s lane=default
func TestV41CompleteAdmissionCausal(t *testing.T) {
	t.Setenv("FAK_V41_STEP_BATCH", "1")
	messages := []Message{{Role: RoleUser, Content: "a"}}
	t.Run("public prefill through real shared head and false control", func(t *testing.T) {
		m := v41CompleteModel(t)
		defer m.CloseWeights()
		serial := &v41CompleteBackend{Backend: compute.Default(), available: true}
		p := v41CompletePlanner(t, m, serial, false)
		want, err := p.Complete(context.Background(), messages, nil, WithRawPrompt())
		if err != nil {
			t.Fatalf("real public Prefill fixture gate: %v", err)
		}
		if len(serial.headOutputs) != 3 || serial.kvCreates.Load() != 1 || serial.kvCloses.Load() != 1 {
			t.Fatalf("serial operations: heads=%d created=%d closed=%d", len(serial.headOutputs), serial.kvCreates.Load(), serial.kvCloses.Load())
		}
		const requestCount = 2
		for _, available := range []bool{true, false} {
			be := &v41CompleteBackend{Backend: compute.Default(), available: available}
			p := v41CompletePlanner(t, m, be, true)
			release := make(chan struct{})
			if available {
				p.coalesceReadyHook = func() { <-release }
			}
			type answer struct {
				c     *Completion
				err   error
				emits int
			}
			done := make(chan answer, requestCount)
			for i := 0; i < requestCount; i++ {
				go func() {
					n := 0
					c, err := p.Complete(context.Background(), messages, nil, WithRawPrompt(), func(sp *SampleParams) {
						// Complete sends token events with empty raw and one final nonempty raw text.
						// Count empty token pieces too; this fixture produces nonempty final text.
						sp.DecodeTokenObserver = func(piece, raw string) {
							if raw == "" {
								n++
							}
						}
					})
					done <- answer{c, err, n}
				}()
			}
			if available {
				deadline := time.Now().Add(3 * time.Second)
				for p.CoalesceReadyLenForTest() != 2 && time.Now().Before(deadline) {
					time.Sleep(time.Millisecond)
				}
				if p.CoalesceReadyLenForTest() != 2 {
					close(release)
					t.Fatal("both Complete requests did not reach the queue")
				}
				close(release)
			}
			for i := 0; i < requestCount; i++ {
				select {
				case got := <-done:
					if got.err != nil || got.c == nil {
						t.Fatalf("Complete: %v", got.err)
					}
					if got.c.Message.Content != want.Message.Content || got.c.Usage.CompletionTokens != 3 || got.emits != 3 {
						t.Fatalf("output/count mismatch: %#v emits=%d", got.c.Message, got.emits)
					}
					if available {
						r := got.c.InKernelBatch
						if r == nil || r.CohortSize != 2 || r.SharedRows != 4 || r.SessionCloses != 2 {
							t.Fatalf("receipt=%+v", r)
						}
					} else if got.c.InKernelBatch != nil {
						t.Fatalf("false capability coalesced: %+v", got.c.InKernelBatch)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("Complete did not finish")
				}
			}
			if be.admissions.Load() != 2 || be.kvCreates.Load() != 2 || be.kvCloses.Load() != 2 {
				t.Fatalf("admissions=%d KV=%d/%d", be.admissions.Load(), be.kvCreates.Load(), be.kvCloses.Load())
			}
			if available {
				if be.headRows != 4 || len(be.headOutputs) != requestCount*len(serial.headOutputs) {
					t.Fatalf("completed shared rows=%d heads=%d", be.headRows, len(be.headOutputs))
				}
				for i, row := range be.headOutputs {
					if !reflect.DeepEqual(row, serial.headOutputs[i/2]) {
						t.Fatalf("shared logits differ at row %d", i)
					}
				}
			} else {
				if be.headRows != 0 {
					t.Fatal("false capability ran shared head")
				}
				for i, row := range be.headOutputs {
					if !reflect.DeepEqual(row, serial.headOutputs[i%3]) {
						t.Fatalf("fallback logits differ at row %d", i)
					}
				}
			}
		}
	})
	t.Run("selected cancellation isolates its lane after one emission", func(t *testing.T) {
		m := v41CompleteModel(t)
		defer m.CloseWeights()
		be := &v41CompleteBackend{Backend: compute.Default(), available: true}
		p := v41CompletePlanner(t, m, be, true)
		release := make(chan struct{})
		p.coalesceReadyHook = func() { <-release }
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		type answer struct {
			lane  int
			c     *Completion
			err   error
			emits int
		}
		done := make(chan answer, 2)
		for i := 0; i < 2; i++ {
			go func(lane int) {
				runCtx := context.Background()
				if lane == 0 {
					runCtx = ctx
				}
				n := 0
				c, err := p.Complete(runCtx, messages, nil, WithRawPrompt(), func(sp *SampleParams) {
					sp.DecodeTokenObserver = func(piece, raw string) {
						if raw != "" {
							return
						}
						n++
						if lane == 0 {
							cancel()
						}
					}
				})
				done <- answer{lane, c, err, n}
			}(i)
		}
		deadline := time.Now().Add(3 * time.Second)
		for p.CoalesceReadyLenForTest() != 2 && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		if p.CoalesceReadyLenForTest() != 2 {
			close(release)
			t.Fatal("both requests did not enqueue")
		}
		close(release)
		for i := 0; i < 2; i++ {
			select {
			case got := <-done:
				if got.lane == 0 {
					if !errors.Is(got.err, context.Canceled) || got.emits != 1 {
						t.Fatalf("cancelled lane: err=%v emits=%d", got.err, got.emits)
					}
				} else if got.err != nil || got.c == nil || got.emits != 3 {
					t.Fatalf("healthy lane: err=%v emits=%d", got.err, got.emits)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("selected cancellation did not join")
			}
		}
		if be.kvCreates.Load() != 2 || be.kvCloses.Load() != 2 {
			t.Fatalf("request replay/leak KV=%d/%d", be.kvCreates.Load(), be.kvCloses.Load())
		}
	})

	t.Run("busy cancellation cannot emit or construct", func(t *testing.T) {
		m := v41CompleteModel(t)
		defer m.CloseWeights()
		be := &v41CompleteBackend{Backend: compute.Default(), available: true}
		p := v41CompletePlanner(t, m, be, true)
		be.query = make(chan struct{}, 1)
		be.owner.Lock()
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		done := make(chan error, 1)
		go func() { _, err := p.Complete(ctx, messages, nil, WithRawPrompt()); done <- err }()
		select {
		case <-be.query:
		case <-time.After(time.Second):
			be.owner.Unlock()
			t.Fatal("no admission")
		}
		cancel()
		select {
		case err := <-done:
			if !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		case <-time.After(time.Second):
			be.owner.Unlock()
			t.Fatal("busy cancellation blocked")
		}
		be.owner.Unlock()
		if be.kvCreates.Load() != 0 || len(be.headOutputs) != 0 {
			t.Fatal("busy admission performed forward")
		}
	})
	t.Run("closing refuses new session and preserves typed error over cleanup", func(t *testing.T) {
		m := v41CompleteModel(t)
		be := &v41CompleteBackend{Backend: compute.Default(), available: true}
		p := v41CompletePlanner(t, m, be, true)
		retired := false
		m.SetWeightCloser(v41CompleteCloser(func() error {
			if !be.owner.TryLock() {
				panic("retired under owner")
			}
			be.owner.Unlock()
			retired = true
			panic("secondary closer")
		}))
		be.probe = func() {
			var active *model.WeightSessionsActiveError
			if !errors.As(m.CloseWeights(), &active) || active.Count != 1 || retired {
				panic("eligibility pin missing")
			}
		}
		_, err := p.Complete(context.Background(), messages, nil, WithRawPrompt())
		var closed *model.ModelWeightsClosedError
		if !errors.As(err, &closed) || !retired || be.kvCreates.Load() != 0 {
			t.Fatalf("err=%v retired=%v KV=%d", err, retired, be.kvCreates.Load())
		}
	})
	t.Run("operation panic remains primary through final retirement", func(t *testing.T) {
		m := v41CompleteModel(t)
		be := &v41CompleteBackend{Backend: compute.Default(), available: true}
		p := v41CompletePlanner(t, m, be, true)
		primary := errors.New("forward panic")
		retired := false
		m.SetWeightCloser(v41CompleteCloser(func() error {
			if !be.owner.TryLock() {
				panic("retired under owner")
			}
			be.owner.Unlock()
			retired = true
			panic("secondary closer")
		}))
		be.fail = func() { _ = m.CloseWeights(); panic(primary) }
		var got any
		func() {
			defer func() { got = recover() }()
			_, _ = p.Complete(context.Background(), messages, nil, WithRawPrompt())
		}()
		if got != primary || !retired || be.kvCreates.Load() != 1 || be.kvCloses.Load() != 1 {
			t.Fatalf("panic=%v retired=%v KV=%d/%d", got, retired, be.kvCreates.Load(), be.kvCloses.Load())
		}
	})
}
