//go:build darwin && arm64 && cgo

package agent

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestInKernelPlannerCoalescesRealQwenMetalTurns is the planner-owned execution
// witness for #9075. It deliberately installs no shared-receipt probe: nonzero
// receipt values can only come back from BatchSession's real Qwen Metal path.
func TestInKernelPlannerCoalescesRealQwenMetalTurns(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	m := newQwen35HybridQ4KMetalFixture(t)
	if got := m.Q4KCount(); got != 15 {
		t.Fatalf("fixture resident Q4_K tensors=%d, want 15", got)
	}
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Fatalf("close model weights: %v", err)
		}
	})

	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "off")
	p := NewInKernelPlannerWithConfig(m, loadProbeTok(t), "synthetic-qwen35-metal-cohort", true, nil, true, InKernelPlannerConfig{Qwen35MetalGDNSequence: true})
	p.maxNew = 2
	p.batchDecode = true

	const (
		requestCount = 5
		activeCount  = 4 // cancel one request; real hybrid Q4_K still receives B=4
	)
	ready := make(chan struct{})
	p.coalesceReadyHook = func() { <-ready }
	var batchMu sync.Mutex
	var batches []int
	p.coalesceBatchHook = func(n int) {
		batchMu.Lock()
		batches = append(batches, n)
		batchMu.Unlock()
	}
	messages := make([][]Message, requestCount)
	for i := range messages {
		messages[i] = []Message{{Role: RoleUser, Content: string(rune('a' + i))}}
	}
	type answer struct {
		completion *Completion
		err        error
	}
	answers := make([]answer, requestCount)
	ctxs := make([]context.Context, requestCount)
	for i := range ctxs {
		ctxs[i] = context.Background()
	}
	cancelCtx, cancel := context.WithCancel(context.Background())
	ctxs[0] = cancelCtx
	var wg sync.WaitGroup
	for i := range answers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			answers[i].completion, answers[i].err = p.Complete(ctxs[i], messages[i], nil)
		}(i)
	}
	deadline := time.After(10 * time.Second)
	for {
		p.coalesceMu.Lock()
		n := len(p.coalesceReady)
		p.coalesceMu.Unlock()
		if n == requestCount {
			break
		}
		select {
		case <-deadline:
			t.Fatalf("planner queued %d/%d requests", n, requestCount)
		default:
			runtime.Gosched()
		}
	}
	cancel()
	close(ready)
	wg.Wait()

	if !errors.Is(answers[0].err, context.Canceled) {
		t.Fatalf("canceled lane err=%v", answers[0].err)
	}
	var cohortID uint64
	for i, answer := range answers[1:] {
		i++
		if answer.err != nil {
			t.Fatalf("coalesced[%d]: %v", i, answer.err)
		}
		if answer.completion == nil || answer.completion.InKernelBatch == nil {
			t.Fatalf("coalesced[%d] missing receipt", i)
		}
		receipt := answer.completion.InKernelBatch
		if receipt.CohortSize != activeCount || receipt.SharedPanels == 0 || receipt.SharedMACs == 0 || receipt.SessionCloses != activeCount {
			t.Fatalf("coalesced[%d] real Metal receipt=%+v", i, receipt)
		}
		if cohortID == 0 {
			cohortID = receipt.CohortID
		} else if receipt.CohortID != cohortID {
			t.Fatalf("coalesced[%d] cohort=%d, want %d", i, receipt.CohortID, cohortID)
		}
	}
	batchMu.Lock()
	defer batchMu.Unlock()
	if len(batches) == 0 || batches[0] != activeCount {
		t.Fatalf("planner batches=%v, want first batch B=%d", batches, activeCount)
	}
}

type qwen35MetalFixtureTensor struct {
	name  string
	shape []int
	q4k   bool
}

// newQwen35HybridQ4KMetalFixture builds the agent integration fixture through
// model's public loader-neutral builder. Test-only declarations in internal/model
// are not importable here, and putting this constructor in model production code
// would expose a synthetic checkpoint surface solely for a test. The Q4_K roster
// matches the real hybrid dispatch split: FFN plus full-attention v/o projections
// use raw Q4_K blocks, while reordered q/k and linear-attention projections use Q8.
func newQwen35HybridQ4KMetalFixture(t *testing.T) *model.Model {
	t.Helper()
	cfg := model.Config{
		HiddenSize:            256,
		NumLayers:             4,
		NumHeads:              4,
		NumKVHeads:            2,
		HeadDim:               64,
		IntermediateSize:      256,
		VocabSize:             512,
		RMSNormEps:            1e-5,
		RopeTheta:             10000,
		TieWordEmbeddings:     false,
		EOSTokenID:            -1,
		ModelType:             "qwen3_5_text",
		LayerTypes:            []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      64,
		LinearNumKeyHeads:     2,
		LinearValueHeadDim:    64,
		LinearNumValueHeads:   4,
		AttnOutputGate:        true,
		FullAttentionInterval: 4,
		NormGain1p:            true,
		QKNorm:                true,
		QKNormEps:             3e-5,
	}
	tensors := qwen35MetalFixtureTensors(cfg)
	builder := model.NewQuantBuilder(cfg, false)
	f32rng := rand.New(rand.NewSource(0x9075))
	q4rng := rand.New(rand.NewSource(99))
	for _, tensor := range tensors {
		values := qwen35MetalFixtureF32(f32rng, tensor)
		if err := builder.AddF32Tensor(tensor.name, tensor.shape, values); err != nil {
			t.Fatalf("add %s: %v", tensor.name, err)
		}
		if tensor.q4k {
			raw := qwen35MetalFixtureQ4K(q4rng, tensor.shape)
			if err := builder.AddResidentQ4K(tensor.name, tensor.shape, raw); err != nil {
				t.Fatalf("add resident Q4_K %s: %v", tensor.name, err)
			}
		}
	}
	m, err := builder.Build()
	if err != nil {
		t.Fatalf("build Qwen hybrid fixture: %v", err)
	}
	return m
}

func qwen35MetalFixtureTensors(cfg model.Config) []qwen35MetalFixtureTensor {
	const (
		keyDim  = 2 * 64
		valDim  = 4 * 64
		convDim = 2*keyDim + valDim
	)
	tensors := []qwen35MetalFixtureTensor{{"model.embed_tokens.weight", []int{cfg.VocabSize, cfg.HiddenSize}, false}}
	for layer, layerType := range cfg.LayerTypes {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		tensors = append(tensors, qwen35MetalFixtureTensor{prefix + "input_layernorm.weight", []int{cfg.HiddenSize}, false})
		if layerType == "linear_attention" {
			tensors = append(tensors,
				qwen35MetalFixtureTensor{prefix + "linear_attn.in_proj_qkv.weight", []int{convDim, cfg.HiddenSize}, false},
				qwen35MetalFixtureTensor{prefix + "linear_attn.in_proj_z.weight", []int{valDim, cfg.HiddenSize}, false},
				qwen35MetalFixtureTensor{prefix + "linear_attn.in_proj_b.weight", []int{cfg.LinearNumValueHeads, cfg.HiddenSize}, false},
				qwen35MetalFixtureTensor{prefix + "linear_attn.in_proj_a.weight", []int{cfg.LinearNumValueHeads, cfg.HiddenSize}, false},
				qwen35MetalFixtureTensor{prefix + "linear_attn.conv1d.weight", []int{convDim * cfg.LinearConvKernelDim}, false},
				qwen35MetalFixtureTensor{prefix + "linear_attn.A_log", []int{cfg.LinearNumValueHeads}, false},
				qwen35MetalFixtureTensor{prefix + "linear_attn.dt_bias", []int{cfg.LinearNumValueHeads}, false},
				qwen35MetalFixtureTensor{prefix + "linear_attn.norm.weight", []int{cfg.LinearValueHeadDim}, false},
				qwen35MetalFixtureTensor{prefix + "linear_attn.out_proj.weight", []int{cfg.HiddenSize, valDim}, false},
			)
		} else {
			tensors = append(tensors,
				qwen35MetalFixtureTensor{prefix + "self_attn.q_proj.weight", []int{2 * cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize}, false},
				qwen35MetalFixtureTensor{prefix + "self_attn.k_proj.weight", []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize}, false},
				qwen35MetalFixtureTensor{prefix + "self_attn.v_proj.weight", []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize}, true},
				qwen35MetalFixtureTensor{prefix + "self_attn.o_proj.weight", []int{cfg.HiddenSize, cfg.NumHeads * cfg.HeadDim}, true},
				qwen35MetalFixtureTensor{prefix + "self_attn.q_norm.weight", []int{cfg.NumHeads * cfg.HeadDim}, false},
				qwen35MetalFixtureTensor{prefix + "self_attn.k_norm.weight", []int{cfg.NumKVHeads * cfg.HeadDim}, false},
			)
		}
		tensors = append(tensors,
			qwen35MetalFixtureTensor{prefix + "post_attention_layernorm.weight", []int{cfg.HiddenSize}, false},
			qwen35MetalFixtureTensor{prefix + "mlp.gate_proj.weight", []int{cfg.IntermediateSize, cfg.HiddenSize}, true},
			qwen35MetalFixtureTensor{prefix + "mlp.up_proj.weight", []int{cfg.IntermediateSize, cfg.HiddenSize}, true},
			qwen35MetalFixtureTensor{prefix + "mlp.down_proj.weight", []int{cfg.HiddenSize, cfg.IntermediateSize}, true},
		)
	}
	return append(tensors,
		qwen35MetalFixtureTensor{"model.norm.weight", []int{cfg.HiddenSize}, false},
		qwen35MetalFixtureTensor{"lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize}, true},
	)
}

func qwen35MetalFixtureF32(rng *rand.Rand, tensor qwen35MetalFixtureTensor) []float32 {
	n := 1
	for _, dim := range tensor.shape {
		n *= dim
	}
	values := make([]float32, n)
	if strings.HasSuffix(tensor.name, "layernorm.weight") || strings.HasSuffix(tensor.name, "linear_attn.norm.weight") || tensor.name == "model.norm.weight" {
		for i := range values {
			values[i] = 1
		}
		return values
	}
	for i := range values {
		values[i] = (rng.Float32()*2 - 1) * 0.1
	}
	return values
}

func qwen35MetalFixtureQ4K(rng *rand.Rand, shape []int) []byte {
	const (
		q4kBlockWeights = 256
		q4kBlockBytes   = 144
	)
	out, in := shape[0], shape[1]
	raw := make([]byte, out*(in/q4kBlockWeights)*q4kBlockBytes)
	for offset := 0; offset < len(raw); offset += q4kBlockBytes {
		block := raw[offset : offset+q4kBlockBytes]
		_, _ = rng.Read(block)
		for scale := 0; scale < 2; scale++ {
			hi := block[scale*2+1]
			exponent := 2 + rng.Intn(5)
			block[scale*2+1] = (hi & 0x83) | byte(exponent<<2)
		}
	}
	return raw
}
