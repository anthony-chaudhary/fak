package agent

import (
	"context"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func newBackendDecodeLane(m *model.Model, prompt []int, maxNew int) (*decodeLane, *batchLaneSink) {
	sink := &batchLaneSink{}
	s := m.NewBackendSession(compute.Pick("cpu-ref"))
	return &decodeLane{
		s:      s,
		logits: s.Prefill(prompt),
		rng:    rand.New(rand.NewSource(0)),
		stops:  map[int]bool{},
		emit:   sink.emit,
		maxNew: maxNew,
	}, sink
}

func TestInKernelSerialGreedyDecodeSelectsOnDevice(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	prompt := synthIDs(cfg.VocabSize, 9, 4242)

	host, hostSink := newDecodeLaneOver(m, prompt, 8, 0)
	inKernelDecodeSerial(context.Background(), host)
	dev, devSink := newBackendDecodeLane(m, prompt, 8)
	inKernelDecodeSerial(context.Background(), dev)

	if host.err != nil || dev.err != nil {
		t.Fatalf("decode errors host=%v device=%v", host.err, dev.err)
	}
	if !eqInts(devSink.toks, hostSink.toks) || len(devSink.toks) != 8 {
		t.Fatalf("device-greedy tokens %v, want host-sampled %v", devSink.toks, hostSink.toks)
	}
	if !dev.greedyReady || dev.logits != nil || dev.vocab != cfg.VocabSize {
		t.Fatalf("device lane greedyReady=%t logits=%d vocab=%d, want token-only forwards over vocab %d",
			dev.greedyReady, len(dev.logits), dev.vocab, cfg.VocabSize)
	}
}

func TestInKernelBatchedSingletonGreedyDecodeSelectsOnDevice(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	prompt := synthIDs(cfg.VocabSize, 11, 515)

	host, hostSink := newDecodeLaneOver(m, prompt, 8, 0)
	inKernelDecodeSerial(context.Background(), host)
	dev, devSink := newBackendDecodeLane(m, prompt, 8)
	inKernelDecodeLanesBatched(context.Background(), []*decodeLane{dev}, m, false)

	if host.err != nil || dev.err != nil {
		t.Fatalf("decode errors host=%v device=%v", host.err, dev.err)
	}
	if !eqInts(devSink.toks, hostSink.toks) || len(devSink.toks) != 8 {
		t.Fatalf("batched device-greedy tokens %v, want host-sampled %v", devSink.toks, hostSink.toks)
	}
	if !dev.greedyReady || dev.logits != nil || dev.vocab != cfg.VocabSize {
		t.Fatalf("batched device lane greedyReady=%t logits=%d vocab=%d, want token-only forwards",
			dev.greedyReady, len(dev.logits), dev.vocab)
	}
}

func TestInKernelGreedyContinuationKeepsLastDistribution(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	prompt := synthIDs(cfg.VocabSize, 10, 9191)
	host, _ := newDecodeLaneOver(m, prompt, 6, 0)
	host.forwarded = []int{}
	inKernelDecodeSerial(context.Background(), host)
	for name, run := range map[string]func(*decodeLane){
		"serial":  func(ln *decodeLane) { inKernelDecodeSerial(context.Background(), ln) },
		"batched": func(ln *decodeLane) { inKernelDecodeLanesBatched(context.Background(), []*decodeLane{ln}, m, false) },
	} {
		dev, _ := newBackendDecodeLane(m, prompt, 6)
		dev.forwarded = []int{}
		run(dev)
		if dev.err != nil || !dev.greedyReady || dev.logits != nil || !eqInts(dev.forwarded, host.forwarded) {
			t.Fatalf("%s: err=%v greedyReady=%t logits=%d forwarded=%v want %v", name, dev.err, dev.greedyReady, len(dev.logits), dev.forwarded, host.forwarded)
		}
		got := dev.continuationLogits()
		if len(got) != len(host.logits) {
			t.Fatalf("%s: continuation logits=%d, want %d", name, len(got), len(host.logits))
		}
		for i := range got {
			if got[i] != host.logits[i] {
				t.Fatalf("%s: continuation logit[%d]=%v want %v", name, i, got[i], host.logits[i])
			}
		}
	}
}

func TestInKernelSerialDecodeKeepsLogitsWhenTheDistributionIsRead(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	prompt := synthIDs(cfg.VocabSize, 7, 777)
	for name, mutate := range map[string]func(*decodeLane){
		"temperature": func(ln *decodeLane) { ln.temp = 0.7 },
		"logit_bias":  func(ln *decodeLane) { ln.logitBias = model.LogitBias{3: 1} },
		"frequency":   func(ln *decodeLane) { ln.freqPenalty, ln.counts = 0.5, make([]int32, cfg.VocabSize) },
		"logprobs":    func(ln *decodeLane) { ln.measurement = &nativeInferenceMeasurement{} },
	} {
		ln, _ := newBackendDecodeLane(m, prompt, 4)
		mutate(ln)
		if ln.deviceGreedy() {
			t.Fatalf("%s: deviceGreedy=true, want host logits", name)
		}
		inKernelDecodeSerial(context.Background(), ln)
		if ln.err != nil || ln.greedyReady || len(ln.logits) != cfg.VocabSize {
			t.Fatalf("%s: err=%v greedyReady=%t logits=%d, want host logits", name, ln.err, ln.greedyReady, len(ln.logits))
		}
	}
}
