package agent

import (
	"context"
	"math/rand"
	"testing"
	"time"

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
	const maxNew = 8
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	prompt := synthIDs(cfg.VocabSize, 9, 4242)

	host, hostSink := newDecodeLaneOver(m, prompt, maxNew, 0)
	inKernelDecodeSerial(context.Background(), host)
	dev, devSink := newBackendDecodeLane(m, prompt, maxNew)
	inKernelDecodeSerial(context.Background(), dev)

	if host.err != nil || dev.err != nil {
		t.Fatalf("decode errors host=%v device=%v", host.err, dev.err)
	}
	if !eqInts(devSink.toks, hostSink.toks) || len(devSink.toks) != maxNew {
		t.Fatalf("device-greedy tokens %v, want host-sampled %v", devSink.toks, hostSink.toks)
	}
	if !dev.greedyReady || dev.logits != nil || dev.vocab != cfg.VocabSize {
		t.Fatalf("device lane greedyReady=%t logits=%d vocab=%d, want token-only forwards over vocab %d",
			dev.greedyReady, len(dev.logits), dev.vocab, cfg.VocabSize)
	}
}

func TestInKernelBatchedSingletonGreedyDecodeSelectsOnDevice(t *testing.T) {
	const maxNew = 8
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	prompt := synthIDs(cfg.VocabSize, 11, 515)

	host, hostSink := newDecodeLaneOver(m, prompt, maxNew, 0)
	inKernelDecodeSerial(context.Background(), host)
	dev, devSink := newBackendDecodeLane(m, prompt, maxNew)
	inKernelDecodeLanesBatched(context.Background(), []*decodeLane{dev}, m, false)

	if host.err != nil || dev.err != nil {
		t.Fatalf("decode errors host=%v device=%v", host.err, dev.err)
	}
	if !eqInts(devSink.toks, hostSink.toks) || len(devSink.toks) != maxNew {
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

// This exercises the real CPU-reference backend's token-only route. The clock
// below is synthetic metadata; it is not GPU or performance evidence.
// fak-test:runtime fast est=200ms lane=default
func TestInKernelTraceOnlyPreservesGreedyRoute(t *testing.T) {
	if ref, ok := compute.Lookup("cpu-ref"); !ok || ref == nil {
		t.Fatal("cpu-ref backend unavailable")
	}
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("close model weights: %v", err)
		}
	})
	prompt := synthIDs(cfg.VocabSize, 7, 7751)
	const tokens = 4
	for _, batched := range []bool{false, true} {
		run := func(ln *decodeLane) {
			if batched {
				inKernelDecodeLanesBatched(context.Background(), []*decodeLane{ln}, m, false)
			} else {
				inKernelDecodeSerial(context.Background(), ln)
			}
		}
		plain, plainSink := newBackendDecodeLane(m, prompt, tokens)
		func() { defer plain.s.Close(); run(plain) }()
		if plain.err != nil || !plain.greedyReady || plain.logits != nil {
			t.Fatalf("batched=%t: plain err=%v greedy=%t logits=%d", batched, plain.err, plain.greedyReady, len(plain.logits))
		}
		for _, tc := range []struct {
			name             string
			ids, fullReceipt bool
		}{
			{name: "trace_only"},
			{name: "trace_and_light_ids", ids: true},
			{name: "full_logprob_receipt", ids: true, fullReceipt: true},
		} {
			t.Run(tc.name+map[bool]string{false: "/serial", true: "/singleton_batch"}[batched], func(t *testing.T) {
				ln, sink := newBackendDecodeLane(m, prompt, tokens)
				t.Cleanup(ln.s.Close)
				clockCalls := 0
				measurement := &nativeInferenceMeasurement{
					startedAt:             time.Unix(0, 0),
					inferenceDisabled:     !tc.fullReceipt,
					decodeTokenIDsEnabled: tc.ids,
					traceNow:              func() time.Time { clockCalls++; return time.Unix(0, int64(clockCalls)) },
				}
				ln.measurement = measurement
				measurement.startDecodeTrace()
				if ln.deviceGreedy() == tc.fullReceipt {
					t.Fatalf("deviceGreedy=%t fullReceipt=%t", ln.deviceGreedy(), tc.fullReceipt)
				}
				run(ln)
				if ln.err != nil || !eqInts(sink.toks, plainSink.toks) || len(sink.toks) != tokens {
					t.Fatalf("err=%v tokens=%v, want %v", ln.err, sink.toks, plainSink.toks)
				}
				if ln.greedyReady == tc.fullReceipt || (len(ln.logits) == 0) == tc.fullReceipt {
					t.Fatalf("greedy=%t logits=%d fullReceipt=%t", ln.greedyReady, len(ln.logits), tc.fullReceipt)
				}
				if clockCalls == 0 || len(measurement.traceEvents) != tokens {
					t.Fatalf("clock calls=%d trace events=%d, want a complete trace", clockCalls, len(measurement.traceEvents))
				}
				if tc.ids && !eqInts(measurement.decodeTokenIDs, plainSink.toks) {
					t.Fatalf("light IDs=%v, want %v", measurement.decodeTokenIDs, plainSink.toks)
				}
				wantFull := 0
				if tc.fullReceipt {
					wantFull = tokens
				}
				if len(measurement.logprobs) != wantFull || len(measurement.tokenIDs) != wantFull {
					t.Fatalf("full token/logprob counts=%d/%d, want %d", len(measurement.tokenIDs), len(measurement.logprobs), wantFull)
				}
				for i, event := range measurement.traceEvents {
					if (event.Forward != nil) != (i < tokens-1) {
						t.Fatalf("trace event %d forward=%+v, final emitted token must have no invented forward", i+1, event.Forward)
					}
				}
			})
		}
	}
}

// greedyRouteProbeBackend counts actual CPU-reference argmax calls. CloneTensor
// preserves the continuation-logit capability rather than forcing its fallback.
type greedyRouteProbeBackend struct {
	compute.Backend
	cloner      compute.TensorCloner
	argmaxCalls int
}

func (b *greedyRouteProbeBackend) Argmax(logits compute.Tensor) int {
	b.argmaxCalls++
	return b.Backend.Argmax(logits)
}

func (b *greedyRouteProbeBackend) CloneTensor(t compute.Tensor) (compute.Tensor, error) {
	return b.cloner.CloneTensor(t)
}

// Exercise actual Complete options -> planner measurement -> real token-only
// forwards. Removing inferenceDisabled: !sp.NativeInferenceReceipt makes the
// trace-only arm lose its argmax calls, even if the direct consumer test passes.
// fak-test:runtime fast est=300ms lane=default
func TestInKernelCompleteTraceOptionsPreserveGreedyRoute(t *testing.T) {
	ref, ok := compute.Lookup("cpu-ref")
	if !ok || ref == nil {
		t.Fatal("cpu-ref backend unavailable")
	}
	cloner, ok := ref.(compute.TensorCloner)
	if !ok {
		t.Fatal("cpu-ref tensor clone capability unavailable")
	}
	tok := loadProbeTok(t)
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	const tokens = 4
	zero := 0.0
	for _, batched := range []bool{false, true} {
		var plainText string
		for _, tc := range []struct {
			name               string
			trace, fullReceipt bool
		}{
			{name: "plain"},
			{name: "trace_and_light_ids", trace: true},
			{name: "full_logprob_receipt", trace: true, fullReceipt: true},
		} {
			t.Run(tc.name+map[bool]string{false: "/serial", true: "/singleton_batch"}[batched], func(t *testing.T) {
				be := &greedyRouteProbeBackend{Backend: ref, cloner: cloner}
				m := model.NewSynthetic(cfg)
				t.Cleanup(func() {
					if err := m.CloseWeights(); err != nil {
						t.Errorf("close model weights: %v", err)
					}
				})
				p := NewInKernelPlanner(m, tok, "synthetic-greedy-route", false, be, false)
				t.Cleanup(p.ReleaseStartupWarm)
				p.batchDecode = batched
				clockCalls := 0
				p.decodeTraceNow = func() time.Time { clockCalls++; return time.Unix(0, int64(clockCalls)) }
				comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "plain"}}, nil,
					WithRawPrompt(), WithMaxTokens(tokens), WithTemperature(&zero),
					WithDecodeTrace(tc.trace), WithNativeDecodeTokenIDs(tc.trace), WithNativeInferenceReceipt(tc.fullReceipt))
				if err != nil {
					t.Fatal(err)
				}
				if comp.Usage.CompletionTokens != tokens {
					t.Fatalf("completion tokens=%d, want %d", comp.Usage.CompletionTokens, tokens)
				}
				if tc.name == "plain" {
					plainText = comp.Message.Content
				} else if comp.Message.Content != plainText {
					t.Fatalf("content=%q, want %q", comp.Message.Content, plainText)
				}
				wantArgmax := tokens - 1
				if tc.fullReceipt {
					wantArgmax = 0
				}
				if be.argmaxCalls != wantArgmax {
					t.Fatalf("backend argmax calls=%d, want %d (fullReceipt=%t)", be.argmaxCalls, wantArgmax, tc.fullReceipt)
				}
				if tc.trace {
					if clockCalls == 0 || comp.DecodeTrace == nil || len(comp.DecodeTrace.Events) != tokens || comp.NativeDecodeTokenIDs == nil || len(comp.NativeDecodeTokenIDs.TokenIDs) != tokens {
						t.Fatalf("trace/token-ID mapping missing: clocks=%d completion=%+v", clockCalls, comp)
					}
				} else if clockCalls != 0 || comp.DecodeTrace != nil || comp.NativeDecodeTokenIDs != nil {
					t.Fatalf("default-off observer was active: clocks=%d", clockCalls)
				}
				if (comp.NativeInference != nil) != tc.fullReceipt {
					t.Fatalf("full receipt presence=%t, want %t", comp.NativeInference != nil, tc.fullReceipt)
				}
			})
		}
	}
}
