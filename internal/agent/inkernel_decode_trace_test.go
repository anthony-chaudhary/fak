package agent

import (
	"context"
	"math/rand"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func decodeTraceClock(offsets ...time.Duration) func() time.Time {
	base := time.Unix(1_000, 0)
	i := 0
	return func() time.Time {
		if i >= len(offsets) {
			panic("decode trace clock exhausted")
		}
		at := base.Add(offsets[i])
		i++
		return at
	}
}

func runDecodeTrace(t *testing.T, batched bool) ([]int, []NativeDecodeTraceEvent) {
	t.Helper()
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	p := &InKernelPlanner{m: model.NewSynthetic(cfg), modelID: "synthetic-trace", quant: false, batchDecode: batched}
	ids := synthIDs(cfg.VocabSize, 12, 9071)
	measurement := &nativeInferenceMeasurement{
		inferenceDisabled: true,
		traceNow: decodeTraceClock(
			0,
			10*time.Nanosecond,
			25*time.Nanosecond,
			25*time.Nanosecond,
			80*time.Nanosecond,
		),
	}
	var emitted []int
	gen, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(
		context.Background(), ids, 4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(tok int) bool {
			emitted = append(emitted, tok)
			return false
		}, measurement,
	)
	if err != nil {
		t.Fatal(err)
	}
	if gen != 4 || len(emitted) != gen {
		t.Fatalf("generated/emitted = %d/%d, want 4/4", gen, len(emitted))
	}
	return emitted, append([]NativeDecodeTraceEvent(nil), measurement.traceEvents...)
}

func TestInKernelDecodeTraceIndicesTimingAndBatchParity(t *testing.T) {
	serialTokens, serialTrace := runDecodeTrace(t, false)
	batchedTokens, batchedTrace := runDecodeTrace(t, true)
	if !eqInts(batchedTokens, serialTokens) {
		t.Fatalf("batched tokens = %v, serial = %v", batchedTokens, serialTokens)
	}
	wantElapsed := []int64{10, 25, 25, 80}
	for name, trace := range map[string][]NativeDecodeTraceEvent{"serial": serialTrace, "batched": batchedTrace} {
		if len(trace) != len(wantElapsed) {
			t.Fatalf("%s trace length = %d, want %d", name, len(trace), len(wantElapsed))
		}
		for i, event := range trace {
			if event.TokenIndex != i+1 || event.ElapsedNS != wantElapsed[i] {
				t.Fatalf("%s event[%d] = %+v, want token_index=%d elapsed_ns=%d", name, i, event, i+1, wantElapsed[i])
			}
		}
	}
}

func TestInKernelDecodeTraceRecordsAfterEmitAndCount(t *testing.T) {
	base := time.Unix(1_000, 0)
	clockCalls := 0
	emitted := false
	measurement := &nativeInferenceMeasurement{inferenceDisabled: true}
	var lane *decodeLane
	measurement.traceNow = func() time.Time {
		clockCalls++
		if clockCalls == 1 {
			return base
		}
		if !emitted || lane.gen != 1 || lane.counts[0] != 1 {
			t.Fatalf("trace clock observed emit/gen/count = %v/%d/%d, want true/1/1", emitted, lane.gen, lane.counts[0])
		}
		return base.Add(9 * time.Nanosecond)
	}
	lane = &decodeLane{
		logits:      []float32{1},
		counts:      make([]int32, 1),
		rng:         rand.New(rand.NewSource(9071)),
		emit:        func(int) bool { emitted = true; return false },
		stops:       map[int]bool{},
		maxNew:      1,
		measurement: measurement,
	}
	measurement.startDecodeTrace()
	if next, advance := lane.decodeOne(context.Background()); next != 0 || advance {
		t.Fatalf("decodeOne = token %d advance=%v, want token 0 advance=false", next, advance)
	}
	if len(measurement.traceEvents) != 1 || measurement.traceEvents[0] != (NativeDecodeTraceEvent{TokenIndex: 1, ElapsedNS: 9}) {
		t.Fatalf("trace = %+v, want one post-commit event", measurement.traceEvents)
	}
}

func TestInKernelDecodeTraceDoesNotRecordStopToken(t *testing.T) {
	base := time.Unix(1_000, 0)
	clockCalls := 0
	measurement := &nativeInferenceMeasurement{
		inferenceDisabled: true,
		traceNow: func() time.Time {
			clockCalls++
			return base
		},
	}
	lane := &decodeLane{
		logits:      []float32{1},
		rng:         rand.New(rand.NewSource(9071)),
		stops:       map[int]bool{0: true},
		maxNew:      1,
		measurement: measurement,
	}
	measurement.startDecodeTrace()
	if next, advance := lane.decodeOne(context.Background()); next != 0 || advance {
		t.Fatalf("decodeOne = token %d advance=%v, want stop without advance", next, advance)
	}
	if lane.gen != 0 || !lane.stopped || len(measurement.traceEvents) != 0 || clockCalls != 1 {
		t.Fatalf("stop state gen/stopped/events/clock = %d/%v/%d/%d, want 0/true/0/1", lane.gen, lane.stopped, len(measurement.traceEvents), clockCalls)
	}
}

func TestInKernelDecodeTraceDefaultOffIsInert(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-default-off", false, nil, false)
	p.maxNew = 2
	clockCalls := 0
	p.decodeTraceNow = func() time.Time {
		clockCalls++
		return time.Now()
	}
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "plain"}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if comp.DecodeTrace != nil || clockCalls != 0 {
		t.Fatalf("default trace/clock calls = %#v/%d, want nil/0", comp.DecodeTrace, clockCalls)
	}
}

type decodeTraceOOMBackend struct {
	compute.Backend
	measurement *nativeInferenceMeasurement
	failed      bool
}

func (b *decodeTraceOOMBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if !b.failed && len(b.measurement.traceEvents) == 1 {
		b.failed = true
		panic(&compute.DeviceAllocError{Bytes: 4096, Site: "decode-trace-after-first-token", Class: compute.MemoryScratchpad})
	}
	return b.Backend.MatMul(w, x)
}

func (b *decodeTraceOOMBackend) Recycle()      {}
func (b *decodeTraceOOMBackend) Trim()         {}
func (b *decodeTraceOOMBackend) TrimLarge(int) {}

func TestInKernelDecodeTraceOOMRetryDropsFirstAttempt(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	measurement := &nativeInferenceMeasurement{
		inferenceDisabled: true,
		traceNow: decodeTraceClock(
			0, 10*time.Nanosecond,
			100*time.Nanosecond, 110*time.Nanosecond, 120*time.Nanosecond, 130*time.Nanosecond,
		),
	}
	be := &decodeTraceOOMBackend{Backend: compute.Default(), measurement: measurement}
	p := &InKernelPlanner{m: m, modelID: "synthetic-trace-retry", backend: be}
	ids := synthIDs(cfg.VocabSize, 8, 9072)
	var emitted []int
	res, err := p.generateReusedWithOOMRetry(context.Background(), ids, 3, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(tok int) bool {
		emitted = append(emitted, tok)
		return false
	}, func() {
		emitted = emitted[:0]
		measurement.reset()
	}, measurement)
	if err != nil {
		t.Fatal(err)
	}
	if !be.failed || res.gen != 3 || len(emitted) != 3 || len(measurement.traceEvents) != 3 {
		t.Fatalf("retry state failed=%v gen=%d emitted=%d events=%d, want true/3/3/3", be.failed, res.gen, len(emitted), len(measurement.traceEvents))
	}
	for i, event := range measurement.traceEvents {
		if event.TokenIndex != i+1 || event.ElapsedNS != int64((i+1)*10) {
			t.Fatalf("retry event[%d] = %+v, want clean second-attempt index/timing", i, event)
		}
	}
}
