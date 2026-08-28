package agent

import (
	"context"
	"math/rand"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
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

type decodeTraceRun struct {
	emitted           []int
	events            []NativeDecodeTraceEvent
	decodeTokenIDs    []int
	fullTokenIDs      []int
	logprobs          []float64
	inferenceDisabled bool
}

func runDecodeTrace(t *testing.T, batched, captureTokenIDs bool) decodeTraceRun {
	t.Helper()
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	p := &InKernelPlanner{m: model.NewSynthetic(cfg), modelID: "synthetic-trace", quant: false, batchDecode: batched}
	ids := synthIDs(cfg.VocabSize, 12, 9071)
	measurement := &nativeInferenceMeasurement{
		inferenceDisabled:     true,
		decodeTokenIDsEnabled: captureTokenIDs,
		traceNow: decodeTraceClock(
			0,
			10*time.Nanosecond,
			11*time.Nanosecond,
			21*time.Nanosecond,
			25*time.Nanosecond,
			26*time.Nanosecond,
			46*time.Nanosecond,
			50*time.Nanosecond,
			55*time.Nanosecond,
			85*time.Nanosecond,
			100*time.Nanosecond,
		),
	}
	if captureTokenIDs {
		measurement.decodeTokenIDs = make([]int, 0, 4)
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
	if captureTokenIDs && cap(measurement.decodeTokenIDs) < 4 {
		t.Fatalf("light token-ID capacity = %d, want at least maxNew=4", cap(measurement.decodeTokenIDs))
	}
	return decodeTraceRun{
		emitted:           append([]int(nil), emitted...),
		events:            append([]NativeDecodeTraceEvent(nil), measurement.traceEvents...),
		decodeTokenIDs:    append([]int(nil), measurement.decodeTokenIDs...),
		fullTokenIDs:      append([]int(nil), measurement.tokenIDs...),
		logprobs:          append([]float64(nil), measurement.logprobs...),
		inferenceDisabled: measurement.inferenceDisabled,
	}
}

func TestInKernelDecodeTraceIndicesTimingAndBatchParity(t *testing.T) {
	serial := runDecodeTrace(t, false, false)
	serialIDs := runDecodeTrace(t, false, true)
	batched := runDecodeTrace(t, true, false)
	batchedIDs := runDecodeTrace(t, true, true)
	if !eqInts(batched.emitted, serial.emitted) || !eqInts(serialIDs.emitted, serial.emitted) || !eqInts(batchedIDs.emitted, serial.emitted) {
		t.Fatalf("serial/batched token mismatch: %v/%v/%v/%v", serial.emitted, serialIDs.emitted, batched.emitted, batchedIDs.emitted)
	}
	wantElapsed := []int64{10, 25, 50, 100}
	for name, trace := range map[string][]NativeDecodeTraceEvent{"serial": serial.events, "serial-ids": serialIDs.events, "batched": batched.events, "batched-ids": batchedIDs.events} {
		if len(trace) != len(wantElapsed) {
			t.Fatalf("%s trace length = %d, want %d", name, len(trace), len(wantElapsed))
		}
		for i, event := range trace {
			if event.TokenIndex != i+1 || event.ElapsedNS != wantElapsed[i] {
				t.Fatalf("%s event[%d] = %+v, want token_index=%d elapsed_ns=%d", name, i, event, i+1, wantElapsed[i])
			}
			if i == len(trace)-1 {
				if event.Forward != nil {
					t.Fatalf("%s final event has nonexistent forward timing: %+v", name, event.Forward)
				}
				continue
			}
			wantDuration := int64((i + 1) * 10)
			wantKind := NativeForwardSessionStep
			if name == "batched" || name == "batched-ids" {
				wantKind = NativeForwardStepBatchActive
			}
			if event.Forward == nil || event.Forward.DurationNS != wantDuration || event.Forward.Kind != wantKind || event.Forward.ActiveLanes != 1 {
				t.Fatalf("%s event[%d] forward = %+v, want kind=%s duration=%dns active_lanes=1", name, i, event.Forward, wantKind, wantDuration)
			}
		}
	}
	if !eqInts(serialIDs.decodeTokenIDs, serial.emitted) || !eqInts(batchedIDs.decodeTokenIDs, batched.emitted) {
		t.Fatalf("light token IDs serial/batched = %v/%v, emitted = %v/%v", serialIDs.decodeTokenIDs, batchedIDs.decodeTokenIDs, serial.emitted, batched.emitted)
	}
	if len(serial.decodeTokenIDs) != 0 || len(batched.decodeTokenIDs) != 0 {
		t.Fatalf("disabled light token IDs serial/batched = %v/%v, want empty", serial.decodeTokenIDs, batched.decodeTokenIDs)
	}
	for name, run := range map[string]decodeTraceRun{"serial-ids": serialIDs, "batched-ids": batchedIDs} {
		if !run.inferenceDisabled || len(run.fullTokenIDs) != 0 || len(run.logprobs) != 0 {
			t.Fatalf("%s inferenceDisabled/full token IDs/logprobs = %v/%v/%v, want true/empty/empty", name, run.inferenceDisabled, run.fullTokenIDs, run.logprobs)
		}
	}
	if !reflect.DeepEqual(serial.events, serialIDs.events) || !reflect.DeepEqual(batched.events, batchedIDs.events) {
		t.Fatalf("light token capture changed fake-clock timings: serial=%v/%v batched=%v/%v", serial.events, serialIDs.events, batched.events, batchedIDs.events)
	}
}

func TestInKernelDecodeTraceExactCachedLogitsBindsOnlyRealSteps(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	cfg := tinyGLMDsaCfg()
	backend := &countingBackend{Backend: compute.Default(), deviceMemory: true}
	p := NewInKernelPlanner(model.NewSyntheticGLMDsa(cfg), nil, "glm-dsa-device-trace", false, backend, false)
	p.quant = false
	ids := synthIDs(cfg.VocabSize, 9, 9754)
	decode(p, ids, 0) // prime exact prompt-final logits without a decode step.

	measurement := &nativeInferenceMeasurement{
		inferenceDisabled: true,
		traceNow: decodeTraceClock(
			0,
			10*time.Nanosecond, 11*time.Nanosecond, 21*time.Nanosecond,
			30*time.Nanosecond, 31*time.Nanosecond, 51*time.Nanosecond,
			60*time.Nanosecond,
		),
	}
	gen, _, _, matched, tier, _, _, _, err := p.generateReusedContextWithBias(
		context.Background(), ids, 3, 0, 0, 0, nil, 0, 0, map[int]bool{}, nil, measurement,
	)
	if err != nil {
		t.Fatal(err)
	}
	if gen != 3 || matched != len(ids) || tier != radixkv.SnapshotTierDeviceL1 {
		t.Fatalf("exact replay generated/matched/tier = %d/%d/%s, want 3/%d/device_l1", gen, matched, tier, len(ids))
	}
	if len(measurement.traceEvents) != 3 {
		t.Fatalf("exact replay trace length = %d, want 3", len(measurement.traceEvents))
	}
	for i, wantDuration := range []int64{10, 20} {
		forward := measurement.traceEvents[i].Forward
		if forward == nil || forward.Kind != NativeForwardSessionStep || forward.DurationNS != wantDuration || forward.ActiveLanes != 1 {
			t.Fatalf("exact replay event[%d] forward = %+v, want first/steady direct Step duration %dns", i, forward, wantDuration)
		}
	}
	if measurement.traceEvents[2].Forward != nil {
		t.Fatalf("exact replay final event has nonexistent Step: %+v", measurement.traceEvents[2].Forward)
	}
}

func TestInKernelDecodeTraceBatchedForwardIsOneSharedWallTime(t *testing.T) {
	cfg := tinyCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	clocks := [][]time.Duration{
		{0, 10 * time.Nanosecond, 11 * time.Nanosecond, 31 * time.Nanosecond, 40 * time.Nanosecond},
		{0, 12 * time.Nanosecond, 42 * time.Nanosecond},
	}
	lanes := make([]*decodeLane, len(clocks))
	for i := range lanes {
		lanes[i], _ = newDecodeLaneOver(m, synthIDs(cfg.VocabSize, 8+i, uint64(97540+i)), 2, 0)
		lanes[i].measurement = &nativeInferenceMeasurement{inferenceDisabled: true, traceNow: decodeTraceClock(clocks[i]...)}
		lanes[i].measurement.startDecodeTrace()
	}
	inKernelDecodeLanesBatched(context.Background(), lanes, m, false)

	for i, lane := range lanes {
		if len(lane.measurement.traceEvents) != 2 {
			t.Fatalf("lane %d trace length = %d, want 2", i, len(lane.measurement.traceEvents))
		}
		forward := lane.measurement.traceEvents[0].Forward
		if forward == nil || forward.Kind != NativeForwardStepBatchActive || forward.DurationNS != 20 || forward.ActiveLanes != 2 {
			t.Fatalf("lane %d shared forward = %+v, want one 20ns StepBatchActive wall time across 2 lanes", i, forward)
		}
		if lane.measurement.traceEvents[1].Forward != nil {
			t.Fatalf("lane %d final event has nonexistent batch forward: %+v", i, lane.measurement.traceEvents[1].Forward)
		}
	}
}

func TestInKernelDecodeTraceRecordsAfterEmitAndCount(t *testing.T) {
	base := time.Unix(1_000, 0)
	clockCalls := 0
	emitted := false
	measurement := &nativeInferenceMeasurement{inferenceDisabled: true, decodeTokenIDsEnabled: true}
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
	if !measurement.inferenceDisabled || !eqInts(measurement.decodeTokenIDs, []int{0}) || len(measurement.tokenIDs) != 0 || len(measurement.logprobs) != 0 {
		t.Fatalf("inferenceDisabled/light/full IDs/logprobs = %v/%v/%v/%v, want true/[0]/empty/empty", measurement.inferenceDisabled, measurement.decodeTokenIDs, measurement.tokenIDs, measurement.logprobs)
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
	if lane.gen != 0 || !lane.stopped || len(measurement.traceEvents) != 0 || len(measurement.decodeTokenIDs) != 0 || clockCalls != 1 {
		t.Fatalf("stop state gen/stopped/events/token_ids/clock = %d/%v/%d/%d/%d, want 0/true/0/0/1", lane.gen, lane.stopped, len(measurement.traceEvents), len(measurement.decodeTokenIDs), clockCalls)
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
	if comp.DecodeTrace != nil || comp.NativeDecodeTokenIDs != nil || clockCalls != 0 {
		t.Fatalf("default trace/token IDs/clock calls = %#v/%#v/%d, want nil/nil/0", comp.DecodeTrace, comp.NativeDecodeTokenIDs, clockCalls)
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
		inferenceDisabled:     true,
		decodeTokenIDsEnabled: true,
		decodeTokenIDs:        make([]int, 0, 3),
		traceNow: decodeTraceClock(
			0, 10*time.Nanosecond, 11*time.Nanosecond,
			100*time.Nanosecond,
			110*time.Nanosecond, 111*time.Nanosecond, 121*time.Nanosecond,
			130*time.Nanosecond, 131*time.Nanosecond, 141*time.Nanosecond,
			150*time.Nanosecond,
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
		if event.TokenIndex != i+1 || event.ElapsedNS != int64((i+1)*20-10) {
			t.Fatalf("retry event[%d] = %+v, want clean second-attempt index/timing", i, event)
		}
	}
	if !measurement.inferenceDisabled || cap(measurement.decodeTokenIDs) < 3 || !eqInts(measurement.decodeTokenIDs, emitted) || len(measurement.tokenIDs) != 0 || len(measurement.logprobs) != 0 {
		t.Fatalf("retry inferenceDisabled/cap/light/full IDs/logprobs = %v/%d/%v/%v/%v, emitted=%v", measurement.inferenceDisabled, cap(measurement.decodeTokenIDs), measurement.decodeTokenIDs, measurement.tokenIDs, measurement.logprobs, emitted)
	}
}

type cudaUploadSnapshotBackend struct {
	compute.Backend
	once          sync.Once
	mu            sync.Mutex
	calls         uint64
	transferBytes uint64
	residentBytes uint64
}

func (b *cudaUploadSnapshotBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.once.Do(func() {
		b.mu.Lock()
		b.calls++
		b.transferBytes += 4096
		b.residentBytes += 2048
		b.mu.Unlock()
	})
	return b.Backend.MatMul(w, x)
}

func (b *cudaUploadSnapshotBackend) CUDAImmutableWeightUploadSnapshot() (calls, transferBytes, residentBytes uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.calls, b.transferBytes, b.residentBytes
}

func TestNativeInferenceReceiptCarriesCumulativeCUDAUploadDelta(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	be := &cudaUploadSnapshotBackend{Backend: compute.Default()}
	p := NewInKernelPlanner(m, loadProbeTok(t), "synthetic-cuda-upload-receipt", false, be, false)
	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "receipt"}}, nil, WithMaxTokens(3), WithNativeInferenceReceipt(true))
	if err != nil {
		t.Fatal(err)
	}
	got := comp.NativeInference.CUDAImmutableWeightUploads
	if got == nil {
		t.Fatal("native inference receipt omitted available CUDA immutable-weight upload counters")
	}
	if got.Before != (NativeCUDAImmutableWeightUploadCounters{}) || got.After != (NativeCUDAImmutableWeightUploadCounters{Calls: 1, TransferBytes: 4096, ResidentBytes: 2048}) || got.Delta != got.After {
		t.Fatalf("CUDA immutable-weight upload window = %+v, want zero before and exact 1/4096/2048 after+delta", got)
	}
}
