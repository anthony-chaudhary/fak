package armbench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestGPUWaveReachesConcurrencyEightAndReceiptsDeterministically(t *testing.T) {
	m, tasks := eightGPUManifest()
	p := newGateProvider(8)
	type result struct {
		run *Run
		err error
	}
	done := make(chan result, 1)
	go func() {
		run, err := Execute(context.Background(), m, tasks, p, &FakeGrader{}, Options{MaxParallel: 8})
		done <- result{run: run, err: err}
	}()

	seenDevices := map[int]bool{}
	for i := 0; i < 8; i++ {
		req := waitRequest(t, p.started)
		if req.GPUIndex == nil || req.CUDAVisibleDevices != fmt.Sprint(*req.GPUIndex) {
			t.Fatalf("provider request has incomplete GPU environment: %+v", req)
		}
		seenDevices[*req.GPUIndex] = true
	}
	if len(seenDevices) != 8 { //boundarylint:ignore CHANGE_DETECTOR_TEST test runs exactly 8 parallel trials
		t.Fatalf("provider saw devices %v, want eight distinct assignments", seenDevices)
	}
	if got := p.maxActive(); got != 8 {
		t.Fatalf("peak concurrency = %d, want 8", got)
	}
	for i := 0; i < 8; i++ {
		p.release <- struct{}{}
	}
	got := waitRun(t, done)
	if got.err != nil {
		t.Fatalf("Execute: %v", got.err)
	}
	if len(got.run.Trials) != 8 { //boundarylint:ignore CHANGE_DETECTOR_TEST test runs exactly 8 parallel trials
		t.Fatalf("ledger rows = %d, want 8", len(got.run.Trials))
	}
	for i, row := range got.run.Trials {
		wantArm := fmt.Sprintf("arm-%d", i)
		if row.ArmID != wantArm {
			t.Fatalf("ledger[%d].arm_id = %q, want deterministic %q", i, row.ArmID, wantArm)
		}
		if row.Launch == nil {
			t.Fatalf("ledger[%d] has no launch receipt", i)
		}
		r := row.Launch
		if r.Wave != 1 || r.GPUIndex == nil || *r.GPUIndex != i || r.CUDAVisibleDevices != fmt.Sprint(i) {
			t.Errorf("ledger[%d] wave/device receipt = %+v", i, r)
		}
		if r.StartedAt == "" || r.EndedAt == "" || r.WallMS < 0 || r.ExitCode != 0 || !r.Reaped || r.ReapOutcome == "" {
			t.Errorf("ledger[%d] incomplete timing/exit/reap receipt = %+v", i, r)
		}
	}
	rep, err := Summarize(got.run)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	for i, arm := range rep.Arms {
		if arm.ArmID != fmt.Sprintf("arm-%d", i) {
			t.Fatalf("report arm order[%d] = %q", i, arm.ArmID)
		}
	}
}

func TestGPUWaveHonorsMaxParallelBound(t *testing.T) {
	m, tasks := eightGPUManifest()
	p := newGateProvider(8)
	type result struct {
		run *Run
		err error
	}
	done := make(chan result, 1)
	go func() {
		run, err := Execute(context.Background(), m, tasks, p, &FakeGrader{}, Options{MaxParallel: 3})
		done <- result{run: run, err: err}
	}()

	for _, waveSize := range []int{3, 3, 2} {
		for i := 0; i < waveSize; i++ {
			waitRequest(t, p.started)
		}
		select {
		case req := <-p.started:
			t.Fatalf("arm %s launched before the wave barrier released", req.ArmID)
		case <-time.After(25 * time.Millisecond):
		}
		for i := 0; i < waveSize; i++ {
			p.release <- struct{}{}
		}
	}
	got := waitRun(t, done)
	if got.err != nil {
		t.Fatalf("Execute: %v", got.err)
	}
	if peak := p.maxActive(); peak != 3 {
		t.Fatalf("peak concurrency = %d, want exact bound 3", peak)
	}
	waves := map[int]int{}
	for _, row := range got.run.Trials {
		waves[row.Launch.Wave]++
	}
	if fmt.Sprint(waves) != "map[1:3 2:3 3:2]" {
		t.Fatalf("wave membership = %v", waves)
	}
}

func TestGPUAssignmentsRefuseCollisionAndUnknownBeforeLaunch(t *testing.T) {
	t.Run("duplicate", func(t *testing.T) {
		m, tasks := eightGPUManifest()
		m.Arms[1].GPUIndex = intPtr(*m.Arms[0].GPUIndex)
		p := &admissionProbe{}
		_, err := Execute(context.Background(), m, tasks, p, &FakeGrader{}, Options{MaxParallel: 8})
		requireReason(t, err, ReasonGPUAssignmentDuplicate)
		if p.calls() != 0 {
			t.Fatalf("provider setup/complete called %d times before collision refusal", p.calls())
		}
	})

	t.Run("unknown", func(t *testing.T) {
		m, tasks := eightGPUManifest()
		m.Arms[7].GPUIndex = nil
		p := &admissionProbe{}
		_, err := Execute(context.Background(), m, tasks, p, &FakeGrader{}, Options{MaxParallel: 8})
		requireReason(t, err, ReasonGPUAssignmentUnknown)
		if p.calls() != 0 {
			t.Fatalf("provider setup/complete called %d times before unknown-assignment refusal", p.calls())
		}
	})
}

func TestGPUWaveFailureCancelsAndReapsEveryLaunch(t *testing.T) {
	m, tasks := eightGPUManifest()
	p := newCancelProvider(8)
	_, err := Execute(context.Background(), m, tasks, p, &FakeGrader{}, Options{MaxParallel: 8})
	if err == nil || !strings.Contains(err.Error(), "synthetic launch failure") {
		t.Fatalf("Execute error = %v, want synthetic launch failure", err)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.active != 0 || p.returned != 8 {
		t.Fatalf("cleanup active=%d returned=%d, want 0/8", p.active, p.returned)
	}
}

func TestSerialDefaultAndLegacyArtifactsRemainCompatible(t *testing.T) {
	m := DemoManifest()
	identity := m.Identity()
	manifestJSON, err := MarshalManifest(m)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(manifestJSON), "gpu_index") {
		t.Fatalf("legacy manifest gained an assignment field:\n%s", manifestJSON)
	}
	back, err := UnmarshalManifest(manifestJSON)
	if err != nil {
		t.Fatalf("legacy manifest decode: %v", err)
	}
	if back.Identity() != identity {
		t.Fatalf("omitted gpu_index moved identity: %s -> %s", identity, back.Identity())
	}

	p := &serialProbe{}
	run, err := Execute(context.Background(), back, DemoCorpus(), p, &FakeGrader{}, Options{})
	if err != nil {
		t.Fatalf("serial Execute: %v", err)
	}
	if p.peak != 1 || run.MaxParallel != 0 {
		t.Fatalf("implicit serial peak/recorded max_parallel = %d/%d, want 1/0", p.peak, run.MaxParallel)
	}
	for _, row := range run.Trials {
		if row.Launch != nil {
			t.Fatalf("implicit legacy row %s gained launch receipt %+v", row.Key(), row.Launch)
		}
		for _, field := range []string{`"wave"`, `"gpu_index"`, `"cuda_visible_devices"`} {
			if strings.Contains(row.Response.RawRequest, field) {
				t.Fatalf("implicit legacy raw request gained %s: %s", field, row.Response.RawRequest)
			}
		}
	}
	rep0, err := Summarize(run)
	if err != nil {
		t.Fatalf("implicit legacy report: %v", err)
	}
	if rep0.MaxParallel != 0 || strings.Contains(Human(rep0), "max_parallel") {
		t.Fatalf("implicit legacy report changed scheduling line:\n%s", Human(rep0))
	}

	// Remove only the additive wave fields to model a pre-#10042 run/1 ledger.
	blob, err := MarshalRun(run)
	if err != nil {
		t.Fatal(err)
	}
	var legacy map[string]any
	if err := json.Unmarshal(blob, &legacy); err != nil {
		t.Fatal(err)
	}
	delete(legacy, "max_parallel")
	for _, raw := range legacy["trials"].([]any) {
		delete(raw.(map[string]any), "launch")
	}
	legacyBlob, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	legacyRun, err := UnmarshalRun(legacyBlob)
	if err != nil {
		t.Fatalf("legacy run decode: %v", err)
	}
	rep, err := Summarize(legacyRun)
	if err != nil {
		t.Fatalf("legacy run report: %v", err)
	}
	if rep.MaxParallel != 0 || rep.TotalTrials != len(run.Trials) {
		t.Fatalf("legacy report max_parallel/rows = %d/%d", rep.MaxParallel, rep.TotalTrials)
	}
}

func TestImplicitOptionsPreserveCommittedSelfcheckWitness(t *testing.T) {
	res, err := Selfcheck()
	if err != nil {
		t.Fatal(err)
	}
	got, err := MarshalSelfcheck(res)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join("..", "..", "docs", "_witnesses", "armbench-selfcheck-2026-08-13.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatal("implicit Options{} changed the committed armbench selfcheck witness")
	}
}

func TestExplicitSerialAndAssignedImplicitModesEmitReceipts(t *testing.T) {
	t.Run("explicit_serial", func(t *testing.T) {
		run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{MaxParallel: 1})
		if err != nil {
			t.Fatal(err)
		}
		if run.MaxParallel != 1 || run.Trials[0].Launch == nil {
			t.Fatalf("explicit serial run omitted new scheduling receipt: max=%d launch=%+v", run.MaxParallel, run.Trials[0].Launch)
		}
		if !strings.Contains(run.Trials[0].Response.RawRequest, `"wave":`) {
			t.Fatalf("explicit raw request omitted wave: %s", run.Trials[0].Response.RawRequest)
		}
	})

	t.Run("assigned_implicit", func(t *testing.T) {
		m, tasks := eightGPUManifest()
		run, err := Execute(context.Background(), m, tasks, &FakeProvider{}, &FakeGrader{}, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if run.MaxParallel != 1 {
			t.Fatalf("assigned implicit run max_parallel=%d, want recorded serial 1", run.MaxParallel)
		}
		for _, row := range run.Trials {
			if row.Launch == nil || row.Launch.GPUIndex == nil || row.Launch.CUDAVisibleDevices == "" {
				t.Fatalf("assigned implicit row omitted receipt/device: %+v", row)
			}
		}
	})
}

func TestCheckRunsComparableRefusesBoundDrift(t *testing.T) {
	a := &Run{Manifest: DemoManifest(), MaxParallel: 0}
	b := &Run{Manifest: DemoManifest(), MaxParallel: 1}
	if fields, err := CheckRunsComparable(a, b); err != nil || len(fields) != 0 {
		t.Fatalf("legacy sentinel and explicit serial should compare: fields=%v err=%v", fields, err)
	}
	b.MaxParallel = 2
	fields, err := CheckRunsComparable(a, b)
	requireReason(t, err, ReasonIncomparableManifest)
	if len(fields) != 1 || fields[0].Field != "run.max_parallel" || fields[0].A != "1" || fields[0].B != "2" {
		t.Fatalf("bound drift = %+v", fields)
	}
}

func TestSummarizeReadmitsConcurrentAssignments(t *testing.T) {
	run, err := Execute(context.Background(), DemoManifest(), DemoCorpus(), &FakeProvider{}, &FakeGrader{}, Options{MaxParallel: 1})
	if err != nil {
		t.Fatal(err)
	}
	run.MaxParallel = 2
	_, err = Summarize(run)
	requireReason(t, err, ReasonGPUAssignmentUnknown)
}

func TestPathologicalMaxParallelCapsToUsefulArms(t *testing.T) {
	m, tasks := eightGPUManifest()
	requested := int(^uint(0) >> 1)
	run, err := Execute(context.Background(), m, tasks, &FakeProvider{}, &FakeGrader{}, Options{MaxParallel: requested})
	if err != nil {
		t.Fatal(err)
	}
	if run.MaxParallel != requested {
		t.Fatalf("recorded max_parallel=%d, want requested %d", run.MaxParallel, requested)
	}
	for _, row := range run.Trials {
		if row.Launch == nil || row.Launch.Wave != 1 {
			t.Fatalf("pathological bound produced wrong wave: %+v", row.Launch)
		}
	}
}

type gateProvider struct {
	mu      sync.Mutex
	active  int
	peak    int
	started chan Request
	release chan struct{}
}

func newGateProvider(capacity int) *gateProvider {
	return &gateProvider{started: make(chan Request, capacity), release: make(chan struct{}, capacity)}
}

func (p *gateProvider) Complete(ctx context.Context, req Request) (Response, error) {
	p.mu.Lock()
	p.active++
	if p.active > p.peak {
		p.peak = p.active
	}
	p.mu.Unlock()
	p.started <- req
	select {
	case <-p.release:
	case <-ctx.Done():
		p.finish()
		return Response{}, ctx.Err()
	}
	p.finish()
	return (&FakeProvider{}).Complete(ctx, req)
}

func (p *gateProvider) finish() {
	p.mu.Lock()
	p.active--
	p.mu.Unlock()
}

func (p *gateProvider) maxActive() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.peak
}

type admissionProbe struct {
	mu    sync.Mutex
	count int
}

func (p *admissionProbe) SetupArm(context.Context, Arm) (SetupCost, error) {
	p.mu.Lock()
	p.count++
	p.mu.Unlock()
	return SetupCost{}, nil
}

func (p *admissionProbe) Complete(ctx context.Context, req Request) (Response, error) {
	p.mu.Lock()
	p.count++
	p.mu.Unlock()
	return (&FakeProvider{}).Complete(ctx, req)
}

func (p *admissionProbe) calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.count
}

type cancelProvider struct {
	mu         sync.Mutex
	active     int
	started    int
	returned   int
	allStarted chan struct{}
	once       sync.Once
	want       int
}

func newCancelProvider(want int) *cancelProvider {
	return &cancelProvider{allStarted: make(chan struct{}), want: want}
}

func (p *cancelProvider) Complete(ctx context.Context, req Request) (Response, error) {
	p.mu.Lock()
	p.active++
	p.started++
	if p.started == p.want {
		p.once.Do(func() { close(p.allStarted) })
	}
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.active--
		p.returned++
		p.mu.Unlock()
	}()
	<-p.allStarted
	if req.ArmID == "arm-0" {
		return Response{}, errors.New("synthetic launch failure")
	}
	<-ctx.Done()
	return Response{}, ctx.Err()
}

type serialProbe struct {
	mu     sync.Mutex
	active int
	peak   int
}

func (p *serialProbe) Complete(ctx context.Context, req Request) (Response, error) {
	p.mu.Lock()
	p.active++
	if p.active > p.peak {
		p.peak = p.active
	}
	p.mu.Unlock()
	time.Sleep(time.Millisecond)
	resp, err := (&FakeProvider{}).Complete(ctx, req)
	p.mu.Lock()
	p.active--
	p.mu.Unlock()
	return resp, err
}

func eightGPUManifest() (*Manifest, []Task) {
	m := DemoManifest()
	tasks := []Task{{ID: "task-only", Input: "exercise all devices", Expect: "task-only"}}
	m.Corpus = Corpus{ID: "eight-gpu", Hash: HashTasks(tasks), TaskCount: len(tasks)}
	m.Trials = Trials{Count: 1, Seed: 11, Order: OrderCounterbalanced, Concurrency: 4}
	m.Arms = make([]Arm, 8)
	for i := range m.Arms {
		kind := ArmFakPassthrough
		if i == 0 {
			kind = ArmBaseline
		}
		m.Arms[i] = Arm{
			ID: fmt.Sprintf("arm-%d", i), Kind: kind,
			PromptHash: fixtureDigest("prompt/eight-gpu"), GPUIndex: intPtr(i),
		}
	}
	return m, tasks
}

func intPtr(v int) *int { return &v }

func waitRequest(t *testing.T, ch <-chan Request) Request {
	t.Helper()
	select {
	case req := <-ch:
		return req
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for provider launch")
		return Request{}
	}
}

func waitRun[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case got := <-ch:
		return got
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for run cleanup")
		var zero T
		return zero
	}
}
