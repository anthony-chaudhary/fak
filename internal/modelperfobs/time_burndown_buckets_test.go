package modelperfobs

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// bucketFixture lays out one request with a prefill kernel of k ms. Native
// wall is 72+k ms, API wall 82+k ms (10 ms outside the receipt).
func bucketFixture(t *testing.T, id string, k float64) (Observation, RequestPhaseReceipt) {
	t.Helper()
	ms := func(v float64) time.Duration { return time.Duration(v * float64(time.Millisecond)) }
	wall := 72 + k
	rec := nativeperf.NewPhaseRecorder("inkernel", "cuda", "fixture", ms(wall))
	for _, iv := range []struct {
		phase, parent nativeperf.Phase
		start, end    float64
		kind          nativeperf.WorkKind
	}{
		{nativeperf.PhaseTokenization, "", 0, 3, nativeperf.WorkActive},
		{nativeperf.PhaseModelLoad, "", 3, 13, nativeperf.WorkWait},
		{nativeperf.PhasePrefill, "", 13, 25 + k, nativeperf.WorkActive},
		{nativeperf.PhaseHostUpload, nativeperf.PhasePrefill, 13, 17, nativeperf.WorkWait},
		{nativeperf.PhaseKernel, nativeperf.PhasePrefill, 17, 17 + k, nativeperf.WorkActive},
		{nativeperf.PhaseDecode, "", 25 + k, 70 + k, nativeperf.WorkActive},
		{nativeperf.PhaseSynchronization, nativeperf.PhaseDecode, 32 + k, 42 + k, nativeperf.WorkWait},
		{nativeperf.PhaseSampling, nativeperf.PhaseDecode, 62 + k, 67 + k, nativeperf.WorkActive},
		{nativeperf.PhaseOutput, "", 70 + k, 72 + k, nativeperf.WorkWait},
	} {
		if err := rec.Add(iv.phase, iv.parent, ms(iv.start), ms(iv.end), iv.kind); err != nil {
			t.Fatal(err)
		}
	}
	receipt, err := rec.Finalize(0)
	if err != nil {
		t.Fatal(err)
	}
	obs := burndownObservation(id, "cold", 82+k, 30)
	obs.PromptTokens, obs.CompletionTokens = 10, 5
	unattributed := 4.0
	return obs, RequestPhaseReceipt{Schema: RequestPhaseSchema, RequestID: id, Model: obs.Model, Backend: obs.Backend, Receipt: receipt, UnattributedMS: &unattributed}
}

func bucketRun(t *testing.T, k float64, baseline *TimeBurndownV2) TimeBurndownV2 {
	t.Helper()
	obs, env := bucketFixture(t, "native", k)
	apiOnly := burndownObservation("api-only", "cold", 50, 20)
	apiOnly.PromptTokens, apiOnly.CompletionTokens = 100, 100
	report, err := ReduceTimeBurndownBuckets([]Observation{obs, apiOnly}, []RequestPhaseReceipt{env}, BurndownTargets{PrefillMS: DefaultPrefillTargetMS, DecodeMS: DefaultDecodeTargetMS}, baseline)
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != TimeBurndownBucketSchema || len(report.BucketGroups) != 1 {
		t.Fatalf("schema=%s groups=%d", report.Schema, len(report.BucketGroups))
	}
	return report
}

func approx(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Abs(b)) }

func bucketCell(g TimeBurndownBucketGroup, stage, bucket string) BurndownBucketRow {
	for _, row := range g.Buckets {
		if row.Stage == stage && row.Bucket == bucket {
			return row
		}
	}
	return BurndownBucketRow{}
}

func stageTotal(g TimeBurndownBucketGroup, stage string) BurndownStageTotal {
	for _, s := range g.Stages {
		if s.Stage == stage {
			return s
		}
	}
	return BurndownStageTotal{}
}

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=5ms
func TestTimeBurndownBucketsGoldenMapAndReconciliation(t *testing.T) {
	g := bucketRun(t, 28, nil).BucketGroups[0]
	want := map[[2]string]float64{
		{StageUnstaged, BucketCPUWork}:     3,  // tokenization active
		{StageUnstaged, BucketWeightLoad}:  10, // model_load wall
		{StageUnstaged, BucketUnexplained}: 62, // output wait 2 + outside native 10 + API-only 50
		{StagePrefill, BucketMemoryCopy}:   4,
		{StagePrefill, BucketGPUKernel}:    28,
		{StagePrefill, BucketCPUWork}:      8, // prefill exclusive active
		{StageDecode, BucketSync}:          10,
		{StageDecode, BucketCPUWork}:       35, // decode exclusive 30 + sampling 5
	}
	if len(g.Buckets) != len(BurndownStages())*len(BurndownBuckets()) {
		t.Fatalf("cells=%d, want every stage x bucket", len(g.Buckets))
	}
	var sum float64
	for _, row := range g.Buckets {
		sum += row.TotalMS
		if !approx(row.TotalMS, want[[2]string{row.Stage, row.Bucket}]) {
			t.Errorf("%s/%s total=%g want %g", row.Stage, row.Bucket, row.TotalMS, want[[2]string{row.Stage, row.Bucket}])
		}
	}
	if !approx(g.APIWallMS, 160) || !approx(g.BucketedMS, g.APIWallMS) || !approx(sum, g.APIWallMS) {
		t.Fatalf("reconciliation api=%g bucketed=%g sum=%g", g.APIWallMS, g.BucketedMS, sum)
	}
	if g.ProducerUnattributedMS == nil || *g.ProducerUnattributedMS != 4 || g.Baseline != "absent" {
		t.Fatalf("producer unattributed=%v baseline=%s", g.ProducerUnattributedMS, g.Baseline)
	}
	cases := []struct {
		stage            string
		units            int64
		perUnit          float64
		target, remain   *float64
		wantTargetAbsent bool
	}{
		{StagePrefill, 10, 4, ptr(4), ptr(0), false},
		{StageDecode, 5, 9, ptr(20), ptr(-11), false},
		{StageUnstaged, 2, 37.5, nil, nil, true},
	}
	for _, tc := range cases {
		s := stageTotal(g, tc.stage)
		if s.Units != tc.units || s.MSPerUnit == nil || !approx(*s.MSPerUnit, tc.perUnit) {
			t.Errorf("%s units=%d perUnit=%v", tc.stage, s.Units, s.MSPerUnit)
		}
		if tc.wantTargetAbsent != (s.TargetMSPerUnit == nil) {
			t.Errorf("%s target presence=%v", tc.stage, s.TargetMSPerUnit)
		}
		if tc.remain != nil && (s.RemainingMSPerUnit == nil || !approx(*s.RemainingMSPerUnit, *tc.remain)) {
			t.Errorf("%s remaining=%v want %g", tc.stage, s.RemainingMSPerUnit, *tc.remain)
		}
	}
	if c := bucketCell(g, StagePrefill, BucketGPUKernel); c.MSPerUnit == nil || !approx(*c.MSPerUnit, 2.8) {
		t.Errorf("prefill kernel ms/token=%v want 2.8", c.MSPerUnit)
	}
}

func ptr(v float64) *float64 { return &v }

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=5ms
func TestTimeBurndownBucketsBaselineDelta(t *testing.T) {
	before := bucketRun(t, 28, nil)
	raw, err := json.Marshal(before)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := ReadTimeBurndownBaseline(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	g := bucketRun(t, 14, baseline).BucketGroups[0]
	if g.Baseline != "matched" {
		t.Fatalf("baseline=%s", g.Baseline)
	}
	k := bucketCell(g, StagePrefill, BucketGPUKernel)
	if k.BeforeMSPerUnit == nil || !approx(*k.BeforeMSPerUnit, 2.8) || !approx(*k.MSPerUnit, 1.4) || k.DeltaPct == nil || !approx(*k.DeltaPct, -50) {
		t.Fatalf("kernel before=%v after=%v delta=%v", k.BeforeMSPerUnit, k.MSPerUnit, k.DeltaPct)
	}
	p := stageTotal(g, StagePrefill)
	if !approx(*p.BeforeMSPerUnit, 4) || !approx(*p.MSPerUnit, 2.6) || !approx(*p.DeltaPct, -35) || !approx(*p.RemainingMSPerUnit, -1.4) {
		t.Fatalf("prefill before=%v after=%v delta=%v remaining=%v", *p.BeforeMSPerUnit, *p.MSPerUnit, *p.DeltaPct, *p.RemainingMSPerUnit)
	}
	if d := stageTotal(g, StageDecode); d.DeltaPct == nil || !approx(*d.DeltaPct, 0) {
		t.Fatalf("decode delta=%v want 0", d.DeltaPct)
	}
	if z := bucketCell(g, StagePrefill, BucketSync); z.DeltaPct != nil {
		t.Fatalf("zero baseline must not mint a delta: %v", *z.DeltaPct)
	}
	if _, err := ReadTimeBurndownBaseline(strings.NewReader(`{"schema":"fak-time-burndown/1"}`)); !errors.Is(err, ErrInvalidTimeBurndown) {
		t.Fatalf("v1 baseline admitted: %v", err)
	}
}

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=5ms
func TestTimeBurndownBucketsRefusesMissingTokens(t *testing.T) {
	obs, env := bucketFixture(t, "native", 28)
	obs.PromptTokens = 0
	report, err := ReduceTimeBurndownBuckets([]Observation{obs}, []RequestPhaseReceipt{env}, BurndownTargets{PrefillMS: 4, DecodeMS: 20}, nil)
	if err != nil {
		t.Fatal(err)
	}
	g := report.BucketGroups[0]
	if s := stageTotal(g, StagePrefill); s.MSPerUnit != nil || s.UnitsUnavailable != "prompt-tokens-missing" || s.RemainingMSPerUnit != nil {
		t.Fatalf("prefill per-token minted without tokens: %+v", s)
	}
	if s := stageTotal(g, StageDecode); s.MSPerUnit == nil {
		t.Fatal("decode per-token lost")
	}
	apiOnly := burndownObservation("api", "warm", 10, 2)
	report, err = ReduceTimeBurndownBuckets([]Observation{apiOnly}, nil, BurndownTargets{PrefillMS: 4, DecodeMS: 20}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if s := stageTotal(report.BucketGroups[0], StageDecode); s.UnitsUnavailable != "no-native-receipts" {
		t.Fatalf("API-only decode=%+v", s)
	}
	if _, err := ReduceTimeBurndownBuckets(nil, nil, BurndownTargets{PrefillMS: math.NaN()}, nil); !errors.Is(err, ErrInvalidTimeBurndown) {
		t.Fatalf("NaN target admitted: %v", err)
	}
}

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=30ms
func TestTimeBurndownBucketsCommand(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, v any) string {
		path := filepath.Join(dir, name)
		var buf bytes.Buffer
		switch x := v.(type) {
		case string:
			buf.WriteString(x)
		default:
			if err := json.NewEncoder(&buf).Encode(x); err != nil {
				t.Fatal(err)
			}
		}
		if err := os.WriteFile(path, buf.Bytes(), 0600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	obs, env := bucketFixture(t, "native", 28)
	input, phases := write("obs.jsonl", obs), write("phases.jsonl", env)
	var v1 bytes.Buffer
	if err := RunTimeBurndown([]string{"--input", input, "--phases", phases, "--format", "json"}, &v1); err != nil || !strings.Contains(v1.String(), `"schema":"fak-time-burndown/1"`) {
		t.Fatalf("v1 default changed: err=%v", err)
	}
	var first bytes.Buffer
	if err := RunTimeBurndown([]string{"--input", input, "--phases", phases, "--buckets", "--format", "json"}, &first); err != nil {
		t.Fatal(err)
	}
	baseline := write("before.json", first.String())
	gcBefore := runtimeobs.Receipt{Schema: runtimeobs.Schema, GoVersion: "go-test", Epoch: "e", Seq: 1, SampledAtUnixNano: 0,
		GC: runtimeobs.GC{Cycles: runtimeobs.U64{V: 3, OK: true}}, CPU: runtimeobs.CPU{TotalSeconds: runtimeobs.F64{V: 10, OK: true}, GCTotalSeconds: runtimeobs.F64{V: 1, OK: true}}}
	gcAfter := gcBefore
	gcAfter.Seq, gcAfter.SampledAtUnixNano = 2, 2e9
	gcAfter.GC.Cycles = runtimeobs.U64{V: 5, OK: true}
	gcAfter.CPU.TotalSeconds, gcAfter.CPU.GCTotalSeconds = runtimeobs.F64{V: 20, OK: true}, runtimeobs.F64{V: 1.5, OK: true}
	rtBefore := write("vars-before.json", map[string]any{"go_runtime": gcBefore, "cmdline": []string{"fak"}})
	rtAfter := write("vars-after.json", gcAfter)
	var out bytes.Buffer
	if err := RunTimeBurndown([]string{"--input", input, "--phases", phases, "--baseline", baseline, "--go-runtime-before", rtBefore, "--go-runtime-after", rtAfter, "--format", "json"}, &out); err != nil {
		t.Fatal(err)
	}
	var report TimeBurndownV2
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != TimeBurndownBucketSchema || report.BucketGroups[0].Baseline != "matched" || report.GoRuntime == nil ||
		!report.GoRuntime.GCCycles.OK || report.GoRuntime.GCCycles.V != 2 || !approx(report.GoRuntime.GCCPUSeconds.V, .5) || !approx(report.GoRuntime.GCCPUFraction.V, .05) {
		t.Fatalf("v2 command contract: %+v", report.GoRuntime)
	}
	var md bytes.Buffer
	if err := RunTimeBurndown([]string{"--input", input, "--phases", phases, "--baseline", baseline}, &md); err != nil || !strings.Contains(md.String(), "# Campaign buckets") {
		t.Fatalf("markdown buckets: err=%v", err)
	}
	for _, args := range [][]string{
		{"--input", input, "--buckets", "--format", "prom"},
		{"--input", input, "--go-runtime-before", rtBefore},
	} {
		if err := RunTimeBurndown(args, &bytes.Buffer{}); err == nil {
			t.Errorf("args %v admitted", args)
		}
	}
}
