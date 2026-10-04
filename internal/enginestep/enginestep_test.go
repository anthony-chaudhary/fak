package enginestep

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

type fakeClock struct{ t time.Time }

func (c *fakeClock) now() time.Time          { return c.t }
func (c *fakeClock) advance(d time.Duration) { c.t = c.t.Add(d) }
func newFakeRecorder(ring int) (*Recorder, *fakeClock) {
	c := &fakeClock{t: time.Unix(1_700_000_000, 0)}
	r := New(ring)
	r.SetClock(c.now)
	return r, c
}

func render(r *Recorder) string {
	var b bytes.Buffer
	r.WritePrometheus(&b)
	return b.String()
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepIdleEmitsEveryFamily(t *testing.T) {
	r, _ := newFakeRecorder(8)
	out := render(r)
	for _, fam := range MetricFamilies {
		if !strings.Contains(out, "# TYPE "+fam+" ") {
			t.Fatalf("idle render missing family %s", fam)
		}
	}
	for _, want := range []string{
		MetricLastStepTimestamp + " 0\n",
		MetricRequestsActive + " 0\n",
		MetricCoalesceQueueDepth + " 0\n",
		MetricPrefillTokensTotal + " 0\n",
		MetricDecodeTokensTotal + `{path="serial"} 0` + "\n",
		MetricDecodeTokensTotal + `{path="batched"} 0` + "\n",
		MetricDecodeStepSeconds + `_count{path="serial"} 0` + "\n",
		MetricPhaseSeconds + `_count{phase="admission_wait"} 0` + "\n",
		MetricCohortSize + "_count 0\n",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("idle render missing %q", want)
		}
	}
	s := r.Snapshot(10, "")
	if s.Schema != SnapshotSchema || s.LastStepUnixNano != 0 || len(s.Recent) != 0 || len(s.Phases) != 0 || len(s.Decode) != 0 {
		t.Fatalf("idle snapshot = %+v", s)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepDecodeStepsFoldByPath(t *testing.T) {
	r, clk := newFakeRecorder(16)
	r.ObserveDecodeStep(PathSerial, 1, 2*time.Millisecond)
	clk.advance(time.Second)
	r.ObserveDecodeStep(PathSerial, 1, 2*time.Millisecond)
	clk.advance(time.Second)
	r.ObserveDecodeStep(PathBatched, 3, 5*time.Millisecond)

	s := r.Snapshot(10, "")
	if got := s.Decode[PathSerial]; got.Steps != 2 || got.Tokens != 2 || got.MaxLanes != 1 || got.MeanLanes != 1 {
		t.Fatalf("serial = %+v, want 2 steps / 2 tokens / 1 lane", got)
	}
	if got := s.Decode[PathBatched]; got.Steps != 1 || got.Tokens != 3 || got.MaxLanes != 3 {
		t.Fatalf("batched = %+v, want 1 step / 3 tokens / max 3 lanes", got)
	}
	if got := s.Decode[PathSerial].P50StepSeconds; got != 0.002 {
		t.Fatalf("serial p50 = %v, want 0.002 (bucket bound clamped to max)", got)
	}
	if s.LastStepUnixNano != clk.t.UnixNano() {
		t.Fatalf("last step = %d, want fake now %d", s.LastStepUnixNano, clk.t.UnixNano())
	}
	if len(s.Recent) != 3 {
		t.Fatalf("recent = %d, want 3", len(s.Recent))
	}
	for i, want := range []string{PathSerial, PathSerial, PathBatched} {
		if rec := s.Recent[i]; rec.Kind != KindDecodeStep || rec.Path != want || rec.Seq != uint64(i+1) {
			t.Fatalf("recent[%d] = %+v, want decode_step on %s seq %d", i, rec, want, i+1)
		}
	}
	if rec := s.Recent[2]; rec.Lanes != 3 || rec.Tokens != 3 || rec.DurationNS != int64(5*time.Millisecond) {
		t.Fatalf("batched record = %+v", rec)
	}

	out := render(r)
	for _, want := range []string{
		MetricDecodeTokensTotal + `{path="serial"} 2` + "\n",
		MetricDecodeTokensTotal + `{path="batched"} 3` + "\n",
		MetricDecodeStepSeconds + `_count{path="serial"} 2` + "\n",
		MetricDecodeStepSeconds + `_bucket{path="serial",le="0.001"} 0` + "\n",
		MetricDecodeStepSeconds + `_bucket{path="serial",le="0.0025"} 2` + "\n",
		MetricDecodeStepSeconds + `_bucket{path="serial",le="+Inf"} 2` + "\n",
		MetricDecodeStepLanes + `_bucket{path="batched",le="2"} 0` + "\n",
		MetricDecodeStepLanes + `_bucket{path="batched",le="3"} 1` + "\n",
		MetricDecodeStepLanes + `_sum{path="batched"} 3` + "\n",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n%s", want, out)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepClosedVocabularyIgnoresUnknown(t *testing.T) {
	r, _ := newFakeRecorder(8)
	r.ObserveDecodeStep("bogus", 4, time.Millisecond)
	r.ObservePhase(Phase("bogus"), time.Millisecond)
	r.ObserveDecodeStep(PathSerial, 0, time.Millisecond)
	r.ObserveDecodeStep(PathBatched, -2, time.Millisecond)
	r.ObservePrefillChunk(0, time.Millisecond)
	r.ObservePrefixMatched(-1)
	r.ObserveCohort(0)

	s := r.Snapshot(10, "")
	if len(s.Recent) != 0 || len(s.Decode) != 0 || len(s.Phases) != 0 || s.PrefillChunks != 0 || s.PrefixMatched != 0 || s.Cohorts.Steps != 0 || s.LastStepUnixNano != 0 {
		t.Fatalf("unknown/empty observations leaked into snapshot: %+v", s)
	}
	if out := render(r); strings.Contains(out, "bogus") {
		t.Fatalf("render carries an unknown label value:\n%s", out)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepRingBoundedKeepsNewestInOrder(t *testing.T) {
	r, _ := newFakeRecorder(4)
	for i := 1; i <= 10; i++ {
		r.ObservePrefillChunk(i, time.Duration(i)*time.Millisecond)
	}
	s := r.Snapshot(100, "")
	if len(s.Recent) != 4 {
		t.Fatalf("recent = %d, want ring size 4", len(s.Recent))
	}
	for i, rec := range s.Recent {
		if want := 7 + i; rec.Tokens != want || rec.Seq != uint64(want) || rec.Kind != KindPrefillChunk {
			t.Fatalf("recent[%d] = %+v, want tokens/seq %d", i, rec, want)
		}
	}
	if s.PrefillTokens != 55 || s.PrefillChunks != 10 {
		t.Fatalf("prefill totals = %d/%d, want 55/10 (counters survive ring eviction)", s.PrefillTokens, s.PrefillChunks)
	}
	two := r.Snapshot(2, "")
	if len(two.Recent) != 2 || two.Recent[0].Tokens != 9 || two.Recent[1].Tokens != 10 {
		t.Fatalf("recent(2) = %+v, want newest two oldest-first [9 10]", two.Recent)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestEngineStepSnapshotRecentClamps(t *testing.T) {
	r, _ := newFakeRecorder(MaxRecent + 10)
	for i := 0; i < MaxRecent+10; i++ {
		r.ObserveCohort(1)
	}
	if got := len(r.Snapshot(MaxRecent+100, "").Recent); got != MaxRecent {
		t.Fatalf("recent over cap = %d, want MaxRecent %d", got, MaxRecent)
	}
	neg := r.Snapshot(-5, "")
	if neg.Recent == nil || len(neg.Recent) != 0 {
		t.Fatalf("recent(-5) = %#v, want empty non-nil", neg.Recent)
	}
	raw, err := json.Marshal(neg)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`"recent":[]`)) {
		t.Fatalf("clamped snapshot JSON lacks an empty recent array: %s", raw)
	}
	if neg.Cohorts.Steps != uint64(MaxRecent+10) {
		t.Fatalf("summary under recent=0 = %+v, want full cohort count", neg.Cohorts)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepSnapshotKindFilter(t *testing.T) {
	r, _ := newFakeRecorder(32)
	r.ObserveCohort(2)
	r.ObserveDecodeStep(PathBatched, 2, time.Millisecond)
	r.ObservePrefillChunk(5, time.Millisecond)
	r.ObservePhase(PhasePrefill, time.Millisecond)
	r.ObserveCohort(3)
	r.ObserveDecodeStep(PathSerial, 1, time.Millisecond)

	cohorts := r.Snapshot(10, KindCohort).Recent
	if len(cohorts) != 2 || cohorts[0].Lanes != 2 || cohorts[1].Lanes != 3 {
		t.Fatalf("cohort filter = %+v, want [2 3]", cohorts)
	}
	for _, kind := range []string{KindDecodeStep, KindPrefillChunk, KindPhase, KindCohort} {
		for _, rec := range r.Snapshot(10, kind).Recent {
			if rec.Kind != kind {
				t.Fatalf("filter %s returned %+v", kind, rec)
			}
		}
	}
	if got := r.Snapshot(1, KindDecodeStep).Recent; len(got) != 1 || got[0].Path != PathSerial {
		t.Fatalf("decode filter n=1 = %+v, want newest serial step", got)
	}
	if got := r.Snapshot(10, "nope").Recent; len(got) != 0 {
		t.Fatalf("unknown kind filter = %+v, want none", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepRequestStartActiveGaugeAndIdempotentDone(t *testing.T) {
	r, clk := newFakeRecorder(8)
	done := r.RequestStart()
	other := r.RequestStart()
	if got := r.Snapshot(0, "").RequestsActive; got != 2 {
		t.Fatalf("active = %d, want 2", got)
	}
	if !strings.Contains(render(r), MetricRequestsActive+" 2\n") {
		t.Fatal("render lacks requests_active 2")
	}
	clk.advance(30 * time.Millisecond)
	done()
	done()
	s := r.Snapshot(10, "")
	if s.RequestsActive != 1 {
		t.Fatalf("active after idempotent done = %d, want 1", s.RequestsActive)
	}
	st, ok := s.Phases[string(PhaseRequest)]
	if !ok || st.Count != 1 || st.SumSeconds != 0.03 {
		t.Fatalf("request phase = %+v ok=%v, want one 30ms record", st, ok)
	}
	if len(s.Recent) != 1 || s.Recent[0].Kind != KindPhase || s.Recent[0].Phase != string(PhaseRequest) || s.Recent[0].DurationNS != int64(30*time.Millisecond) {
		t.Fatalf("recent = %+v, want one request phase record", s.Recent)
	}
	other()
	if got := r.Snapshot(0, "").RequestsActive; got != 0 {
		t.Fatalf("active = %d, want 0", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepSamplePhaseIsHistogramOnly(t *testing.T) {
	r, _ := newFakeRecorder(8)
	for i := 0; i < 3; i++ {
		r.ObservePhase(PhaseSample, time.Millisecond)
	}
	s := r.Snapshot(10, "")
	if st := s.Phases[string(PhaseSample)]; st.Count != 3 {
		t.Fatalf("sample count = %d, want 3", st.Count)
	}
	if len(s.Recent) != 0 {
		t.Fatalf("sample phases entered the ring: %+v", s.Recent)
	}
	r.ObservePhase(PhaseDecode, time.Millisecond)
	if got := r.Snapshot(10, "").Recent; len(got) != 1 || got[0].Phase != string(PhaseDecode) {
		t.Fatalf("non-sample phase ring = %+v", got)
	}
	if !strings.Contains(render(r), MetricPhaseSeconds+`_count{phase="sample"} 3`+"\n") {
		t.Fatal("render lacks sample phase count 3")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepQueueDepthClampsNegative(t *testing.T) {
	r, _ := newFakeRecorder(8)
	r.SetQueueDepth(5)
	if got := r.Snapshot(0, "").CoalesceQueueDepth; got != 5 {
		t.Fatalf("queue depth = %d, want 5", got)
	}
	r.SetQueueDepth(-3)
	if got := r.Snapshot(0, "").CoalesceQueueDepth; got != 0 {
		t.Fatalf("queue depth = %d, want 0", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepCompactIsOneBoundedLine(t *testing.T) {
	r, clk := newFakeRecorder(8)
	idle := r.Snapshot(0, "").Compact()
	if !strings.HasPrefix(idle, "ENGINE ") || !strings.Contains(idle, "last_step=never") || strings.Contains(idle, "\n") {
		t.Fatalf("idle compact = %q", idle)
	}
	for _, p := range Phases {
		r.ObservePhase(p, time.Millisecond)
	}
	for i := 0; i < 200; i++ {
		r.ObserveDecodeStep(PathSerial, 1, time.Millisecond)
		r.ObserveDecodeStep(PathBatched, 4, time.Millisecond)
		r.ObserveCohort(4)
		r.ObservePrefillChunk(64, time.Millisecond)
		r.ObservePrefixMatched(16)
	}
	r.SetQueueDepth(3)
	clk.advance(1500 * time.Millisecond)
	line := r.Snapshot(MaxRecent, "").Compact()
	if !strings.HasPrefix(line, "ENGINE active=0 queue=3") || strings.Contains(line, "\n") {
		t.Fatalf("compact = %q", line)
	}
	for _, want := range []string{"last_step=1.5s ago", "serial steps=200", "batched steps=200 tok=800", "cohorts=200", "prefill tok=12800 chunks=200 prefix_hit_tok=3200", "request="} {
		if !strings.Contains(line, want) {
			t.Fatalf("compact %q missing %q", line, want)
		}
	}
	if len(line) > 1024 {
		t.Fatalf("compact line is %d bytes, want bounded (<=1024)", len(line))
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepNilRecorderIsSafe(t *testing.T) {
	var r *Recorder
	r.SetClock(time.Now)
	r.ObservePhase(PhaseDecode, time.Millisecond)
	r.ObserveDecodeStep(PathSerial, 1, time.Millisecond)
	r.ObservePrefillChunk(3, time.Millisecond)
	r.ObservePrefixMatched(3)
	r.ObserveCohort(2)
	r.SetQueueDepth(2)
	r.RequestStart()()
	s := r.Snapshot(10, "")
	if s.Schema != SnapshotSchema || s.Recent == nil || s.Phases == nil || s.Decode == nil {
		t.Fatalf("nil snapshot = %+v", s)
	}
	if out := render(r); out != "" {
		t.Fatalf("nil render = %q, want empty", out)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepNewClampsRingSize(t *testing.T) {
	r, _ := newFakeRecorder(0)
	r.ObserveCohort(1)
	r.ObserveCohort(2)
	if got := r.Snapshot(10, "").Recent; len(got) != 1 || got[0].Lanes != 2 {
		t.Fatalf("ring(0) recent = %+v, want one newest record", got)
	}
}
