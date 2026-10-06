package enginestep

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepParityIdleZeroSeries(t *testing.T) {
	r, _ := newFakeRecorder(8)
	out := render(r)
	for _, want := range []string{
		MetricPrefixQueriedTotal + " 0\n",
		MetricPreemptionsTotal + `{reason="swap"} 0` + "\n",
		MetricPreemptionsTotal + `{reason="recompute"} 0` + "\n",
		MetricPreemptionsTotal + `{reason="gpudirect_swap"} 0` + "\n",
		MetricIterationTokens + "_count 0\n",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("idle render missing %q", want)
		}
	}
	s := r.Snapshot(0, "")
	if s.PrefixQueried != 0 || s.PrefixHitRate != 0 || s.IterationTokens.Count != 0 || len(s.Preemptions) != len(PreemptionReasons) {
		t.Fatalf("idle parity snapshot = %+v", s)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepPrefixHitRateMatchedOverQueried(t *testing.T) {
	r, _ := newFakeRecorder(8)
	r.ObservePrefixQueried(100)
	r.ObservePrefixMatched(60)
	r.ObservePrefixQueried(100) // a full miss still widens the denominator
	r.ObservePrefixMatched(0)
	r.ObservePrefixQueried(-5) // ignored
	out := render(r)
	if !strings.Contains(out, MetricPrefixQueriedTotal+" 200\n") || !strings.Contains(out, MetricPrefixMatchedTotal+" 60\n") {
		t.Fatalf("prefix counters wrong:\n%s", out)
	}
	s := r.Snapshot(0, "")
	if s.PrefixQueried != 200 || s.PrefixMatched != 60 || s.PrefixHitRate != 0.3 {
		t.Fatalf("snapshot prefix = queried %d matched %d rate %v, want 200/60/0.3", s.PrefixQueried, s.PrefixMatched, s.PrefixHitRate)
	}
	raw, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{`"prefix_queried_tokens":200`, `"prefix_hit_rate":0.3`, `"preemptions":{`, `"iteration_tokens":{`} {
		if !strings.Contains(string(raw), k) {
			t.Fatalf("snapshot JSON missing %s: %s", k, raw)
		}
	}
	if c := s.Compact(); !strings.Contains(c, "prefix_hit_tok=60/200 (30%)") {
		t.Fatalf("compact missing hit rate: %s", c)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepPreemptionsByReasonClosedVocabulary(t *testing.T) {
	r, _ := newFakeRecorder(8)
	r.ObservePreemption(PreemptSwap)
	r.ObservePreemption(PreemptSwap)
	r.ObservePreemption(PreemptRecompute)
	r.ObservePreemption("bogus")
	var nilRec *Recorder
	nilRec.ObservePreemption(PreemptSwap)
	out := render(r)
	for _, want := range []string{
		MetricPreemptionsTotal + `{reason="swap"} 2` + "\n",
		MetricPreemptionsTotal + `{reason="recompute"} 1` + "\n",
		MetricPreemptionsTotal + `{reason="gpudirect_swap"} 0` + "\n",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n%s", want, out)
		}
	}
	if strings.Contains(out, "bogus") {
		t.Fatal("unknown reason leaked into series set")
	}
	s := r.Snapshot(0, "")
	if s.Preemptions[PreemptSwap] != 2 || s.Preemptions[PreemptRecompute] != 1 {
		t.Fatalf("snapshot preemptions = %v", s.Preemptions)
	}
	if c := s.Compact(); !strings.Contains(c, "preempt=3 (swap=2 recompute=1 gpudirect=0)") {
		t.Fatalf("compact missing preempt segment: %s", c)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestEngineStepIterationTokensPerStep(t *testing.T) {
	r, _ := newFakeRecorder(8)
	r.ObserveDecodeStep(PathBatched, 4, time.Millisecond) // 4 lanes -> 4 tokens
	r.ObserveDecodeStep(PathSerial, 1, time.Millisecond)  // 1 token
	r.ObservePrefillChunk(512, time.Millisecond)          // 512-token chunk
	r.ObserveSpeculativeRound(3, 2, 3, time.Millisecond)  // verify 3 drafts + bonus = 4
	out := render(r)
	for _, want := range []string{
		MetricIterationTokens + "_count 4\n",
		MetricIterationTokens + "_sum 521\n",
		MetricIterationTokens + `_bucket{le="1"} 1` + "\n",
		MetricIterationTokens + `_bucket{le="4"} 3` + "\n",
		MetricIterationTokens + `_bucket{le="512"} 4` + "\n",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q\n%s", want, out)
		}
	}
	it := r.Snapshot(0, "").IterationTokens
	if it.Count != 4 || it.Max != 512 || it.P50 != 4 || it.P99 != 512 {
		t.Fatalf("iteration snapshot = %+v", it)
	}
}
