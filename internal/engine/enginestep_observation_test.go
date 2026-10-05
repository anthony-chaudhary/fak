package engine

// enginestep_observation_test.go — instrumentation witness for the wire engine.
//
// It drives a real BatchingEngine step loop (chunked prefill then decode) with an
// injected private enginestep.Recorder and asserts the PromQL-visible families
// advanced. It probes only the recorder surface, never implementation internals.

import (
	"bytes"
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

// promCount sums every sample of the histogram family `base` whose field is
// `base_count` (any label set). It fails the test if the family is absent.
func promCount(t *testing.T, body, base string) float64 {
	t.Helper()
	var total float64
	found := false
	for _, line := range strings.Split(body, "\n") {
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		field := line
		if i := strings.IndexByte(line, ' '); i >= 0 {
			field = line[:i]
		}
		name := field
		if i := strings.IndexByte(name, '{'); i >= 0 {
			name = name[:i]
		}
		if name != base+"_count" {
			continue
		}
		i := strings.LastIndexByte(line, ' ')
		if i < 0 {
			t.Fatalf("malformed metric line %q", line)
		}
		v, err := strconv.ParseFloat(line[i+1:], 64)
		if err != nil {
			t.Fatalf("parse %q value: %v", base, err)
		}
		total += v
		found = true
	}
	if !found {
		t.Fatalf("metric family %q absent from Prometheus output", base)
	}
	return total
}

// promValue returns the trailing numeric value of the first line whose field is
// exactly `name` (unlabelled). It fails the test if absent.
func promValue(t *testing.T, body, name string) float64 {
	t.Helper()
	for _, line := range strings.Split(body, "\n") {
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, name+" ") {
			i := strings.LastIndexByte(line, ' ')
			v, err := strconv.ParseFloat(line[i+1:], 64)
			if err != nil {
				t.Fatalf("parse %q value: %v", name, err)
			}
			return v
		}
	}
	t.Fatalf("metric %q absent from Prometheus output", name)
	return 0
}

// fak-test:runtime fast est=3s
func TestBatchingEngine_RecordsStepObservations(t *testing.T) {
	const target = 4
	rec := enginestep.New(64)
	cfg := DefaultBatchingEngineConfig()
	cfg.Batcher.MaxSlots = 2
	cfg.Batcher.PrefillBudget = 2 // prompt (5) > budget => real prefill arm
	cfg.StepInterval = 0
	cfg.recorder = rec // inject a private recorder, not process-wide Default

	eng, err := NewBatchingEngine(cfg)
	if err != nil {
		t.Fatalf("NewBatchingEngine failed: %v", err)
	}
	defer func() { _ = eng.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	args, err := json.Marshal(map[string]any{
		"session_id":      "observe-wire",
		"prompt_tokens":   []int{1, 2, 3, 4, 5},
		"target_tokens":   target,
		"chunked_prefill": true,
	})
	if err != nil {
		t.Fatalf("marshal args failed: %v", err)
	}
	req, err := eng.Admit(ctx, &abi.ToolCall{Tool: "t", Args: abi.Ref{Kind: abi.RefInline, Inline: args}})
	if err != nil {
		t.Fatalf("Admit failed: %v", err)
	}

	// Drive the loop to completion.
	for range req.Tokens() {
	}
	if _, err := req.Result(); err != nil {
		t.Fatalf("Result failed: %v", err)
	}

	var buf bytes.Buffer
	rec.WritePrometheus(&buf)
	body := buf.String()

	if got := promCount(t, body, enginestep.MetricCohortSize); got < 1 {
		t.Fatalf("fak_engine_cohort_size_count = %v, want >= 1", got)
	}
	if got := promCount(t, body, enginestep.MetricDecodeStepSeconds); got < 1 {
		t.Fatalf("fak_engine_decode_step_seconds_count = %v, want >= 1", got)
	}
	if got := promCount(t, body, enginestep.MetricDecodeStepLanes); got < 1 {
		t.Fatalf("fak_engine_decode_step_lanes_count = %v, want >= 1", got)
	}
	if got := promValue(t, body, enginestep.MetricPrefillChunksTotal); got < 1 {
		t.Fatalf("fak_engine_prefill_chunks_total = %v, want >= 1", got)
	}
	if got := promValue(t, body, enginestep.MetricLastStepTimestamp); got <= 0 {
		t.Fatalf("fak_engine_last_step_timestamp_seconds = %v, want > 0", got)
	}

	snap := rec.Snapshot(0, "")
	if snap.Cohorts.Steps < 1 {
		t.Fatalf("snapshot cohorts = %d, want >= 1", snap.Cohorts.Steps)
	}
	if st := snap.Decode[enginestep.PathBatched]; st.Steps < 1 {
		t.Fatalf("snapshot batched decode steps = %d, want >= 1", st.Steps)
	}
	if snap.PrefillChunks < 1 {
		t.Fatalf("snapshot prefill chunks = %d, want >= 1", snap.PrefillChunks)
	}
	if snap.LastStepUnixNano == 0 {
		t.Fatal("snapshot last step timestamp is zero")
	}
}
