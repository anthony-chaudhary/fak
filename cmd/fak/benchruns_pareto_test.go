package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchruns"
)

func paretoRun(id string, kv ...any) benchruns.Run {
	r := benchruns.Run{"run_id": id}
	for i := 0; i+1 < len(kv); i += 2 {
		r[kv[i].(string)] = kv[i+1]
	}
	return r
}

func frontierIDs(res benchRunsParetoResult) []string {
	ids := []string{}
	for _, p := range res.Frontier {
		ids = append(ids, p.RunID)
	}
	return ids
}

func TestParetoBenchRuns(t *testing.T) {
	const tp, ttft = "peak_tok_per_sec", "ttft_p99_ms"
	cases := []struct {
		name        string
		runs        []benchruns.Run
		wantIDs     []string
		wantSkipped []string
	}{
		{
			// Upstream doctest: c30 (140, 90) is dominated by c20 (150, 80).
			name: "upstream_doctest",
			runs: []benchruns.Run{
				paretoRun("c10", tp, 100.0, ttft, 50.0),
				paretoRun("c20", tp, 150.0, ttft, 80.0),
				paretoRun("c30", tp, 140.0, ttft, 90.0),
			},
			wantIDs: []string{"c10", "c20"},
		},
		{
			name: "strict_dominance_both_axes",
			runs: []benchruns.Run{
				paretoRun("a", tp, 200.0, ttft, 50.0),
				paretoRun("b", tp, 100.0, ttft, 80.0),
			},
			wantIDs: []string{"a"},
		},
		{
			// Better throughput but worse latency: neither dominates.
			name: "trade_off_keeps_both",
			runs: []benchruns.Run{
				paretoRun("a", tp, 200.0, ttft, 80.0),
				paretoRun("b", tp, 100.0, ttft, 50.0),
			},
			wantIDs: []string{"b", "a"},
		},
		{
			// Equal on every objective: no strictly-better axis, both stay.
			name: "exact_tie_keeps_both",
			runs: []benchruns.Run{
				paretoRun("x", tp, 100.0, ttft, 50.0),
				paretoRun("y", tp, 100.0, ttft, 50.0),
			},
			wantIDs: []string{"x", "y"},
		},
		{
			// Tie on throughput, strictly better latency: dominates.
			name: "tie_one_axis_strict_other",
			runs: []benchruns.Run{
				paretoRun("x", tp, 100.0, ttft, 50.0),
				paretoRun("y", tp, 100.0, ttft, 60.0),
			},
			wantIDs: []string{"x"},
		},
		{
			name:    "single_run",
			runs:    []benchruns.Run{paretoRun("only", tp, 100.0, ttft, 50.0)},
			wantIDs: []string{"only"},
		},
		{
			name:    "empty",
			runs:    nil,
			wantIDs: []string{},
		},
		{
			name: "missing_metric_skipped",
			runs: []benchruns.Run{
				paretoRun("a", tp, 100.0, ttft, 50.0),
				paretoRun("no-ttft", tp, 999.0),
				paretoRun("no-tp", ttft, 1.0),
				paretoRun("zero-lat", tp, 999.0, ttft, 0.0),
			},
			wantIDs:     []string{"a"},
			wantSkipped: []string{"no-ttft", "no-tp", "zero-lat"},
		},
		{
			name: "json_number_and_int_values",
			runs: []benchruns.Run{
				paretoRun("n", tp, json.Number("120"), ttft, 40),
				paretoRun("m", tp, 110.0, ttft, json.Number("45")),
			},
			wantIDs: []string{"n"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := benchRunsPareto(tc.runs, tp, ttft)
			if got := frontierIDs(res); !reflect.DeepEqual(got, tc.wantIDs) {
				t.Fatalf("frontier = %v, want %v", got, tc.wantIDs)
			}
			if !reflect.DeepEqual(res.Skipped, tc.wantSkipped) {
				t.Fatalf("skipped = %v, want %v", res.Skipped, tc.wantSkipped)
			}
		})
	}
}

func TestParetoBenchRunsLatencyKey(t *testing.T) {
	cases := []struct {
		name string
		runs []benchruns.Run
		want string
	}{
		{"ttft_everywhere", []benchruns.Run{
			paretoRun("a", "ttft_p99_ms", 1.0, "request_latency_p99_ms", 2.0),
			paretoRun("b", "ttft_p99_ms", 1.0),
		}, "ttft_p99_ms"},
		// Upstream fallback: non-streaming sweeps carry only request latency.
		{"fallback_request_latency", []benchruns.Run{
			paretoRun("a", "request_latency_p99_ms", 75.0),
		}, "request_latency_p99_ms"},
		// TTFT on only some runs must not mix axes; fall back to the common key.
		{"partial_ttft_falls_back", []benchruns.Run{
			paretoRun("a", "ttft_p99_ms", 1.0, "request_latency_p99_ms", 2.0),
			paretoRun("b", "request_latency_p99_ms", 3.0),
		}, "request_latency_p99_ms"},
		{"none_common", []benchruns.Run{
			paretoRun("a", "ttft_p99_ms", 1.0),
			paretoRun("b", "request_latency_p99_ms", 3.0),
		}, ""},
		{"empty", nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveBenchRunsLatencyKey(tc.runs); got != tc.want {
				t.Fatalf("latency key = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestParetoBenchRunsCommand(t *testing.T) {
	root := t.TempDir()
	benchDir := filepath.Join(root, "experiments", "benchmark")
	if err := os.MkdirAll(benchDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeBenchRunsJSON(t, filepath.Join(benchDir, "catalog.json"), map[string]any{"runs": []map[string]any{
		{"run_id": "c10", "machine_id": "box", "model": "m", "peak_tok_per_sec": 100.0, "ttft_p99_ms": 50.0},
		{"run_id": "c20", "machine_id": "box", "model": "m", "peak_tok_per_sec": 150.0, "ttft_p99_ms": 80.0},
		{"run_id": "c30", "machine_id": "box", "model": "m", "peak_tok_per_sec": 140.0, "ttft_p99_ms": 90.0},
	}})

	var out, errb bytes.Buffer
	if code := runBenchRuns(&out, &errb, []string{"pareto", "--workspace", root, "--format", "json"}); code != 0 {
		t.Fatalf("pareto code=%d stderr=%s", code, errb.String())
	}
	var res benchRunsParetoResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("decode: %v\n%s", err, out.String())
	}
	if res.LatencyKey != "ttft_p99_ms" || res.Considered != 3 {
		t.Fatalf("result = %+v", res)
	}
	if got := frontierIDs(res); !reflect.DeepEqual(got, []string{"c10", "c20"}) {
		t.Fatalf("frontier = %v", got)
	}

	out.Reset()
	errb.Reset()
	if code := runBenchRuns(&out, &errb, []string{"pareto", "--workspace", root}); code != 0 {
		t.Fatalf("table code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "c20") || strings.Contains(out.String(), "c30") {
		t.Fatalf("table output:\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	if code := runBenchRuns(&out, &errb, []string{"pareto", "--workspace", root, "--latency", "", "--throughput", "peak_tok_per_sec", "--model", "nomatch"}); code != 1 {
		t.Fatalf("no-latency code=%d, want 1; stderr=%s", code, errb.String())
	}
	if code := runBenchRuns(&out, &errb, []string{"pareto", "--workspace", root, "--format", "xml"}); code != 2 {
		t.Fatalf("bad format code=%d, want 2", code)
	}
}
