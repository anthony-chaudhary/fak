package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/webbench"
)

func TestWebbenchServingSweepCLIEmitsCapacityBoundReceipt(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	dataset := filepath.Join(t.TempDir(), "tasks.jsonl")
	if err := os.WriteFile(dataset, []byte("{\"task_id\":\"t1\",\"description\":\"return ok\"}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "sweep.json")
	cmdWebbenchServing([]string{
		"--dataset", dataset,
		"--tracks", "ours",
		"--endpoints", "ours=" + server.URL + "/v1",
		"--model", "fixture-model",
		"--agents", "4",
		"--max-output-tokens", "1",
		"--concurrencies", "1,2",
		"--batch-capacities", "ours=2",
		"--capacity-sources", "ours=fixture-declaration",
		"--engines", "ours=fak-native",
		"--engine-receipts", "ours=sha256:1111111111111111111111111111111111111111111111111111111111111111",
		"--ttft-p99-budget-ms", "10000",
		"--timeout-sec", "5",
		"--out", out,
	})

	report, err := webbench.LoadServingSweepReport(out)
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != webbench.ServingSweepSchema || report.Workload.Digest == "" {
		t.Fatalf("receipt header = %#v", report)
	}
	if !reflect.DeepEqual(report.Workload.Concurrencies, []int{1, 2}) || len(report.Points) != 2 {
		t.Fatalf("receipt points = %#v", report.Workload)
	}
	if len(report.Tracks) != 1 || report.Tracks[0].Status != "measured" || report.Tracks[0].Peak == nil || report.Tracks[0].SLAKnee == nil {
		t.Fatalf("receipt summary = %#v", report.Tracks)
	}
	for _, point := range report.Points {
		if len(point.Tracks) != 1 || point.Tracks[0].Status != "valid" || point.Tracks[0].Engine != "fak-native" {
			t.Fatalf("point = %#v", point)
		}
	}
}

func TestServingSweepCLIParsersRejectInvalidCapacity(t *testing.T) {
	got, err := parseServingTrackIntMap("ours=8,vllm=16")
	if err != nil {
		t.Fatal(err)
	}
	if got[webbench.TrackOurs] != 8 || got[webbench.TrackVLLM] != 16 {
		t.Fatalf("capacity map = %#v", got)
	}
	for _, input := range []string{"ours=0", "ours=-1", "ours=bad", "unknown=8"} {
		if _, err := parseServingTrackIntMap(input); err == nil {
			t.Fatalf("parseServingTrackIntMap(%q) succeeded", input)
		}
	}
	if got, err := parsePositiveIntList("8,1,4"); err != nil || !reflect.DeepEqual(got, []int{8, 1, 4}) {
		t.Fatalf("positive list = %v, %v", got, err)
	}
}

func TestValidateWebbenchClaimConsumesServingSweepReceipt(t *testing.T) {
	throughputs := []float64{80, 120}
	ttft := []float64{10, 20}
	digest := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	report := &webbench.ServingSweepReport{
		Schema:   webbench.ServingSweepSchema,
		Workload: webbench.ServingSweepWorkload{Digest: "workload-1", Concurrencies: []int{1, 2}},
		SLA:      webbench.ServingSweepSLA{TTFTP99Millis: 50},
		Contracts: []webbench.ServingSweepTrackContract{{
			Track:               webbench.TrackOurs,
			Model:               "qwen3.8",
			Engine:              "fak-native",
			EngineReceiptDigest: digest,
			BatchCapacity:       2,
			CapacitySource:      "fixture",
		}},
	}
	for i, concurrency := range []int{1, 2} {
		report.Points = append(report.Points, webbench.ServingSweepPoint{
			Concurrency: concurrency, WorkloadDigest: report.Workload.Digest,
			Tracks: []webbench.ServingSweepTrackPoint{{
				Track: webbench.TrackOurs, Model: "qwen3.8", Engine: "fak-native",
				EngineReceiptDigest: digest, BatchCapacity: 2, CapacitySource: "fixture",
				MeasurementStatus: "measured",
				Stats: webbench.ServingStats{
					OK:                1,
					ThroughputTokensS: webbench.ScalarMetric{Status: "measured", Value: &throughputs[i]},
					TTFTMillis:        webbench.QuantileMetric{Status: "measured", P99: &ttft[i]},
				},
			}},
		})
	}
	path := filepath.Join(t.TempDir(), "sweep.json")
	if err := webbench.WriteServingSweepReport(report, path); err != nil {
		t.Fatal(err)
	}
	if err := validateWebbenchClaim("ours capacity-valid peak is 120 tok/s", path); err != nil {
		t.Fatalf("valid sweep claim rejected: %v", err)
	}
	if err := validateWebbenchClaim("ours p99 SLA knee is concurrency 2", path); err != nil {
		t.Fatalf("valid SLA-knee claim rejected: %v", err)
	}
	if err := validateWebbenchClaim("ours serving peak is 120 tok/s", ""); err == nil {
		t.Fatal("missing sweep receipt accepted")
	}
}

func TestValidateWebbenchClaimPreservesParityGate(t *testing.T) {
	report := &webbench.ServingParityReport{
		Schema: webbench.ServingParitySchema,
		Tracks: []webbench.ServingTrackResult{
			{Track: webbench.TrackVLLM, Status: "measured", Stats: webbench.ServingStats{OK: 1}},
			{Track: webbench.TrackSGLang, Status: "measured", Stats: webbench.ServingStats{OK: 1}},
			{Track: webbench.TrackFakFrontsFleet, Status: "measured", Stats: webbench.ServingStats{OK: 1}},
		},
	}
	path := filepath.Join(t.TempDir(), "parity.json")
	if err := webbench.WriteServingParityReport(report, path); err != nil {
		t.Fatal(err)
	}
	if err := validateWebbenchClaim("fak is parity or better on serving", path); err != nil {
		t.Fatalf("valid parity claim rejected: %v", err)
	}
	report.Tracks[2].Status = "not_measured"
	if err := webbench.WriteServingParityReport(report, path); err != nil {
		t.Fatal(err)
	}
	if err := validateWebbenchClaim("fak is parity or better on serving", path); err == nil {
		t.Fatal("unmeasured parity track accepted")
	}
	if err := validateWebbenchClaim("this report records a planned comparison", ""); err != nil {
		t.Fatalf("unrelated prose rejected: %v", err)
	}
}
