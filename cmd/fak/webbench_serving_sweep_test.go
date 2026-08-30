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
