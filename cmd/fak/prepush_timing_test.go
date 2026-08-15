package main

import (
	"encoding/json"
	"testing"
	"time"
)

func TestPrepushBuildReportsCriticalPathPhases(t *testing.T) {
	setupHappyPrepushSeams(t)
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 0 || !res.OK {
		t.Fatalf("code=%d result=%+v", code, res)
	}
	want := []string{"extract_tip", "list_graph", "build_selected"}
	if len(res.Phases) != len(want) {
		t.Fatalf("phases=%v, want %v", res.Phases, want)
	}
	for i, name := range want {
		if res.Phases[i].Name != name || res.Phases[i].ElapsedMS < 0 {
			t.Fatalf("phase[%d]=%+v, want name=%q and non-negative elapsed", i, res.Phases[i], name)
		}
	}
	b, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(b, &payload); err != nil {
		t.Fatal(err)
	}
	if _, ok := payload["phases"]; !ok {
		t.Fatalf("JSON report omitted additive phases: %s", b)
	}
}
