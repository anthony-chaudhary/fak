package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func adaptiveScaleOKSLA() adaptiveScaleSLA {
	return adaptiveScaleSLA{MaxP95TTFTMS: 100, MaxP95ITLMS: 40, MaxErrors: 0}
}

// TestAdaptiveScaleRampUntilFail pins the controller contract: the cap ramps
// while windows pass, the first failing window marks the boundary, the
// controller sustains the last passing level, and later windows are ignored.
func TestAdaptiveScaleRampUntilFail(t *testing.T) {
	windows := []adaptiveScaleWindow{
		{Level: 4, P95TTFT: 20, P95ITL: 5},
		{Level: 8, P95TTFT: 35, P95ITL: 9},
		{Level: 16, P95TTFT: 60, P95ITL: 15},
		{Level: 32, P95TTFT: 140, P95ITL: 15},
		{Level: 64, P95TTFT: 10, P95ITL: 1},
	}
	res, err := adaptiveScaleRamp(windows, adaptiveScaleOKSLA())
	if err != nil {
		t.Fatalf("ramp: %v", err)
	}
	want := adaptiveScaleResult{
		Schema:    adaptiveScaleSchemaName,
		Boundary:  16,
		FailedAt:  32,
		Sustained: 16,
		Events: []adaptiveScaleDecision{
			{Window: 1, Level: 4, Action: adaptiveScaleActionRamp, NextLevel: 8},
			{Window: 2, Level: 8, Action: adaptiveScaleActionRamp, NextLevel: 16},
			{Window: 3, Level: 16, Action: adaptiveScaleActionRamp, NextLevel: 32},
			{Window: 4, Level: 32, Action: adaptiveScaleActionSustain, NextLevel: 16},
		},
	}
	if !reflect.DeepEqual(res, want) {
		t.Fatalf("result = %+v, want %+v", res, want)
	}
}

// TestAdaptiveScaleSustains covers the all-pass ramp (inclusive filter) and a
// first-window failure, where there is no passing level to sustain.
func TestAdaptiveScaleSustains(t *testing.T) {
	allPass := []adaptiveScaleWindow{
		{Level: 4, P95TTFT: 20, P95ITL: 5},
		{Level: 8, P95TTFT: 35, P95ITL: 9},
		{Level: 16, P95TTFT: 100, P95ITL: 40},
	}
	res, err := adaptiveScaleRamp(allPass, adaptiveScaleOKSLA())
	if err != nil {
		t.Fatalf("ramp: %v", err)
	}
	want := adaptiveScaleResult{
		Schema:    adaptiveScaleSchemaName,
		Boundary:  16,
		FailedAt:  0,
		Sustained: 16,
		Events: []adaptiveScaleDecision{
			{Window: 1, Level: 4, Action: adaptiveScaleActionRamp, NextLevel: 8},
			{Window: 2, Level: 8, Action: adaptiveScaleActionRamp, NextLevel: 16},
			{Window: 3, Level: 16, Action: adaptiveScaleActionSustain, NextLevel: 0},
		},
	}
	if !reflect.DeepEqual(res, want) {
		t.Fatalf("all-pass result = %+v, want %+v", res, want)
	}

	res, err = adaptiveScaleRamp([]adaptiveScaleWindow{{Level: 2, P95TTFT: 101, P95ITL: 0}}, adaptiveScaleOKSLA())
	if err != nil {
		t.Fatalf("ramp: %v", err)
	}
	want = adaptiveScaleResult{
		Schema:    adaptiveScaleSchemaName,
		Boundary:  0,
		FailedAt:  2,
		Sustained: 0,
		Events: []adaptiveScaleDecision{
			{Window: 1, Level: 2, Action: adaptiveScaleActionSustain, NextLevel: 0},
		},
	}
	if !reflect.DeepEqual(res, want) {
		t.Fatalf("first-fail result = %+v, want %+v", res, want)
	}
}

// TestAdaptiveScaleRefuses pins the closed sentinel-error contract; no result
// may be trusted from a refused ramp.
func TestAdaptiveScaleRefuses(t *testing.T) {
	ok := adaptiveScaleOKSLA()
	good := adaptiveScaleWindow{Level: 4, P95TTFT: 10, P95ITL: 1}
	cases := []struct {
		name    string
		windows []adaptiveScaleWindow
		sla     adaptiveScaleSLA
		want    error
	}{
		{"no_windows", nil, ok, errAdaptiveScaleWindows},
		{"zero_level", []adaptiveScaleWindow{{Level: 0}}, ok, errAdaptiveScaleLevels},
		{"duplicate_level", []adaptiveScaleWindow{good, good}, ok, errAdaptiveScaleLevels},
		{"descending_level", []adaptiveScaleWindow{{Level: 8}, {Level: 4}}, ok, errAdaptiveScaleLevels},
		{"nan_ttft", []adaptiveScaleWindow{{Level: 4, P95TTFT: math.NaN()}}, ok, errAdaptiveScaleMeasurement},
		{"inf_itl", []adaptiveScaleWindow{{Level: 4, P95ITL: math.Inf(1)}}, ok, errAdaptiveScaleMeasurement},
		{"negative_errors", []adaptiveScaleWindow{{Level: 4, Errors: -1}}, ok, errAdaptiveScaleMeasurement},
		{"negative_ttft_sla", []adaptiveScaleWindow{good}, adaptiveScaleSLA{MaxP95TTFTMS: -1}, errAdaptiveScaleSLA},
		{"nan_itl_sla", []adaptiveScaleWindow{good}, adaptiveScaleSLA{MaxP95ITLMS: math.NaN()}, errAdaptiveScaleSLA},
		{"negative_errors_sla", []adaptiveScaleWindow{good}, adaptiveScaleSLA{MaxErrors: -1}, errAdaptiveScaleSLA},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := adaptiveScaleRamp(tc.windows, tc.sla)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want sentinel %v", err, tc.want)
			}
			if res.Boundary != 0 || res.FailedAt != 0 || res.Sustained != 0 || len(res.Events) != 0 {
				t.Fatalf("refused ramp must yield an empty result, got %+v", res)
			}
		})
	}
}

// TestAdaptiveScaleJSONL pins the one-decision-per-line rendering and the
// --ramp CLI path end to end.
func TestAdaptiveScaleJSONL(t *testing.T) {
	events := []adaptiveScaleDecision{
		{Window: 1, Level: 4, Action: adaptiveScaleActionRamp, NextLevel: 8},
		{Window: 2, Level: 8, Action: adaptiveScaleActionSustain},
	}
	var buf bytes.Buffer
	if err := writeAdaptiveScaleJSONL(&buf, events); err != nil {
		t.Fatalf("write: %v", err)
	}
	lines := strings.Split(strings.TrimSuffix(buf.String(), "\n"), "\n")
	if len(lines) != len(events) {
		t.Fatalf("lines = %d, want %d:\n%s", len(lines), len(events), buf.String())
	}
	for i, line := range lines {
		var got adaptiveScaleDecision
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("line %d: %v", i+1, err)
		}
		if got != events[i] {
			t.Fatalf("line %d = %+v, want %+v", i+1, got, events[i])
		}
	}
}

func TestAdaptiveScaleCommand(t *testing.T) {
	dir := t.TempDir()
	doc := filepath.Join(dir, "ramp.json")
	raw := `{"schema":"fak.speed-ab.ramp.v1","windows":[` +
		`{"level":4,"p95_ttft_ms":20,"p95_itl_ms":5,"errors":0},` +
		`{"level":8,"p95_ttft_ms":35,"p95_itl_ms":9,"errors":0},` +
		`{"level":16,"p95_ttft_ms":60,"p95_itl_ms":15,"errors":0},` +
		`{"level":32,"p95_ttft_ms":140,"p95_itl_ms":15,"errors":0}],` +
		`"sla":{"max_p95_ttft_ms":100,"max_p95_itl_ms":40,"max_errors":0}}`
	if err := os.WriteFile(doc, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runSpeedAB(&out, &errb, []string{"--manifest", doc, "--ramp"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	lines := strings.Split(strings.TrimSuffix(out.String(), "\n"), "\n")
	var res adaptiveScaleResult
	if err := json.Unmarshal([]byte(lines[len(lines)-1]), &res); err != nil {
		t.Fatalf("result decode: %v", err)
	}
	if res.Schema != adaptiveScaleSchemaName || res.Boundary != 16 || res.FailedAt != 32 || res.Sustained != 16 || len(res.Events) != 4 {
		t.Fatalf("result = %+v", res)
	}
	if len(lines) != len(res.Events)+1 {
		t.Fatalf("stdout lines = %d, want one line per ramp event (%d) plus the result:\n%s", len(lines), len(res.Events), out.String())
	}

	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte(`{"schema":"nope"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	if code := runSpeedAB(&out, &errb, []string{"--manifest", bad, "--ramp"}); code != 2 {
		t.Fatalf("bad schema code=%d, want 2", code)
	}
	if _, err := parseAdaptiveScaleInput([]byte(`{"schema":"nope"}`)); !errors.Is(err, errAdaptiveScaleSchema) {
		t.Fatalf("parse err = %v, want errAdaptiveScaleSchema", err)
	}
}
