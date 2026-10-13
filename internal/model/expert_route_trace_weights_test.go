package model

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestExpertRouteTraceGateWeights(t *testing.T) {
	for _, tc := range []struct {
		name    string
		weights []float32
		want    []float32
	}{
		{"signed", []float32{0, -0.25, 1.5}, []float32{0, -0.25, 1.5}},
		{"zeros", []float32{0, 0, 0}, []float32{0, 0, 0}},
		{"nan", []float32{0.5, float32(math.NaN()), 0.25}, nil},
		{"positive infinity", []float32{0.5, float32(math.Inf(1)), 0.25}, nil},
		{"negative infinity", []float32{0.5, float32(math.Inf(-1)), 0.25}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{}
			m.v41SetExpertFaultPhase(V41PhaseDecode)
			var ev ExpertRouteEvent
			m.SetExpertRouteEventObserver(func(got ExpertRouteEvent) { ev = got })
			picks := []routePick{{expert: 7, weight: tc.weights[0]}, {expert: 2, weight: tc.weights[1]}, {expert: 9, weight: tc.weights[2]}}
			m.emitExpertRouteAt(3, 11, picks)
			// The event must own its arrays, independent of later router reuse.
			picks[0] = routePick{expert: 99, weight: 99}
			if ev.Layer != 3 || ev.Token != 11 || ev.Phase != "decode" || !reflect.DeepEqual(ev.Experts, []int{7, 2, 9}) {
				t.Fatalf("event = %+v, want layer 3 token 11 decode IDs [7 2 9]", ev)
			}
			path := filepath.Join(t.TempDir(), "trace.jsonl")
			f, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			sink := &expertRouteTraceSink{f: f}
			sink.write(m, ev)
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			if sink.err != nil {
				t.Fatal(sink.err)
			}
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
			if len(lines) != 2 {
				t.Fatalf("trace has %d lines, want header and event: %s", len(lines), raw)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal([]byte(lines[1]), &fields); err != nil {
				t.Fatal(err)
			}
			if _, present := fields["weights"]; present != (tc.want != nil) {
				t.Fatalf("weights presence = %v, want %v: %s", present, tc.want != nil, lines[1])
			}
			if tc.want != nil {
				var weights []float32
				if err := json.Unmarshal(fields["weights"], &weights); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(weights, tc.want) {
					t.Fatalf("weights = %v, want %v", weights, tc.want)
				}
			}
			trace, err := ReadExpertRouteTrace(strings.NewReader(string(raw)))
			if err != nil {
				t.Fatal(err)
			}
			if len(trace.Events) != 1 || !reflect.DeepEqual(trace.Events[0], ev) {
				t.Fatalf("round trip = %+v, want %+v", trace.Events, ev)
			}
		})
	}
	for _, tc := range []struct {
		name  string
		field string
		want  []float32
		bad   bool
	}{
		{"legacy absent", "", nil, false},
		{"legacy null", `,"weights":null`, nil, false},
		{"explicit zero", `,"weights":[0]`, []float32{0}, false},
		{"empty", `,"weights":[]`, nil, true},
		{"misaligned", `,"weights":[1,2]`, nil, true},
		{"null element", `,"weights":[null]`, nil, true},
		{"positive float32 overflow", `,"weights":[3.5e38]`, nil, true},
		{"negative float32 overflow", `,"weights":[-3.5e38]`, nil, true},
		{"string element", `,"weights":["0.5"]`, nil, true},
		{"boolean element", `,"weights":[true]`, nil, true},
		{"object element", `,"weights":[{}]`, nil, true},
		{"array element", `,"weights":[[]]`, nil, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := `{"layer":0,"token":0,"phase":"decode","experts":[7]` + tc.field + `}`
			trace, err := ReadExpertRouteTrace(strings.NewReader(raw))
			if tc.bad {
				if !errors.Is(err, ErrExpertRouteTrace) {
					t.Fatalf("error = %v, want ErrExpertRouteTrace", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := json.Marshal(trace.Events[0])
			if err != nil {
				t.Fatal(err)
			}
			var got struct {
				Weights []float32 `json:"weights"`
			}
			if err := json.Unmarshal(encoded, &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Weights, tc.want) {
				t.Fatalf("weights = %v, want %v", got.Weights, tc.want)
			}
		})
	}
}
