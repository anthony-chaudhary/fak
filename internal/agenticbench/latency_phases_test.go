package agenticbench

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

func TestLatencyPhaseFixtureRoundTripsThroughJSONAndMarkdown(t *testing.T) {
	root := fixtureRoot(t)
	write(t, root, "experiments/agent-live/result-opus/compare.json", `{"ok": true}`)
	fixture, err := os.ReadFile("testdata/latency_good.json")
	if err != nil {
		t.Fatal(err)
	}
	write(t, root, "experiments/agent-live/agentic-benchmark-result-packets/good.json", string(fixture))

	report, err := Build(root, time.Unix(0, 0))
	if err != nil {
		t.Fatal(err)
	}
	packet := report.ResultPackets[0]
	if packet.Gate != "PASS_RESULT" || len(packet.Latency) != 2 {
		t.Fatalf("good latency fixture did not graduate: %+v", packet)
	}
	if got := packet.Latency[0].Total.Duration; got == nil || *got != 140 {
		t.Fatalf("raw total = %v, want 140", got)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"queue_wait":{"duration":10`, `"source":"ale.queue_wait_s"`, `"additive":false`} {
		if !strings.Contains(string(encoded), want) {
			t.Fatalf("JSON rollup missing %q: %s", want, encoded)
		}
	}
	markdown := RenderMarkdown(report)
	for _, want := range []string{"Latency Phases", "10 s (source: ale.queue_wait_s)", "100 s (source: ale.execution_duration)", "30 s (source: ale.evaluation_duration)", "140 s (source: ale.lifecycle)", "nested, non-additive"} {
		if !strings.Contains(markdown, want) {
			t.Fatalf("Markdown rollup missing %q:\n%s", want, markdown)
		}
	}
}

func TestLatencyPhaseDerivedDurationsNeedNoSyntheticIntervals(t *testing.T) {
	root := fixtureRoot(t)
	write(t, root, "experiments/agent-live/result-opus/compare.json", `{"ok": true}`)
	doc := loadLatencyFixture(t, "testdata/latency_good.json")
	for _, armRaw := range anySlice(doc["arms"]) {
		latency := mapValue(mapValue(armRaw)["latency"])
		for _, name := range append([]string{"total"}, requiredLatencyPhases...) {
			measurement := mapValue(latency[name])
			delete(measurement, "start")
			delete(measurement, "end")
		}
		for _, gatewayRaw := range anySlice(latency["gateway_requests"]) {
			gateway := mapValue(gatewayRaw)
			delete(gateway, "start")
			delete(gateway, "end")
		}
	}
	packet := checkResultPacket(root, "derived-duration.json", doc)
	if packet.Gate != "PASS_RESULT" {
		t.Fatalf("source-attributed derived durations should pass without invented intervals: %+v", packet)
	}
}

func TestBuildRejectsOverlappingLatencyIntervals(t *testing.T) {
	root := fixtureRoot(t)
	write(t, root, "experiments/agent-live/result-opus/compare.json", `{"ok": true}`)
	fixture, err := os.ReadFile("testdata/latency_overlapping.json")
	if err != nil {
		t.Fatal(err)
	}
	write(t, root, "experiments/agent-live/agentic-benchmark-result-packets/overlap.json", string(fixture))

	report, err := Build(root, time.Unix(0, 0))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(report.ResultPackets); got != 1 {
		t.Fatalf("result packets = %d, want 1", got)
	}
	packet := report.ResultPackets[0]
	if packet.Gate != "FAIL" || !strings.Contains(packet.Detail, "LATENCY_INTERVAL_OVERLAP") {
		t.Fatalf("overlapping phases must be refused with LATENCY_INTERVAL_OVERLAP: %+v", packet)
	}
}

func TestLatencyPhaseRefusalVocabulary(t *testing.T) {
	root := fixtureRoot(t)
	write(t, root, "experiments/agent-live/result-opus/compare.json", `{"ok": true}`)
	tests := []struct {
		name   string
		want   string
		mutate func(map[string]any)
	}{
		{
			name: "unknown is preserved rather than coerced to zero",
			want: "LATENCY_PHASE_UNKNOWN",
			mutate: func(doc map[string]any) {
				measurement := fixtureMeasurement(doc, 0, "evaluation")
				delete(measurement, "duration")
				delete(measurement, "start")
				delete(measurement, "end")
				measurement["unknown_reason"] = "upstream harness omitted evaluation telemetry"
			},
		},
		{
			name: "negative",
			want: "LATENCY_PHASE_NEGATIVE",
			mutate: func(doc map[string]any) {
				fixtureMeasurement(doc, 0, "queue_wait")["duration"] = -1.0
			},
		},
		{
			name: "unitless",
			want: "LATENCY_PHASE_UNITLESS",
			mutate: func(doc map[string]any) {
				fixtureMeasurement(doc, 0, "evaluation")["unit"] = ""
			},
		},
		{
			name: "total inconsistent",
			want: "LATENCY_TOTAL_INCONSISTENT",
			mutate: func(doc map[string]any) {
				fixtureMeasurement(doc, 0, "total")["duration"] = 141.0
			},
		},
		{
			name: "gateway double counted",
			want: "GATEWAY_INTERVAL_ADDITIVE",
			mutate: func(doc map[string]any) {
				latency := fixtureLatency(doc, 1)
				gateway := mapValue(anySlice(latency["gateway_requests"])[0])
				gateway["additive"] = true
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			doc := loadLatencyFixture(t, "testdata/latency_good.json")
			test.mutate(doc)
			packet := checkResultPacket(root, test.name+".json", doc)
			if packet.Gate != "FAIL" || !strings.Contains(packet.Detail, test.want) {
				t.Fatalf("want stable refusal %s: %+v", test.want, packet)
			}
			if test.want == "LATENCY_PHASE_UNKNOWN" {
				unknown := packet.Latency[0].Evaluation
				if unknown.Duration != nil || unknown.UnknownReason == "" {
					t.Fatalf("unknown phase was not preserved distinctly: %+v", unknown)
				}
			}
		})
	}
}

func loadLatencyFixture(t *testing.T, path string) map[string]any {
	t.Helper()
	doc, err := readJSON(path)
	if err != nil {
		t.Fatal(err)
	}
	return doc
}

func fixtureLatency(doc map[string]any, arm int) map[string]any {
	return mapValue(mapValue(anySlice(doc["arms"])[arm])["latency"])
}

func fixtureMeasurement(doc map[string]any, arm int, name string) map[string]any {
	return mapValue(fixtureLatency(doc, arm)[name])
}
