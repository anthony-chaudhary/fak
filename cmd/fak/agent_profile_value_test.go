package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestAgentProfileValueReadoutAcrossSourcesAndOff(t *testing.T) {
	tests := []struct {
		name       string
		configBody string
		flags      []string
		want       []string
	}{
		{
			name: "shipped defaults",
			want: []string{
				"response=caveman:native:medium (source=shipped-default; value=concise response shape with correctness carve-outs)",
				"work=ponytail:native:medium (source=shipped-default; value=simplicity ladder with correctness carve-outs)",
			},
		},
		{
			name:  "explicit independent axes",
			flags: []string{"--output-style", "caveman:low", "--work-profile", "ponytail:high"},
			want: []string{
				"response=caveman:native:low (source=cli; value=light response compression)",
				"work=ponytail:native:high (source=cli; value=actively resist avoidable complexity)",
			},
		},
		{
			name:       "persisted response",
			configBody: `{"pane_defaults":{"adapt":{"output-style":"caveman:high"}}}`,
			want: []string{
				"response=caveman:native:high (source=persisted; value=strong response compression)",
				"work=ponytail:native:medium (source=shipped-default; value=simplicity ladder with correctness carve-outs)",
			},
		},
		{
			name:  "explicit off",
			flags: []string{"--output-style", "full", "--work-profile", "standard"},
			want: []string{
				"response=full (source=cli; value=no response-shape steering)",
				"work=standard (source=cli; value=no implementation-policy steering)",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			config := filepath.Join(dir, "console.json")
			if tt.configBody != "" {
				if err := os.WriteFile(config, []byte(tt.configBody), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			out := filepath.Join(dir, "agent-report.json")
			args := []string{"--offline", "--code-tools=false", "--console-config", config, "--out", out}
			args = append(args, tt.flags...)
			_, stderr := captureAgentStdio(t, func() { cmdAgent(args) })

			if got := strings.Count(stderr, "fak agent profile value:"); got != 1 {
				t.Fatalf("profile readout count=%d, want 1:\n%s", got, stderr)
			}
			for _, want := range append(tt.want,
				"off means response=full adds no response-shape steering and work=standard adds no implementation-policy steering",
				"inspect sweep: fak agent profiles --sweep",
			) {
				if !strings.Contains(stderr, want) {
					t.Errorf("profile readout missing %q:\n%s", want, stderr)
				}
			}
		})
	}
}

func TestAgentProfileSweepJSONRowsAreStableAndUnmeasured(t *testing.T) {
	var first, second bytes.Buffer
	if err := printAgentOutputProfiles(&first, []string{"--sweep", "--json"}); err != nil {
		t.Fatal(err)
	}
	if err := printAgentOutputProfiles(&second, []string{"--sweep", "--json"}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatalf("two profile sweep renders differ:\nfirst:\n%s\nsecond:\n%s", first.Bytes(), second.Bytes())
	}

	var got agentProfileSweepPlan
	if err := json.Unmarshal(first.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-agent-profile-sweep/1" || got.Axes != "independent" || !strings.Contains(got.ResultSemantics, "not a benchmark result") {
		t.Fatalf("sweep contract = %+v", got)
	}
	want := []agentProfileSweepRow{
		{Axis: "response", Rung: "off", Selection: "full", Canonical: "full", Meaning: "No response-shape steering.", Command: "fak agent --output-style full --work-profile ponytail:medium", Result: "not-measured"},
		{Axis: "response", Rung: "low", Selection: "caveman:low", Canonical: "caveman:native:low", Meaning: "Light response compression.", Command: "fak agent --output-style caveman:low --work-profile ponytail:medium", Result: "not-measured"},
		{Axis: "response", Rung: "medium", Selection: "caveman:medium", Canonical: "caveman:native:medium", Meaning: "Concise response shape with correctness carve-outs.", Command: "fak agent --output-style caveman:medium --work-profile ponytail:medium", Result: "not-measured"},
		{Axis: "response", Rung: "high", Selection: "caveman:high", Canonical: "caveman:native:high", Meaning: "Strong response compression.", Command: "fak agent --output-style caveman:high --work-profile ponytail:medium", Result: "not-measured"},
		{Axis: "work", Rung: "off", Selection: "standard", Canonical: "standard", Meaning: "No implementation-policy steering.", Command: "fak agent --work-profile standard --output-style caveman:medium", Result: "not-measured"},
		{Axis: "work", Rung: "low", Selection: "ponytail:low", Canonical: "ponytail:native:low", Meaning: "Brief simplicity check before adding machinery.", Command: "fak agent --work-profile ponytail:low --output-style caveman:medium", Result: "not-measured"},
		{Axis: "work", Rung: "medium", Selection: "ponytail:medium", Canonical: "ponytail:native:medium", Meaning: "Simplicity ladder with correctness carve-outs.", Command: "fak agent --work-profile ponytail:medium --output-style caveman:medium", Result: "not-measured"},
		{Axis: "work", Rung: "high", Selection: "ponytail:high", Canonical: "ponytail:native:high", Meaning: "Actively resist avoidable complexity.", Command: "fak agent --work-profile ponytail:high --output-style caveman:medium", Result: "not-measured"},
	}
	if len(got.Rows) != len(want) {
		t.Fatalf("sweep rows=%d, want %d: %+v", len(got.Rows), len(want), got.Rows)
	}
	for i := range want {
		if got.Rows[i] != want[i] {
			t.Errorf("sweep[%d]=%+v, want %+v", i, got.Rows[i], want[i])
		}
	}

	var human bytes.Buffer
	if err := printAgentOutputProfiles(&human, []string{"--sweep"}); err != nil {
		t.Fatal(err)
	}
	for _, text := range []string{
		"independent controls",
		"response off",
		"response high",
		"work     off",
		"work     high",
		"No benchmark results are bundled",
		"fak agent profiles --sweep --json",
	} {
		if !strings.Contains(human.String(), text) {
			t.Errorf("human sweep missing %q:\n%s", text, human.String())
		}
	}
}
