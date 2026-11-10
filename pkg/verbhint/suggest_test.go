package verbhint

import (
	"reflect"
	"testing"
)

// benchVerbs models a representative CLI verb surface for benchmarks; it is
// local to the test file because the lane must not import cmd/.
var benchVerbs = []string{
	"repo", "push", "preflight", "provenance", "model-quirks", "opencode",
	"agentctl", "artifact", "audit", "assurance", "backlog", "bet",
	"bottleneck", "boundary", "build-lease", "cadence", "concept-coverage",
	"coordination", "deploy", "dispatch", "distributed-land", "dos",
	"doctor", "driver", "feature", "fleet", "flow", "gate", "gitd", "goal",
	"gpu", "grafana", "handoff", "harness", "hitl", "hwbench", "hwenv",
	"hwutil", "index", "instruction", "issue", "lessons", "lock",
	"mac-identity", "manifest-gate", "maturity", "memory", "nebius",
	"node-secrets", "node-spec", "nvlease", "ops", "ops-log", "perf-claims",
	"portfolio", "port", "promptfloor", "queue", "release", "router",
	"scorecards", "self", "shadowvcs", "skill-tool-surface",
	"skills-posture", "stack", "steer", "study", "timing",
	"tool-coverage-audit", "toolperf", "trajectory-hil", "update", "up",
	"verify", "version", "vm", "watchdog", "wired", "workstation",
}

// fak-test:runtime fast est=1s
func TestSuggest(t *testing.T) {
	cases := []struct {
		name  string
		token string
		known []string
		maxD  int
		limit int
		want  []string
	}{
		{"single hit", "doctro", benchVerbs, 2, 3, []string{"doctor"}},
		{"tie-break ordering and limit trim", "ax",
			[]string{"ae", "ad", "ac", "ab", "aa"}, 2, 3, []string{"aa", "ab", "ac"}},
		{"cap filtering", "stauts",
			[]string{"state", "status", "stat", "statex", "stop"}, 2, 3,
			[]string{"stat", "state", "status"}},
		{"case-insensitive input", "AX",
			[]string{"ae", "ad", "ac", "ab", "aa"}, 2, 3, []string{"aa", "ab", "ac"}},
		{"far miss", "zzzqqqxx", benchVerbs, 2, 3, nil},
		{"empty token", "", benchVerbs, 2, 3, nil},
		{"empty known", "doctro", nil, 2, 3, nil},
		{"empty known entry skipped", "ab", []string{""}, 2, 3, nil},
		{"duplicate known entries pass through", "ax",
			[]string{"aa", "aa"}, 2, 3, []string{"aa", "aa"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Suggest(tc.token, tc.known, tc.maxD, tc.limit)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("Suggest(%q, %v, %d, %d) = %v, want %v",
					tc.token, tc.known, tc.maxD, tc.limit, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestSuggestNilGuards(t *testing.T) {
	verbs := []string{"doctor", "repo"}
	if got := Suggest("doctro", verbs, 2, 0); got != nil {
		t.Errorf("limit 0 = %v, want nil", got)
	}
	if got := Suggest("doctro", verbs, -1, 3); got != nil {
		t.Errorf("negative maxDist = %v, want nil", got)
	}
}

func BenchmarkSuggestVerbSurface(b *testing.B) {
	b.ReportAllocs()
	known := benchVerbs
	for i := 0; i < b.N; i++ {
		Suggest("doctro", known, DefaultMaxDistance, 3)
	}
}

func BenchmarkLevenshteinLimited(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Levenshtein("provenence", "provenance", DefaultMaxDistance)
	}
}
