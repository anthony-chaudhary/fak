package model

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// TestGlueCensusEnvGateOffByDefault proves the measured-mode hook is OFF when
// FAK_GLUE_CENSUS is unset: in a fresh process with the variable explicitly
// removed, counting calls must leave every total at zero. This runs the test
// binary as a SUBPROCESS so the package-level glueCensusOn init re-reads the
// environment from scratch (the parent test process may have the variable set
// by an operator and cannot prove the default itself).
func TestGlueCensusEnvGateOffByDefault(t *testing.T) {
	total, sites := runGlueCensusProbe(t, false)
	if total != 0 {
		t.Fatalf("FAK_GLUE_CENSUS unset: counted %d round-trips, want 0", total)
	}
	if len(sites) != 0 {
		t.Fatalf("FAK_GLUE_CENSUS unset: saw sites %v, want none", sites)
	}
}

// TestGlueCensusEnvGateOn proves the same probe counts when FAK_GLUE_CENSUS=1,
// aggregates per-layer names into their projection class, and renders the
// stderr flush line shape.
func TestGlueCensusEnvGateOn(t *testing.T) {
	const n = 5
	total, sites := runGlueCensusProbe(t, true)
	if total != int64(n) {
		t.Fatalf("FAK_GLUE_CENSUS=1: counted %d round-trips, want %d", total, n)
	}
	if got := sites["self_attn.q_a_proj.weight"]; got != int64(n) {
		t.Fatalf("normalized site count = %d, want %d (sites=%v)", got, n, sites)
	}
}

// runGlueCensusProbe spawns the test binary with TestGlueCensusProbe, which
// performs `probeCount` device-seam counts and prints "probe <total> <json-sites>".
func runGlueCensusProbe(t *testing.T, on bool) (int64, map[string]int64) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestGlueCensusProbe$")
	env := []string{"FAK_GLUE_CENSUS_PROBE=1"}
	if on {
		env = append(env, "FAK_GLUE_CENSUS=1")
	} else {
		// Explicitly strip any inherited FAK_GLUE_CENSUS so the probe sees the
		// default, regardless of how the outer suite was launched.
		for _, e := range os.Environ() {
			if strings.HasPrefix(e, "FAK_GLUE_CENSUS=") {
				continue
			}
			env = append(env, e)
		}
	}
	cmd.Env = env
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		t.Fatalf("probe subprocess failed: %v\nstderr:\n%s", err, errb.String())
	}
	line := strings.TrimSpace(out.String())
	if !strings.HasPrefix(line, "probe ") {
		t.Fatalf("probe output %q does not start with \"probe \"", line)
	}
	var total int64
	var raw string
	if _, err := fmt.Sscanf(line, "probe %d %s", &total, &raw); err != nil {
		t.Fatalf("parse probe output %q: %v", line, err)
	}
	sites := parseProbeSites(raw)
	return total, sites
}

// TestGlueCensusProbe is the child side of the env-gate probes; it only runs
// when invoked with FAK_GLUE_CENSUS_PROBE=1.
func TestGlueCensusProbe(t *testing.T) {
	if os.Getenv("FAK_GLUE_CENSUS_PROBE") == "" {
		t.Skip("subprocess probe target; run via TestGlueCensusEnvGate*")
	}
	for i := 0; i < 5; i++ {
		glueCensusCount("model.layers.0.self_attn.q_a_proj.weight")
	}
	total, sites := glueCensusTotals()
	fmt.Printf("probe %d %s\n", total, renderProbeSites(sites))
	t.Skip("probe reported")
}

func renderProbeSites(sites map[string]int64) string {
	if len(sites) == 0 {
		return "{}"
	}
	parts := make([]string, 0, len(sites))
	for s, n := range sites {
		parts = append(parts, fmt.Sprintf("%q:%d", s, n))
	}
	return "{" + strings.Join(parts, ",") + "}"
}

func parseProbeSites(raw string) map[string]int64 {
	out := map[string]int64{}
	raw = strings.TrimPrefix(strings.TrimSuffix(raw, "}"), "{")
	if raw == "" {
		return out
	}
	for _, part := range strings.Split(raw, ",") {
		idx := strings.LastIndex(part, ":")
		if idx < 0 {
			continue
		}
		key := strings.Trim(part[:idx], `"`)
		var n int64
		fmt.Sscanf(part[idx+1:], "%d", &n)
		out[key] = n
	}
	return out
}

// TestGlueCensusNormalizeSite pins the per-layer name aggregation rule.
func TestGlueCensusNormalizeSite(t *testing.T) {
	cases := map[string]string{
		"model.layers.0.self_attn.q_a_proj.weight": "self_attn.q_a_proj.weight",
		"model.layers.91.mlp.gate.weight":          "mlp.gate.weight",
		"dsa_sparse_attend":                        "dsa_sparse_attend",
		"short.name":                               "short.name",
	}
	for in, want := range cases {
		if got := normalizeGlueCensusSite(in); got != want {
			t.Errorf("normalizeGlueCensusSite(%q) = %q, want %q", in, got, want)
		}
	}
}
