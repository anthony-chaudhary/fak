package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

func TestDispatchStalePythonProvenanceParity(t *testing.T) {
	t.Setenv(allowBinSkewEnv, "")
	pre := map[string]any{
		"verdict": "SPAWN_OK",
		"reason":  "capacity available",
		"fak_bin": map[string]any{
			"resolvers": map[string]any{
				"preflight_gate": map[string]any{"path": "/repo/fak", "resolved": true, "build": "111111111111", "dirty": false},
				"worker_guard":   map[string]any{"path": "/repo/tools/.bin/fak", "resolved": true, "build": "111111111111", "dirty": false},
			},
			"expected_head":       "222222222222",
			"observed_build":      "111111111111",
			"repository_relation": "BEHIND",
			"resolved_count":      3,
		},
	}

	prov := dispatchGateProvenance(pre, func(string) string { return "" })
	if prov.RepoHead != "222222222222" || prov.RepoRelation != "BEHIND" || prov.ResolvedCount != 3 {
		t.Fatalf("Python provenance was not lifted intact: %+v", prov)
	}
	if !dispatchApplyBinSkew(pre, "SPAWN_OK") {
		t.Fatal("stale Python provenance did not change the spawn verdict")
	}
	if got := dispatchMapString(pre, "verdict"); got != selfinstall.RefuseBinStale {
		t.Fatalf("verdict = %q, want %q", got, selfinstall.RefuseBinStale)
	}
	reason := dispatchMapString(pre, "reason")
	for _, want := range []string{"111111111111", "222222222222", "fak self-update --force --root ."} {
		if !strings.Contains(reason, want) {
			t.Errorf("reason %q does not contain %q", reason, want)
		}
	}
}
