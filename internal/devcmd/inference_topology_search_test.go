package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
)

// fak-test:runtime fast est=20ms lane=default
func TestInferenceTopologySearchCommand(t *testing.T) {
	fixture := filepath.Join("..", "inferencetopology", "testdata", "runtime_search.json")
	var out, stderr bytes.Buffer
	code := RunInferenceTopology(&out, &stderr, []string{"search", "--manifest", fixture, "--at", "2026-11-02T05:00:30Z", "--json"})
	var report inferencetopology.RuntimeSearchReport
	if code != 0 || json.Unmarshal(out.Bytes(), &report) != nil || report.Chosen == nil || report.Chosen.Config.TopologyID != "colocated" {
		t.Fatalf("command code=%d stderr=%s", code, stderr.String())
	}
	if report.Chosen.UsefulTokensPerSecond != 70 || report.Evidence != "SUPPLIED_OBSERVATIONS_NOT_HARDWARE_QUALIFICATION" {
		t.Fatal("observed result or qualification boundary lost")
	}
	out.Reset()
	stderr.Reset()
	code = RunInferenceTopology(&out, &stderr, []string{"search", "--manifest", fixture, "--at", "2026-11-02T05:02:00Z"})
	if code != 1 || json.Unmarshal(out.Bytes(), &report) != nil || report.Status != "unknown" {
		t.Fatal("stale snapshot did not produce a structured unknown refusal")
	}
	// Decode into a fresh value: omitted chosen cannot retain a previous result.
	var refused inferencetopology.RuntimeSearchReport
	if json.Unmarshal(out.Bytes(), &refused) != nil || refused.Chosen != nil {
		t.Fatal("stale command selected capacity")
	}
	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte(`{"schema":"fak-runtime-topology-search/1","unexpected":true}`), 0600); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		args []string
		code int
	}{
		{[]string{"search", "--manifest", bad, "--at", "2026-11-02T05:00:30Z"}, 1},
		{[]string{"search", "--manifest", fixture}, 2},
		{[]string{"search", "--manifest", fixture, "--at", "not-a-time"}, 2},
		{[]string{"search", "--manifest", fixture, "--at", "2026-11-02T05:00:30Z", "extra"}, 2},
	} {
		out.Reset()
		stderr.Reset()
		if got := RunInferenceTopology(&out, &stderr, tc.args); got != tc.code || out.Len() != 0 {
			t.Fatalf("invalid command code=%d want=%d", got, tc.code)
		}
	}
}
