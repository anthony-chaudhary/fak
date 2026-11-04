package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
)

// runtimeInventoryFixtureDocument renders one minimal eligible RuntimeTopology
// document: a single prefill rank on one observed device, observed at a pinned
// time with TTL 60s and epoch 7.
func runtimeInventoryFixtureDocument(t *testing.T) []byte {
	t.Helper()
	top := inferencetopology.RuntimeTopology{
		Schema: inferencetopology.RuntimeTopologySchema,
		ID:     "fixture",
		Configured: inferencetopology.RuntimeTopologyConfigured{
			Groups: []inferencetopology.RuntimeTopologyGroup{{
				ID:       "prefill-0",
				Role:     inferencetopology.RuntimeRolePrefill,
				TP:       1,
				PP:       1,
				EP:       1,
				Layers:   4,
				KVFormat: "paged-v1",
				RankIDs:  []string{"prefill-0-r0-s0"},
			}},
			Ranks: []inferencetopology.RuntimeTopologyRank{{
				ID:                "prefill-0-r0-s0",
				Incarnation:       "rank-inc-1",
				GroupID:           "prefill-0",
				Role:              inferencetopology.RuntimeRolePrefill,
				DeviceID:          "gpu-prefill-0-r0-s0",
				DeviceIncarnation: "device-inc-1",
				PPStage:           0,
				LayerStart:        0,
				LayerEnd:          4,
				KVOwnership:       []string{"kv-prefill-0-r0-s0"},
				Commitment:        inferencetopology.Resources{ResidentWeightBytes: 40, KVResidentBytes: 5},
			}},
		},
		Observed: inferencetopology.RuntimeTopologyObserved{
			At:         "2026-10-26T10:00:00Z",
			TTLSeconds: 60,
			Epoch:      7,
			Provenance: "synthetic-test-observation",
			Visibility: []string{"gpu-prefill-0-r0-s0"},
			Devices: []inferencetopology.RuntimeObservedDevice{{
				ID:             "gpu-prefill-0-r0-s0",
				Incarnation:    "device-inc-1",
				Lifecycle:      "active",
				Health:         "healthy",
				CapacityBytes:  80,
				FreeBytes:      75,
				FreeBytesKnown: true,
			}},
		},
		Supported: []inferencetopology.RuntimeTopologySupport{{
			DeviceID:    "gpu-prefill-0-r0-s0",
			Incarnation: "device-inc-1",
			State:       "supported",
		}},
		Available: []inferencetopology.RuntimeTopologyAvailable{{
			DeviceID:    "gpu-prefill-0-r0-s0",
			Incarnation: "device-inc-1",
			State:       "available",
		}},
	}
	raw, err := json.Marshal(top)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// fak-test:runtime fast est=20ms lane=default
func TestInferenceTopologyRuntimeInventoryCommand(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "topology.json")
	if err := os.WriteFile(manifest, runtimeInventoryFixtureDocument(t), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, stderr bytes.Buffer
	code := RunInferenceTopology(&out, &stderr, []string{"runtime-inventory", "--manifest", manifest, "--at", "2026-10-26T10:00:30Z"})
	if code != 0 {
		t.Fatalf("eligible code=%d stderr=%q", code, stderr.String())
	}
	var receipt map[string]any
	if err := json.Unmarshal(out.Bytes(), &receipt); err != nil {
		t.Fatalf("stdout is not JSON: %v (%q)", err, out.String())
	}
	if receipt["schema"] != "fak-runtime-topology-eligibility/1" || receipt["topology_id"] != "fixture" || receipt["observation_epoch"] != float64(7) {
		t.Fatalf("receipt identity drifted: %q", out.String())
	}
	stdoutBytes := out.Bytes()

	outPath := filepath.Join(dir, "receipt.json")
	out.Reset()
	stderr.Reset()
	code = RunInferenceTopology(&out, &stderr, []string{"runtime-inventory", "--manifest", manifest, "--at", "2026-10-26T10:00:30Z", "--out", outPath})
	if code != 0 || out.Len() != 0 {
		t.Fatalf("--out code=%d stdout=%q stderr=%q", code, out.String(), stderr.String())
	}
	written, err := os.ReadFile(outPath)
	if err != nil || !bytes.Equal(written, stdoutBytes) {
		t.Fatalf("--out content diverged from stdout: err=%v", err)
	}

	out.Reset()
	stderr.Reset()
	code = RunInferenceTopology(&out, &stderr, []string{"runtime-inventory", "--manifest", manifest, "--at", "2026-10-26T10:02:00Z"})
	if code != 1 || out.Len() != 0 || !strings.Contains(stderr.String(), "stale") {
		t.Fatalf("stale code=%d stdout=%q stderr=%q", code, out.String(), stderr.String())
	}

	bad := filepath.Join(dir, "bad.json")
	doc := strings.TrimSuffix(string(runtimeInventoryFixtureDocument(t)), "}")
	if err := os.WriteFile(bad, []byte(doc+`,"unknown":true}`), 0o600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	stderr.Reset()
	code = RunInferenceTopology(&out, &stderr, []string{"runtime-inventory", "--manifest", bad, "--at", "2026-10-26T10:00:30Z"})
	if code != 1 || out.Len() != 0 || !strings.Contains(stderr.String(), "malformed-input") {
		t.Fatalf("unknown-field code=%d stdout=%q stderr=%q", code, out.String(), stderr.String())
	}

	for _, tc := range []struct {
		name string
		args []string
	}{
		{name: "missing_at", args: []string{"runtime-inventory", "--manifest", manifest}},
		{name: "unparseable_at", args: []string{"runtime-inventory", "--manifest", manifest, "--at", "not-a-time"}},
		{name: "missing_manifest", args: []string{"runtime-inventory", "--at", "2026-10-26T10:00:30Z"}},
		{name: "extra_positional", args: []string{"runtime-inventory", "--manifest", manifest, "--at", "2026-10-26T10:00:30Z", "extra"}},
		{name: "unknown_verb", args: []string{"bogus"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, stderr bytes.Buffer
			code := RunInferenceTopology(&out, &stderr, tc.args)
			if code != 2 || !strings.Contains(stderr.String(), "usage:") {
				t.Fatalf("code=%d stderr=%q", code, stderr.String())
			}
		})
	}
}
