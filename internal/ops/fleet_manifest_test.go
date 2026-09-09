package ops

import (
	"context"
	"strings"
	"testing"
	"time"
)

// mockNodeProber simulates remote node probing for capability discovery.
type mockNodeProber struct {
	results map[string]*NodeProbeResult
}

func (m *mockNodeProber) ProbeNode(_ context.Context, host string) (*NodeProbeResult, error) {
	if res, ok := m.results[host]; ok {
		return res, nil
	}
	return &NodeProbeResult{Reachable: false}, nil
}

func TestFleetManifestValidation(t *testing.T) {
	t.Run("ValidManifest", func(t *testing.T) {
		manifest := &FleetOpsManifest{
			Schema:    FleetOpsSchemaV1,
			ClusterID: "lan-cluster-alpha",
			Version:   "1.0.0",
			Nodes: []FleetNodeSpec{
				{
					ID:           "node-strix-1",
					Host:         "192.168.1.101:8080",
					Role:         "worker",
					OS:           "linux",
					Capabilities: []string{"vulkan", "avx512", "uma_128gb"},
				},
				{
					ID:           "node-win-1",
					Host:         "192.168.1.102:8080",
					Role:         "coordinator",
					OS:           "windows",
					Capabilities: []string{"cpu", "scheduler"},
				},
			},
			Tasks: []FleetScheduleTaskSpec{
				{
					ID:        "task-compute-sync",
					NodeID:    "node-strix-1",
					Command:   "fak compute bench",
					Interval:  Duration(15 * time.Minute),
					Timeout:   Duration(2 * time.Minute),
					LaneScope: []string{"compute"},
					Active:    true,
				},
				{
					ID:        "task-gateway-poll",
					NodeID:    "node-win-1",
					Command:   "fak gateway check",
					Interval:  Duration(5 * time.Minute),
					Timeout:   Duration(30 * time.Second),
					LaneScope: []string{"gateway"},
					Active:    true,
				},
			},
		}

		if err := manifest.Validate(); err != nil {
			t.Fatalf("expected valid manifest, got error: %v", err)
		}
	})

	t.Run("SchemaMismatch", func(t *testing.T) {
		manifest := &FleetOpsManifest{
			Schema: "invalid.schema/v0",
		}
		err := manifest.Validate()
		if err == nil || !strings.Contains(err.Error(), "invalid or missing schema") {
			t.Fatalf("expected schema mismatch error, got: %v", err)
		}
	})

	t.Run("EmptyNodeID", func(t *testing.T) {
		manifest := &FleetOpsManifest{
			Schema: FleetOpsSchemaV1,
			Nodes: []FleetNodeSpec{
				{ID: "", Host: "192.168.1.100", OS: "linux"},
			},
		}
		err := manifest.Validate()
		if err == nil || !strings.Contains(err.Error(), "NODE_ID_EMPTY") {
			t.Fatalf("expected empty node id error, got: %v", err)
		}
	})

	t.Run("DuplicateNodeRejection", func(t *testing.T) {
		manifest := &FleetOpsManifest{
			Schema: FleetOpsSchemaV1,
			Nodes: []FleetNodeSpec{
				{ID: "node-duplicate", Host: "192.168.1.101", OS: "linux"},
				{ID: "node-duplicate", Host: "192.168.1.102", OS: "linux"},
			},
		}
		err := manifest.Validate()
		if err == nil || !strings.Contains(err.Error(), "NODE_ALREADY_EXISTS") {
			t.Fatalf("expected NODE_ALREADY_EXISTS error, got: %v", err)
		}

		// Also verify AddNode rejects duplicate node ID
		baseManifest := NewFleetOpsManifest("test-cluster")
		err = baseManifest.AddNode(FleetNodeSpec{ID: "node-unique", Host: "192.168.1.101", OS: "linux"})
		if err != nil {
			t.Fatalf("expected initial node add to succeed, got: %v", err)
		}
		err = baseManifest.AddNode(FleetNodeSpec{ID: "node-unique", Host: "192.168.1.102", OS: "linux"})
		if err == nil || !strings.Contains(err.Error(), "NODE_ALREADY_EXISTS") {
			t.Fatalf("expected AddNode duplicate rejection (NODE_ALREADY_EXISTS), got: %v", err)
		}
	})

	t.Run("InvalidOSTarget", func(t *testing.T) {
		manifest := &FleetOpsManifest{
			Schema: FleetOpsSchemaV1,
			Nodes: []FleetNodeSpec{
				{ID: "node-solaris", Host: "192.168.1.100", OS: "solaris"},
			},
		}
		err := manifest.Validate()
		if err == nil || !strings.Contains(err.Error(), "INVALID_OS_TARGET") {
			t.Fatalf("expected INVALID_OS_TARGET error, got: %v", err)
		}
	})

	t.Run("UnknownNodeReference", func(t *testing.T) {
		manifest := &FleetOpsManifest{
			Schema: FleetOpsSchemaV1,
			Nodes: []FleetNodeSpec{
				{ID: "node-strix-1", Host: "192.168.1.101", OS: "linux"},
			},
			Tasks: []FleetScheduleTaskSpec{
				{
					ID:       "task-1",
					NodeID:   "non-existent-node",
					Interval: Duration(10 * time.Minute),
				},
			},
		}
		err := manifest.Validate()
		if err == nil || !strings.Contains(err.Error(), "UNKNOWN_NODE") {
			t.Fatalf("expected UNKNOWN_NODE error, got: %v", err)
		}
	})

	t.Run("InvalidInterval", func(t *testing.T) {
		manifest := &FleetOpsManifest{
			Schema: FleetOpsSchemaV1,
			Nodes: []FleetNodeSpec{
				{ID: "node-1", Host: "192.168.1.101", OS: "linux"},
			},
			Tasks: []FleetScheduleTaskSpec{
				{
					ID:       "task-1",
					NodeID:   "node-1",
					Interval: Duration(0),
				},
			},
		}
		err := manifest.Validate()
		if err == nil || !strings.Contains(err.Error(), "INVALID_INTERVAL") {
			t.Fatalf("expected INVALID_INTERVAL error, got: %v", err)
		}
	})

	t.Run("LaneCollisionDetection", func(t *testing.T) {
		// Two tasks with overlapping run windows and conflicting lane scopes (gateway vs gateway/server)
		manifest := &FleetOpsManifest{
			Schema: FleetOpsSchemaV1,
			Nodes: []FleetNodeSpec{
				{ID: "node-1", Host: "192.168.1.101", OS: "linux"},
				{ID: "node-2", Host: "192.168.1.102", OS: "windows"},
			},
			Tasks: []FleetScheduleTaskSpec{
				{
					ID:        "task-gateway-broad",
					NodeID:    "node-1",
					Interval:  Duration(10 * time.Minute),
					LaneScope: []string{"gateway"},
					RunWindow: &RunWindowSpec{Start: "08:00", End: "12:00"},
				},
				{
					ID:        "task-gateway-sublane",
					NodeID:    "node-2",
					Interval:  Duration(10 * time.Minute),
					LaneScope: []string{"gateway/server"},
					RunWindow: &RunWindowSpec{Start: "10:00", End: "14:00"}, // overlaps with 08:00-12:00
				},
			},
		}

		err := manifest.Validate()
		if err == nil || !strings.Contains(err.Error(), "FLEET_LANE_COLLISION") {
			t.Fatalf("expected FLEET_LANE_COLLISION error, got: %v", err)
		}

		// Now change run windows so they are completely non-overlapping
		manifest.Tasks[1].RunWindow = &RunWindowSpec{Start: "13:00", End: "17:00"}
		if err := manifest.Validate(); err != nil {
			t.Fatalf("expected disjoint run windows to pass validation without collision, got: %v", err)
		}
	})

	t.Run("RemoteHostReachabilityAndCapabilityIngestion", func(t *testing.T) {
		prober := &mockNodeProber{
			results: map[string]*NodeProbeResult{
				"192.168.1.150": {
					Reachable:    true,
					OS:           "linux",
					Silicon:      "strix-halo-ryzen-ai-max-395",
					UmaGB:        128,
					Engine:       "fak-native",
					Capabilities: []string{"vulkan", "avx512", "wmma", "wave32"},
				},
			},
		}

		ctx := context.Background()
		res, err := prober.ProbeNode(ctx, "192.168.1.150")
		if err != nil {
			t.Fatalf("probe failed: %v", err)
		}
		if !res.Reachable {
			t.Fatalf("expected node to be reachable")
		}

		manifest := NewFleetOpsManifest("lan-cluster")
		node := FleetNodeSpec{
			ID:   "strix-halo-node",
			Host: "192.168.1.150",
			OS:   res.OS,
		}
		if err := manifest.AddNode(node); err != nil {
			t.Fatalf("failed to add node: %v", err)
		}

		// Ingest capabilities discovered from remote probe
		if err := manifest.IngestNodeCapabilities("strix-halo-node", res.Capabilities); err != nil {
			t.Fatalf("failed to ingest capabilities: %v", err)
		}

		if len(manifest.Nodes[0].Capabilities) != 4 {
			t.Fatalf("expected 4 capabilities ingested, got: %v", manifest.Nodes[0].Capabilities)
		}
	})

	t.Run("JSONRoundTripAndParsing", func(t *testing.T) {
		jsonData := []byte(`{
  "schema": "fak.fleet.ops/v1",
  "cluster_id": "test-cluster",
  "nodes": [
    {
      "id": "node-1",
      "host": "localhost:8080",
      "os": "linux"
    }
  ],
  "tasks": [
    {
      "id": "task-1",
      "node_id": "node-1",
      "interval": "10m",
      "timeout": "1m",
      "lane_scope": ["strix"]
    }
  ]
}`)

		manifest, err := ParseFleetOpsManifest(jsonData)
		if err != nil {
			t.Fatalf("failed to parse manifest json: %v", err)
		}
		if manifest.Tasks[0].Interval.Duration() != 10*time.Minute {
			t.Fatalf("expected interval 10m, got: %v", manifest.Tasks[0].Interval)
		}
		if manifest.Tasks[0].Timeout.Duration() != 1*time.Minute {
			t.Fatalf("expected timeout 1m, got: %v", manifest.Tasks[0].Timeout)
		}
		if err := manifest.Validate(); err != nil {
			t.Fatalf("expected valid manifest from json, got: %v", err)
		}
	})
}
