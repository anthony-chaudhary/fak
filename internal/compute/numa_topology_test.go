package compute

import (
	"bytes"
	"os"
	"testing"
)

func TestSynthesizeNUMATopology(t *testing.T) {
	if topo := SynthesizeNUMATopology(1); topo != nil {
		t.Fatalf("SynthesizeNUMATopology(1) = %v, want nil", topo)
	}

	for _, nodes := range []int{2, 4, 8} {
		topo := SynthesizeNUMATopology(nodes)
		if len(topo) != nodes {
			t.Fatalf("nodes=%d: len(topo)=%d, want %d", nodes, len(topo), nodes)
		}
		seen := make(map[int]bool)
		for i, n := range topo {
			if n.NodeID != i {
				t.Fatalf("node index %d has NodeID %d, want %d", i, n.NodeID, i)
			}
			if len(n.CPUs) == 0 {
				t.Fatalf("node %d has empty CPUs", n.NodeID)
			}
			for _, cpu := range n.CPUs {
				if seen[cpu] {
					t.Fatalf("CPU %d duplicated across nodes", cpu)
				}
				seen[cpu] = true
			}
		}
	}
}

func TestResolveNUMAReplicaConfig(t *testing.T) {
	for _, disabled := range []string{"off", "0", "1", "false", "none", "disabled"} {
		cfg := ResolveNUMAReplicaConfig(disabled)
		if cfg.Enabled {
			t.Fatalf("ResolveNUMAReplicaConfig(%q).Enabled = true, want false", disabled)
		}
	}

	for _, forced := range []string{"all", "on", "true"} {
		cfg := ResolveNUMAReplicaConfig(forced)
		if !cfg.Enabled || cfg.Replicas < 2 || len(cfg.Topology) < 2 {
			t.Fatalf("ResolveNUMAReplicaConfig(%q) = %+v, want enabled with >= 2 replicas", forced, cfg)
		}
	}

	for _, nStr := range []string{"2", "4"} {
		cfg := ResolveNUMAReplicaConfig(nStr)
		if !cfg.Enabled {
			t.Fatalf("ResolveNUMAReplicaConfig(%q) enabled = false", nStr)
		}
		if cfg.Replicas != len(cfg.Topology) {
			t.Fatalf("ResolveNUMAReplicaConfig(%q) replicas=%d != len(topo)=%d", nStr, cfg.Replicas, len(cfg.Topology))
		}
	}

	if cfg := ResolveNUMAReplicaConfig("invalid_xyz"); cfg.Enabled {
		t.Fatalf("ResolveNUMAReplicaConfig(invalid) enabled = true")
	}

	// Test environment variable FAK_NUMA_REPLICAS fallback
	orig := os.Getenv("FAK_NUMA_REPLICAS")
	defer os.Setenv("FAK_NUMA_REPLICAS", orig)

	os.Setenv("FAK_NUMA_REPLICAS", "2")
	cfg := ResolveNUMAReplicaConfig("")
	if !cfg.Enabled || cfg.Replicas != 2 {
		t.Fatalf("env FAK_NUMA_REPLICAS=2 resolved to %+v", cfg)
	}

	os.Setenv("FAK_NUMA_REPLICAS", "off")
	cfg = ResolveNUMAReplicaConfig("")
	if cfg.Enabled {
		t.Fatalf("env FAK_NUMA_REPLICAS=off resolved to enabled")
	}
}

func TestPlanAndBuildNUMAReplicasForTopology(t *testing.T) {
	topo := SynthesizeNUMATopology(3)
	src := make([]byte, 1024)
	for i := range src {
		src[i] = byte(i*17 + 5)
	}

	set, err := BuildNUMAReplicasForTopology(src, topo)
	if err != nil {
		t.Fatalf("BuildNUMAReplicasForTopology failed: %v", err)
	}
	defer set.Free()

	if set.Len() != 3 {
		t.Fatalf("set.Len() = %d, want 3", set.Len())
	}
	if err := VerifyNUMAReplicas(src, set); err != nil {
		t.Fatalf("VerifyNUMAReplicas failed: %v", err)
	}

	for node := 0; node < 3; node++ {
		replica := set.For(node)
		if !bytes.Equal(replica, src) {
			t.Fatalf("node %d replica not byte-identical", node)
		}
	}

	// Distinct backing stores
	r0, r1 := set.For(0), set.For(1)
	r0[0] ^= 0xFF
	if r1[0] == r0[0] || src[0] == r0[0] {
		t.Fatalf("replicas share memory with each other or source")
	}
	r0[0] ^= 0xFF // restore
}
