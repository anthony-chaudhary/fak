package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestClusterInfoSubprocess(t *testing.T) {
	if format := os.Getenv("FAK_CLUSTER_INFO_TEST_CHILD"); format != "" {
		args := []string{"info"}
		if format == "json" {
			args = append(args, "--json")
		}
		cmdCluster(args)
		os.Exit(0)
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestClusterInfoSubprocess$")
	cmd.Env = append(os.Environ(), "FAK_CLUSTER_INFO_TEST_CHILD=json")
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("cluster info subprocess: %v", err)
	}
	var report clusterInfoReport
	if err := json.Unmarshal(out, &report); err != nil {
		t.Fatalf("decode cluster info output %q: %v", out, err)
	}
	if report.Schema != "fak.cluster-local-discovery/1" || report.Source != "local_system_discovery" {
		t.Fatalf("unexpected provenance: schema=%q source=%q", report.Schema, report.Source)
	}
	if report.ProofScope != "configuration_only_not_data_transfer_or_working_mesh" {
		t.Fatalf("proof_scope = %q", report.ProofScope)
	}
	if report.Discovery == nil {
		t.Fatal("cluster info omitted discovery payload")
	}

	cmd = exec.Command(os.Args[0], "-test.run=^TestClusterInfoSubprocess$")
	cmd.Env = append(os.Environ(), "FAK_CLUSTER_INFO_TEST_CHILD=text")
	out, err = cmd.Output()
	if err != nil {
		t.Fatalf("cluster info text subprocess: %v", err)
	}
	text := string(out)
	if !strings.Contains(text, "Local Discovery") || !strings.Contains(text, "does not prove data transfer or a working mesh") {
		t.Fatalf("cluster info text lacks honest proof scope: %q", text)
	}
}

// cluster_test.go — gates `fak cluster`, the runnable multi-node collective. The
// command is a thin shell over internal/model's DistComm (whose own tests prove the
// cross-process collective is byte-identical to LocalCollective); these tests pin the
// shell: the loopback selftest harness the operator runs before a two-node launch
// really asserts bit-exactness, the vec/width parsing fails closed on junk, and a
// ragged reduce is refused on every rank rather than producing a wrong answer.

// TestClusterSelftestPasses is the gate behind `fak cluster selftest`: for rank counts
// 1..4 the cross-process allreduce AND allgather over a real loopback socket must be
// bit-for-bit equal to the in-process LocalCollective reference. A regression in the
// wire codec, the rank ordering, or the orchestration breaks max|Δ|=0 here, before any
// operator launches the same path on two machines.
func TestClusterSelftestPasses(t *testing.T) {
	if err := clusterSelftest(4, 17); err != nil {
		t.Fatalf("clusterSelftest(4,17): %v", err)
	}
	// A length that is a single round word and a longer one both pass.
	if err := clusterSelftest(3, 1); err != nil {
		t.Fatalf("clusterSelftest(3,1): %v", err)
	}
}

// TestRunLoopbackGroupAllReduce checks the harness directly: three ranks, each holding
// only its own part, end with the same rank-order sum LocalCollective computes.
func TestRunLoopbackGroupAllReduce(t *testing.T) {
	parts := [][]float32{{1, 2, 3}, {10, 20, 30}, {100, 200, 300}}
	want, err := model.LocalCollective{}.AllReduceSum(parts)
	if err != nil {
		t.Fatalf("local: %v", err)
	}
	got, errs := runLoopbackGroup(3, func(g *model.DistComm) ([]float32, error) {
		return g.AllReduceSum(parts[g.Rank()])
	})
	for r := 0; r < 3; r++ {
		if errs[r] != nil {
			t.Fatalf("rank %d: %v", r, errs[r])
		}
		if err := assertBitExact(got[r], want); err != nil {
			t.Fatalf("rank %d: %v", r, err)
		}
	}
}

// TestRunLoopbackGroupRaggedFailsClosed proves a malformed collective (parts of
// mismatched length) is refused on EVERY rank without a peer deadlocking on a response
// that never comes — the fail-closed contract DistComm guarantees, exercised through the
// command's own harness.
func TestRunLoopbackGroupRaggedFailsClosed(t *testing.T) {
	parts := [][]float32{{1, 2, 3}, {1, 2}} // rank 1 is short
	_, errs := runLoopbackGroup(2, func(g *model.DistComm) ([]float32, error) {
		return g.AllReduceSum(parts[g.Rank()])
	})
	for r := 0; r < 2; r++ {
		if errs[r] == nil {
			t.Fatalf("rank %d: ragged reduce returned nil error, want fail-closed", r)
		}
	}
}

func TestParseVec(t *testing.T) {
	cases := []struct {
		in   string
		want []float32
		ok   bool
	}{
		{"1,2,3", []float32{1, 2, 3}, true},
		{" 1.5 , -2 , 0 ", []float32{1.5, -2, 0}, true},
		{"", []float32{}, true},
		{"1,x,3", nil, false},
	}
	for _, c := range cases {
		got, err := parseVec(c.in)
		if c.ok != (err == nil) {
			t.Fatalf("parseVec(%q) err = %v, ok = %v", c.in, err, c.ok)
		}
		if !c.ok {
			continue
		}
		if err := assertBitExact(got, c.want); err != nil {
			t.Fatalf("parseVec(%q): %v", c.in, err)
		}
	}
}

func TestPlanFromWidths(t *testing.T) {
	p, err := planFromWidths([]int{2, 3, 1})
	if err != nil {
		t.Fatalf("planFromWidths: %v", err)
	}
	if p.Dim != 6 {
		t.Fatalf("Dim = %d, want 6", p.Dim)
	}
	if len(p.Shards) != 3 || p.Shards[1].Lo != 2 || p.Shards[1].Hi != 5 {
		t.Fatalf("shards = %+v, want shard 1 = [2,5)", p.Shards)
	}
	// A zero width leaves a rank with no work; Validate must reject it.
	if _, err := planFromWidths([]int{0, 3}); err == nil {
		t.Fatalf("planFromWidths with a zero shard returned nil error, want fail-closed")
	}
}

func TestParseWidths(t *testing.T) {
	if w, err := parseWidths("2,3,1"); err != nil || len(w) != 3 || w[2] != 1 {
		t.Fatalf("parseWidths(2,3,1) = %v, %v", w, err)
	}
	if _, err := parseWidths(""); err == nil {
		t.Fatalf("parseWidths(empty) returned nil error")
	}
	if _, err := parseWidths("2,-1"); err == nil {
		t.Fatalf("parseWidths with a negative width returned nil error")
	}
}
