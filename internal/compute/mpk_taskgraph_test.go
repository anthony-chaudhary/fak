package compute

import (
	"strings"
	"testing"
)

func TestMPKTaskgraph(t *testing.T) {
	t.Run("GraphSchemaShape", func(t *testing.T) {
		g, p := MPKBuildDemoGraph()
		if len(g.Nodes) != 3 {
			t.Fatalf("demo graph must have 3 nodes, got %d", len(g.Nodes))
		}
		if len(g.Edges) != 2 {
			t.Fatalf("demo graph must have 2 edges, got %d", len(g.Edges))
		}
		kindByID := map[string]MPKTaskKind{}
		for _, n := range g.Nodes {
			kindByID[n.ID] = n.Kind
			if n.SMs <= 0 {
				t.Fatalf("task %q must occupy a positive number of SMs", n.ID)
			}
		}
		if kindByID["producer"] != MPKTaskCompute || kindByID["consumer"] != MPKTaskCompute {
			t.Fatalf("producer/consumer must be compute tasks")
		}
		if kindByID["comm"] != MPKTaskComm {
			t.Fatalf("comm must be a communication task")
		}
		for _, e := range g.Edges {
			if e.Bytes <= 0 {
				t.Fatalf("demo edges carry a positive byte count, got %d", e.Bytes)
			}
		}
		if p.SMs != 2 {
			t.Fatalf("demo partition budget must be 2 SMs, got %d", p.SMs)
		}
		// The demo graph must validate cleanly.
		if v := MPKValidateGraph(g); !v.OK {
			t.Fatalf("demo graph must validate: %s", v.Error())
		}
	})

	t.Run("PartitionInvariantsHold", func(t *testing.T) {
		g, p := MPKBuildDemoGraph()
		if v := MPKValidatePartition(g, p); !v.OK {
			t.Fatalf("demo partition must satisfy the invariants: %s", v.Error())
		}
		// A partition that moves the consumer to SM 0 keeps the cross-SM
		// handoff on the comm task, so it must still hold.
		p2 := &MPKPartition{
			Assign: map[string]int{"producer": 0, "comm": 1, "consumer": 0},
			SMs:    2,
		}
		if v := MPKValidatePartition(g, p2); !v.OK {
			t.Fatalf("consumer on SM 0 must still validate: %s", v.Error())
		}
	})

	t.Run("TaskDescriptorContract", func(t *testing.T) {
		g, p := MPKBuildDemoGraph()
		// The comm descriptor must match the graph node and partition, and
		// reference exactly one input and one output tensor.
		commDesc := &MPKTaskDescriptor{
			TaskID:     "comm",
			Kind:       MPKTaskComm,
			Op:         "sm2sm",
			SM:         1,
			InTensors:  []string{"act"},
			OutTensors: []string{"act"},
		}
		if v := MPKValidateDescriptor(g, p, commDesc); !v.OK {
			t.Fatalf("comm descriptor must satisfy the contract: %s", v.Error())
		}
		// The producer descriptor has one output tensor and no inputs.
		prodDesc := &MPKTaskDescriptor{
			TaskID:     "producer",
			Kind:       MPKTaskCompute,
			Op:         "gemm",
			SM:         0,
			OutTensors: []string{"act"},
		}
		if v := MPKValidateDescriptor(g, p, prodDesc); !v.OK {
			t.Fatalf("producer descriptor must satisfy the contract: %s", v.Error())
		}
		// A descriptor with the wrong SM must be refused.
		badSM := &MPKTaskDescriptor{
			TaskID:     "producer",
			Kind:       MPKTaskCompute,
			Op:         "gemm",
			SM:         1,
			OutTensors: []string{"act"},
		}
		if v := MPKValidateDescriptor(g, p, badSM); v.OK {
			t.Fatalf("descriptor with wrong SM must be refused")
		}
		// A comm descriptor missing its output tensor must be refused.
		missingOut := &MPKTaskDescriptor{
			TaskID:     "comm",
			Kind:       MPKTaskComm,
			Op:         "sm2sm",
			SM:         1,
			InTensors:  []string{"act"},
			OutTensors: nil,
		}
		if v := MPKValidateDescriptor(g, p, missingOut); v.OK {
			t.Fatalf("comm descriptor missing its output tensor must be refused")
		}
	})

	t.Run("FailClosedOnInvalidPartition", func(t *testing.T) {
		g, _ := MPKBuildDemoGraph()
		// A partition that leaves the consumer unassigned must be refused.
		partial := &MPKPartition{
			Assign: map[string]int{"producer": 0, "comm": 1},
			SMs:    2,
		}
		if v := MPKValidatePartition(g, partial); v.OK {
			t.Fatalf("partition missing the consumer must be refused")
		}
		// A partition that assigns a task outside the SM budget must be refused.
		outOfBudget := &MPKPartition{
			Assign: map[string]int{"producer": 0, "comm": 1, "consumer": 5},
			SMs:    2,
		}
		if v := MPKValidatePartition(g, outOfBudget); v.OK {
			t.Fatalf("partition assigning SM 5 under a 2-SM budget must be refused")
		}
		// The oracle must refuse to run on the invalid partition.
		if _, err := MPKTaskgraphOracle(g, partial); err == nil {
			t.Fatalf("oracle must refuse an invalid partition")
		} else if !strings.Contains(err.Error(), "mpk taskgraph: invalid") {
			t.Fatalf("oracle refusal must carry the invalid sentinel, got: %v", err)
		}
	})

	t.Run("OracleSimulatesPipeline", func(t *testing.T) {
		g, p := MPKBuildDemoGraph()
		receipt, err := MPKTaskgraphOracle(g, p)
		if err != nil {
			t.Fatalf("oracle must run on the demo graph: %v", err)
		}
		if receipt.Tasks != 3 {
			t.Fatalf("oracle must schedule all 3 tasks, got %d", receipt.Tasks)
		}
		// producer (SM 0) runs 0-1; comm (SM 1) waits on producer and runs
		// 1-2; consumer (SM 1) waits on comm and runs 2-3. Makespan is 3.
		if receipt.Makespan != 3 {
			t.Fatalf("demo pipeline makespan must be 3, got %d", receipt.Makespan)
		}
		// The comm task serializes behind the producer on the critical path,
		// so no compute task overlaps it in this graph.
		if receipt.Overlap != 0 {
			t.Fatalf("demo pipeline must report 0 overlap, got %d", receipt.Overlap)
		}
	})

	t.Run("OracleReportsOverlap", func(t *testing.T) {
		// A comm task on its own SM that runs concurrently with compute
		// tasks on other SMs: the cross-operator compute-communication
		// overlap the MPK runtime is built to exploit.
		g := &MPKTaskGraph{
			Nodes: []MPKTaskNode{
				{ID: "c1", Kind: MPKTaskCompute, Op: "gemm", SMs: 1, Shared: 0},
				{ID: "c2", Kind: MPKTaskCompute, Op: "gemm", SMs: 1, Shared: 0},
				{ID: "comm", Kind: MPKTaskComm, Op: "sm2sm", SMs: 1, Shared: 0},
				{ID: "consumer", Kind: MPKTaskCompute, Op: "gemm", SMs: 1, Shared: 0},
			},
			Edges: []MPKTaskEdge{
				{From: "comm", To: "consumer", Tensor: "t", Bytes: 4096},
			},
		}
		p := &MPKPartition{
			Assign: map[string]int{"c1": 0, "c2": 1, "comm": 2, "consumer": 1},
			SMs:    3,
		}
		if v := MPKValidateGraph(g); !v.OK {
			t.Fatalf("overlap graph must validate: %s", v.Error())
		}
		if v := MPKValidatePartition(g, p); !v.OK {
			t.Fatalf("overlap partition must validate: %s", v.Error())
		}
		receipt, err := MPKTaskgraphOracle(g, p)
		if err != nil {
			t.Fatalf("oracle must run on the overlap graph: %v", err)
		}
		// comm (SM 2) runs 0-1 concurrently with c1 (SM 0) and c2 (SM 1).
		if receipt.Overlap != 1 {
			t.Fatalf("overlap graph must report exactly 1 compute-communication overlap, got %d", receipt.Overlap)
		}
		if receipt.Makespan != 2 {
			t.Fatalf("overlap graph makespan must be 2, got %d", receipt.Makespan)
		}
	})

	t.Run("OracleRefusesCyclicGraph", func(t *testing.T) {
		g := &MPKTaskGraph{
			Nodes: []MPKTaskNode{
				{ID: "a", Kind: MPKTaskCompute, Op: "gemm", SMs: 1, Shared: 0},
				{ID: "b", Kind: MPKTaskCompute, Op: "gemm", SMs: 1, Shared: 0},
			},
			Edges: []MPKTaskEdge{
				{From: "a", To: "b", Tensor: "t", Bytes: 0},
				{From: "b", To: "a", Tensor: "t", Bytes: 0},
			},
		}
		if v := MPKValidateGraph(g); v.OK {
			t.Fatalf("cyclic graph must be refused")
		}
		p := &MPKPartition{Assign: map[string]int{"a": 0, "b": 0}, SMs: 1}
		if _, err := MPKTaskgraphOracle(g, p); err == nil {
			t.Fatalf("oracle must refuse a cyclic graph")
		}
	})
}
