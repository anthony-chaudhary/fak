package vcachechain

import (
	"fmt"
	"testing"
)

var (
	benchErrSink         error
	benchChainSink       []ChainNode
	benchTokensSink      int64
	benchReplayPlanSink  ReplayPlan
	benchBreakpointsSink []int
	benchRecallPlanSink  RecallPlan
	benchRecallProofSink RecallProof
	benchDescriptorSink  WarmPrefixDescriptor
	benchVerifySink      bool
	benchFactsSink       []FactRecord
	benchRefreshEvalSink RefreshEvaluation
)

func benchmarkDAG() PrefixDAG {
	nodes := []ChainNode{
		{ID: "anchor", ParentID: "", Tokens: 16384, Blocks: 10},
		{ID: "branch-a", ParentID: "anchor", Tokens: 4096, Blocks: 5},
		{ID: "branch-b", ParentID: "anchor", Tokens: 4096, Blocks: 5},
	}
	for i := 1; i <= 6; i++ {
		parent := "branch-a"
		if i > 3 {
			parent = "branch-b"
		}
		nodes = append(nodes, ChainNode{
			ID:       fmt.Sprintf("node-%d", i),
			ParentID: parent,
			Tokens:   1024,
			Blocks:   3,
		})
	}
	for i := 1; i <= 6; i++ {
		nodes = append(nodes, ChainNode{
			ID:       fmt.Sprintf("leaf-%d", i),
			ParentID: fmt.Sprintf("node-%d", i),
			Tokens:   128,
			Blocks:   1,
		})
	}
	return PrefixDAG{Nodes: nodes}
}

func benchmarkCorrectionChain() (CorrectionChain, RefreshPolicy) {
	base := make([]FactRecord, 20)
	for i := 0; i < 20; i++ {
		base[i] = FactRecord{
			Key:   fmt.Sprintf("config.key.%02d", i),
			Value: fmt.Sprintf("initial-value-%d", i),
		}
	}
	chain := NewCorrectionChain(base)
	for s := 0; s < 5; s++ {
		facts := []FactRecord{
			{Key: fmt.Sprintf("config.key.%02d", s*2), Value: fmt.Sprintf("updated-%d", s)},
			{Key: fmt.Sprintf("config.key.%02d", s*2+1), Value: fmt.Sprintf("updated-%d", s)},
		}
		chain = chain.AppendCorrection(CorrectionSegment{
			ID:      fmt.Sprintf("seg-%d", s),
			Facts:   facts,
			ByteLen: 64,
		})
	}
	policy := RefreshPolicy{
		MaxCorrectionCount: 10,
		MaxCorrectionBytes: 1024,
	}
	return chain, policy
}

func BenchmarkPrefixDAGValidate(b *testing.B) {
	dag := benchmarkDAG()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchErrSink = dag.Validate()
	}
}

func BenchmarkChainTo(b *testing.B) {
	dag := benchmarkDAG()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchChainSink, benchErrSink = dag.ChainTo("leaf-6")
	}
}

func BenchmarkPrefixTokens(b *testing.B) {
	dag := benchmarkDAG()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchTokensSink, benchErrSink = dag.PrefixTokens("leaf-6")
	}
}

func BenchmarkTopologicalReplay(b *testing.B) {
	dag := benchmarkDAG()
	targets := []string{"leaf-1", "leaf-2", "leaf-3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReplayPlanSink, benchErrSink = dag.TopologicalReplay(targets, 1)
	}
}

func BenchmarkPlaceBreakpoints(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBreakpointsSink = PlaceBreakpoints(120)
	}
}

func BenchmarkMergeBreakpoints(b *testing.B) {
	a := []int{15, 45, 75}
	c := []int{30, 45, 60, 90}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBreakpointsSink = MergeBreakpoints(a, c)
	}
}

func BenchmarkPlanRecall_SingleUnitRefused(b *testing.B) {
	dag := benchmarkDAG()
	req := RecallRequest{
		TargetNodeID:     "leaf-6",
		SiblingsRecalled: 1,
		ReadMult:         0.1,
		WarmDepth:        1,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRecallPlanSink, benchErrSink = PlanRecall(dag, req, true)
	}
}

func BenchmarkPlanRecall_AmortizedRebuild(b *testing.B) {
	dag := benchmarkDAG()
	req := RecallRequest{
		TargetNodeID:     "leaf-6",
		SiblingsRecalled: 500,
		ReadMult:         0.1,
		WarmDepth:        1,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRecallPlanSink, benchErrSink = PlanRecall(dag, req, true)
	}
}

func BenchmarkProveRecall(b *testing.B) {
	in := ProveRecallInput{
		PrefixTokens: 30000,
		UnitTokens:   10,
		ReadMult:     0.1,
		Siblings:     1,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRecallProofSink = ProveRecall(in)
	}
}

func BenchmarkDescribeWarmPrefix(b *testing.B) {
	prefix := []byte("System preamble for vcache model execution and agent interaction.\nTool schema definitions: read, write, edit, bash, glob, grep.\nInitial conversation context and system instructions.")
	spans := []SpanBoundary{
		{Kind: SpanSystem, Start: 0, End: 66},
		{Kind: SpanTools, Start: 66, End: 129},
		{Kind: SpanSystem, Start: 129, End: len(prefix)},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDescriptorSink = DescribeWarmPrefix(prefix, spans)
	}
}

func BenchmarkVerifyWarmPrefixReplay(b *testing.B) {
	prefix := []byte("System preamble for vcache model execution and agent interaction.\nTool schema definitions: read, write, edit, bash, glob, grep.\nInitial conversation context and system instructions.")
	desc := DescribeWarmPrefix(prefix, nil)
	replayed := append([]byte(nil), prefix...)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerifySink = VerifyWarmPrefixReplay(desc, replayed)
	}
}

func BenchmarkCorrectionChainEffectiveFacts(b *testing.B) {
	chain, _ := benchmarkCorrectionChain()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFactsSink = chain.EffectiveFacts()
	}
}

func BenchmarkCorrectionChainEvaluateRefresh(b *testing.B) {
	chain, policy := benchmarkCorrectionChain()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRefreshEvalSink = chain.EvaluateRefresh(policy)
	}
}
