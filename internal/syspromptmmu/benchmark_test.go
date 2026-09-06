package syspromptmmu

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/capindex"
)

var (
	benchSegmentsSink []Segment
	benchPlanSink     []cachemeta.PromptSegment
	benchStringSink   string
	benchBytesSink    []byte
	benchSpliceSink   SpliceResult
	benchAuditSink    PrefixAudit
	benchOverlaySink  OverlaySelection
	benchVerdictSink  EditVerdict
	benchStyleSink    StyleReadout
	benchProfileSink  WorkProfileReadout
)

func buildBenchmarkRequestBody(sysValue []byte) []byte {
	obj := map[string]json.RawMessage{
		"model":    json.RawMessage(`"claude-3-5-sonnet"`),
		"system":   json.RawMessage(sysValue),
		"messages": json.RawMessage(`[{"role":"user","content":"benchmark request"}]`),
	}
	raw, _ := json.Marshal(obj)
	return raw
}

func BenchmarkBaseContext(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSegmentsSink = BaseContext()
	}
}

func BenchmarkBaseContextPlan(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchPlanSink = BaseContextPlan()
	}
}

func BenchmarkPlanDigest(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = PlanDigest()
	}
}

func BenchmarkWitnessFor(b *testing.B) {
	sample := []byte(spineIdentity)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = WitnessFor(sample)
	}
}

func BenchmarkBuildSystemValue_BaseOnly(b *testing.B) {
	plan := BaseContextPlan()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBytesSink = BuildSystemValue(plan, nil)
	}
}

func BenchmarkBuildSystemValue_WithOverlay(b *testing.B) {
	plan := BaseContextPlan()
	overlay := []cachemeta.PromptSegment{
		{
			Kind:    cachemeta.SegMessage,
			Content: []byte("You are a specialized coding agent operating with strict boundary discipline."),
			Tokens:  20,
			Witness: WitnessFor([]byte("You are a specialized coding agent operating with strict boundary discipline.")),
		},
		{
			Kind:    cachemeta.SegMessage,
			Content: []byte("Output compact receipts with exact command and exit status."),
			Tokens:  15,
			Witness: WitnessFor([]byte("Output compact receipts with exact command and exit status.")),
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBytesSink = BuildSystemValue(plan, overlay)
	}
}

func BenchmarkSpliceSystemOverlay(b *testing.B) {
	plan := BaseContextPlan()
	oldOverlay := []cachemeta.PromptSegment{
		{
			Kind:    cachemeta.SegMessage,
			Content: []byte("initial overlay content"),
			Tokens:  5,
			Witness: WitnessFor([]byte("initial overlay content")),
		},
	}
	sysVal := BuildSystemValue(plan, oldOverlay)
	raw := buildBenchmarkRequestBody(sysVal)

	newOverlay := []cachemeta.PromptSegment{
		{
			Kind:    cachemeta.SegMessage,
			Content: []byte("updated overlay segment for subsequent turn"),
			Tokens:  10,
			Witness: WitnessFor([]byte("updated overlay segment for subsequent turn")),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSpliceSink = SpliceSystemOverlay(raw, plan, newOverlay, nil)
	}
}

func BenchmarkSpliceSystemOverlay_WithDecode(b *testing.B) {
	plan := BaseContextPlan()
	oldOverlay := []cachemeta.PromptSegment{
		{
			Kind:    cachemeta.SegMessage,
			Content: []byte("initial overlay content"),
			Tokens:  5,
			Witness: WitnessFor([]byte("initial overlay content")),
		},
	}
	sysVal := BuildSystemValue(plan, oldOverlay)
	raw := buildBenchmarkRequestBody(sysVal)

	newOverlay := []cachemeta.PromptSegment{
		{
			Kind:    cachemeta.SegMessage,
			Content: []byte("updated overlay segment for subsequent turn"),
			Tokens:  10,
			Witness: WitnessFor([]byte("updated overlay segment for subsequent turn")),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSpliceSink = SpliceSystemOverlay(raw, plan, newOverlay, decodeOK)
	}
}

func BenchmarkAuditRealizedPrefix(b *testing.B) {
	plan := BaseContextPlan()
	overlay := []cachemeta.PromptSegment{
		{
			Kind:    cachemeta.SegMessage,
			Content: []byte("turn overlay"),
			Tokens:  5,
			Witness: WitnessFor([]byte("turn overlay")),
		},
	}
	sysVal := BuildSystemValue(plan, overlay)
	raw := buildBenchmarkRequestBody(sysVal)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchAuditSink = AuditRealizedPrefix(raw, plan)
	}
}

func BenchmarkAuditBaseContext(b *testing.B) {
	plan := BaseContextPlan()
	sysVal := BuildSystemValue(plan, nil)
	raw := buildBenchmarkRequestBody(sysVal)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchAuditSink = AuditBaseContext(raw)
	}
}

func BenchmarkSelectOverlay(b *testing.B) {
	cat := capindex.NewCatalog()
	resolves := 0
	r := newFakeResolver(20, 128, "agent task", &resolves)
	cat.AddResolver(capindex.CapKindSkill, r)
	cat.Sync()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchOverlaySink = SelectOverlay(cat, "agent task", 1000)
	}
}

func BenchmarkOverlayCache_Hit(b *testing.B) {
	cat := capindex.NewCatalog()
	resolves := 0
	r := newFakeResolver(20, 128, "agent task", &resolves)
	cat.AddResolver(capindex.CapKindSkill, r)
	cat.Sync()

	oc := NewOverlayCache()
	_, _ = oc.GetOrSelect(cat, "agent task", 1000)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchOverlaySink, _ = oc.GetOrSelect(cat, "agent task", 1000)
	}
}

func BenchmarkGateEdit(b *testing.B) {
	edit := BaseEdit{
		Op:      EditAdd,
		Tier:    TierPolicy,
		Content: []byte("learned rule: verify tests before declaring complete"),
		Version: "v2",
	}
	witness := func(BaseEdit) bool { return true }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchVerdictSink = GateEdit(edit, witness)
	}
}

func BenchmarkApplyEdit(b *testing.B) {
	base := BaseContext()
	edit := BaseEdit{
		Op:      EditAdd,
		Tier:    TierPolicy,
		Content: []byte("learned rule: verify tests before declaring complete"),
		Version: "v2",
	}
	witness := func(BaseEdit) bool { return true }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSegmentsSink, benchVerdictSink = ApplyEdit(base, edit, witness)
	}
}

func BenchmarkDescribeStyle(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStyleSink = DescribeStyle("caveman:native:medium")
	}
}

func BenchmarkDescribeWorkProfile(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchProfileSink = DescribeWorkProfile(WorkProfilePonytailNativeMed)
	}
}

func TestBenchmarkOperationsSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}
	benchmarks := []struct {
		name string
		fn   func(*testing.B)
	}{
		{"BenchmarkBaseContext", BenchmarkBaseContext},
		{"BenchmarkBaseContextPlan", BenchmarkBaseContextPlan},
		{"BenchmarkPlanDigest", BenchmarkPlanDigest},
		{"BenchmarkWitnessFor", BenchmarkWitnessFor},
		{"BenchmarkBuildSystemValue_BaseOnly", BenchmarkBuildSystemValue_BaseOnly},
		{"BenchmarkBuildSystemValue_WithOverlay", BenchmarkBuildSystemValue_WithOverlay},
		{"BenchmarkSpliceSystemOverlay", BenchmarkSpliceSystemOverlay},
		{"BenchmarkSpliceSystemOverlay_WithDecode", BenchmarkSpliceSystemOverlay_WithDecode},
		{"BenchmarkAuditRealizedPrefix", BenchmarkAuditRealizedPrefix},
		{"BenchmarkAuditBaseContext", BenchmarkAuditBaseContext},
		{"BenchmarkSelectOverlay", BenchmarkSelectOverlay},
		{"BenchmarkOverlayCache_Hit", BenchmarkOverlayCache_Hit},
		{"BenchmarkGateEdit", BenchmarkGateEdit},
		{"BenchmarkApplyEdit", BenchmarkApplyEdit},
		{"BenchmarkDescribeStyle", BenchmarkDescribeStyle},
		{"BenchmarkDescribeWorkProfile", BenchmarkDescribeWorkProfile},
	}

	for _, tc := range benchmarks {
		t.Run(tc.name, func(t *testing.T) {
			res := testing.Benchmark(tc.fn)
			if res.N <= 0 {
				t.Fatalf("%s failed to execute any iterations: %+v", tc.name, res)
			}
		})
	}
}
