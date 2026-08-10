package vcachestar

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

type ComparisonArm struct {
	Name                    string
	Kind                    string
	Available               bool
	Correct                 bool
	Latency                 time.Duration
	Demoted                 bool
	Alarmed                 bool
	FirstDivergeSegment     int
	FirstDivergeTokenOffset int64
	FirstDivergeByteOffset  int
	BookedUncachedTokens    int64
	RebateTokens            int64
	CPUSeconds              float64
	PeakRSSBytes            int64
	TelemetryBytes          int64
	NetworkBytes            int64
	StorageBytes            int64
	OperatorSeconds         float64
	CostUSD                 float64
	Note                    string
}

type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func reconciliationFixture() (Belief, Telemetry) {
	oldPrefix := []cachemeta.PromptSegment{{Kind: cachemeta.SegStable, Tokens: 100, Content: []byte("system")}, {Kind: cachemeta.SegMessage, Tokens: 50, Content: []byte("old")}}
	newPrefix := []cachemeta.PromptSegment{{Kind: cachemeta.SegStable, Tokens: 100, Content: []byte("system")}, {Kind: cachemeta.SegMessage, Tokens: 50, Content: []byte("new")}}
	return Belief{Warm: true, LastPrefix: oldPrefix, LastPrefixBytes: []byte("system|old")}, Telemetry{CacheReadInputTokens: 0, UncachedInputTokens: 150, CurrentPrefix: newPrefix, CurrentPrefixBytes: []byte("system|new")}
}

func correctReconciliation(f FoldResult) bool {
	return f.Demoted && f.Alarm && !f.ConfirmedHit && !f.Belief.Warm && f.Reason == ReasonBelievedWarmZeroRead && f.Divergence.FirstDivergeSeg == 1 && f.FirstDivergeTokenOffset == 100 && f.FirstDivergeByteOffset == 7 && f.Cost.BookedUncachedTokens == 150 && f.Cost.RebateTokens == 0
}

// CompareTelemetryLocal executes native reconciliation and the no-read-back
// manifest-trust baseline. Real providers and observability products stay at
// zero until their actual boundaries run this fixture.
func CompareTelemetryLocal() ComparisonResult {
	belief, telemetry := reconciliationFixture()
	start := time.Now()
	fold := FoldTelemetry(belief, telemetry)
	nativeLatency := time.Since(start)
	start = time.Now()
	baseline := FoldResult{Belief: belief, ConfirmedHit: true, FirstDivergeByteOffset: -1, Cost: CostBooking{RebateTokens: 150}}
	baselineLatency := time.Since(start)
	return ComparisonResult{Workload: "reconcile one believed-warm divergent prefix against zero provider cache-read tokens and book only confirmed savings", Arms: []ComparisonArm{
		{Name: "fak native cache telemetry reconciliation", Kind: "native", Available: true, Correct: correctReconciliation(fold), Latency: nativeLatency, Demoted: fold.Demoted, Alarmed: fold.Alarm, FirstDivergeSegment: fold.Divergence.FirstDivergeSeg, FirstDivergeTokenOffset: fold.FirstDivergeTokenOffset, FirstDivergeByteOffset: fold.FirstDivergeByteOffset, BookedUncachedTokens: fold.Cost.BookedUncachedTokens, RebateTokens: fold.Cost.RebateTokens, Note: "provider read-back dominates manifest belief; token and byte divergence are reported"},
		{Name: "trust warm manifest and modeled savings", Kind: "baseline", Available: true, Correct: correctReconciliation(baseline), Latency: baselineLatency, Demoted: baseline.Demoted, Alarmed: baseline.Alarm, FirstDivergeSegment: baseline.Divergence.FirstDivergeSeg, FirstDivergeTokenOffset: baseline.FirstDivergeTokenOffset, FirstDivergeByteOffset: baseline.FirstDivergeByteOffset, BookedUncachedTokens: baseline.Cost.BookedUncachedTokens, RebateTokens: baseline.Cost.RebateTokens, Note: "no-feature baseline falsely keeps warmth and rebates modeled rather than confirmed tokens"},
		{Name: "fak + Anthropic prompt caching", Kind: "integration", Note: "requires real Anthropic usage read-back"},
		{Name: "fak + OpenAI prompt caching", Kind: "integration", Note: "requires real OpenAI usage read-back"},
		{Name: "fak + Gemini context caching", Kind: "integration", Note: "requires real Gemini usage read-back"},
		{Name: "fak + Prometheus", Kind: "integration", Note: "requires real metric export, rule, and read-back"},
		{Name: "fak + OpenTelemetry", Kind: "integration", Note: "requires real collector/exporter and read-back"},
		{Name: "Prometheus recording and alert rules", Kind: "external", Note: "requires real Prometheus rule evaluation"},
		{Name: "Datadog monitors", Kind: "external", Note: "requires real Datadog ingestion and monitor evaluation"},
		{Name: "LangSmith traces", Kind: "external", Note: "requires real trace ingestion and query"},
	}}
}
