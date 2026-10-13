package deepseekv4moe

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// scoredTestEpsilon is the tolerance for VersusOPT upper-bound checks. The ratio is an
// exact float division of ints, but a tiny slack documents intent and absorbs any future
// oracle normalization change without hiding a real >1 violation.
const scoredTestEpsilon = 1e-9

// scoredTestRow finds the row for policy p, failing when it is absent.
func scoredTestRow(t *testing.T, receipt CachePolicyBenchReceipt, p CachePolicy) CachePolicyRow {
	t.Helper()
	for _, row := range receipt.Rows {
		if row.Policy == p {
			return row
		}
	}
	t.Fatalf("receipt has no row for policy %s (%d)", p, int(p))
	return CachePolicyRow{}
}

// scoredTestAssertRowInvariants checks the access-accounting contract every row must hold.
func scoredTestAssertRowInvariants(t *testing.T, row CachePolicyRow, context string) {
	t.Helper()
	if row.Hits < 0 {
		t.Fatalf("%s policy %s: Hits = %d, want >= 0", context, row.Policy, row.Hits)
	}
	if row.Misses < 0 {
		t.Fatalf("%s policy %s: Misses = %d, want >= 0", context, row.Policy, row.Misses)
	}
	if row.Hits+row.Misses != row.Accesses {
		t.Fatalf("%s policy %s: Hits(%d)+Misses(%d) = %d, want Accesses %d",
			context, row.Policy, row.Hits, row.Misses, row.Hits+row.Misses, row.Accesses)
	}
	if row.Evictions < 0 {
		t.Fatalf("%s policy %s: Evictions = %d, want >= 0", context, row.Policy, row.Evictions)
	}
	if row.Accesses > 0 {
		want := float64(row.Hits) / float64(row.Accesses)
		if !row.HitRateKnown {
			t.Fatalf("%s policy %s: HitRateKnown = false, want true (Accesses %d)",
				context, row.Policy, row.Accesses)
		}
		if row.HitRate != want {
			t.Fatalf("%s policy %s: HitRate = %v, want %v", context, row.Policy, row.HitRate, want)
		}
	} else {
		if row.HitRateKnown {
			t.Fatalf("%s policy %s: HitRateKnown = true on empty stream", context, row.Policy)
		}
		if row.HitRate != 0 {
			t.Fatalf("%s policy %s: HitRate = %v on empty stream, want 0", context, row.Policy, row.HitRate)
		}
	}
}

// TestCachePolicyScoredEnumStability pins the numeric values of the original five arms.
// A renumbering would silently invalidate every persisted CachePolicyBenchReceipt.
func TestCachePolicyScoredEnumStability(t *testing.T) {
	cases := []struct {
		p    CachePolicy
		want int
	}{
		{CachePolicyLRU, 0},
		{CachePolicyLFU, 1},
		{CachePolicyHeat, 2},
		{CachePolicyRandom, 3},
		{CachePolicyOPT, 4},
	}
	for _, tc := range cases {
		if int(tc.p) != tc.want {
			t.Fatalf("%s = %d, want %d (arm renumbering breaks persisted receipts)",
				tc.p, int(tc.p), tc.want)
		}
	}
}

// TestCachePolicyScoredArmsPartition checks that the scored arms are appended after the
// original five and are disjoint from them.
func TestCachePolicyScoredArmsPartition(t *testing.T) {
	orig := CachePolicyArms()
	// Relation, not a frozen count: CachePolicyArms is exactly the five original arms.
	wantOrig := []CachePolicy{CachePolicyLRU, CachePolicyLFU, CachePolicyHeat, CachePolicyRandom, CachePolicyOPT}
	if !reflect.DeepEqual(orig, wantOrig) {
		t.Fatalf("CachePolicyArms() = %v, want %v", orig, wantOrig)
	}

	scored := CachePolicyScoredArms()
	wantScored := []CachePolicy{CachePolicyLFRU, CachePolicyScoreEMA}
	if !reflect.DeepEqual(scored, wantScored) {
		t.Fatalf("CachePolicyScoredArms() = %v, want %v", scored, wantScored)
	}
	if len(scored)+len(orig) != len(wantScored)+len(wantOrig) {
		t.Fatalf("scored(%d)+original(%d) arms = %d, want %d", len(scored), len(orig), len(scored)+len(orig), len(wantScored)+len(wantOrig))
	}
	for _, p := range scored {
		if p == CachePolicyOPT {
			t.Fatalf("CachePolicyScoredArms() contains OPT")
		}
		for _, o := range orig {
			if p == o {
				t.Fatalf("CachePolicyScoredArms() contains original arm %s", o)
			}
		}
	}

	if got := CachePolicyLFRU.String(); got != "lfru" {
		t.Fatalf("CachePolicyLFRU.String() = %q, want %q", got, "lfru")
	}
	if got := CachePolicyScoreEMA.String(); got != "scoreema" {
		t.Fatalf("CachePolicyScoreEMA.String() = %q, want %q", got, "scoreema")
	}
	if got := CachePolicy(99).String(); got != "unknown" {
		t.Fatalf("CachePolicy(99).String() = %q, want %q", got, "unknown")
	}
	if CachePolicyLFRU.String() == "unknown" || CachePolicyScoreEMA.String() == "unknown" {
		t.Fatalf("scored arm String() collides with the unknown sentinel")
	}
}

// TestCachePolicyScoredReplayRegression checks that the original entry point still emits
// exactly the original five arms and agrees with the generalized entry point.
func TestCachePolicyScoredReplayRegression(t *testing.T) {
	// Keep the distinct-group pool small: the replayoracle exact Belady DP is
	// exponential in distinct groups, and these tests assert the exact bound.
	stream := mustSyntheticStream(t, "scored-regression", 11, 1, 8, 2, 80)

	receipt, err := CachePolicyReplay(stream, 5)
	if err != nil {
		t.Fatalf("CachePolicyReplay error = %v", err)
	}
	if len(receipt.Rows) != len(CachePolicyArms()) {
		t.Fatalf("len(receipt.Rows) = %d, want len(CachePolicyArms()) = %d", len(receipt.Rows), len(CachePolicyArms()))
	}
	for i, row := range receipt.Rows {
		if row.Policy != CachePolicyArms()[i] {
			t.Fatalf("row[%d].Policy = %s, want %s", i, row.Policy, CachePolicyArms()[i])
		}
	}

	with, err := CachePolicyReplayWith(stream, 5, CachePolicyArms())
	if err != nil {
		t.Fatalf("CachePolicyReplayWith error = %v", err)
	}
	if with.OPTHits != receipt.OPTHits {
		t.Fatalf("OPT hits = %d, want %d", with.OPTHits, receipt.OPTHits)
	}
	if len(with.Rows) != len(receipt.Rows) {
		t.Fatalf("len(with.Rows) = %d, want %d", len(with.Rows), len(receipt.Rows))
	}
	for i := range receipt.Rows {
		if with.Rows[i].Hits != receipt.Rows[i].Hits {
			t.Fatalf("row %s: With Hits = %d, Replay Hits = %d",
				receipt.Rows[i].Policy, with.Rows[i].Hits, receipt.Rows[i].Hits)
		}
	}
}

// TestCachePolicyReplayWithContract checks requested-order output, header parity with the
// original replay, the OPT row's normalization, and capacity fail-closed behavior.
func TestCachePolicyReplayWithContract(t *testing.T) {
	stream := mustSyntheticStream(t, "scored-contract", 23, 1, 8, 2, 60)

	base, err := CachePolicyReplay(stream, 4)
	if err != nil {
		t.Fatalf("CachePolicyReplay error = %v", err)
	}

	// Reordered subset that includes OPT.
	requested := []CachePolicy{CachePolicyHeat, CachePolicyOPT, CachePolicyLRU}
	with, err := CachePolicyReplayWith(stream, 4, requested)
	if err != nil {
		t.Fatalf("CachePolicyReplayWith error = %v", err)
	}
	if len(with.Rows) != len(requested) {
		t.Fatalf("len(with.Rows) = %d, want %d", len(with.Rows), len(requested))
	}
	for i, p := range requested {
		if with.Rows[i].Policy != p {
			t.Fatalf("row[%d].Policy = %s, want %s (requested order not preserved)",
				i, with.Rows[i].Policy, p)
		}
	}
	if with.Schema != base.Schema {
		t.Fatalf("Schema = %q, want %q", with.Schema, base.Schema)
	}
	if with.Measurement != base.Measurement {
		t.Fatalf("Measurement = %q, want %q", with.Measurement, base.Measurement)
	}
	if with.Stream != base.Stream {
		t.Fatalf("Stream = %q, want %q", with.Stream, base.Stream)
	}
	if with.Capacity != base.Capacity {
		t.Fatalf("Capacity = %d, want %d", with.Capacity, base.Capacity)
	}
	if with.AccessCount != base.AccessCount {
		t.Fatalf("AccessCount = %d, want %d", with.AccessCount, base.AccessCount)
	}
	if with.OPTExact != base.OPTExact {
		t.Fatalf("OPTExact = %v, want %v", with.OPTExact, base.OPTExact)
	}
	if with.OPTHits != base.OPTHits {
		t.Fatalf("OPTHits = %d, want %d", with.OPTHits, base.OPTHits)
	}
	if opt := scoredTestRow(t, with, CachePolicyOPT); opt.VersusOPT != 1.0 {
		t.Fatalf("OPT VersusOPT = %v, want 1.0", opt.VersusOPT)
	}

	// Subset excluding OPT.
	without, err := CachePolicyReplayWith(stream, 4, []CachePolicy{CachePolicyScoreEMA, CachePolicyLFRU})
	if err != nil {
		t.Fatalf("CachePolicyReplayWith(no OPT) error = %v", err)
	}
	if len(without.Rows) != 2 {
		t.Fatalf("len(without.Rows) = %d, want 2", len(without.Rows))
	}
	if without.Rows[0].Policy != CachePolicyScoreEMA || without.Rows[1].Policy != CachePolicyLFRU {
		t.Fatalf("without-OPT order = [%s %s], want [%s %s]",
			without.Rows[0].Policy, without.Rows[1].Policy, CachePolicyScoreEMA, CachePolicyLFRU)
	}
	if without.OPTHits != base.OPTHits {
		t.Fatalf("without-OPT OPTHits = %d, want %d (oracle computed regardless)", without.OPTHits, base.OPTHits)
	}

	if _, err := CachePolicyReplayWith(stream, 0, CachePolicyArms()); !errors.Is(err, ErrInvalidTraceCapacity) {
		t.Fatalf("capacity 0 error = %v, want %v", err, ErrInvalidTraceCapacity)
	}
	if _, err := CachePolicyReplayWith(stream, -3, CachePolicyScoredArms()); !errors.Is(err, ErrInvalidTraceCapacity) {
		t.Fatalf("capacity -3 error = %v, want %v", err, ErrInvalidTraceCapacity)
	}
}

// distinctTraceGroups counts the distinct (layer, expert) pairs touched by a stream.
func distinctTraceGroups(stream CachePolicyStream) int {
	seen := make(map[ExpertGroup]struct{}, len(stream.Groups))
	for _, g := range stream.Groups {
		seen[g] = struct{}{}
	}
	return len(seen)
}

// TestCachePolicyScoredBeladyBound checks that no online arm, including the two scored
// arms, ever exceeds the OPT oracle (exact or greedy-fallback).
func TestCachePolicyScoredBeladyBound(t *testing.T) {
	policies := append(CachePolicyScoredArms(), CachePolicyArms()...)

	exact := mustSyntheticStream(t, "scored-bound-exact", 31, 1, 12, 3, 24)
	if d := distinctTraceGroups(exact); d > 63 {
		t.Fatalf("exact-stream distinct groups = %d, want <= 63", d)
	}
	assertBound := func(t *testing.T, stream CachePolicyStream, capacity int) {
		t.Helper()
		receipt, err := CachePolicyReplayWith(stream, capacity, policies)
		if err != nil {
			t.Fatalf("CachePolicyReplayWith error = %v", err)
		}
		for _, row := range receipt.Rows {
			if row.Policy == CachePolicyOPT {
				continue
			}
			if row.Hits < 0 || row.Hits > receipt.OPTHits {
				t.Fatalf("stream %q cap %d policy %s: Hits = %d, want in [0, %d]",
					stream.Name, capacity, row.Policy, row.Hits, receipt.OPTHits)
			}
			if row.VersusOPT < 0 || row.VersusOPT > 1.0+scoredTestEpsilon {
				t.Fatalf("stream %q cap %d policy %s: VersusOPT = %v, want in [0, 1]",
					stream.Name, capacity, row.Policy, row.VersusOPT)
			}
		}
	}
	receiptExact, err := CachePolicyReplayWith(exact, 3, policies)
	if err != nil {
		t.Fatalf("exact CachePolicyReplayWith error = %v", err)
	}
	if !receiptExact.OPTExact {
		t.Fatalf("exact stream OPTExact = false, want true")
	}
	assertBound(t, exact, 3)

	large := mustSyntheticStream(t, "scored-bound-greedy", 47, 8, 32, 2, 200)
	if d := distinctTraceGroups(large); d <= 63 {
		t.Fatalf("large-stream distinct groups = %d, want > 63 to exercise the greedy fallback", d)
	}
	receiptLarge, err := CachePolicyReplayWith(large, 7, policies)
	if err != nil {
		t.Fatalf("large CachePolicyReplayWith error = %v", err)
	}
	if receiptLarge.OPTExact {
		t.Fatalf("large stream OPTExact = true, want false (greedy fallback)")
	}
	assertBound(t, large, 7)

	// Tightest capacity must still respect the bound.
	assertBound(t, exact, 1)
}

// TestCachePolicyScoredRowInvariants checks access accounting across arms and capacities.
func TestCachePolicyScoredRowInvariants(t *testing.T) {
	stream := mustSyntheticStream(t, "scored-invariants", 59, 1, 8, 2, 120)
	policies := append(CachePolicyScoredArms(), CachePolicyArms()...)
	for _, capacity := range []int{1, 2, 5, 17, 64} {
		receipt, err := CachePolicyReplayWith(stream, capacity, policies)
		if err != nil {
			t.Fatalf("capacity %d: CachePolicyReplayWith error = %v", capacity, err)
		}
		for _, row := range receipt.Rows {
			scoredTestAssertRowInvariants(t, row, "capacity "+itoaTest(capacity))
		}
	}
}

// itoaTest is a tiny local int-to-string to avoid importing strconv.
func itoaTest(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// TestCachePolicyScoredDeterminism checks that both new entry points are pure.
func TestCachePolicyScoredDeterminism(t *testing.T) {
	stream := mustSyntheticStream(t, "scored-determinism", 67, 1, 6, 3, 60)
	policies := append(CachePolicyScoredArms(), CachePolicyArms()...)

	first, err := CachePolicyReplayWith(stream, 6, policies)
	if err != nil {
		t.Fatalf("first CachePolicyReplayWith error = %v", err)
	}
	second, err := CachePolicyReplayWith(stream, 6, policies)
	if err != nil {
		t.Fatalf("second CachePolicyReplayWith error = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("CachePolicyReplayWith not deterministic:\nfirst  = %+v\nsecond = %+v", first, second)
	}

	routes := scoredTestRoutesFromStream(t, stream, 3)
	traceA, err := SimulateExpertCacheTrace(routes, 1, 6, 3, 6)
	if err != nil {
		t.Fatalf("first SimulateExpertCacheTrace error = %v", err)
	}
	traceB, err := SimulateExpertCacheTrace(routes, 1, 6, 3, 6)
	if err != nil {
		t.Fatalf("second SimulateExpertCacheTrace error = %v", err)
	}
	if !reflect.DeepEqual(traceA, traceB) {
		t.Fatalf("SimulateExpertCacheTrace not deterministic:\nfirst  = %+v\nsecond = %+v", traceA, traceB)
	}
}

// TestCachePolicyScoredLFRUEdgeCases covers single-slot thrash and the empty stream.
func TestCachePolicyScoredLFRUEdgeCases(t *testing.T) {
	stream := mustSyntheticStream(t, "scored-lfru-edge", 71, 1, 8, 2, 50)
	receipt, err := CachePolicyReplayWith(stream, 1, []CachePolicy{CachePolicyLFRU})
	if err != nil {
		t.Fatalf("capacity 1 CachePolicyReplayWith error = %v", err)
	}
	row := scoredTestRow(t, receipt, CachePolicyLFRU)
	scoredTestAssertRowInvariants(t, row, "LFRU capacity 1")

	empty, err := CachePolicyReplayWith(CachePolicyStream{Name: "empty"}, 4, []CachePolicy{CachePolicyLFRU})
	if err != nil {
		t.Fatalf("empty CachePolicyReplayWith error = %v", err)
	}
	emptyRow := scoredTestRow(t, empty, CachePolicyLFRU)
	if emptyRow.Accesses != 0 {
		t.Fatalf("empty stream Accesses = %d, want 0", emptyRow.Accesses)
	}
	if emptyRow.HitRateKnown {
		t.Fatalf("empty stream HitRateKnown = true, want false")
	}
	scoredTestAssertRowInvariants(t, emptyRow, "LFRU empty")
}

// TestCachePolicyScoredScoreEMAEdgeCases covers single-slot thrash, an all-distinct
// stream below capacity, and determinism.
func TestCachePolicyScoredScoreEMAEdgeCases(t *testing.T) {
	stream := mustSyntheticStream(t, "scored-ema-edge", 73, 1, 8, 2, 50)
	receipt, err := CachePolicyReplayWith(stream, 1, []CachePolicy{CachePolicyScoreEMA})
	if err != nil {
		t.Fatalf("capacity 1 CachePolicyReplayWith error = %v", err)
	}
	scoredTestAssertRowInvariants(t, scoredTestRow(t, receipt, CachePolicyScoreEMA), "ScoreEMA capacity 1")

	const distinct = 12
	const capacity = 5
	groups := make([]ExpertGroup, 0, distinct)
	for i := 0; i < distinct; i++ {
		groups = append(groups, ExpertGroup{Layer: 0, Expert: i})
	}
	allDistinct := CachePolicyStream{Name: "all-distinct", Groups: groups}
	distinctReceipt, err := CachePolicyReplayWith(allDistinct, capacity, []CachePolicy{CachePolicyScoreEMA})
	if err != nil {
		t.Fatalf("all-distinct CachePolicyReplayWith error = %v", err)
	}
	distinctRow := scoredTestRow(t, distinctReceipt, CachePolicyScoreEMA)
	if distinctRow.Hits != 0 {
		t.Fatalf("ScoreEMA all-distinct Hits = %d, want 0 (no reuse below capacity)", distinctRow.Hits)
	}
	scoredTestAssertRowInvariants(t, distinctRow, "ScoreEMA all-distinct")

	again, err := CachePolicyReplayWith(allDistinct, capacity, []CachePolicy{CachePolicyScoreEMA})
	if err != nil {
		t.Fatalf("repeat CachePolicyReplayWith error = %v", err)
	}
	if !reflect.DeepEqual(distinctReceipt, again) {
		t.Fatalf("ScoreEMA not deterministic:\nfirst  = %+v\nsecond = %+v", distinctReceipt, again)
	}
}

// scoredTestRoutesFromStream chunks a synthetic stream (topK consecutive unit accesses per
// route, all at one layer) back into ExpertRoute values, asserting the chunk stays layer-
// homogeneous.
func scoredTestRoutesFromStream(t *testing.T, stream CachePolicyStream, topK int) []ExpertRoute {
	t.Helper()
	if len(stream.Groups)%topK != 0 {
		t.Fatalf("stream groups %d not divisible by topK %d", len(stream.Groups), topK)
	}
	routes := make([]ExpertRoute, 0, len(stream.Groups)/topK)
	for i := 0; i < len(stream.Groups); i += topK {
		layer := stream.Groups[i].Layer
		experts := make([]int, 0, topK)
		for j := 0; j < topK; j++ {
			g := stream.Groups[i+j]
			if g.Layer != layer {
				t.Fatalf("chunk %d spans layers %d and %d; synthetic stream not layer-homogeneous",
					i/topK, layer, g.Layer)
			}
			experts = append(experts, g.Expert)
		}
		routes = append(routes, ExpertRoute{Layer: layer, Experts: experts})
	}
	return routes
}

// TestSimulateExpertCacheTraceIdentity checks the real-trace adapter is an identity-
// preserving LRU replay, agrees with the original SimulateExpertCache entry point, and
// fails closed on malformed input.
func TestSimulateExpertCacheTraceIdentity(t *testing.T) {
	stream := mustSyntheticStream(t, "scored-trace-adapter", 83, 2, 6, 2, 40)
	const topK = 2
	const capacity = 4
	routes := scoredTestRoutesFromStream(t, stream, topK)

	trace, err := SimulateExpertCacheTrace(routes, 2, 6, topK, capacity)
	if err != nil {
		t.Fatalf("SimulateExpertCacheTrace error = %v", err)
	}

	lruReceipt, err := CachePolicyReplayWith(stream, capacity, []CachePolicy{CachePolicyLRU})
	if err != nil {
		t.Fatalf("CachePolicyReplayWith error = %v", err)
	}
	lru := scoredTestRow(t, lruReceipt, CachePolicyLRU)
	if trace.Hits != int64(lru.Hits) {
		t.Fatalf("trace.Hits = %d, want LRU Hits %d", trace.Hits, lru.Hits)
	}
	if trace.PageIns != int64(lru.Misses) {
		t.Fatalf("trace.PageIns = %d, want LRU Misses %d", trace.PageIns, lru.Misses)
	}

	// The real identity claim: the adapter must reproduce the ORIGINAL
	// SimulateExpertCache(routes, capacity, layers, expertsPerLayer, topK) trace on the
	// same routes/capacity, including the resident set. The adapter was added alongside
	// the original entry point, so any divergence here is a silent behaviour fork.
	original, err := SimulateExpertCache(routes, capacity, 2, 6, topK)
	if err != nil {
		t.Fatalf("SimulateExpertCache error = %v", err)
	}
	if trace.Hits != original.Hits {
		t.Fatalf("adapter trace.Hits = %d, SimulateExpertCache.Hits = %d", trace.Hits, original.Hits)
	}
	if trace.PageIns != original.PageIns {
		t.Fatalf("adapter trace.PageIns = %d, SimulateExpertCache.PageIns = %d", trace.PageIns, original.PageIns)
	}
	if trace.Evictions != original.Evictions {
		t.Fatalf("adapter trace.Evictions = %d, SimulateExpertCache.Evictions = %d", trace.Evictions, original.Evictions)
	}
	if trace.PeakResident != original.PeakResident {
		t.Fatalf("adapter trace.PeakResident = %d, SimulateExpertCache.PeakResident = %d", trace.PeakResident, original.PeakResident)
	}
	if len(trace.FinalResident) != len(original.FinalResident) {
		t.Fatalf("adapter len(FinalResident) = %d, SimulateExpertCache len(FinalResident) = %d",
			len(trace.FinalResident), len(original.FinalResident))
	}
	for i := range trace.FinalResident {
		if trace.FinalResident[i] != original.FinalResident[i] {
			t.Fatalf("adapter FinalResident[%d] = %+v, SimulateExpertCache FinalResident[%d] = %+v",
				i, trace.FinalResident[i], i, original.FinalResident[i])
		}
	}

	if len(trace.FinalResident) > capacity {
		t.Fatalf("len(FinalResident) = %d, want <= capacity %d", len(trace.FinalResident), capacity)
	}
	if trace.PeakResident > capacity {
		t.Fatalf("PeakResident = %d, want <= capacity %d", trace.PeakResident, capacity)
	}
	seen := make(map[ExpertGroup]struct{}, len(trace.FinalResident))
	for i, g := range trace.FinalResident {
		if _, dup := seen[g]; dup {
			t.Fatalf("FinalResident has duplicate %+v", g)
		}
		seen[g] = struct{}{}
		if i > 0 && !groupLess(trace.FinalResident[i-1], g) {
			t.Fatalf("FinalResident not ascending at %d: %+v then %+v",
				i, trace.FinalResident[i-1], g)
		}
	}

	// Fail-closed matrix.
	good := []ExpertRoute{{Layer: 0, Experts: []int{0, 1, 2}}}
	cases := []struct {
		name    string
		routes  []ExpertRoute
		layers  int
		experts int
		topK    int
		cap     int
		want    error
	}{
		{"zero capacity", good, 1, 4, 3, 0, ErrInvalidTraceCapacity},
		{"negative shape layers", good, 0, 4, 3, 2, ErrInvalidTraceShape},
		{"negative shape experts", good, 1, 0, 3, 2, ErrInvalidTraceShape},
		{"top-k exceeding experts", good, 1, 4, 5, 2, ErrInvalidTraceShape},
		{"wrong top-k width", []ExpertRoute{{Layer: 0, Experts: []int{0, 1}}}, 1, 4, 3, 2, ErrInvalidTraceRoute},
		{"layer out of range", []ExpertRoute{{Layer: 5, Experts: []int{0, 1, 2}}}, 1, 4, 3, 2, ErrInvalidTraceRoute},
		{"expert out of range", []ExpertRoute{{Layer: 0, Experts: []int{0, 1, 4}}}, 1, 4, 3, 2, ErrInvalidTraceRoute},
		{"duplicate expert", []ExpertRoute{{Layer: 0, Experts: []int{1, 1, 2}}}, 1, 4, 3, 2, ErrDuplicateTraceExpert},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := SimulateExpertCacheTrace(tc.routes, tc.layers, tc.experts, tc.topK, tc.cap); !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestCacheTraceScoredArmsAreLive is the fak-private#3342 DS4.1 P6 liveness witness for
// score-aware eviction. It deterministically enumerates a bounded small-sequence space
// (alphabet {0,1,2,3}, lengths 6..9, capacities {2,3}; no time, no randomness) and makes
// two claims:
//
//  1. LFRU is live: at least one stream exists where lfru.Hits != lru.Hits.
//  2. ScoreEMA is hit-equivalent to LRU in this space: every stream searched must have
//     scoreema.Hits == lru.Hits. With alpha=0.5 and a unit access indicator, the uniform
//     decay is a positive affine scaling that preserves the argmin, and the touched group
//     resets to 0.5 regardless of prior value, so the victim order is exactly recency
//     order -- i.e. hit-equivalent to LRU. A single counterexample would refute that
//     equivalence and is reported as a hard failure rather than silently weakening the
//     check.
//
// fak-test:runtime medium est=6s lane=default
func TestCacheTraceScoredArmsAreLive(t *testing.T) {
	const (
		alphabet = 4
		minLen   = 6
		maxLen   = 9
	)
	capacities := []int{2, 3}
	policies := append(CachePolicyScoredArms(), CachePolicyArms()...)

	var witnessSeq []int
	var witnessCap int
	var wLRU, wLFRU, wHeat, wScoreEMA int
	searched := 0

	// Fixed enumeration order: length, then capacity, then lexicographic sequence.
	for length := minLen; length <= maxLen; length++ {
		for _, capacity := range capacities {
			seq := make([]int, length)
			var walk func(pos int)
			walk = func(pos int) {
				if pos == length {
					searched++
					groups := make([]ExpertGroup, length)
					for i, e := range seq {
						groups[i] = ExpertGroup{Layer: 0, Expert: e}
					}
					stream := CachePolicyStream{Name: "scored-live-search", Groups: groups}
					receipt, err := CachePolicyReplayWith(stream, capacity, policies)
					if err != nil {
						t.Fatalf("CachePolicyReplayWith error = %v", err)
					}
					lru := scoredTestRow(t, receipt, CachePolicyLRU)
					lfru := scoredTestRow(t, receipt, CachePolicyLFRU)
					heat := scoredTestRow(t, receipt, CachePolicyHeat)
					scoreEMA := scoredTestRow(t, receipt, CachePolicyScoreEMA)

					// The authoritative finding for fak-private#3342: ScoreEMA with alpha=0.5 and
					// a unit access indicator is hit-equivalent to LRU. Any difference refutes it
					// and must be reported, not weakened.
					if scoreEMA.Hits != lru.Hits {
						t.Fatalf("ScoreEMA/LRU equivalence refuted: cap=%d seq=%v lru=%d scoreema=%d; alpha=0.5 uniform decay should preserve the LRU argmin. Report this counterexample; do not weaken the assertion.",
							capacity, seq, lru.Hits, scoreEMA.Hits)
					}

					if witnessSeq == nil && lfru.Hits != lru.Hits {
						witnessSeq = append([]int(nil), seq...)
						witnessCap = capacity
						wLRU, wLFRU, wHeat, wScoreEMA = lru.Hits, lfru.Hits, heat.Hits, scoreEMA.Hits
					}
					return
				}
				for e := 0; e < alphabet; e++ {
					seq[pos] = e
					walk(pos + 1)
				}
			}
			walk(0)
		}
	}

	if witnessSeq == nil {
		t.Fatalf("no LFRU liveness witness found in bounded search: alphabet {0,1,2,3}, lengths %d..%d, capacities %v (%d streams). LFRU may be an alias of LRU; do not weaken this test silently.",
			minLen, maxLen, capacities, searched)
	}
	if wLFRU == wLRU {
		t.Fatalf("pinned witness cap=%d seq=%v: LFRU Hits = %d equals LRU Hits; LFRU may be an alias",
			witnessCap, witnessSeq, wLFRU)
	}
	t.Logf("live-witness seq=%v cap=%d lru=%d lfru=%d heat=%d scoreema=%d searched=%d",
		witnessSeq, witnessCap, wLRU, wLFRU, wHeat, wScoreEMA, searched)
}

// TestScoreEMAVictimSelectionIsLive is a same-package white-box witness that the ScoreEMA
// victim picker reads the ScoreEMA score map rather than degenerating into LRU's
// residentEntry-only recency picker. It calls pickScoreEMAVictim directly with hand-built
// resident/score maps whose orderings deliberately contradict each other, so an
// implementation that ignored score would return the wrong group.
func TestScoreEMAVictimSelectionIsLive(t *testing.T) {
	groupA := ExpertGroup{Layer: 0, Expert: 0}
	groupB := ExpertGroup{Layer: 0, Expert: 1}
	groupC := ExpertGroup{Layer: 0, Expert: 2}

	// Case 1: recency order contradicts score order. A is the most recently used but has
	// the smallest score, so a score-reading picker returns A while an LRU-alias picker
	// would return C (oldest lastUse).
	resident := map[ExpertGroup]residentEntry{
		groupA: {group: groupA, lastUse: 30},
		groupB: {group: groupB, lastUse: 20},
		groupC: {group: groupC, lastUse: 10},
	}
	score := map[ExpertGroup]float64{
		groupA: 0.125,
		groupB: 0.875,
		groupC: 0.5,
	}
	if got := pickScoreEMAVictim(resident, score); got != groupA {
		t.Fatalf("pickScoreEMAVictim = %+v, want %+v (smallest score, newest lastUse); picker may be an LRU alias that ignores score",
			got, groupA)
	}

	// Case 2: tie on score; the older lastUse must win.
	tieScore := map[ExpertGroup]float64{
		groupA: 0.5,
		groupB: 0.5,
	}
	tieResident := map[ExpertGroup]residentEntry{
		groupA: {group: groupA, lastUse: 5},
		groupB: {group: groupB, lastUse: 9},
	}
	if got := pickScoreEMAVictim(tieResident, tieScore); got != groupA {
		t.Fatalf("pickScoreEMAVictim = %+v, want %+v (score tie -> older lastUse)", got, groupA)
	}

	// Case 3: score and lastUse both tie; groupLess must break it (smaller group wins).
	// groupLess orders by Layer then Expert, so {0,0} beats {0,1}.
	allTie := map[ExpertGroup]float64{
		groupB: 0.25,
		groupA: 0.25,
	}
	allTieResident := map[ExpertGroup]residentEntry{
		groupB: {group: groupB, lastUse: 7},
		groupA: {group: groupA, lastUse: 7},
	}
	if got := pickScoreEMAVictim(allTieResident, allTie); got != groupA {
		t.Fatalf("pickScoreEMAVictim = %+v, want %+v (score+lastUse tie -> groupLess)", got, groupA)
	}

	// The online ScoreEMA replay path must account for every access.
	pinned := []ExpertGroup{groupA, groupB, groupA}
	hits, misses, evictions := replayOnlineScoreEMA(pinned, 2)
	if hits+misses != len(pinned) {
		t.Fatalf("replayOnlineScoreEMA Hits(%d)+Misses(%d) = %d, want accesses %d",
			hits, misses, hits+misses, len(pinned))
	}
	if evictions != 0 {
		t.Fatalf("replayOnlineScoreEMA Evictions = %d on capacity-2 stream of 2 distinct groups, want 0", evictions)
	}
}

// TestCacheTracePolicySimReport is the fak-private#3342 DS4.1 P6 score-aware-eviction offline
// sim witness: it emits the V4.1 geometry scan table as evidence. The numbers are diagnostic;
// only the row invariants and the OPT bound are asserted.
// fak-test:runtime medium est=25s lane=default
func TestCacheTracePolicySimReport(t *testing.T) {
	stream, err := SyntheticCachePolicyStream("v41-halo-zipf", 3342, 40, 384, 6, 4000)
	if err != nil {
		t.Fatalf("SyntheticCachePolicyStream error = %v", err)
	}
	policies := append(CachePolicyScoredArms(), CachePolicyArms()...)
	groupBytes := V41ExpertGroupBytes(0.5)

	t.Logf("V4.1 geometry: 40 layers x 384 experts, topK 6, 4000 accesses; FP4 group = %d bytes (16.875 MiB)", groupBytes)
	t.Logf("| capacity | ring_bytes | lru | lfu | heat | random | lfru | scoreema | opt | OPTExact |")
	t.Logf("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
	for _, capacity := range []int{4, 8, 16, 32, 64, 128, 256, 512} {
		receipt, err := CachePolicyReplayWith(stream, capacity, policies)
		if err != nil {
			t.Fatalf("capacity %d: CachePolicyReplayWith error = %v", capacity, err)
		}
		rate := func(p CachePolicy) string {
			row := scoredTestRow(t, receipt, p)
			return formatRate(row)
		}
		t.Logf("| %d | %d | %s | %s | %s | %s | %s | %s | %s | %v |",
			capacity, int64(capacity)*groupBytes,
			rate(CachePolicyLRU), rate(CachePolicyLFU), rate(CachePolicyHeat), rate(CachePolicyRandom),
			rate(CachePolicyLFRU), rate(CachePolicyScoreEMA), rate(CachePolicyOPT), receipt.OPTExact)
		for _, row := range receipt.Rows {
			scoredTestAssertRowInvariants(t, row, "v41 report")
			// The comment above claims the OPT bound is asserted; assert it here using the
			// rows already produced (no extra replay). CachePolicyReplayWith always computes
			// the Belady oracle, so VersusOPT is a real ratio, not the zero placeholder from
			// CachePolicyReplayOnline.
			if row.Policy == CachePolicyOPT {
				continue
			}
			if row.VersusOPT > 1.0+scoredTestEpsilon {
				t.Fatalf("v41 report capacity %d policy %s: VersusOPT = %v, want <= 1.0 (OPT bound)",
					capacity, row.Policy, row.VersusOPT)
			}
		}
	}
}

// formatRate renders a row's hit rate compactly for the markdown table.
func formatRate(row CachePolicyRow) string {
	if !row.HitRateKnown {
		return "n/a"
	}
	// Three decimal places, rounded, without importing strconv/str formatting elsewhere.
	scaled := int64(math.Round(row.HitRate * 1000)) // e.g. 12345 -> 12.345
	whole := scaled / 1000
	frac := scaled % 1000
	return itoaTest(int(whole)) + "." + pad3Test(frac)
}

// pad3Test formats a 0..999 value as exactly three digits.
func pad3Test(v int64) string {
	s := itoaTest(int(v))
	for len(s) < 3 {
		s = "0" + s
	}
	return s
}
