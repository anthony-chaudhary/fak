package radixkv

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestValueFrequencyAdmissionRejectsOneShotScanBeforeEviction(t *testing.T) {
	tree := NewWithEvictionPolicy(20, EvictionCostAware)
	hotA := distinctReq(0, 10)
	hotB := distinctReq(1, 10)

	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, hotA)
	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, hotB)
	before := tree.Stats()

	for i := 0; i < 32; i++ {
		touchPure(tree, distinctReq(100+i, 10))
	}

	if got := tree.MatchLen(hotA); got != len(hotA) {
		t.Fatalf("hot A lost under one-shot scan: matched %d/%d", got, len(hotA))
	}
	if got := tree.MatchLen(hotB); got != len(hotB) {
		t.Fatalf("hot B lost under one-shot scan: matched %d/%d", got, len(hotB))
	}
	after := tree.Stats()
	if after.Evictions != before.Evictions {
		t.Fatalf("one-shot scan churned resident state: evictions %d -> %d", before.Evictions, after.Evictions)
	}
	if after.Tokens != 20 {
		t.Fatalf("resident tokens=%d, want bounded hot working set 20", after.Tokens)
	}
	if after.AdmissionCandidates != 32 || after.AdmissionRejected != 32 || after.AdmissionHotProtected != 32 {
		t.Fatalf("admission scan stats=%+v, want 32 candidates/rejects/hot protections", after)
	}
	if after.AdmissionRejectedTokens != 320 || after.AdmissionRejectedBytes != 320 {
		t.Fatalf("avoided token/byte writes=%d/%d, want 320/320", after.AdmissionRejectedTokens, after.AdmissionRejectedBytes)
	}
}

func TestValueFrequencyAdmissionPreservesRepresentativeAgentPrefixes(t *testing.T) {
	const sharedTokens = 8
	tree := NewWithEvictionPolicy(16, EvictionCostAware)
	shared := seq(1, sharedTokens)
	hotA := cat(shared, distinctReq(1, 4))
	hotB := cat(shared, distinctReq(2, 4))

	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, hotA)
	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, hotB)
	before := tree.Stats()

	for i := 0; i < 24; i++ {
		req := cat(shared, distinctReq(100+i, 4))
		if got := tree.MatchLen(req); got != sharedTokens {
			t.Fatalf("scan request %d matched %d, want only shared agent prefix %d", i, got, sharedTokens)
		}
		touchPure(tree, req)
	}

	if got := tree.MatchLen(hotA); got != len(hotA) {
		t.Fatalf("agent prefix A lost under sibling scan: matched %d/%d", got, len(hotA))
	}
	if got := tree.MatchLen(hotB); got != len(hotB) {
		t.Fatalf("agent prefix B lost under sibling scan: matched %d/%d", got, len(hotB))
	}
	after := tree.Stats()
	if after.Evictions != before.Evictions {
		t.Fatalf("agent-prefix scan churned resident state: evictions %d -> %d", before.Evictions, after.Evictions)
	}
	if after.Tokens != 16 {
		t.Fatalf("resident tokens=%d, want bounded shared+hot-tail working set 16", after.Tokens)
	}
	if after.AdmissionCandidates != 24 || after.AdmissionRejected != 24 {
		t.Fatalf("agent-prefix admission candidates/rejects=%d/%d, want 24/24",
			after.AdmissionCandidates, after.AdmissionRejected)
	}
	if after.AdmissionRejectedTokens != 96 {
		t.Fatalf("agent-prefix avoided suffix writes=%d tokens, want 96", after.AdmissionRejectedTokens)
	}
}

func TestValueFrequencyAdmissionJournalsRejectsAndRecovery(t *testing.T) {
	tree := NewWithEvictionPolicy(20, EvictionCostAware)
	hotA := distinctReq(0, 10)
	hotB := distinctReq(1, 10)
	candidate := distinctReq(50, 10)

	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, hotA)
	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, hotB)

	for attempt := 1; attempt <= 3; attempt++ {
		touchPure(tree, candidate)
		if got := tree.MatchLen(candidate); got != 0 {
			t.Fatalf("attempt %d admitted candidate too early: matched %d/%d", attempt, got, len(candidate))
		}
	}
	rejected := tree.Stats()
	if rejected.AdmissionRejected != 3 || rejected.AdmissionJournalPending != 1 || rejected.AdmissionRecoveries != 0 {
		t.Fatalf("rejected journal stats=%+v, want 3 rejects / 1 pending / 0 recoveries", rejected)
	}
	if rejected.LastAdmissionFrequency != 3 || rejected.LastAdmissionReason != "colder_than_victim" {
		t.Fatalf("third decision=%d/%q, want frequency 3 / colder_than_victim",
			rejected.LastAdmissionFrequency, rejected.LastAdmissionReason)
	}

	touchPure(tree, candidate) // frequency 4: value 4 now outranks either hit-2 victim's value 3.
	if got := tree.MatchLen(candidate); got != len(candidate) {
		t.Fatalf("recovered candidate matched %d/%d", got, len(candidate))
	}
	recovered := tree.Stats()
	if recovered.AdmissionAdmitted != 1 || recovered.AdmissionRecoveries != 1 || recovered.AdmissionJournalPending != 0 {
		t.Fatalf("recovery stats=%+v, want 1 admitted/recovered and no pending journal row", recovered)
	}
	if recovered.AdmissionRecoveryGapLast == 0 || recovered.AdmissionRecoveryGapMax != recovered.AdmissionRecoveryGapLast {
		t.Fatalf("recovery gaps last/max=%d/%d, want one non-zero deterministic gap",
			recovered.AdmissionRecoveryGapLast, recovered.AdmissionRecoveryGapMax)
	}
	if recovered.LastAdmissionFrequency != 4 || recovered.LastAdmissionReason != "outranks_victim" {
		t.Fatalf("recovery decision=%d/%q, want frequency 4 / outranks_victim",
			recovered.LastAdmissionFrequency, recovered.LastAdmissionReason)
	}
}

func TestValueFrequencyAdmissionStateIsBounded(t *testing.T) {
	tree := NewWithEvictionPolicy(20, EvictionCostAware)
	hotA := distinctReq(0, 10)
	hotB := distinctReq(1, 10)
	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, hotA)
	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, hotB)

	const candidates = 3 * admissionJournalCap
	for i := 0; i < candidates; i++ {
		touchPure(tree, distinctReq(1000+i, 10))
	}
	st := tree.Stats()
	if st.AdmissionSketchCells != admissionSketchWidth*admissionSketchDepth {
		t.Fatalf("sketch cells=%d, want fixed %d", st.AdmissionSketchCells, admissionSketchWidth*admissionSketchDepth)
	}
	if st.AdmissionJournalPending != admissionJournalCap {
		t.Fatalf("pending journal=%d, want hard cap %d", st.AdmissionJournalPending, admissionJournalCap)
	}
	if st.AdmissionJournalDropped != candidates-admissionJournalCap {
		t.Fatalf("dropped journal rows=%d, want %d", st.AdmissionJournalDropped, candidates-admissionJournalCap)
	}
	if st.AdmissionCandidates != candidates || st.AdmissionRejected != candidates {
		t.Fatalf("decision work=%d candidates/%d rejects, want %d/%d",
			st.AdmissionCandidates, st.AdmissionRejected, candidates, candidates)
	}
	if st.AdmissionObservations < candidates {
		t.Fatalf("observations=%d, want at least %d demand touches accounted", st.AdmissionObservations, candidates)
	}
	if st.Tokens != 20 || st.Evictions != 0 {
		t.Fatalf("resident state churned under bounded journal pressure: tokens=%d evictions=%d", st.Tokens, st.Evictions)
	}
}

func TestValueFrequencyAdmissionRejectsOversizedCandidate(t *testing.T) {
	tree := NewWithEvictionPolicy(10, EvictionCostAware)
	oversized := distinctReq(0, 20)
	touchPure(tree, oversized)
	st := tree.Stats()
	if st.Tokens != 0 || st.Nodes != 0 || st.Fills != 0 {
		t.Fatalf("oversized candidate mutated bounded tree: tokens=%d nodes=%d fills=%d", st.Tokens, st.Nodes, st.Fills)
	}
	if st.AdmissionRejected != 1 || st.LastAdmissionReason != admissionReasonInsufficientCapacity {
		t.Fatalf("oversized admission stats=%+v, want one insufficient-capacity rejection", st)
	}
}

func TestValueFrequencyAdmissionComparesEveryDisplacedVictim(t *testing.T) {
	tree := NewWithEvictionPolicy(30, EvictionCostAware)
	cold := distinctReq(0, 10)
	hotA := distinctReq(1, 10)
	hotB := distinctReq(2, 10)
	candidate := distinctReq(50, 20)
	touchPure(tree, cold)
	touchPure(tree, hotA)
	touchPure(tree, hotB)
	for i := 0; i < 5; i++ {
		touchPure(tree, hotA)
		touchPure(tree, hotB)
	}

	touchPure(tree, candidate) // frequency 1: rejected before value comparison
	touchPure(tree, candidate) // frequency 2: beats cold, but not either hit-5 hot victim
	if got := tree.MatchLen(candidate); got != 0 {
		t.Fatalf("multi-victim candidate admitted after comparing only the cold victim: matched %d", got)
	}
	for name, req := range map[string][]int{"cold": cold, "hotA": hotA, "hotB": hotB} {
		if got := tree.MatchLen(req); got != len(req) {
			t.Fatalf("%s resident lost to rejected multi-victim candidate: matched %d/%d", name, got, len(req))
		}
	}
	st := tree.Stats()
	if st.Tokens != 30 || st.Evictions != 0 || st.AdmissionRejected != 2 {
		t.Fatalf("multi-victim admission stats=%+v, want bounded 30 tokens / 0 evictions / 2 rejects", st)
	}
	if st.LastAdmissionReason != "colder_than_victim" {
		t.Fatalf("second multi-victim reason=%q, want colder_than_victim", st.LastAdmissionReason)
	}
}

func TestValueFrequencyAdmissionTelemetryAbsentFallbackIsDeterministic(t *testing.T) {
	run := func() (Stats, [3]int) {
		tree := NewWithEvictionPolicy(20, EvictionCostAware)
		hotA := distinctReq(0, 10)
		hotB := distinctReq(1, 10)
		candidate := distinctReq(2, 10)
		touchPure(tree, hotA)
		touchPure(tree, hotB)
		touchPure(tree, hotA)
		touchPure(tree, hotA)
		touchPure(tree, hotB)
		touchPure(tree, hotB)

		boundary, matched := tree.Lookup(candidate)
		tree.admissionSketch = nil // deterministic telemetry outage after the demand probe
		leaf := tree.Insert(boundary, candidate[matched:], nil)
		tree.Done(leaf)
		return tree.Stats(), [3]int{tree.MatchLen(hotA), tree.MatchLen(hotB), tree.MatchLen(candidate)}
	}

	firstStats, firstResident := run()
	secondStats, secondResident := run()
	if !reflect.DeepEqual(firstStats, secondStats) || firstResident != secondResident {
		t.Fatalf("telemetry-absent fallback drifted:\nfirst=%+v resident=%v\nsecond=%+v resident=%v",
			firstStats, firstResident, secondStats, secondResident)
	}
	if firstStats.AdmissionTelemetryFallbacks != 1 || firstStats.AdmissionAdmitted != 1 {
		t.Fatalf("fallback stats=%+v, want one fail-open admission", firstStats)
	}
	if firstResident[2] != 10 || firstStats.Evictions != 1 {
		t.Fatalf("fallback did not reproduce insert-always: resident=%v evictions=%d", firstResident, firstStats.Evictions)
	}
}

func TestValueFrequencyAdmissionRollbackRestoresInsertAlways(t *testing.T) {
	tree := NewWithEvictionPolicy(20, EvictionCostAware)
	tree.SetAdmissionEnabled(false)
	hotA := distinctReq(0, 10)
	hotB := distinctReq(1, 10)
	candidate := distinctReq(2, 10)
	touchPure(tree, hotA)
	touchPure(tree, hotB)
	touchPure(tree, candidate)

	st := tree.Stats()
	if st.AdmissionEnabled || st.AdmissionCandidates != 0 || st.AdmissionSketchCells != 0 {
		t.Fatalf("rollback left admission work armed: %+v", st)
	}
	if tree.MatchLen(candidate) != len(candidate) || st.Evictions != 1 {
		t.Fatalf("rollback did not restore insert-always: candidate=%d/%d evictions=%d",
			tree.MatchLen(candidate), len(candidate), st.Evictions)
	}
}

func TestValueFrequencyAdmissionHeatIsNamespaceScoped(t *testing.T) {
	tree := NewWithEvictionPolicy(10, EvictionCostAware)
	req := distinctReq(0, 10)
	for i := 0; i < 4; i++ {
		_, leaf := serveNS(tree, "tenant-A", req)
		tree.Done(leaf)
	}

	_, leaf := serveNS(tree, "tenant-B", req)
	tree.Done(leaf)
	if got := tree.MatchLenNS("tenant-A", req); got != len(req) {
		t.Fatalf("tenant-A hot prefix lost: matched %d/%d", got, len(req))
	}
	if got := tree.MatchLenNS("tenant-B", req); got != 0 {
		t.Fatalf("tenant-B inherited tenant-A heat: matched %d/%d", got, len(req))
	}
	if tree.Namespaces() != 1 {
		t.Fatalf("empty rejected namespace root leaked: Namespaces=%d, want only tenant-A", tree.Namespaces())
	}
	if st := tree.Stats(); st.AdmissionRejected != 1 || st.LastAdmissionFrequency != 1 {
		t.Fatalf("namespace-scoped admission stats=%+v, want cold tenant-B frequency 1 rejection", st)
	}
}

func TestValueFrequencyAdmissionGatesSnapshotOwnershipPath(t *testing.T) {
	tree := NewWithTierBudgetsAndEvictionPolicy(1000, 64, 0, EvictionCostAware)
	logits := make([]float32, 16)
	hotIDs := []int{1}
	candidateIDs := []int{2}

	hot := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
	boundary, matched := tree.Lookup(hotIDs)
	leaf, err := tree.InsertSnapshot(boundary, hotIDs[matched:], hot, logits)
	if err != nil {
		t.Fatal(err)
	}
	tree.Done(leaf)
	for i := 0; i < 2; i++ {
		n, snap, got, err := tree.LookupSnapshot(hotIDs)
		if err != nil {
			t.Fatal(err)
		}
		if snap == nil || got != len(hotIDs) {
			t.Fatalf("hot snapshot lookup %d: snapshot=%v matched=%d", i, snap, got)
		}
		snap.Close()
		tree.Done(n)
	}

	for attempt := 1; attempt <= 3; attempt++ {
		candidate := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
		boundary, matched = tree.Lookup(candidateIDs)
		leaf, err = tree.InsertSnapshot(boundary, candidateIDs[matched:], candidate, logits)
		if err != nil {
			t.Fatal(err)
		}
		tree.Done(leaf)
		if candidate.Cache != nil {
			t.Fatalf("bypassed snapshot attempt %d retained caller/device ownership", attempt)
		}
		if got := tree.MatchLen(candidateIDs); got != 0 {
			t.Fatalf("snapshot candidate admitted too early on attempt %d: matched %d", attempt, got)
		}
	}

	recovered := model.NewHostPrefixSnapshotForTest(model.NewKVCache(model.Config{}))
	boundary, matched = tree.Lookup(candidateIDs)
	leaf, err = tree.InsertSnapshot(boundary, candidateIDs[matched:], recovered, logits)
	if err != nil {
		t.Fatal(err)
	}
	tree.Done(leaf)
	n, snap, got, err := tree.LookupSnapshot(candidateIDs)
	if err != nil {
		t.Fatal(err)
	}
	if snap == nil || got != len(candidateIDs) {
		t.Fatalf("recovered snapshot=%v matched=%d, want full candidate", snap, got)
	}
	snap.Close()
	tree.Done(n)
	st := tree.Stats()
	if st.SnapshotBytes != 64 || st.AdmissionRejected != 3 || st.AdmissionRecoveries != 1 {
		t.Fatalf("snapshot admission stats=%+v, want 64 bytes / 3 rejects / 1 recovery", st)
	}
}

func TestValueFrequencyAdmissionBypassDoesNotChangeModelOutput(t *testing.T) {
	m := newSyntheticTiny()
	tree := NewWithEvictionPolicy(8, EvictionCostAware)
	hot := seq(1, 8)
	candidate := seq(20, 8)

	hotSession := m.NewSession()
	hotLogits := hotSession.Prefill(hot)
	boundary, matched := tree.Lookup(hot)
	leaf := tree.InsertWithLogits(boundary, hot[matched:], hotSession.Cache.Clone(), hotLogits)
	tree.Done(leaf)
	touchPure(tree, hot)
	touchPure(tree, hot)

	boundary, matched = tree.Lookup(candidate)
	if matched != 0 {
		t.Fatalf("candidate unexpectedly matched %d tokens", matched)
	}
	request := m.NewSession()
	gotLogits := request.Prefill(candidate)
	callerOwned := request.Cache.Clone()
	leaf = tree.InsertWithLogits(boundary, candidate, callerOwned, gotLogits)
	if leaf != boundary {
		t.Fatal("bypass must return the still-leased lookup boundary")
	}
	tree.Done(leaf)

	fresh := m.NewSession()
	wantLogits := fresh.Prefill(candidate)
	if got, want := argmax(gotLogits), argmax(wantLogits); got != want {
		t.Fatalf("bypass argmax=%d, fresh recompute=%d", got, want)
	}
	if diff := maxAbsDiff(gotLogits, wantLogits); diff != 0 {
		t.Fatalf("bypass changed logits: max|delta|=%g", diff)
	}
	if callerOwned.Len() != len(candidate) {
		t.Fatalf("caller-owned KV length=%d, want unchanged %d", callerOwned.Len(), len(candidate))
	}
	if tree.MatchLen(candidate) != 0 || tree.MatchLen(hot) != len(hot) {
		t.Fatalf("bypass residency drift: candidate=%d hot=%d/%d",
			tree.MatchLen(candidate), tree.MatchLen(hot), len(hot))
	}
}
