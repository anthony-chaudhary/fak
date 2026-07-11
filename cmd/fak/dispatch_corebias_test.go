package main

import "testing"

// The tests below pin the core-source ordering bias on the two LIVE dispatch entry points
// (the wave pricer's order stamp and the single-tick comparator): under the default
// throughput goal a trunk-safe core lane must outrank the coarse docs/tools buckets
// regardless of step budget, while under high-priority the priority label still leads and
// core-ness only breaks ties. This is what stops the unattended loop from churning
// docs/tools while fragmented per-leaf core work starves behind the big buckets.

func TestDispatchWaveOrderStampPrefersCoreUnderThroughput(t *testing.T) {
	coreSmall := dispatchWaveOrderStamp(dispatchGoalProfileThroughput, 0, 1, true)
	docsBig := dispatchWaveOrderStamp(dispatchGoalProfileThroughput, 0, 999, false)
	if !(coreSmall > docsBig) {
		t.Fatalf("throughput: core(step=1) stamp %d should outrank docs(step=999) stamp %d", coreSmall, docsBig)
	}
	// Within the core class, richer step budget still leads (richest-first is preserved).
	coreBig := dispatchWaveOrderStamp(dispatchGoalProfileThroughput, 0, 5, true)
	if !(coreBig > coreSmall) {
		t.Fatalf("throughput: core(step=5) %d should outrank core(step=1) %d", coreBig, coreSmall)
	}
}

func TestDispatchWaveOrderStampPriorityLeadsUnderHighPriority(t *testing.T) {
	// A high-priority docs lane must still outrank a default-priority core lane.
	docsHi := dispatchWaveOrderStamp(dispatchGoalProfileHighPriority, 100, 0, false)
	coreDefault := dispatchWaveOrderStamp(dispatchGoalProfileHighPriority, 1, 0, true)
	if !(docsHi > coreDefault) {
		t.Fatalf("high-priority: docs(prio=100) %d should outrank core(prio=1) %d", docsHi, coreDefault)
	}
	// At equal priority, core-source breaks the tie.
	coreTie := dispatchWaveOrderStamp(dispatchGoalProfileHighPriority, 50, 0, true)
	docsTie := dispatchWaveOrderStamp(dispatchGoalProfileHighPriority, 50, 0, false)
	if !(coreTie > docsTie) {
		t.Fatalf("high-priority tie: core %d should outrank docs %d at equal priority", coreTie, docsTie)
	}
}

func TestDispatchLaneBetterForGoalPrefersCoreUnderThroughput(t *testing.T) {
	// Candidate = trunk-safe core lane with a tiny budget; incumbent best = big docs lane.
	better := dispatchLaneBetterForGoal(dispatchGoalProfileThroughput,
		0, 1, 1, true, "gateway",
		0, 999, 50, false, "docs")
	if !better {
		t.Fatal("throughput: a core lane should beat a richer docs lane")
	}
	// The reverse: a big docs lane must NOT displace an already-chosen core lane.
	worse := dispatchLaneBetterForGoal(dispatchGoalProfileThroughput,
		0, 999, 50, false, "docs",
		0, 1, 1, true, "gateway")
	if worse {
		t.Fatal("throughput: a docs lane should not displace a chosen core lane")
	}
}

func TestDispatchLaneBetterForGoalPriorityLeadsUnderHighPriority(t *testing.T) {
	// Under high-priority a higher-priority docs lane beats a default-priority core lane.
	better := dispatchLaneBetterForGoal(dispatchGoalProfileHighPriority,
		100, 1, 1, false, "docs",
		1, 1, 1, true, "gateway")
	if !better {
		t.Fatal("high-priority: a higher-priority docs lane should beat a default-priority core lane")
	}
	// At equal priority, core-source breaks the tie in the shared tail.
	tie := dispatchLaneBetterForGoal(dispatchGoalProfileHighPriority,
		50, 1, 1, true, "gateway",
		50, 1, 1, false, "docs")
	if !tie {
		t.Fatal("high-priority tie: a core lane should beat a docs lane at equal priority")
	}
}
