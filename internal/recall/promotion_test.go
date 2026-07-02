package recall

import "testing"

func TestRecallPromotionCandidatesSurfaceFreshWitnessedMemory(t *testing.T) {
	events := []RecallPromotionEvent{
		{MemoryID: "scar:windows-go-test", AgentStore: "agent-a", TriggerContext: "go test refused", Verdict: "recall fresh", Witness: "dos-recall:a"},
		{MemoryID: "scar:windows-go-test", AgentStore: "agent-b", TriggerContext: "native test blocked", Verdict: "RECALL_FRESH", Witness: "dos-recall:b"},
		{MemoryID: "scar:windows-go-test", AgentStore: "agent-a", TriggerContext: "use WSL", Verdict: "recall-fresh", Witness: "dos-recall:c"},
		{MemoryID: "scar:other", Verdict: "RECALL_FRESH", Witness: "dos-recall:x"},
	}

	got := RecallPromotionCandidates(events, PromotionPolicy{MinFreshRecalls: 3})
	if len(got) != 1 {
		t.Fatalf("candidates len = %d, want 1: %+v", len(got), got)
	}
	c := got[0]
	if !c.Ready {
		t.Fatalf("candidate not ready: %+v", c)
	}
	if c.MemoryID != "scar:windows-go-test" || c.FreshRecalls != 3 {
		t.Fatalf("candidate = %+v, want memory scar with 3 fresh recalls", c)
	}
	if joined(c.AgentStores) != "agent-a,agent-b" {
		t.Fatalf("stores = %v, want sorted unique stores", c.AgentStores)
	}
	if joined(c.Witnesses) != "dos-recall:a,dos-recall:b,dos-recall:c" {
		t.Fatalf("witnesses = %v, want sorted unique witnesses", c.Witnesses)
	}

	lesson, ok := c.FleetLesson()
	if !ok {
		t.Fatal("ready candidate did not produce fleet lesson row")
	}
	if lesson.Source != "recall-promotion" || lesson.MemoryID != c.MemoryID {
		t.Fatalf("lesson = %+v, want recall-promotion source for candidate memory", lesson)
	}
	if len(lesson.TriggerContext) != 3 {
		t.Fatalf("lesson trigger context = %v, want all unique triggers", lesson.TriggerContext)
	}
}

func TestRecallPromotionCandidatesIgnoreWitnesslessAndHoldLatestStale(t *testing.T) {
	events := []RecallPromotionEvent{
		{MemoryID: "memory-a", Verdict: "RECALL_FRESH", Witness: "w1"},
		{MemoryID: "memory-a", Verdict: "RECALL_FRESH"}, // witnessless: does not count
		{MemoryID: "memory-a", Verdict: "RECALL_FRESH", Witness: "w2"},
		{MemoryID: "memory-a", Verdict: "RECALL_FRESH", Witness: "w3"},
		{MemoryID: "memory-a", Verdict: "RECALL_STALE", Witness: "stale-witness"},
	}

	got := RecallPromotionCandidates(events, PromotionPolicy{MinFreshRecalls: 3})
	if len(got) != 1 {
		t.Fatalf("candidates len = %d, want held candidate crossing count: %+v", len(got), got)
	}
	if got[0].Ready {
		t.Fatalf("latest-stale memory must be held, got ready candidate: %+v", got[0])
	}
	if _, ok := got[0].FleetLesson(); ok {
		t.Fatalf("held candidate produced a fleet lesson row: %+v", got[0])
	}
	if got[0].FreshRecalls != 3 {
		t.Fatalf("fresh recalls = %d, want witnessless row ignored and 3 witnessed rows counted", got[0].FreshRecalls)
	}
}

func joined(values []string) string {
	out := ""
	for i, value := range values {
		if i > 0 {
			out += ","
		}
		out += value
	}
	return out
}
