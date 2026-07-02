package recall

import (
	"sort"
	"strings"
)

const (
	// RecallFresh is the only verdict that can count toward fleet promotion.
	RecallFresh = "RECALL_FRESH"
	// DefaultPromotionMinFreshRecalls is the conservative auto-propose threshold.
	DefaultPromotionMinFreshRecalls = 3
)

// RecallPromotionEvent is the durable read-back of one per-agent memory recall.
// Fresh recalls must be independently re-verified, for example by dos_recall.
type RecallPromotionEvent struct {
	MemoryID       string `json:"memory_id"`
	AgentStore     string `json:"agent_store,omitempty"`
	TriggerContext string `json:"trigger_context,omitempty"`
	Verdict        string `json:"verdict"`
	Witness        string `json:"witness,omitempty"`
}

// PromotionPolicy controls when a per-agent memory is worth surfacing to the
// fleet lessons ledger.
type PromotionPolicy struct {
	MinFreshRecalls int `json:"min_fresh_recalls"`
}

// FleetLessonCandidate is the operator-facing proposal. It is intentionally a
// proposal, not a silent write: callers may auto-promote it or require human
// confirmation before appending to the fleet lessons ledger.
type FleetLessonCandidate struct {
	MemoryID       string   `json:"memory_id"`
	FreshRecalls   int      `json:"fresh_recalls"`
	AgentStores    []string `json:"agent_stores,omitempty"`
	TriggerContext []string `json:"trigger_context,omitempty"`
	Witnesses      []string `json:"witnesses"`
	Ready          bool     `json:"ready"`
	Reason         string   `json:"reason"`
}

// FleetLessonEntry is the ledger row a ready candidate can become after the
// operator or caller accepts the promotion.
type FleetLessonEntry struct {
	MemoryID       string   `json:"memory_id"`
	Source         string   `json:"source"`
	TriggerContext []string `json:"trigger_context,omitempty"`
	Witnesses      []string `json:"witnesses"`
}

// RecallPromotionCandidates folds recall telemetry into promotion proposals. A
// memory is ready only when it crossed the fresh recall threshold and its latest
// observed verdict is still RECALL_FRESH. Witnessless fresh-looking events do not
// count, because promotion must not rest on a memory's own claim.
func RecallPromotionCandidates(events []RecallPromotionEvent, policy PromotionPolicy) []FleetLessonCandidate {
	minFresh := policy.MinFreshRecalls
	if minFresh <= 0 {
		minFresh = DefaultPromotionMinFreshRecalls
	}

	byMemory := map[string]*promotionFold{}
	order := []string{}
	for _, event := range events {
		id := normalizeMemoryID(event.MemoryID)
		if id == "" {
			continue
		}
		fold := byMemory[id]
		if fold == nil {
			fold = &promotionFold{
				memoryID:  id,
				stores:    map[string]bool{},
				triggers:  map[string]bool{},
				witnesses: map[string]bool{},
			}
			byMemory[id] = fold
			order = append(order, id)
		}
		fold.latestVerdict = normalizeRecallVerdict(event.Verdict)
		if s := strings.TrimSpace(event.AgentStore); s != "" {
			fold.stores[s] = true
		}
		if t := strings.TrimSpace(event.TriggerContext); t != "" {
			fold.triggers[t] = true
		}
		witness := strings.TrimSpace(event.Witness)
		if fold.latestVerdict == RecallFresh && witness != "" {
			fold.freshRecalls++
			fold.witnesses[witness] = true
		}
	}

	var out []FleetLessonCandidate
	for _, id := range order {
		fold := byMemory[id]
		if fold.freshRecalls < minFresh {
			continue
		}
		candidate := FleetLessonCandidate{
			MemoryID:       fold.memoryID,
			FreshRecalls:   fold.freshRecalls,
			AgentStores:    sortedKeys(fold.stores),
			TriggerContext: sortedKeys(fold.triggers),
			Witnesses:      sortedKeys(fold.witnesses),
		}
		if fold.latestVerdict == RecallFresh {
			candidate.Ready = true
			candidate.Reason = "ready: memory crossed fresh recall threshold and latest recall is still verified fresh"
		} else {
			candidate.Reason = "held: memory crossed threshold but latest recall is no longer verified fresh"
		}
		out = append(out, candidate)
	}
	return out
}

// FleetLesson returns the accepted ledger row for a ready candidate.
func (c FleetLessonCandidate) FleetLesson() (FleetLessonEntry, bool) {
	if !c.Ready {
		return FleetLessonEntry{}, false
	}
	return FleetLessonEntry{
		MemoryID:       c.MemoryID,
		Source:         "recall-promotion",
		TriggerContext: append([]string(nil), c.TriggerContext...),
		Witnesses:      append([]string(nil), c.Witnesses...),
	}, true
}

type promotionFold struct {
	memoryID      string
	freshRecalls  int
	latestVerdict string
	stores        map[string]bool
	triggers      map[string]bool
	witnesses     map[string]bool
}

func normalizeMemoryID(id string) string {
	return strings.TrimSpace(id)
}

func normalizeRecallVerdict(verdict string) string {
	verdict = strings.TrimSpace(verdict)
	verdict = strings.ReplaceAll(verdict, "-", "_")
	verdict = strings.ReplaceAll(verdict, " ", "_")
	return strings.ToUpper(verdict)
}

func sortedKeys(set map[string]bool) []string {
	if len(set) == 0 {
		return nil
	}
	out := make([]string, 0, len(set))
	for key := range set {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}
