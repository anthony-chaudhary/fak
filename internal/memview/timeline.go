package memview

import (
	"fmt"
	"sort"
	"strings"
)

// timeline.go — issue #1599: managed-context needs a debug view that renders WHY a
// durable fact is what it is, over TIME, not just its latest promotion snapshot.
// #1595 (memq/promotion.go) already ships PromotionRecord — a durable fact's
// admission event — and `fak memory explain-promotion` already renders the LATEST
// one. #1594 (recall/stalefact.go) already ships StaleFactDecision — the runtime
// verdict that a recalled fact is (or is no longer) safe to treat as current. A
// "provenance timeline" is neither of those data models; it is a VIEW that orders a
// cell's promotions and stale-fact transitions into one chronological account: when
// it entered durable memory, why, and what happened to its validity afterward.
//
// This is deliberately a VIEW, not a new fact type: memview stays tier 2 (mechanism)
// and must not import memq or recall (both tier 3) — see internal/architest's tier
// map and AGENTS.md's layering contract. So the timeline operates on a small,
// generic ProvenanceEvent that a tier-3 caller (memq, recall) or a tier-4 caller
// (cmd/fak) translates its OWN typed record into. The translation is a pure,
// lossless field copy — no new vocabulary is invented here, and no producer/consent/
// reason string is reinterpreted; RenderProvenanceTimeline only orders and formats
// what the caller already decided.
//
// Never a raw payload: ProvenanceEvent carries only safe, already-extractive
// metadata (digest, descriptor, producer, reason) — the same "never the sealed
// bytes themselves" posture memq.SourceSpan and recall.Page.Descriptor already take.
// A caller that has only a raw body must digest/describe it before building an
// event; the type has no field a body could be smuggled into.

// ProvenanceEventKind is the closed vocabulary of timeline entries. The set is
// intentionally the two things #1599's "In scope" line names — promotions and
// invalidations — plus a neutral "observed" for any other structured checkpoint a
// future caller wants to place on the same timeline without inventing a new kind.
type ProvenanceEventKind string

const (
	// EventPromotion: the fact (re-)entered durable memory — one memq.PromotionRecord.
	EventPromotion ProvenanceEventKind = "promotion"
	// EventStaleTransition: a runtime staleness verdict was produced for the fact —
	// one recall.StaleFactDecision whose Outcome is anything but "fresh" (a fresh
	// verdict is not a transition worth timelining; see NewStaleTransitionEvent).
	EventStaleTransition ProvenanceEventKind = "stale_transition"
	// EventObserved is a neutral placeholder for a caller-supplied checkpoint that is
	// neither a promotion nor a stale-fact transition (e.g. a future fold/revision
	// event). Renders plainly; never produced by this package itself.
	EventObserved ProvenanceEventKind = "observed"
)

var validProvenanceEventKinds = map[ProvenanceEventKind]bool{
	EventPromotion:       true,
	EventStaleTransition: true,
	EventObserved:        true,
}

// ValidProvenanceEventKind reports whether k is a member of the closed vocabulary.
func ValidProvenanceEventKind(k ProvenanceEventKind) bool { return validProvenanceEventKinds[k] }

func (k ProvenanceEventKind) String() string {
	if ValidProvenanceEventKind(k) {
		return string(k)
	}
	if k == "" {
		return "(unset)"
	}
	return "unknown(" + string(k) + ")"
}

// ProvenanceEvent is one chronological entry on a fact's timeline. Every field is
// either copied verbatim from a promotion/stale-fact record or supplied by the
// translating caller — RenderProvenanceTimeline synthesizes nothing beyond ordering
// and layout. Digest identifies the source bytes WITHOUT carrying them.
type ProvenanceEvent struct {
	// Seq is the caller's stable ordering key (memq.PromotionRecord.Seq for a
	// promotion; a monotonic counter the caller assigns for anything else). Ties are
	// broken by Step, then by insertion order, so a replay over the same inputs
	// always renders the same order.
	Seq int
	// Step is the originating turn/session step, if known (0 = unknown/not
	// applicable — e.g. a stale-fact check has no promotion step of its own).
	Step int
	Kind ProvenanceEventKind
	// Durability is the class in force at this event (turn|session|bounded|durable;
	// already-normalized vocabulary from memq/ctxmmu/recall — this package does not
	// re-validate it).
	Durability string
	// Producer names what authored this event (a promotion's producer, or the
	// detector name for a stale-fact transition, e.g. "recall.DetectStaleFact").
	Producer string
	// Consent is the promotion consent class (explicit|inferred|unknown); empty for
	// a non-promotion event.
	Consent string
	// Digest is the content address of the source bytes this event's fact traces to,
	// if known. Never the bytes themselves.
	Digest string
	// Descriptor is a safe, already-extractive description (never raw payload) — a
	// memq.SourceSpan.Descriptor or a recall.Page.Descriptor, verbatim.
	Descriptor string
	// Outcome carries a stale-fact outcome string (e.g. "expired_must_query") for an
	// EventStaleTransition; empty otherwise.
	Outcome string
	// Reason is the free-text justification already recorded upstream (a
	// promotion's Reason, or a StaleFactDecision's Reason) — rendered alongside the
	// structured fields, never as a substitute for them.
	Reason string
}

// Timeline is the ordered, rendered account of one cell/fact's provenance history.
type Timeline struct {
	CellID string            `json:"cell_id"`
	Events []ProvenanceEvent `json:"events"`
}

// BuildTimeline sorts a cell's provenance events into chronological order (by Seq,
// then Step, ties broken stably) and returns the Timeline. It performs NO filtering
// and NO reinterpretation — every event the caller supplies appears exactly once, in
// order. An empty/nil events slice yields an empty Timeline (not an error): a fact
// with no recorded provenance is a valid, renderable state ("no history found"),
// mirroring memq.Explanation's Found=false posture rather than refusing outright.
func BuildTimeline(cellID string, events []ProvenanceEvent) Timeline {
	out := make([]ProvenanceEvent, len(events))
	copy(out, events)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Seq != out[j].Seq {
			return out[i].Seq < out[j].Seq
		}
		return out[i].Step < out[j].Step
	})
	return Timeline{CellID: cellID, Events: out}
}

// Render formats the timeline as a human-readable, chronological account — from
// structured fields ONLY, via fmt/strings formatting. It never calls a model to
// narrate or summarize; it is the timeline analogue of memq.Explanation.Narrative
// and follows the same fixed-template posture. A digest is always truncated to a
// short prefix (never a raw payload field exists on ProvenanceEvent to leak in the
// first place, but truncation keeps the rendered line scannable).
func (t Timeline) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "== provenance timeline: %s ==\n", t.CellID)
	if len(t.Events) == 0 {
		b.WriteString("no provenance events recorded for this cell\n")
		return b.String()
	}
	for i, e := range t.Events {
		fmt.Fprintf(&b, "[%d] step=%d kind=%-16s durability=%-9s", i, e.Step, e.Kind, e.Durability)
		if e.Producer != "" {
			fmt.Fprintf(&b, " producer=%q", e.Producer)
		}
		if e.Consent != "" {
			fmt.Fprintf(&b, " consent=%s", e.Consent)
		}
		if e.Outcome != "" {
			fmt.Fprintf(&b, " outcome=%s", e.Outcome)
		}
		if e.Digest != "" {
			fmt.Fprintf(&b, " digest=%s", shortDigest(e.Digest))
		}
		b.WriteByte('\n')
		if e.Descriptor != "" {
			fmt.Fprintf(&b, "      source: %s\n", e.Descriptor)
		}
		if e.Reason != "" {
			fmt.Fprintf(&b, "      reason: %s\n", e.Reason)
		}
	}
	return b.String()
}

// shortDigest truncates a hex digest to a stable, scannable prefix (mirroring the
// recall.Digest[:12] convention cmd/fak/debug.go already uses for previews).
func shortDigest(d string) string {
	const n = 12
	if len(d) <= n {
		return d
	}
	return d[:n]
}
