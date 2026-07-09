package metrics

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
)

// Generation lifecycle events + durable ledger (issue #1663, gen/second-next).
//
// A generation portfolio moves options between horizons and out of the portfolio:
// an option is PROMOTED toward a nearer horizon, DEMOTED toward a farther one,
// RELABELED (its identity/name changes without a horizon move), or RETIRED (it
// leaves the portfolio). Today those changes are invisible — a gen/* label flips on
// an issue and nothing witnesses when, why, or on what evidence. Promotion, demotion,
// relabeling, and retirement should be observable EVENTS, and the sequence of them
// should be a durable, append-only LEDGER an agent can replay to recover an option's
// current horizon without rereading the whole generation epic.
//
// This file is the SHAPE of those events and that ledger, expressed as a pure,
// stdlib-only model (no clock, no disk, no default exposure) in the same idiom as
// generation_roadmap.go and generation_dependency.go. It names a closed event
// vocabulary, enforces an evidence-and-direction discipline before an event may enter
// the ledger (a lifecycle change with no witness is refused, not silently recorded),
// serializes each event as a versioned durable JSONL row, folds a ledger down to the
// current per-option state, and renders a deterministic report. No clock lives here:
// ordering is a caller-supplied Seq and the timestamp is a caller-supplied string, so
// the model stays replayable and testable to the byte.
//
// Generation stays orthogonal to three things, and the model is built to keep them
// separate (OrthogonalityNote, shared with the roadmap and dependency models, is
// rendered in the report header):
//
//   - PRIORITY — an event records a horizon change, never a priority change. Promotion
//     moves an option to a nearer horizon on maturation evidence; it does not raise the
//     option's priority, and a demoted gen/future option is not thereby deprioritized.
//     The ledger carries no priority field, so a horizon move can never leak into one.
//   - SHARED TRUNK — every option lives on main. An event is a label-to-label transition
//     over one trunk, not a branch cut; a promotion is a relabel of an issue on the
//     shared trunk, never a new per-generation branch or worktree.
//   - RUNTIME FEATURE GATES — an event changes WHEN an option is expected to mature, not
//     WHETHER it is exposed at runtime. Promoting gen/second-next -> gen/next does not
//     flip a feature on; default exposure stays an explicit, independent feature gate.
//
// Promotion / demotion / invalidating-assumption for this artifact itself:
//   - Promotion evidence: a real relabel source (the devindex gen/* label history, or a
//     GitHub label-change webhook) folds actual transitions into a LifecycleLedger, the
//     durable JSONL rows land under docs/nightrun (next to the memory-value ledger), and
//     an operator surface renders it so an agent replays an option's horizon from events.
//   - Demotion/retirement evidence: retire this model if the milestone/devindex ledger
//     grows its own witnessed label-change history that subsumes it, or if measurement
//     shows lifecycle changes are so rare the ledger is ceremony. A promotion whose
//     Evidence never resolves to a real witness is itself the signal to demote the event
//     back to advisory (Validate refuses evidence-free transitions to make that visible).
//   - Invalidating assumption: that a lifecycle change is a discrete, single-option event
//     with a nameable evidence witness and a total now->future horizon order (so promote =
//     nearer rank, demote = farther rank). If real changes are bulk relabels with no
//     per-option witness, or if the horizon vocabulary stops being a total order, the
//     direction check and the one-event-per-change shape must move before this is promoted.

// LifecycleEventKind is the closed set of generation lifecycle changes. A value
// outside this set is a bug, not a new kind — the same closed-vocabulary discipline
// the roadmap and dependency models apply.
type LifecycleEventKind string

const (
	// EventPromotion moves an option to a NEARER horizon (lower now->future rank) on
	// maturation evidence — the option is expected to mature sooner than before.
	EventPromotion LifecycleEventKind = "promotion"
	// EventDemotion moves an option to a FARTHER horizon (higher rank) — it cannot
	// honestly mature on its old timeline; the demotion evidence names why.
	EventDemotion LifecycleEventKind = "demotion"
	// EventRelabeling changes an option's identity/name without moving its horizon —
	// a rename, not a promote or demote. From and To are labels, not horizons.
	EventRelabeling LifecycleEventKind = "relabeling"
	// EventRetirement removes an option from the portfolio entirely. It leaves the
	// portfolio (To is empty); From names its last horizon.
	EventRetirement LifecycleEventKind = "retirement"
)

// LifecycleEventKinds is the ordered, closed event vocabulary, so a test or a
// renderer can enumerate every lifecycle change an option can undergo.
var LifecycleEventKinds = []LifecycleEventKind{
	EventPromotion,
	EventDemotion,
	EventRelabeling,
	EventRetirement,
}

// Label is the short human label for an event kind.
func (k LifecycleEventKind) Label() string {
	switch k {
	case EventPromotion:
		return "Promotion"
	case EventDemotion:
		return "Demotion"
	case EventRelabeling:
		return "Relabeling"
	case EventRetirement:
		return "Retirement"
	default:
		return string(k)
	}
}

// LifecycleLedgerSchema is the versioned tag every durable row carries, in the same
// schema/N shape as the other durable ledgers in this repo (e.g. the memory-value
// ledger). A consumer keys parsing off this and refuses an unknown schema.
const LifecycleLedgerSchema = "generation-lifecycle-event/1"

// LifecycleRefusal is the closed set of reasons a lifecycle event is rejected before
// it may enter the durable ledger. A malformed or unwitnessed change is refused with a
// reason, never silently appended — the same structured-refusal discipline the kernel
// uses. The empty string means the event validated.
type LifecycleRefusal string

const (
	// RefusalNoOption — the event names no option; there is nothing to record.
	RefusalNoOption LifecycleRefusal = "LIFECYCLE_NO_OPTION"
	// RefusalUnknownKind — the event kind is outside LifecycleEventKinds.
	RefusalUnknownKind LifecycleRefusal = "LIFECYCLE_UNKNOWN_KIND"
	// RefusalNoEvidence — a horizon-moving or retiring event carries no evidence
	// witness. An observable lifecycle event with no witness is exactly the
	// self-authored claim the repo distrusts, so it is refused here.
	RefusalNoEvidence LifecycleRefusal = "LIFECYCLE_NO_EVIDENCE"
	// RefusalBadDirection — a promotion that does not move nearer, or a demotion that
	// does not move farther, on the now->future horizon rank (or an endpoint outside
	// the closed horizon vocabulary).
	RefusalBadDirection LifecycleRefusal = "LIFECYCLE_BAD_DIRECTION"
	// RefusalBadRelabel — a relabeling whose From and To are equal or empty: nothing
	// actually changed.
	RefusalBadRelabel LifecycleRefusal = "LIFECYCLE_BAD_RELABEL"
	// RefusalBadRetirement — a retirement that still names a destination horizon (To);
	// a retired option leaves the portfolio, it does not move within it.
	RefusalBadRetirement LifecycleRefusal = "LIFECYCLE_BAD_RETIREMENT"
)

// LifecycleEvent is one observable generation lifecycle change. It is a plain data
// snapshot — a caller folds a real label-change source into it; this package reads no
// clock and no disk. Seq is the caller-supplied monotonic order and At is a
// caller-supplied timestamp string, so the model stays replayable and byte-testable.
type LifecycleEvent struct {
	// Schema is the durable-row schema tag; MarshalLine stamps it to LifecycleLedgerSchema.
	Schema string `json:"schema"`
	// Seq is the caller-supplied monotonic sequence number (this package has no clock).
	Seq int `json:"seq"`
	// At is a caller-supplied timestamp (e.g. RFC3339), or "" — the model never reads it.
	At string `json:"at,omitempty"`
	// Option is the stable key of the portfolio option the event is about.
	Option string `json:"option"`
	// Kind is the lifecycle change (one of LifecycleEventKinds).
	Kind LifecycleEventKind `json:"kind"`
	// From is the prior horizon (a stream in RoadmapGenerations) or prior label.
	From string `json:"from,omitempty"`
	// To is the new horizon or label; empty for a retirement.
	To string `json:"to,omitempty"`
	// Evidence is the witness that authorizes this change (promotion/demotion/retirement
	// evidence). Required for every horizon-moving or retiring event.
	Evidence string `json:"evidence,omitempty"`
	// Reason is an optional one-line rationale carried into the rendered report.
	Reason string `json:"reason,omitempty"`
}

// Validate returns the refusal reason for a malformed event, or "" if the event may
// enter the ledger. Direction is decided by horizon rank alone (now->future); priority
// never enters here. This is the admission gate: the ledger only records witnessed,
// well-formed changes.
func (e LifecycleEvent) Validate() LifecycleRefusal {
	if strings.TrimSpace(e.Option) == "" {
		return RefusalNoOption
	}
	switch e.Kind {
	case EventPromotion, EventDemotion:
		if strings.TrimSpace(e.Evidence) == "" {
			return RefusalNoEvidence
		}
		fr, ok1 := horizonRank(e.From)
		tr, ok2 := horizonRank(e.To)
		if !ok1 || !ok2 {
			return RefusalBadDirection
		}
		if e.Kind == EventPromotion && tr >= fr {
			return RefusalBadDirection
		}
		if e.Kind == EventDemotion && tr <= fr {
			return RefusalBadDirection
		}
		return ""
	case EventRetirement:
		if strings.TrimSpace(e.Evidence) == "" {
			return RefusalNoEvidence
		}
		if strings.TrimSpace(e.To) != "" {
			return RefusalBadRetirement
		}
		return ""
	case EventRelabeling:
		if strings.TrimSpace(e.From) == "" || strings.TrimSpace(e.To) == "" || e.From == e.To {
			return RefusalBadRelabel
		}
		return ""
	default:
		return RefusalUnknownKind
	}
}

// MarshalLine returns the deterministic durable JSONL row for the event, with the
// schema tag stamped. One event is one line; the concatenation of lines is the durable
// ledger a future agent replays. Pure (encoding/json over a fixed struct), so the bytes
// are stable for a test.
func (e LifecycleEvent) MarshalLine() (string, error) {
	e.Schema = LifecycleLedgerSchema
	b, err := json.Marshal(e)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// LifecycleLedger is an append-only sequence of validated lifecycle events — the
// durable record of how a portfolio's options moved between horizons over time.
type LifecycleLedger struct {
	// Events is the accepted event list, in append order.
	Events []LifecycleEvent `json:"events"`
}

// Append validates the event and, if it passes, records it and returns "". A refused
// event is NOT recorded and its reason is returned, so a caller can route the refusal
// instead of writing an unwitnessed change to the durable ledger.
func (l *LifecycleLedger) Append(e LifecycleEvent) LifecycleRefusal {
	if r := e.Validate(); r != "" {
		return r
	}
	l.Events = append(l.Events, e)
	return ""
}

// CountByKind rolls the ledger up per event kind. Every kind in LifecycleEventKinds is
// present, including the zero ones — an absent kind must read as "0", not vanish.
func (l LifecycleLedger) CountByKind() map[LifecycleEventKind]int {
	out := make(map[LifecycleEventKind]int, len(LifecycleEventKinds))
	for _, k := range LifecycleEventKinds {
		out[k] = 0
	}
	for _, e := range l.Events {
		out[e.Kind]++
	}
	return out
}

// OptionState is an option's current standing after replaying every event about it:
// its current horizon (Stream), current label, whether it has been retired, and the
// last change's kind and evidence. It is what a future agent reads instead of the epic.
type OptionState struct {
	// Option is the stable option key.
	Option string `json:"option"`
	// Stream is the current horizon after the last promotion/demotion, "" if retired.
	Stream string `json:"stream,omitempty"`
	// Label is the current label after the last relabeling, "" if never relabeled.
	Label string `json:"label,omitempty"`
	// Retired is true once a retirement event has been replayed for the option.
	Retired bool `json:"retired"`
	// LastKind is the most recent lifecycle change applied to the option.
	LastKind LifecycleEventKind `json:"last_kind"`
	// LastEvidence is the witness behind the most recent change.
	LastEvidence string `json:"last_evidence,omitempty"`
	// Events counts how many lifecycle events have touched this option.
	Events int `json:"events"`
}

// Fold replays the ledger down to the current state of each option, returned sorted by
// option key for determinism. Promotion/demotion set the current Stream to the event's
// To; relabeling sets the current Label; retirement marks the option retired and clears
// its Stream. It is pure — the same events always fold to the same states.
func (l LifecycleLedger) Fold() []OptionState {
	byOption := make(map[string]*OptionState)
	order := make([]string, 0)
	for _, e := range l.Events {
		st, ok := byOption[e.Option]
		if !ok {
			st = &OptionState{Option: e.Option}
			byOption[e.Option] = st
			order = append(order, e.Option)
		}
		st.Events++
		st.LastKind = e.Kind
		if e.Evidence != "" {
			st.LastEvidence = e.Evidence
		}
		switch e.Kind {
		case EventPromotion, EventDemotion:
			st.Stream = e.To
			st.Retired = false
		case EventRelabeling:
			st.Label = e.To
		case EventRetirement:
			st.Retired = true
			st.Stream = ""
		}
	}
	sort.Strings(order)
	out := make([]OptionState, 0, len(order))
	for _, k := range order {
		out = append(out, *byOption[k])
	}
	return out
}

// Render produces a deterministic text report: the orthogonality header, a per-kind
// count line, each event with its transition and evidence, then the folded current
// state per option. Pure (no clock, no disk), so a test can assert its bytes and an
// observability surface can mount it.
func (l LifecycleLedger) Render() string {
	var b strings.Builder
	b.WriteString("Generation lifecycle ledger (" + strconv.Itoa(len(l.Events)) + " events, schema " + LifecycleLedgerSchema + ")\n")
	b.WriteString(OrthogonalityNote)
	b.WriteString("\n\n")

	b.WriteString("  counts:")
	counts := l.CountByKind()
	for _, k := range LifecycleEventKinds {
		b.WriteString(" " + string(k) + "=" + strconv.Itoa(counts[k]))
	}
	b.WriteString("\n\n")

	b.WriteString("  events:\n")
	if len(l.Events) == 0 {
		b.WriteString("    - (none)\n")
	}
	for _, e := range l.Events {
		b.WriteString("    - #" + strconv.Itoa(e.Seq) + " " + pad(e.Kind.Label(), lifecycleKindWidth) + " " +
			e.Option + ": " + transition(e))
		if e.Evidence != "" {
			b.WriteString(" | evidence: " + e.Evidence)
		}
		if e.Reason != "" {
			b.WriteString(" | " + e.Reason)
		}
		b.WriteString("\n")
	}

	b.WriteString("\n  current state:\n")
	states := l.Fold()
	if len(states) == 0 {
		b.WriteString("    - (none)\n")
	}
	for _, s := range states {
		b.WriteString("    - " + s.Option + ": " + optionStanding(s) + "\n")
	}
	return b.String()
}

// transition renders the from->to clause for an event, adapting to the kind: a
// retirement leaves the portfolio, a horizon move shows gen/from -> gen/to, a relabel
// shows the bare label change.
func transition(e LifecycleEvent) string {
	switch e.Kind {
	case EventRetirement:
		if e.From != "" {
			return "gen/" + e.From + " -> retired"
		}
		return "retired"
	case EventPromotion, EventDemotion:
		return "gen/" + e.From + " -> gen/" + e.To
	default:
		return e.From + " -> " + e.To
	}
}

// optionStanding renders the folded current standing of an option.
func optionStanding(s OptionState) string {
	if s.Retired {
		return "retired (" + strconv.Itoa(s.Events) + " events)"
	}
	horizon := "unplaced"
	if s.Stream != "" {
		horizon = "gen/" + s.Stream
	}
	out := horizon
	if s.Label != "" {
		out += " label=" + s.Label
	}
	return out + " (" + strconv.Itoa(s.Events) + " events)"
}

const lifecycleKindWidth = 12
