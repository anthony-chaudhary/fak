// ContextPlaybook (#538) — a kernel-gated, witness-counted, self-improving CONTEXT.
//
// fak already self-improves CODE (#388) and materializes provenance-bound,
// read-only VIEWS over immutable recorded CDB pages (ViewSnippet/ViewSummary +
// the #512 indexer views). What it did NOT have is a self-improving CONTEXT: an
// agent-edited, accumulating, counter-ranked store of reusable strategy knowledge
// that refines across runs without being summarized away by the residency levers.
//
// ACE (arXiv 2510.04618, ICLR 2026) treats context as an evolving PLAYBOOK of
// itemized bullets ("[id] helpful=X harmful=Y :: content", grouped by section).
// A Generator runs trajectories, a Reflector increments per-bullet counters from
// observed success/failure, and a Curator applies STRUCTURED INCREMENTAL DELTA
// updates (bullet-localized add/edit + deterministic de-dup + counter-ranked
// pruning under a token budget) instead of monolithic rewrites. The delta-
// localized edits are the explicit anti-context-collapse mechanism.
//
// The fak twist is the kernel/honesty floor — every ACE step is kernel-checked,
// never self-certified:
//
//  1. Curator deltas pass the ADMIT GATE. Each bullet add/edit is screened by the
//     SAME ctxmmu.Admit gate #522 uses for compaction span-swaps, so a poisoned /
//     laundered "strategy" cannot silently enter the playbook (it is quarantined).
//     Deltas stay bullet-localized (ACE's anti-collapse property) AND adjudicated;
//     a whole-playbook rewrite is structurally REFUSED.
//  2. Counters are WITNESSED, not Reflector-asserted. A helpful/harmful increment
//     must be EARNED from independent evidence — a replay-oracle metric move
//     (#502-#505) or a dos-verify ship — never from the model's own claim that an
//     item "helped". A self-graded counter is the same false positive
//     verify-don't-believe refuses, so a Reflector-only claim yields NO change.
//  3. Pruning is counter-ranked and LEGIBLE. A low-helpful bullet is evicted
//     through a path that RETURNS what it dropped, not deleted by an opaque rewrite.
//
// This is co-located with the ViewType enum (it materializes as ViewPlaybook) and
// composes ctxmmu.Admit; it is the genuinely new accumulating, counter-ranked,
// delta-curated, admit-gated artifact no existing view provided.
package contextq

import (
	"context"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// PlaybookBullet is one itemized strategy item in the ACE sense. It renders as
// "[id] helpful=X harmful=Y :: content" and is grouped by Section. Taint records
// the admit-gate verdict the bullet entered under (trusted once it cleared the
// gate; the fail-closed default is tainted).
type PlaybookBullet struct {
	ID      string         `json:"id"`
	Section string         `json:"section"`
	Helpful int            `json:"helpful"`
	Harmful int            `json:"harmful"`
	Content string         `json:"content"`
	Taint   abi.TaintLabel `json:"taint"`
}

// score is the counter-ranked priority used by Prune: net helpfulness.
func (b PlaybookBullet) score() int { return b.Helpful - b.Harmful }

// PlaybookDeltaKind is the CLOSED set of bullet-LOCALIZED edits a Curator may
// apply. There is deliberately NO whole-playbook-rewrite kind — that is ACE's
// anti-context-collapse property made structural (a rewrite cannot be expressed).
type PlaybookDeltaKind string

const (
	DeltaAdd  PlaybookDeltaKind = "add"  // introduce a new bullet (admit-gated, deduped)
	DeltaEdit PlaybookDeltaKind = "edit" // localized edit of one bullet's content; counters preserved
)

// PlaybookDelta is a single bullet-localized Curator edit. Content is PLAIN
// strategy text; it may never itself carry rendered "[id] helpful=N harmful=N ::"
// bullet lines (that shape is a whole-playbook rewrite — or a counter-injection —
// and is refused, see rewriteShaped).
type PlaybookDelta struct {
	Kind     PlaybookDeltaKind `json:"kind"`
	Section  string            `json:"section"`
	BulletID string            `json:"bullet_id,omitempty"` // target for edit
	Content  string            `json:"content"`
}

// PlaybookVerdict is the Curator/Reflector adjudication outcome, reusing the
// package's MaterializationKind vocabulary so the playbook speaks the same closed
// verdict set as the rest of contextq:
//
//	FAULT     — a new bullet was materialized (admit-gated add)
//	HIT       — served against an existing bullet (dedup, or a witnessed counter move)
//	RECOMPUTE — a bullet was edited in place
//	REFUSE    — quarantined by the admit gate, a rewrite-shaped delta, or an unwitnessed counter
//	ABSTAIN   — not adjudicable (no such bullet / unknown kind / witness moved nothing)
type PlaybookVerdict struct {
	Kind     MaterializationKind `json:"kind"`
	Reason   string              `json:"reason"`
	BulletID string              `json:"bullet_id,omitempty"`
}

// WitnessKind names the INDEPENDENT evidence a counter increment must be earned
// from. WitnessNone is a Reflector-only (self-graded) claim and is refused.
type WitnessKind string

const (
	WitnessNone      WitnessKind = ""           // Reflector self-report — REFUSED, no counter change
	WitnessReplay    WitnessKind = "replay"     // a #502-#505 replay-oracle metric move
	WitnessDosVerify WitnessKind = "dos_verify" // a witnessed (fak <leaf>) ship
)

// CounterEvidence is the witness backing a counter increment. The increment
// DIRECTION is DERIVED from the evidence, never taken from a Reflector's claim
// that an item "helped" — that is the verify-don't-believe core.
type CounterEvidence struct {
	Kind WitnessKind `json:"kind"`
	// MetricDelta is the recorded-run metric change attributable to this bullet
	// under WitnessReplay: > 0 ⇒ helpful++, < 0 ⇒ harmful++, == 0 ⇒ no change.
	MetricDelta float64 `json:"metric_delta,omitempty"`
	// ShipSHA is the independently-verified ship commit under WitnessDosVerify; a
	// non-empty sha earns one helpful (the strategy demonstrably shipped).
	ShipSHA string `json:"ship_sha,omitempty"`
}

// ContextPlaybook is the admit-gated, witness-counted ACE playbook. It is safe
// for concurrent use.
type ContextPlaybook struct {
	mu      sync.Mutex
	gate    *ctxmmu.MMU
	bullets map[string]*PlaybookBullet
	order   []string // insertion order — deterministic render / dedup / prune
	nextID  int
}

// NewContextPlaybook builds an empty playbook backed by a fresh ctxmmu admit gate.
func NewContextPlaybook() *ContextPlaybook { return NewContextPlaybookWithGate(ctxmmu.New()) }

// NewContextPlaybookWithGate is the seam a caller (or a test) uses to supply the
// admit gate — e.g. the live process MMU rather than a fresh one. A nil gate
// falls back to a fresh ctxmmu.New so the playbook is never ungated.
func NewContextPlaybookWithGate(gate *ctxmmu.MMU) *ContextPlaybook {
	if gate == nil {
		gate = ctxmmu.New()
	}
	return &ContextPlaybook{gate: gate, bullets: map[string]*PlaybookBullet{}}
}

// rewriteBulletLine matches a RENDERED bullet line ("[id] helpful=N harmful=N ::").
// Delta content must be plain strategy text; carrying a rendered bullet line means
// the caller is feeding the playbook back to itself (a monolithic rewrite) or
// trying to set counters directly — both are refused.
var rewriteBulletLine = regexp.MustCompile(`(?m)^\s*\[[^\]]*\]\s+helpful=\d+\s+harmful=\d+\s*::`)

func rewriteShaped(content string) bool { return rewriteBulletLine.MatchString(content) }

// ApplyDelta is the Curator entry point. It applies a single bullet-localized
// delta through the admit gate. add/edit are the only kinds; there is no
// whole-playbook setter, and a rewrite-shaped delta is refused.
func (p *ContextPlaybook) ApplyDelta(ctx context.Context, d PlaybookDelta) PlaybookVerdict {
	switch d.Kind {
	case DeltaAdd:
		return p.add(ctx, d)
	case DeltaEdit:
		return p.edit(ctx, d)
	default:
		return PlaybookVerdict{Kind: MaterializationAbstain, Reason: "unknown delta kind " + string(d.Kind)}
	}
}

// admitContent screens bullet content through the REAL ctxmmu.Admit gate (the same
// gate the compaction span-swap path uses), carrying the content as an inline
// result payload. A Quarantine verdict means the bytes were held out as a
// secret / prompt-injection / pollution payload.
func (p *ContextPlaybook) admitContent(ctx context.Context, content string) abi.Verdict {
	r := &abi.Result{Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(content), Len: int64(len(content))}}
	return p.gate.Admit(ctx, &abi.ToolCall{Tool: "playbook.curate"}, r)
}

// screenDelta runs the shared content gate for a bullet mutation: it refuses a
// whole-playbook rewrite shape and quarantines content the ctxmmu.Admit gate
// holds out. It returns a non-nil refusal verdict to surface, or nil to proceed.
func (p *ContextPlaybook) screenDelta(ctx context.Context, content string) *PlaybookVerdict {
	if rewriteShaped(content) {
		return &PlaybookVerdict{Kind: MaterializationRefuse,
			Reason: "REWRITE_REFUSED: delta content carries rendered bullet lines — a whole-playbook rewrite is not a bullet-localized delta"}
	}
	if v := p.admitContent(ctx, content); v.Kind == abi.VerdictQuarantine {
		return &PlaybookVerdict{Kind: MaterializationRefuse,
			Reason: "QUARANTINED by ctxmmu.Admit: " + abi.ReasonName(v.Reason)}
	}
	return nil
}

func (p *ContextPlaybook) add(ctx context.Context, d PlaybookDelta) PlaybookVerdict {
	if v := p.screenDelta(ctx, d.Content); v != nil {
		return *v
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if dup := p.findDupLocked(d.Section, d.Content); dup != nil {
		// Deterministic de-dup: an identical bullet in the same section is not a
		// second entry — the add folds into the existing one (served as a HIT).
		return PlaybookVerdict{Kind: MaterializationHit, Reason: "deduped into existing bullet", BulletID: dup.ID}
	}
	p.nextID++
	id := fmt.Sprintf("b%d", p.nextID)
	p.bullets[id] = &PlaybookBullet{ID: id, Section: d.Section, Content: d.Content, Taint: abi.TaintTrusted}
	p.order = append(p.order, id)
	return PlaybookVerdict{Kind: MaterializationFault, Reason: "bullet added (admit-gated)", BulletID: id}
}

func (p *ContextPlaybook) edit(ctx context.Context, d PlaybookDelta) PlaybookVerdict {
	if v := p.screenDelta(ctx, d.Content); v != nil {
		return *v
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	b, ok := p.bullets[d.BulletID]
	if !ok {
		return PlaybookVerdict{Kind: MaterializationAbstain, Reason: "no such bullet " + d.BulletID}
	}
	// Localized edit: replace content in place; counters are PRESERVED (a localized
	// edit is not a reset — eroding earned counters would be the context collapse
	// ACE's delta-localization exists to prevent).
	b.Content = d.Content
	return PlaybookVerdict{Kind: MaterializationRecompute, Reason: "bullet edited in place (counters preserved)", BulletID: b.ID}
}

// Reflect adjudicates a per-bullet counter increment — the honesty core. The
// increment is REFUSED unless backed by an independent witness; the direction is
// derived from the evidence, never from a Reflector's self-graded claim. A
// Reflector-only claim (WitnessNone) yields NO counter change.
func (p *ContextPlaybook) Reflect(bulletID string, ev CounterEvidence) PlaybookVerdict {
	p.mu.Lock()
	defer p.mu.Unlock()
	b, ok := p.bullets[bulletID]
	if !ok {
		return PlaybookVerdict{Kind: MaterializationAbstain, Reason: "no such bullet " + bulletID, BulletID: bulletID}
	}
	switch ev.Kind {
	case WitnessReplay:
		switch {
		case ev.MetricDelta > 0:
			b.Helpful++
			return PlaybookVerdict{Kind: MaterializationHit, Reason: "replay-witnessed helpful", BulletID: bulletID}
		case ev.MetricDelta < 0:
			b.Harmful++
			return PlaybookVerdict{Kind: MaterializationHit, Reason: "replay-witnessed harmful", BulletID: bulletID}
		default:
			return PlaybookVerdict{Kind: MaterializationAbstain, Reason: "replay metric unchanged — no counter move", BulletID: bulletID}
		}
	case WitnessDosVerify:
		if strings.TrimSpace(ev.ShipSHA) == "" {
			return PlaybookVerdict{Kind: MaterializationRefuse, Reason: "dos_verify witness carried no ship sha", BulletID: bulletID}
		}
		b.Helpful++
		return PlaybookVerdict{Kind: MaterializationHit, Reason: "dos-verify-witnessed helpful", BulletID: bulletID}
	default:
		// Reflector-only / unwitnessed claim — refused, NO counter change.
		return PlaybookVerdict{Kind: MaterializationRefuse,
			Reason: "counter increment refused — no replay/dos-verify witness (a Reflector self-report is not evidence)", BulletID: bulletID}
	}
}

// Prune evicts the lowest-scoring bullets (net helpfulness, ties broken oldest-
// first) until the rendered playbook fits within budgetTokens, and RETURNS the
// evicted bullets. Returning rather than silently dropping is the legible-eviction
// property: a low-helpful bullet is demoted through a path a caller can observe and
// persist, not erased by an opaque rewrite. budgetTokens <= 0 evicts everything.
func (p *ContextPlaybook) Prune(budgetTokens int) []PlaybookBullet {
	p.mu.Lock()
	defer p.mu.Unlock()
	var evicted []PlaybookBullet
	for len(p.order) > 0 && p.tokenEstimateLocked() > budgetTokens {
		victim := p.lowestScoreLocked()
		if victim == "" {
			break
		}
		evicted = append(evicted, *p.bullets[victim])
		p.removeLocked(victim)
	}
	return evicted
}

// lowestScoreLocked returns the id of the lowest-scoring bullet (ties: oldest by
// insertion order). Caller holds p.mu.
func (p *ContextPlaybook) lowestScoreLocked() string {
	worst := ""
	worstScore := 0
	for _, id := range p.order {
		b := p.bullets[id]
		if b == nil {
			continue
		}
		if worst == "" || b.score() < worstScore {
			worst, worstScore = id, b.score()
		}
	}
	return worst
}

// removeLocked drops a bullet from both the map and the order slice. Caller holds p.mu.
func (p *ContextPlaybook) removeLocked(id string) {
	delete(p.bullets, id)
	for i, x := range p.order {
		if x == id {
			p.order = append(p.order[:i], p.order[i+1:]...)
			break
		}
	}
}

// findDupLocked returns an existing bullet in the same section whose normalized
// content matches, or nil. Caller holds p.mu.
func (p *ContextPlaybook) findDupLocked(section, content string) *PlaybookBullet {
	key := normalize(content)
	for _, id := range p.order {
		b := p.bullets[id]
		if b != nil && b.Section == section && normalize(b.Content) == key {
			return b
		}
	}
	return nil
}

// normalize folds whitespace and case so a trivially-reworded duplicate dedups.
func normalize(s string) string { return strings.ToLower(strings.Join(strings.Fields(s), " ")) }

// Bullets returns a deterministic (section, insertion-order) snapshot copy of the
// current bullets. Read-only: it copies, so a caller cannot mutate the live store.
func (p *ContextPlaybook) Bullets() []PlaybookBullet {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.orderedLocked()
}

// orderedLocked returns bullets sorted by section then insertion order. Caller holds p.mu.
func (p *ContextPlaybook) orderedLocked() []PlaybookBullet {
	type pos struct {
		b   *PlaybookBullet
		idx int
	}
	all := make([]pos, 0, len(p.order))
	for i, id := range p.order {
		if b := p.bullets[id]; b != nil {
			all = append(all, pos{b, i})
		}
	}
	sort.SliceStable(all, func(i, j int) bool {
		if all[i].b.Section != all[j].b.Section {
			return all[i].b.Section < all[j].b.Section
		}
		return all[i].idx < all[j].idx
	})
	out := make([]PlaybookBullet, 0, len(all))
	for _, p := range all {
		out = append(out, *p.b)
	}
	return out
}

// Render emits the playbook in ACE form: bullets grouped by section, each line
// "[id] helpful=X harmful=Y :: content", sections and bullets in deterministic
// order. This is the surface that enters a prompt and the body of the ViewPlaybook
// view.
func (p *ContextPlaybook) Render() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.renderLocked()
}

func (p *ContextPlaybook) renderLocked() string {
	var sb strings.Builder
	section := ""
	first := true
	for _, b := range p.orderedLocked() {
		if first || b.Section != section {
			if !first {
				sb.WriteByte('\n')
			}
			section = b.Section
			sb.WriteString("## ")
			sb.WriteString(section)
			sb.WriteByte('\n')
			first = false
		}
		fmt.Fprintf(&sb, "[%s] helpful=%d harmful=%d :: %s\n", b.ID, b.Helpful, b.Harmful, b.Content)
	}
	return sb.String()
}

// tokenEstimateLocked is a coarse token estimate of the rendered playbook
// (~4 bytes/token). Caller holds p.mu.
func (p *ContextPlaybook) tokenEstimateLocked() int {
	return len(p.renderLocked())/4 + 1
}

// Snapshot materializes the playbook as a first-class ViewPlaybook record. Unlike
// the read-only views over recorded pages, this view's body is the rendered
// agent-edited store; the record carries the count of bullets that survived the
// admit gate so a consumer can see the artifact is gated, not free-text.
func (p *ContextPlaybook) Snapshot(producer string) (MemoryViewRecord, []byte) {
	p.mu.Lock()
	defer p.mu.Unlock()
	body := p.renderLocked()
	if producer == "" {
		producer = "playbook"
	}
	rec := MemoryViewRecord{
		ViewID:    "playbook",
		ViewType:  ViewPlaybook,
		Producer:  producer,
		Scope:     abi.ScopeAgent,
		Taint:     abi.TaintTrusted,
		SourceLen: int64(len(body)),
		Coverage:  1.0,
		Labels:    map[string]string{"bullets": fmt.Sprintf("%d", len(p.bullets))},
	}
	return rec, []byte(body)
}
