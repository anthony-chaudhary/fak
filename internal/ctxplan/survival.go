package ctxplan

import "sort"

// survival.go — the SURVIVAL CLASS (#2421): an explicit, kernel-assigned answer to the one
// question an eviction has to ask about every context page — "may this page be dropped, and if
// it is, is anything lost?"
//
// # Why a class, and not a hope
//
// Managed context is otherwise a single undifferentiated pool: whatever fits, fits. Everything
// is implicitly evictable, so the pieces a session cannot survive losing — the active steer, the
// live continuation seed, the standing safety invariants — are protected only by HAPPENING to
// sit inside whatever window the compactor kept. That is the failure this leaf exists to remove:
// an eviction eats the standing instruction, and every turn after it is confidently wrong about
// what it was asked to do, with nothing in the transcript saying a thing was lost. Recall is not
// the answer either — a page the model no longer knows to ask for is a page it will not recall.
//
// A class turns the hope into a checkable property. It is assigned DETERMINISTICALLY from the
// page's KIND, which the kernel stamps at admission — never from model output, and never from a
// caller's per-page assertion. That direction matters: if a page could declare itself pinned,
// "pinned" would mean whatever the last thing to write into context said it meant, and a
// prompt-injected page could pin itself into every future turn.
//
// # The three classes, and the one distinction that is easy to get wrong
//
//	PINNED      must survive byte-identical. Dropping it is a REFUSAL (ReasonPinEvictRefused),
//	            not a trade: system invariants, the active steer, the continuation seed.
//	REPLAYABLE  may be dropped, because its full bytes are recoverable — it is backed by the
//	            lossless store and a content-addressed handle pages it straight back in.
//	EVICTABLE   may be dropped, and is then genuinely gone: aged transcript prose whose value
//	            was in being read once.
//
// REPLAYABLE and EVICTABLE are both "may be dropped", which is exactly why they must not be one
// class. Collapsing them loses the ORDER: a planner that cannot tell them apart sheds a
// re-derivable tool result and an unrecoverable turn of reasoning with equal willingness, when
// the first costs a page fault and the second costs the fact. PlanEviction therefore always
// exhausts the EVICTABLE set before it touches a single REPLAYABLE page — the cheap loss first.
//
// This file is pure classification and planning over IDs and token costs: no bytes, no store, no
// I/O, consistent with the rest of the leaf (stdlib only, imports nothing internal). The caller
// that owns the bytes — internal/gateway's compaction path — maps its own pages onto these kinds
// and enforces the verdict on the wire.
type SurvivalClass int

// The survival classes, ordered by how strongly a page resists eviction. The zero value is
// ClassEvictable, which is the fail-closed direction for this vocabulary: an unrecognised or
// unstamped page can only ever be treated as ordinary droppable prose, never silently promoted
// into the protected set.
const (
	// ClassEvictable — droppable, and gone once dropped. Aged transcript prose.
	ClassEvictable SurvivalClass = iota
	// ClassReplayable — droppable, but the full bytes stay recoverable through the lossless
	// store behind a content-addressed handle. Dropping one costs a page fault, never a fact.
	ClassReplayable
	// ClassPinned — must survive byte-identical. A plan that would evict one is refused.
	ClassPinned
)

// String renders the class as the operator-facing token used in refusals and readouts.
func (c SurvivalClass) String() string {
	switch c {
	case ClassPinned:
		return "PINNED"
	case ClassReplayable:
		return "REPLAYABLE"
	default:
		return "EVICTABLE"
	}
}

// The page-kind vocabulary. A kind names WHAT a page is; the class follows from it by the table
// below, so the survival decision is a property of the page's provenance rather than a judgement
// made fresh (and possibly differently) at each eviction.
const (
	// KindSystemInvariant — standing instructions that govern every later turn (the safety and
	// policy floor). PINNED: an eviction here does not shorten the session, it changes what the
	// session is allowed to do.
	KindSystemInvariant = "system_invariant"
	// KindActiveSteer — the operator's live standing instruction for this session (fak marks it
	// on the wire with the goal marker). PINNED: it is the thing every later turn is measured
	// against, so laundering it is indistinguishable from silently changing the task.
	KindActiveSteer = "active_steer"
	// KindContinuationSeed — the state the NEXT turn continues from (the live tail of the
	// conversation). PINNED: without it a resumed turn has no anchor to continue from at all.
	KindContinuationSeed = "continuation_seed"
	// KindToolSchema — tool definitions. REPLAYABLE: they are re-sent verbatim by the caller
	// every turn, so an evicted copy costs nothing to restore.
	KindToolSchema = "tool_schema"
	// KindSystemDef — non-invariant structural prompt material (formatting, environment
	// description). REPLAYABLE for the same reason: re-derivable from the same source that
	// produced it.
	KindSystemDef = "system_def"
	// KindCASResult — a page whose full bytes live in the content-addressed store behind a
	// restore handle: a tool result, or an originating task an eviction tombstoned WITH its
	// handle. REPLAYABLE: recoverable exactly, on demand.
	KindCASResult = "cas_result"
	// KindTranscriptProse — ordinary conversation turns. EVICTABLE: the default, and the bulk.
	KindTranscriptProse = "transcript_prose"
)

// pageKindClass is the whole classification rule: kind → class, total and constant. It is a
// closed table on purpose — a kind not listed here is not a kind this kernel stamped, so
// ClassOf refuses to grant it anything.
var pageKindClass = map[string]SurvivalClass{
	KindSystemInvariant:  ClassPinned,
	KindActiveSteer:      ClassPinned,
	KindContinuationSeed: ClassPinned,
	KindToolSchema:       ClassReplayable,
	KindSystemDef:        ClassReplayable,
	KindCASResult:        ClassReplayable,
	KindTranscriptProse:  ClassEvictable,
}

// ClassOf maps a page kind to its survival class. It is deterministic and total.
//
// An UNKNOWN kind — a typo, an adapter that has not been taught the vocabulary, or a string that
// reached here from model output — falls to ClassEvictable. That is the only safe direction: the
// alternative (fail closed to PINNED) would let any unrecognised string pin itself into every
// future turn, which is precisely the model-controlled protection this class exists to replace.
// Failing to EVICTABLE can only ever cost residency, never authority.
func ClassOf(kind string) SurvivalClass {
	if c, ok := pageKindClass[kind]; ok {
		return c
	}
	return ClassEvictable
}

// PageKinds returns the closed kind vocabulary, sorted, so a consumer (a readout, a test, a
// schema) can enumerate it instead of re-typing it. A fresh slice each call — the table is
// process-global and a caller that sorted or appended to a shared array would corrupt every
// later reader.
func PageKinds() []string {
	out := make([]string, 0, len(pageKindClass))
	for k := range pageKindClass {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Page is one classifiable unit of resident context: a stable address, the kind that fixes its
// class, and its resident token cost. It carries no bytes — the caller that owns them keeps
// them, exactly as Span keeps only SAFE metadata.
// RetentionIntent is the closed, advisory vocabulary used to order eviction candidates
// inside one survival class. It never changes the class itself: keep is not pinning, and drop
// cannot make a pinned page droppable.
type RetentionIntent string

const (
	RetentionKeep    RetentionIntent = "keep"
	RetentionNeutral RetentionIntent = "neutral"
	RetentionDrop    RetentionIntent = "drop"
)

const (
	maxRetentionAnnotations = 8
	maxRetentionSourceLen   = 96
	maxRetentionReasonLen   = 64
)

// RetentionAnnotation is bounded decision metadata, not prompt content. Source identifies either
// a deterministic rule or a bounded agent; ReasonCode is an optional machine token and must not
// contain arbitrary prose.
type RetentionAnnotation struct {
	Intent     RetentionIntent `json:"intent"`
	Source     string          `json:"source"`
	ReasonCode string          `json:"reason_code,omitempty"`
}

type Page struct {
	ID        string                `json:"id"`
	Kind      string                `json:"kind"`
	Tokens    int                   `json:"tokens"`
	Retention []RetentionAnnotation `json:"retention,omitempty"`
}

// Class is the page's survival class, derived from its kind.
func (p Page) Class() SurvivalClass { return ClassOf(p.Kind) }

// ReasonPinEvictRefused is the closed refusal token for "this plan would evict a PINNED page".
// It is registered in the repo's refusal vocabulary (dos.toml [reasons.PIN_EVICT_REFUSED], so
// `dos man wedge PIN_EVICT_REFUSED --explain` resolves it) and is the same token the gateway's
// compaction path returns, so one string names the refusal from the planner through the wire to
// the operator.
const ReasonPinEvictRefused = "PIN_EVICT_REFUSED"

// ReasonRetentionAnnotationInvalid is the closed refusal token for malformed, unbounded, or
// conflicting retention metadata. Planning validates every page before choosing any eviction,
// so this refusal always carries an empty Evict set.
const ReasonRetentionAnnotationInvalid = "RETENTION_ANNOTATION_INVALID"

// EvictionPlan is the verdict of planning one eviction down to a token budget. A plan with a
// non-empty Refusal evicted NOTHING: the refusal is the whole answer, and the caller must leave
// the context as it found it rather than apply a partial, lossy drop.
type EvictionPlan struct {
	Keep         []string `json:"keep"`              // page IDs that stay resident, in input order
	Evict        []string `json:"evict"`             // page IDs to drop, in input order; empty on a refusal
	Refusal      string   `json:"refusal,omitempty"` // "" or ReasonPinEvictRefused
	PinnedTokens int      `json:"pinned_tokens"`     // the floor the budget must clear
	KeptTokens   int      `json:"kept_tokens"`       // resident cost of Keep
}

// PlanEviction plans the drop that brings pages down to budgetTokens while honouring every
// page's survival class.
//
// It refuses rather than trades. If the PINNED set alone does not fit the budget there is no
// honest plan — every remaining option evicts something that must survive — so it returns
// ReasonPinEvictRefused with an empty Evict set. A caller that wanted a smaller context gets a
// refusal it can act on (raise the budget, or shed load elsewhere) instead of a silently lossy
// compaction whose damage only shows up turns later.
//
// Otherwise it retries against the EVICTABLE SET ONLY: pinned pages are never candidates, and
// the shed walks EVICTABLE pages first, then REPLAYABLE ones, stopping the moment the kept set
// fits. Within a class, input order breaks ties — history arrives oldest-first, so the oldest
// prose goes first — which makes the plan deterministic for a given input.
//
// A budget the non-pinned set cannot be shed below is impossible here: evicting everything
// non-pinned leaves exactly PinnedTokens, which the guard above already proved fits.
func PlanEviction(pages []Page, budgetTokens int) EvictionPlan {
	plan := EvictionPlan{}
	intents := make([]RetentionIntent, len(pages))
	for i, p := range pages {
		intent, valid := pageRetentionIntent(p)
		if !valid {
			plan.Refusal = ReasonRetentionAnnotationInvalid
			return plan
		}
		intents[i] = intent
	}
	total := 0
	for _, p := range pages {
		t := pageTokens(p)
		total += t
		if p.Class() == ClassPinned {
			plan.PinnedTokens += t
		}
	}
	if plan.PinnedTokens > budgetTokens {
		for _, p := range pages {
			if p.Class() == ClassPinned {
				plan.Keep = append(plan.Keep, p.ID)
			}
		}
		plan.Refusal = ReasonPinEvictRefused
		plan.KeptTokens = plan.PinnedTokens
		return plan
	}

	evicted := make(map[int]bool, len(pages))
	kept := total
	for _, class := range []SurvivalClass{ClassEvictable, ClassReplayable} {
		for _, intent := range []RetentionIntent{RetentionDrop, RetentionNeutral, RetentionKeep} {
			for i, p := range pages {
				if kept <= budgetTokens {
					break
				}
				if evicted[i] || p.Class() != class || intents[i] != intent {
					continue
				}
				evicted[i] = true
				kept -= pageTokens(p)
			}
		}
	}
	for i, p := range pages {
		if evicted[i] {
			plan.Evict = append(plan.Evict, p.ID)
			continue
		}
		plan.Keep = append(plan.Keep, p.ID)
	}
	plan.KeptTokens = kept
	return plan
}

func pageRetentionIntent(p Page) (RetentionIntent, bool) {
	if len(p.Retention) == 0 {
		return RetentionNeutral, true
	}
	if len(p.Retention) > maxRetentionAnnotations {
		return "", false
	}
	intent := RetentionIntent("")
	for _, annotation := range p.Retention {
		if !validRetentionIntent(annotation.Intent) || !validRetentionSource(annotation.Source) ||
			!validRetentionToken(annotation.ReasonCode, maxRetentionReasonLen, true) {
			return "", false
		}
		if intent != "" && annotation.Intent != intent {
			return "", false
		}
		intent = annotation.Intent
	}
	return intent, true
}

func validRetentionIntent(intent RetentionIntent) bool {
	switch intent {
	case RetentionKeep, RetentionNeutral, RetentionDrop:
		return true
	default:
		return false
	}
}

func validRetentionSource(source string) bool {
	const deterministic = "deterministic:"
	const agentSource = "agent:"
	switch {
	case len(source) > len(deterministic) && len(source) <= maxRetentionSourceLen && source[:len(deterministic)] == deterministic:
		return validRetentionToken(source[len(deterministic):], maxRetentionSourceLen-len(deterministic), false)
	case len(source) > len(agentSource) && len(source) <= maxRetentionSourceLen && source[:len(agentSource)] == agentSource:
		return validRetentionToken(source[len(agentSource):], maxRetentionSourceLen-len(agentSource), false)
	default:
		return false
	}
}

func validRetentionToken(value string, maxLen int, optional bool) bool {
	if value == "" {
		return optional
	}
	if len(value) > maxLen {
		return false
	}
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '.' || r == '_' || r == '/' || r == '-' {
			continue
		}
		return false
	}
	return true
}

// CheckEviction adjudicates a plan somebody ELSE produced: it reports ReasonPinEvictRefused when
// any ID in evictIDs names a PINNED page, and "" otherwise. This is the verification half of the
// contract — a compactor that plans its own drop by other means (byte splicing on a wire body,
// say) still has to pass its outcome through here, so the guarantee holds for plans this package
// did not author. An ID that names no page in pages is not this page set's business and is
// ignored.
func CheckEviction(pages []Page, evictIDs []string) string {
	if len(evictIDs) == 0 {
		return ""
	}
	pinned := make(map[string]bool, len(pages))
	for _, p := range pages {
		if p.Class() == ClassPinned {
			pinned[p.ID] = true
		}
	}
	for _, id := range evictIDs {
		if pinned[id] {
			return ReasonPinEvictRefused
		}
	}
	return ""
}

// pageTokens floors a page's declared cost at zero, so a negative estimate from a sloppy adapter
// cannot buy budget headroom for the pages around it.
func pageTokens(p Page) int {
	if p.Tokens < 0 {
		return 0
	}
	return p.Tokens
}
