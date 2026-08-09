package chatops

// approval.go is the door's SECOND fence — the two-turn approval contract (#2266, epic
// #2259 leaf C6). Parse decides whether a message is a well-formed command from an
// authorized operator; this half decides whether that command may fire at all on the one
// line that carried it.
//
// # Why two turns
//
// The adjudicator's reversibility classifier (#2156) labels a tool call reversible,
// irreversible, or outward-facing. The same taxonomy applies to chatops verbs, and the
// same conclusion follows: some actions must never fire from a single chat line, because
// a single line is exactly what an injected message looks like. So a non-reversible verb
// gets a two-turn contract — propose → approve → execute — where the two turns are
// separate messages, separately authorized, separately audited.
//
// The door CONSUMES those labels and never re-derives them: each grammar row declares its
// Risk, and Risk.Gated() is the whole policy. There is no classifier here, and adding one
// would be a second source of truth for a question #2156 already answers.
//
// # The ordered approval fence
//
// Adjudicate applies its checks in a fixed order, the same discipline as Parse's fence,
// so the refusal a caller sees is the FIRST gate it failed:
//
//  1. the parse fence's own refusal   — a refused reply never reaches the ledger
//  2. UNKNOWN_VERB                    — the reply is not `approve`/`deny`
//  3. NOT_ADMIN                       — fail-closed allowlist, re-checked here so a shell
//     that calls Adjudicate directly cannot skip authz
//  4. APPROVAL_UNKNOWN_NONCE          — no pending proposal under that nonce
//  5. APPROVAL_FOREIGN_THREAD         — the nonce is real but belongs to another channel
//  6. APPROVAL_REPLAYED               — single-use: the nonce was already burned
//  7. APPROVAL_EXPIRED                — TTL-bounded: past (or missing) the deadline
//  8. APPROVAL_SELF                   — proposer == approver on a multi-operator fleet
//
// Every one of those tokens is an OPERATOR_GATE refusal: a human decision is the only
// thing that clears it, which is the whole point of the gate.
//
// # Fail-closed, three times over
//
//   - RiskUnclassified is the ZERO value, and it gates. Adding a verb and forgetting to
//     label it makes it require approval; it can never make it fire unattended.
//   - Outcome's zero value is OutcomeRefused, so a dropped field in a Verdict reads as a
//     refusal, never an accidental execute.
//   - A missing (zero) ExpiresAt counts as EXPIRED, not as "never expires". A proposal
//     that lost its deadline is dead, not immortal.
//
// # Purity
//
// Tier 1, like the rest of the package: state in, decision out. The clock arrives as a
// `now` argument and the pending set arrives as a value, so a test pins the entire
// approval boundary — TTL, replay, self-approval — with plain structs, no Slack and no
// wall clock. The impure shell owns the journal file, the card post, and the execution.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"time"
)

// Risk is the reversibility label a grammar row carries. The values are the adjudicator's
// ReversibilityClass vocabulary (#2156) copied VERBATIM as strings rather than imported:
// internal/adjudicator is tier 2 and this package is tier 1, so an import would be an
// upward layer violation. The string values ARE the binding, so they are pinned as
// literals by TestRiskVocabularyIsTheAdjudicatorsClasses against
// internal/adjudicator/reversibility.go's ReversibilityReversible / ReversibilityIrreversible
// / ReversibilityOutwardFacing — a drift there must fail here rather than silently
// re-classify the fleet's chat verbs.
type Risk string

const (
	// RiskUnclassified is the zero value ON PURPOSE: a verb nobody labeled is gated, so
	// forgetting to classify a new verb fails CLOSED (it needs approval) instead of
	// letting it fire on one chat line.
	RiskUnclassified Risk = ""
	// RiskReversible is the ONLY label that runs on a single line: the effect can be
	// undone, so a bad line costs an undo, not an incident.
	RiskReversible Risk = "reversible"
	// RiskIrreversible names an effect that cannot be undone once started — metered
	// spend, a destroyed artifact.
	RiskIrreversible Risk = "irreversible"
	// RiskOutwardFacing names an effect other people or systems can observe — a push to
	// the shared trunk, a comment on a public issue.
	RiskOutwardFacing Risk = "outward-facing"
)

// Gated reports whether this label demands the two-turn contract. Written as "everything
// except reversible" rather than "irreversible or outward-facing" so that a new label
// added upstream gates by default instead of silently slipping through.
func (r Risk) Gated() bool { return r != RiskReversible }

// The approval gate's closed refusal vocabulary — the OPERATOR_GATE family. Reasons()
// returns these alongside the parse fence's tokens as one flat closed set.
const (
	// ReasonApprovalUnknownNonce: no pending proposal is held under that nonce. Covers a
	// typo, a nonce the shell already reaped, and a fabricated one.
	ReasonApprovalUnknownNonce = "APPROVAL_UNKNOWN_NONCE"
	// ReasonApprovalForeignThread: the nonce names a real proposal, but the reply arrived
	// in a different channel than the one the proposal card was posted in. Approvals are
	// bound to their thread so a nonce overheard elsewhere cannot be spent.
	ReasonApprovalForeignThread = "APPROVAL_FOREIGN_THREAD"
	// ReasonApprovalReplayed: the nonce is single-use and was already burned by an
	// earlier approve/deny. This is the replay fence.
	ReasonApprovalReplayed = "APPROVAL_REPLAYED"
	// ReasonApprovalExpired: the proposal's TTL has run out (or it never carried a
	// deadline, which counts as expired). A stale proposal must be re-proposed, not
	// resurrected.
	ReasonApprovalExpired = "APPROVAL_EXPIRED"
	// ReasonApprovalSelf: proposer == approver on a fleet with more than one operator.
	// Explicitly ALLOWED when the allowlist names exactly one operator (the fleet today),
	// in which case the audit row records SelfApproved.
	ReasonApprovalSelf = "APPROVAL_SELF"
)

// DefaultTTL is how long a proposal stays approvable. Long enough that an operator can
// step away from the channel and still land the approval, short enough that a stale
// proposal cannot be spent by someone scrolling back through history days later.
const DefaultTTL = 15 * time.Minute

// The verdict strings recorded on a resolved proposal and its audit row.
const (
	VerdictApproved = "approved"
	VerdictDenied   = "denied"
	VerdictRefused  = "refused"
)

// The audit journal's closed event vocabulary. Proposal, approval and execution are
// SEPARATE events by design: the journal must never conflate "an operator approved this"
// with "this actually ran".
const (
	EventPropose = "propose"
	EventApprove = "approve"
	EventDeny    = "deny"
	EventRefuse  = "refuse"
	EventExecute = "execute"
)

// Proposal is the pending first turn of the contract: a gated command, frozen with the
// nonce that will authorize it and the deadline after which it cannot be. The shell holds
// these (rebuilt from the journal by Pending, which is the source of truth) and hands the
// matching one to Adjudicate.
type Proposal struct {
	Nonce     string    // the single-use approval nonce (see Propose for its derivation)
	Verb      Verb      // the gated verb that would run
	Operand   string    // its operand — the issue, run id, or bench name
	Risk      Risk      // the label that made this gate in the first place
	Proposer  string    // the Slack user id that asked for it
	Channel   string    // the channel the proposal card was posted in
	ThreadTS  string    // the originating message ts — the thread approvals reply in
	At        time.Time // when the proposal was minted
	ExpiresAt time.Time // the TTL deadline; zero counts as EXPIRED (fail-closed)
	Resolved  bool      // single-use burn flag: set once approved or denied
	Verdict   string    // VerdictApproved / VerdictDenied, set with Resolved
}

// Pendingable reports whether this proposal can still be spent at `now` — held, unburned,
// and inside its TTL.
func (p Proposal) Pendingable(now time.Time) bool {
	return p.Nonce != "" && !p.Resolved && !p.ExpiresAt.IsZero() && now.Before(p.ExpiresAt)
}

// AuditRow is one immutable journal row. The journal — not any in-memory map — is the
// source of truth for the pending set (see Pending), so every field the set needs to be
// rebuilt is carried here. Rows answer the four questions the contract exists to answer:
// who proposed, who approved, when, and what ran.
type AuditRow struct {
	Event        string    // one of the Event* constants
	Nonce        string    // the approval nonce this row is keyed on
	Verb         Verb      // the gated verb
	Operand      string    // its operand
	Risk         Risk      // the reversibility label that made it gate
	Proposer     string    // who asked
	Approver     string    // who approved/denied; empty on a propose row
	Verdict      string    // approved / denied / refused
	Reason       string    // the closed refusal token, set iff Verdict == VerdictRefused
	Channel      string    // where it happened
	ThreadTS     string    // the proposal thread, so Pending can rebuild the binding
	RunID        string    // what actually ran; set only on an EventExecute row
	At           time.Time // when
	SelfApproved bool      // proposer == approver, allowed under the one-operator policy
}

// Outcome is what the shell must do with an approve/deny reply.
type Outcome int

const (
	// OutcomeRefused is the ZERO value, so a Verdict with a dropped field reads as a
	// refusal and never as an accidental execute.
	OutcomeRefused Outcome = iota
	// OutcomeDenied: an authorized operator said no. The proposal is burned; nothing runs.
	OutcomeDenied
	// OutcomeExecute: the contract is satisfied. This is the ONLY outcome that releases a
	// gated command to run.
	OutcomeExecute
)

// String renders the outcome token for logs and test failures.
func (o Outcome) String() string {
	switch o {
	case OutcomeDenied:
		return "DENIED"
	case OutcomeExecute:
		return "EXECUTE"
	default:
		return "REFUSED"
	}
}

// Verdict is the decision for one approve/deny reply, plus the audit row the shell must
// journal for it. It carries the RESOLVED proposal so the shell writes the burn back
// rather than recomputing it.
type Verdict struct {
	Outcome      Outcome
	Reason       string   // a closed token from Reasons(); set iff OutcomeRefused
	Proposal     Proposal // the proposal with its single-use burn applied (on execute/deny)
	Approver     string   // who replied
	SelfApproved bool     // recorded when the one-operator policy allowed a self-approval
	Audit        AuditRow // the approval/denial/refusal row — NOT the execution row
}

// Propose mints the pending proposal for a gated command and the journal row that records
// it. It returns ok=false for anything that must not reach a proposal card: a refused
// parse, or a reversible verb that should just run.
//
// The nonce is derived DETERMINISTICALLY from the proposing message (proposer, channel,
// message ts, verb, operand) rather than drawn at random. Two consequences, both wanted:
// a re-delivered Slack message re-mints the SAME nonce and therefore cannot open a second
// pending proposal for one command (the idempotency discipline internal/chatopsdetach
// holds for dispatch), and the whole kernel stays pure — no RNG, no clock beyond the
// `now` argument. The nonce is a BINDING token, not a secret: authorization is the admin
// allowlist, checked independently, so knowing a nonce grants nothing to someone who
// could not already approve.
func Propose(res Result, now time.Time, ttl time.Duration) (Proposal, AuditRow, bool) {
	if !res.Gated() {
		return Proposal{}, AuditRow{}, false
	}
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	p := Proposal{
		Nonce:     mintNonce(res),
		Verb:      res.Verb,
		Operand:   res.Operand,
		Risk:      res.Risk,
		Proposer:  res.User,
		Channel:   res.Channel,
		ThreadTS:  res.Nonce, // the Slack ts of the proposing message
		At:        now,
		ExpiresAt: now.Add(ttl),
	}
	row := AuditRow{
		Event:    EventPropose,
		Nonce:    p.Nonce,
		Verb:     p.Verb,
		Operand:  p.Operand,
		Risk:     p.Risk,
		Proposer: p.Proposer,
		Channel:  p.Channel,
		ThreadTS: p.ThreadTS,
		At:       now,
	}
	return p, row, true
}

// mintNonce derives the 8-hex-char approval nonce. The fields are NUL-joined so two
// different field splits cannot collide into one digest, and the timestamps are excluded
// so a re-delivery of the same message mints the same nonce.
func mintNonce(res Result) string {
	sum := sha256.Sum256([]byte(strings.Join([]string{
		res.User, res.Channel, res.Nonce, string(res.Verb), res.Operand,
	}, "\x00")))
	return hex.EncodeToString(sum[:4])
}

// Adjudicate folds one parsed `approve <nonce>` / `deny <nonce>` reply against the pending
// proposal the shell looked up, and returns the verdict plus its audit row. It is total
// and pure: same inputs, same Verdict.
//
// `p` is the proposal held under the replied-to nonce; the zero Proposal means the shell
// found none. Passing the wrong proposal is safe — the nonce is re-compared here.
func Adjudicate(res Result, p Proposal, cfg Config, now time.Time) Verdict {
	// 1. A reply that never survived the parse fence never reaches the ledger; its own
	//    refusal token is threaded through unchanged.
	if res.Refused {
		return refuseApproval(res, p, res.Reason, now)
	}
	// 2. Only the two approval verbs adjudicate. Anything else is not a second turn.
	if res.Verb != VerbApprove && res.Verb != VerbDeny {
		return refuseApproval(res, p, ReasonUnknownVerb, now)
	}
	// 3. Authorization, re-checked fail-closed. Parse already ran this fence, but a shell
	//    that calls Adjudicate directly must not be able to skip it — and an empty
	//    allowlist must refuse here exactly as it does there.
	if !isAdmin(res.User, cfg.Admins) {
		return refuseApproval(res, p, ReasonNotAdmin, now)
	}
	// 4. The nonce must name a proposal the shell is actually holding.
	nonce := strings.TrimSpace(res.Operand)
	if p.Nonce == "" || !strings.EqualFold(p.Nonce, nonce) {
		return refuseApproval(res, p, ReasonApprovalUnknownNonce, now)
	}
	// 5. Thread binding: a nonce is spendable only where its card was posted.
	if p.Channel != "" && res.Channel != p.Channel {
		return refuseApproval(res, p, ReasonApprovalForeignThread, now)
	}
	// 6. Single use. Checked before the TTL so an operator who double-taps learns they
	//    already answered, rather than being told the window closed.
	if p.Resolved {
		return refuseApproval(res, p, ReasonApprovalReplayed, now)
	}
	// 7. TTL. A zero deadline is treated as expired, never as unlimited.
	if p.ExpiresAt.IsZero() || !now.Before(p.ExpiresAt) {
		return refuseApproval(res, p, ReasonApprovalExpired, now)
	}
	// 8. Self-approval is explicit policy, not an accident: allowed only while the
	//    allowlist names exactly ONE operator (the fleet today), and recorded as such in
	//    the audit row. Distinct ids are counted, so a duplicated entry cannot fake a
	//    second seat — and cannot lock the single operator out of their own fleet either.
	self := p.Proposer == res.User
	if self && distinctAdmins(cfg.Admins) != 1 {
		return refuseApproval(res, p, ReasonApprovalSelf, now)
	}

	resolved := p
	resolved.Resolved = true
	outcome, event := OutcomeExecute, EventApprove
	resolved.Verdict = VerdictApproved
	if res.Verb == VerbDeny {
		outcome, event = OutcomeDenied, EventDeny
		resolved.Verdict = VerdictDenied
	}
	return Verdict{
		Outcome:      outcome,
		Proposal:     resolved,
		Approver:     res.User,
		SelfApproved: self,
		Audit: AuditRow{
			Event:        event,
			Nonce:        p.Nonce,
			Verb:         p.Verb,
			Operand:      p.Operand,
			Risk:         p.Risk,
			Proposer:     p.Proposer,
			Approver:     res.User,
			Verdict:      resolved.Verdict,
			Channel:      p.Channel,
			ThreadTS:     p.ThreadTS,
			At:           now,
			SelfApproved: self,
		},
	}
}

// refuseApproval builds a refusal verdict carrying the closed reason token and its audit
// row. The proposal is threaded back UNBURNED: a refusal must never consume the nonce, or
// a mistyped channel would destroy a still-valid approval.
func refuseApproval(res Result, p Proposal, reason string, now time.Time) Verdict {
	return Verdict{
		Outcome:  OutcomeRefused,
		Reason:   reason,
		Proposal: p,
		Approver: res.User,
		Audit: AuditRow{
			Event:    EventRefuse,
			Nonce:    p.Nonce,
			Verb:     p.Verb,
			Operand:  p.Operand,
			Risk:     p.Risk,
			Proposer: p.Proposer,
			Approver: res.User,
			Verdict:  VerdictRefused,
			Reason:   reason,
			Channel:  res.Channel,
			ThreadTS: p.ThreadTS,
			At:       now,
		},
	}
}

// ExecuteRow is the SEPARATE journal row for the execution itself, written by the shell
// once it has actually handed the approved command off and holds a run id. It is not part
// of the Verdict on purpose: the kernel decides that a command MAY run, it does not
// witness that it DID. Keeping the two rows apart is what stops the journal from claiming
// an execution that never happened.
//
// ok is false for any verdict other than OutcomeExecute — a denied or refused proposal has
// no execution to record.
func (v Verdict) ExecuteRow(runID string, at time.Time) (AuditRow, bool) {
	if v.Outcome != OutcomeExecute {
		return AuditRow{}, false
	}
	return AuditRow{
		Event:        EventExecute,
		Nonce:        v.Proposal.Nonce,
		Verb:         v.Proposal.Verb,
		Operand:      v.Proposal.Operand,
		Risk:         v.Proposal.Risk,
		Proposer:     v.Proposal.Proposer,
		Approver:     v.Approver,
		Verdict:      VerdictApproved,
		Channel:      v.Proposal.Channel,
		ThreadTS:     v.Proposal.ThreadTS,
		RunID:        runID,
		At:           at,
		SelfApproved: v.SelfApproved,
	}, true
}

// Pending replays the audit journal into the live pending set at `now` — the proposals
// that are still approvable. The JOURNAL is the source of truth, not an in-memory map: a
// door that restarts rebuilds its pending set by replaying rows, so a crash between the
// proposal card and the approval cannot silently un-gate or double-gate a command.
//
// Rows are applied in order: a propose row opens a proposal, an approve/deny row burns it,
// a refuse row leaves it standing (a refusal never consumes the nonce), and anything past
// its TTL is dropped. The result keeps journal order so a `pending` readout lists the
// oldest waiting approval first — a slow approval stays visible instead of getting lost.
func Pending(journal []AuditRow, now time.Time) []Proposal {
	held := map[string]*Proposal{}
	order := []string{}
	for _, row := range journal {
		if row.Nonce == "" {
			continue
		}
		switch row.Event {
		case EventPropose:
			if _, dup := held[row.Nonce]; dup {
				continue // a re-delivered proposal re-mints the same nonce; not a second one
			}
			p := Proposal{
				Nonce:    row.Nonce,
				Verb:     row.Verb,
				Operand:  row.Operand,
				Risk:     row.Risk,
				Proposer: row.Proposer,
				Channel:  row.Channel,
				ThreadTS: row.ThreadTS,
				At:       row.At,
			}
			// The propose row carries `At`; the deadline is re-derived rather than
			// journaled so a hand-edited row cannot extend a TTL.
			p.ExpiresAt = row.At.Add(DefaultTTL)
			held[row.Nonce] = &p
			order = append(order, row.Nonce)
		case EventApprove, EventDeny:
			if p, ok := held[row.Nonce]; ok {
				p.Resolved = true
				p.Verdict = row.Verdict
			}
		}
	}
	out := []Proposal{}
	for _, nonce := range order {
		if p := held[nonce]; p.Pendingable(now) {
			out = append(out, *p)
		}
	}
	return out
}

// Card renders the in-thread proposal card for a pending gated command: what would run,
// the blast-radius line the grammar row declares, who asked, and the nonce with its
// deadline. Deterministic and pure — the same proposal always renders byte-identically,
// so a re-posted card coalesces on the durable outbox and a test can pin the whole
// operator-facing transcript as a fixture.
//
// This is the v0 (reply-keyword) surface. v1 replaces the two backtick commands on the
// last line with Block Kit buttons and is blocked-by the transport spike #2267; every
// other line, and the entire contract behind it, is unchanged by that swap.
func Card(p Proposal) string {
	var b strings.Builder
	fmt.Fprintf(&b, "approval required — `%s %s`\n", p.Verb, p.Operand)
	fmt.Fprintf(&b, "blast radius: %s — %s\n", riskLabel(p.Risk), blastOf(p.Verb))
	fmt.Fprintf(&b, "proposed by <@%s> · nonce `%s` · expires %s\n",
		p.Proposer, p.Nonce, p.ExpiresAt.UTC().Format(time.RFC3339))
	fmt.Fprintf(&b, "reply `approve %s` or `deny %s` in this thread", p.Nonce, p.Nonce)
	return b.String()
}

// riskLabel renders a Risk for the card, naming the unlabeled case explicitly rather than
// printing an empty string — an operator must be able to see that a verb gated because
// nobody classified it.
func riskLabel(r Risk) string {
	if r == RiskUnclassified {
		return "unclassified"
	}
	return string(r)
}

// blastOf reads the blast-radius sentence off the grammar row — the single source of
// truth. A verb with no declared sentence falls back to an explicit unknown-radius line
// rather than a blank, so a card can never imply "no consequences" by omission.
func blastOf(v Verb) string {
	if spec, ok := lookup(string(v)); ok && spec.Blast != "" {
		return spec.Blast
	}
	return "unknown — this verb declares no blast radius"
}

// distinctAdmins counts the distinct non-empty operator ids in the allowlist. Used only by
// the self-approval policy: a duplicated entry is one operator listed twice, not a second
// seat, and must not flip the one-operator fleet into the multi-operator rule.
func distinctAdmins(admins []string) int {
	seen := map[string]bool{}
	for _, a := range admins {
		if a != "" {
			seen[a] = true
		}
	}
	return len(seen)
}
