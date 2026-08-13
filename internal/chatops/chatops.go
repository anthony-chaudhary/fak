// Package chatops is the inbound chatops DOOR — epic #2259 leaf C4 (#2264): the
// pure fold that turns one raw Slack message into either a closed-grammar verb the
// fleet operator controls understand, or a structured refusal. It is the READ/authz
// half of the control surface; the ACT half's execution kernel (internal/chatopsdetach,
// C5) consumes the Command this door produces.
//
// Tier: foundation (1) — see internal/architest. State in, decision out, no I/O: the
// door takes a Message + a Config and returns a Result, touching no wire, no clock,
// and no git. That is the same discipline internal/chatopsdetach and
// internal/dispatchtick's preflight hold, and it is what lets a test pin the whole
// security boundary with plain structs and no Slack.
//
// # Why a door at all
//
// "Control from Slack" is a lethal-trifecta surface (SLACK-CONTROL-FOUNDATION note,
// docs/notes/): channel text is attacker-influenced, the bot holds fleet-mutating
// power, and it can post out. The door denies one leg by construction — CHANNEL TEXT
// IS NEVER INSTRUCTIONS. A mention parses as exactly one member of a CLOSED verb set
// or it is refused; there is no natural-language execution path, so injected prose in
// the channel cannot become a command. Free-form text reaches an agent only later, as
// data inside an already-authorized task, never through this door.
//
// # The ordered fence
//
// Parse applies the checks in a fixed order so the refusal a caller sees is the FIRST
// gate it failed, and so nothing about the grammar leaks to an unauthorized sender:
//
//  1. BOT_LOOP        — a message from any bot/app (or the door's own user) is ignored,
//     the loop fence that stops the door answering itself.
//  2. WRONG_CHANNEL   — the door listens in exactly one invited control channel.
//  3. NOT_ADDRESSED   — the door acts only on an explicit @mention of its bot user.
//  4. EMPTY           — a bare mention with no verb is a no-op, not a command.
//  5. NOT_ADMIN       — authorization is a FAIL-CLOSED allowlist of immutable Slack
//     user ids (Uxxxx), never display names. Checked before the verb
//     is even looked up, so an unauthorized sender learns nothing
//     about the grammar.
//  6. UNKNOWN_VERB    — the first token is not a member of the closed set.
//  7. MISSING_OPERAND — the verb needs an argument (e.g. `dispatch #2265`) and got none.
//
// A Result that survives all seven carries the verb, its operand, the idempotency
// nonce (the Slack ts — internal/slackwire's IdemEventType contract), and the channel
// and user for the ack/audit trail. For an act verb the impure shell lifts those
// fields into an internal/chatopsdetach.Command (the Verb string values are shared by
// design) and routes it through guarded dispatch; a read verb is answered inline —
// notably `status`/`fleet`, which surface the gateway's cross-machine SessionFleet
// aggregate straight into the channel.
//
// # The second fence: approvals (#2266)
//
// Surviving the parse fence is NOT permission to run. A verb whose reversibility label
// is anything but `reversible` takes a second, TWO-TURN fence — propose → approve →
// execute — implemented in approval.go and documented there. The one-line summary: the
// grammar row carries a Risk consumed verbatim from the adjudicator's reversibility
// vocabulary (#2156), Result.Gated() reports whether the two-turn contract applies,
// Propose mints a single-use TTL-bounded nonce plus its proposal card, and Adjudicate
// folds an `approve <nonce>` / `deny <nonce>` reply into an execute/deny/refuse verdict
// with the audit rows the journal is built from. That fence is why the parse fence's
// output can be routed at all: a single chat line can never fire an irreversible verb.
//
// v0 (this file plus approval.go) is the reply-keyword surface, which works on the
// polling transport and needs nothing new on the wire. v1 swaps keywords for Block Kit
// buttons and is BLOCKED-BY the inbound transport spike #2267: `block_actions` payloads
// require an interactive transport (Socket Mode child) that does not exist yet. v1
// changes the INPUT SURFACE ONLY — the nonce, the TTL, the self-approval policy and the
// audit rows are identical — so every v0 test here keeps guarding the v1 contract.
package chatops

import "strings"

// Class is the coarse routing lane a parsed verb falls into — the one bit the shell
// needs to decide whether to answer inline, dispatch detached, or hit the kill-switch.
type Class int

const (
	// ClassRefused is the parse/authz rejection lane: Reason carries the closed token.
	ClassRefused Class = iota
	// ClassRead is a read-only query answered inline (status, fleet, help, ping) — no
	// mutation, no detached run, no idempotency ledger needed.
	ClassRead
	// ClassControl is a control-plane verb that acts on the door itself rather than
	// dispatching work — the halt kill-switch.
	ClassControl
	// ClassAct is a mutating verb the shell must DETACH through guarded dispatch
	// (internal/chatopsdetach): ack now, witnessed completion out-of-band. It is placed
	// last so the zero Class is ClassRefused — a zero Result is a refusal, never an
	// accidental accept.
	ClassAct
)

// String renders a Class for logs and audit rows.
func (c Class) String() string {
	switch c {
	case ClassRead:
		return "read"
	case ClassControl:
		return "control"
	case ClassAct:
		return "act"
	default:
		return "refused"
	}
}

// Verb is one member of the CLOSED chatops grammar. The set is deliberately small:
// it IS the command language, so an attacker cannot widen it with prose.
type Verb string

const (
	VerbHelp     Verb = "help"     // list the verbs the door accepts
	VerbPing     Verb = "ping"     // liveness probe — the door replies pong
	VerbStatus   Verb = "status"   // session + fleet status snapshot
	VerbFleet    Verb = "fleet"    // the cross-machine fleet aggregate (gateway.SessionFleet)
	VerbDispatch Verb = "dispatch" // start a detached issue-resolution run
	VerbResume   Verb = "resume"   // resume a stalled loop/run
	VerbBench    Verb = "bench"    // kick a background benchmark
	VerbHalt     Verb = "halt"     // kill-switch: stop the door acting on further commands
	VerbApprove  Verb = "approve"  // second turn of the approval contract: `approve <nonce>`
	VerbDeny     Verb = "deny"     // second turn of the approval contract: `deny <nonce>`
)

// The closed refusal vocabulary. Every rejection carries exactly one of these tokens,
// so the reason is a first-class value a downstream audit or `dos man wedge <TOKEN> --explain` can
// bind rather than free-text prose.
const (
	ReasonBotLoop        = "BOT_LOOP"
	ReasonWrongChannel   = "WRONG_CHANNEL"
	ReasonNotAddressed   = "NOT_ADDRESSED"
	ReasonEmpty          = "EMPTY"
	ReasonNotAdmin       = "NOT_ADMIN"
	ReasonUnknownVerb    = "UNKNOWN_VERB"
	ReasonMissingOperand = "MISSING_OPERAND"
)

// VerbSpec is one row of the closed grammar: the verb, its routing class, the
// reversibility label the approval gate reads (#2156's vocabulary, consumed verbatim —
// see Risk), whether it requires an operand, a one-line help string, and the blast-radius
// sentence the proposal card prints for a gated verb. Grammar() is the single source of
// truth the door parses against, the `help` verb renders from, AND the approval gate
// labels from, so the three can never drift.
type VerbSpec struct {
	Verb         Verb
	Class        Class
	Risk         Risk // the consumed reversibility label; the zero value gates (fail-closed)
	NeedsOperand bool
	Help         string
	// Blast is the one-line blast-radius sentence Card() prints for a gated verb —
	// what actually happens if this runs, not how to type it. Empty for a reversible
	// verb, which never reaches a proposal card.
	Blast string
}

var grammar = []VerbSpec{
	{Verb: VerbHelp, Class: ClassRead, Risk: RiskReversible,
		Help: "list the verbs the door accepts"},
	{Verb: VerbPing, Class: ClassRead, Risk: RiskReversible,
		Help: "liveness check — the door replies pong"},
	{Verb: VerbStatus, Class: ClassRead, Risk: RiskReversible,
		Help: "current session + fleet status snapshot"},
	{Verb: VerbFleet, Class: ClassRead, Risk: RiskReversible,
		Help: "the cross-machine fleet aggregate: verdict, machine count, stale/action split"},
	{Verb: VerbDispatch, Class: ClassAct, Risk: RiskOutwardFacing, NeedsOperand: true,
		Help:  "start a detached run for an issue, e.g. `dispatch #2265`",
		Blast: "starts a detached worker that commits and pushes on the shared trunk and comments on the issue"},
	{Verb: VerbResume, Class: ClassAct, Risk: RiskOutwardFacing, NeedsOperand: true,
		Help:  "resume a stalled loop or run by id",
		Blast: "wakes a stalled worker that resumes committing and pushing on the shared trunk"},
	{Verb: VerbBench, Class: ClassAct, Risk: RiskIrreversible, NeedsOperand: true,
		Help:  "kick a background benchmark, e.g. `bench frontierswe`",
		Blast: "burns metered model/GPU spend that cannot be un-spent once the run starts"},
	// The kill-switch is deliberately REVERSIBLE, so it fires on one line. Halting is
	// the safe direction (it stops the door acting), it is undone by re-enabling the
	// door, and a brake you must approve twice is a brake that does not work.
	{Verb: VerbHalt, Class: ClassControl, Risk: RiskReversible,
		Help: "kill-switch: stop the door acting on further commands"},
	// The second turn of the contract itself. Reversible by construction: gating an
	// approval verb would be an infinite regress (approving the approval). Authorization
	// is still the same fail-closed admin allowlist, checked twice — once by Parse's
	// NOT_ADMIN fence and again by Adjudicate.
	{Verb: VerbApprove, Class: ClassControl, Risk: RiskReversible, NeedsOperand: true,
		Help: "approve a pending gated command in its thread, e.g. `approve a1b2c3d4`"},
	{Verb: VerbDeny, Class: ClassControl, Risk: RiskReversible, NeedsOperand: true,
		Help: "deny a pending gated command in its thread, e.g. `deny a1b2c3d4`"},
}

// Grammar returns a copy of the closed verb set, best-first for help rendering. The
// slice is copied so a caller cannot mutate the door's grammar.
func Grammar() []VerbSpec { return append([]VerbSpec(nil), grammar...) }

// Reasons returns the package's closed refusal vocabulary — every token Parse or
// Adjudicate can emit. The first seven are the door's ordered parse fence; the rest are
// the approval gate's OPERATOR_GATE family (see approval.go). One flat set, so a
// downstream audit or `dos man wedge <TOKEN> --explain` can bind any refusal this package produces
// without knowing which half emitted it.
func Reasons() []string {
	return []string{
		ReasonBotLoop, ReasonWrongChannel, ReasonNotAddressed,
		ReasonEmpty, ReasonNotAdmin, ReasonUnknownVerb, ReasonMissingOperand,
		ReasonApprovalUnknownNonce, ReasonApprovalForeignThread,
		ReasonApprovalReplayed, ReasonApprovalExpired, ReasonApprovalSelf,
	}
}

// Message is one inbound Slack message as the transport delivered it — the raw fields
// the door needs, nothing more. Text INCLUDES the leading `<@Uxxxx>` mention; the door
// strips it. TS is the Slack message timestamp, which doubles as the idempotency nonce
// for act verbs.
type Message struct {
	User    string // the immutable Slack user id (Uxxxx) of the sender
	BotID   string // non-empty iff this message was posted BY a bot/app (loop fence)
	Channel string // the channel id the message landed in
	TS      string // the Slack ts — the idempotency nonce for act verbs
	Text    string // the full message text, mention included
}

// Config is the door's authorization boundary: the one control channel and the
// fail-closed admin allowlist. Admins are IMMUTABLE user ids (Uxxxx), never display
// names — a display name is renameable and spoofable. An empty Admins set refuses
// EVERY command: the door is dark until an operator seeds it.
type Config struct {
	BotUserID      string   // the bot's own user id, for mention-stripping + the self-loop fence
	ControlChannel string   // the one channel the door listens in; "" disables the channel gate
	Admins         []string // allowlisted sender user ids; empty ⇒ refuse all (fail-closed)
}

// Result is the door's decision for one Message.
type Result struct {
	Class   Class
	Verb    Verb
	Risk    Risk   // the verb's consumed reversibility label; drives Gated()
	Operand string // the verb's argument (issue id, run id, …); "" when the verb takes none
	Nonce   string // the idempotency key (the message TS) for act verbs
	Channel string // where the reply/ack posts
	User    string // who issued it (for the audit trail)
	Refused bool
	Reason  string // a closed refusal token (one of Reasons()); set iff Refused
}

// Gated reports whether this parsed command must take the two-turn approval contract
// (propose → approve → execute) instead of executing on the one chat line that carried
// it. A refusal is never gated — it already stopped at the fence.
func (r Result) Gated() bool { return !r.Refused && r.Risk.Gated() }

// Parse folds one inbound Message through the ordered fence and returns the decision.
// It is total and pure: same inputs, same Result, no side effects.
func Parse(m Message, cfg Config) Result {
	// 1. Loop fence — never act on a bot/app message or the door's own user.
	if m.BotID != "" || (cfg.BotUserID != "" && m.User == cfg.BotUserID) {
		return refuse(m, ReasonBotLoop)
	}
	// 2. Channel gate — only the one invited control channel (when configured).
	if cfg.ControlChannel != "" && m.Channel != cfg.ControlChannel {
		return refuse(m, ReasonWrongChannel)
	}
	// 3. Addressing — the door acts only on an explicit @mention of its bot user.
	body, addressed := stripMention(m.Text, cfg.BotUserID)
	if cfg.BotUserID != "" && !addressed {
		return refuse(m, ReasonNotAddressed)
	}
	// 4. A bare mention with no verb is a no-op, not a command.
	fields := strings.Fields(body)
	if len(fields) == 0 {
		return refuse(m, ReasonEmpty)
	}
	// 5. Authorization — fail-closed allowlist on the immutable user id, BEFORE the
	//    verb is looked up so an unauthorized sender learns nothing about the grammar.
	if !isAdmin(m.User, cfg.Admins) {
		return refuse(m, ReasonNotAdmin)
	}
	// 6. The first token must be a member of the closed grammar.
	spec, ok := lookup(fields[0])
	if !ok {
		return refuse(m, ReasonUnknownVerb)
	}
	// 7. Operand requirement for the verbs that mutate on one.
	operand := strings.TrimSpace(strings.Join(fields[1:], " "))
	if spec.NeedsOperand && operand == "" {
		return refuse(m, ReasonMissingOperand)
	}
	return Result{
		Class:   spec.Class,
		Verb:    spec.Verb,
		Risk:    spec.Risk,
		Operand: operand,
		Nonce:   m.TS,
		Channel: m.Channel,
		User:    m.User,
	}
}

// refuse builds a rejection Result carrying the closed reason token, still threading
// the channel/user/nonce so the shell can post the structured refusal back and audit it.
func refuse(m Message, reason string) Result {
	return Result{
		Class:   ClassRefused,
		Refused: true,
		Reason:  reason,
		Channel: m.Channel,
		User:    m.User,
		Nonce:   m.TS,
	}
}

// lookup resolves a lowercased first token to its grammar row (case-insensitive: Slack
// clients love to auto-capitalize a sentence-leading word).
func lookup(tok string) (VerbSpec, bool) {
	tok = strings.ToLower(tok)
	for _, s := range grammar {
		if string(s.Verb) == tok {
			return s, true
		}
	}
	return VerbSpec{}, false
}

// isAdmin is the fail-closed membership test: an empty user or an empty allowlist can
// never match. Slack user ids are case-sensitive, so the comparison is exact.
func isAdmin(user string, admins []string) bool {
	if user == "" {
		return false
	}
	for _, a := range admins {
		if a == user {
			return true
		}
	}
	return false
}

// stripMention removes a leading-or-embedded `<@BOTID>` (or `<@BOTID|label>`) mention
// from text and reports whether one was present. When botID is empty the addressing
// gate is disabled (a degenerate/test config): the whole text is the body and it is
// treated as addressed.
func stripMention(text, botID string) (body string, addressed bool) {
	if botID == "" {
		return strings.TrimSpace(text), true
	}
	open := "<@" + botID
	i := strings.Index(text, open)
	if i < 0 {
		return strings.TrimSpace(text), false
	}
	// The char right after the id must close the token (`>`) or introduce a label
	// (`|`); otherwise we matched a longer id that merely shares this prefix.
	rest := text[i+len(open):]
	if rest == "" || (rest[0] != '>' && rest[0] != '|') {
		return strings.TrimSpace(text), false
	}
	close := strings.IndexByte(rest, '>')
	if close < 0 {
		return strings.TrimSpace(text), false
	}
	body = strings.TrimSpace(text[:i] + " " + rest[close+1:])
	return body, true
}
