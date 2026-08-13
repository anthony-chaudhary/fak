package chatops

import (
	"strings"
	"testing"
	"time"
)

// t0 is the fixed wall clock every approval test folds against — the kernel takes `now`
// as an argument, so the whole TTL/replay boundary is pinnable with no real clock.
var t0 = time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)

// proposeMsg is the first turn: an admin asking for a gated verb in the control channel.
func proposeMsg(text string) Message {
	return Message{User: "UADMIN", Channel: "CTRL", TS: "1754568000.000100", Text: text}
}

// replyMsg is the second turn: an approve/deny reply in the proposal's thread.
func replyMsg(user, channel, text string) Message {
	return Message{User: user, Channel: channel, TS: "1754568060.000200", Text: text}
}

// gatedProposal runs the real first turn for `dispatch #2265` — the one real gated verb
// this suite wires end to end — and returns its proposal and journal row.
func gatedProposal(t *testing.T) (Proposal, AuditRow) {
	t.Helper()
	res := Parse(proposeMsg("<@UBOT> dispatch #2265"), baseCfg())
	if res.Refused {
		t.Fatalf("the proposing command must parse; got refusal %q", res.Reason)
	}
	p, row, ok := Propose(res, t0, DefaultTTL)
	if !ok {
		t.Fatal("dispatch is an outward-facing verb and must mint a proposal")
	}
	return p, row
}

// ---------------------------------------------------------------------------
// The consumed label vocabulary (#2156)
// ---------------------------------------------------------------------------

// The Risk values ARE the adjudicator's ReversibilityClass strings. chatops is tier 1 and
// internal/adjudicator is tier 2, so the vocabulary is copied rather than imported and
// pinned here as literals: if internal/adjudicator/reversibility.go renames a class, this
// test fails instead of the door quietly labelling verbs against a dead taxonomy.
func TestRiskVocabularyIsTheAdjudicatorsClasses(t *testing.T) {
	for _, tc := range []struct {
		got  Risk
		want string
	}{
		{RiskReversible, "reversible"},
		{RiskIrreversible, "irreversible"},
		{RiskOutwardFacing, "outward-facing"},
	} {
		if string(tc.got) != tc.want {
			t.Errorf("Risk %q must equal the adjudicator's class %q", tc.got, tc.want)
		}
	}
}

// Only an explicitly reversible verb escapes the two-turn contract. The ZERO Risk gates,
// so a new grammar row nobody labeled fails closed.
func TestRisk_OnlyReversibleIsUngated(t *testing.T) {
	for r, wantGated := range map[Risk]bool{
		RiskUnclassified:                true,
		RiskIrreversible:                true,
		RiskOutwardFacing:               true,
		RiskReversible:                  false,
		Risk("invented-upstream-label"): true,
	} {
		if got := r.Gated(); got != wantGated {
			t.Errorf("Risk(%q).Gated() = %v; want %v", r, got, wantGated)
		}
	}
	var zero Risk
	if !zero.Gated() {
		t.Fatal("the zero Risk must gate — an unlabeled verb can never fire on one line")
	}
}

// Every grammar row is labeled, every gated row declares a blast radius, and no MUTATING
// verb is labeled reversible. That last clause is the real invariant: relabelling an act
// verb `reversible` is exactly how this gate would be defeated by a one-word edit.
func TestGrammar_LabelsAreCompleteAndActVerbsGate(t *testing.T) {
	for _, s := range Grammar() {
		if s.Risk == RiskUnclassified {
			t.Errorf("verb %q carries no reversibility label", s.Verb)
		}
		if s.Risk.Gated() && s.Blast == "" {
			t.Errorf("gated verb %q declares no blast-radius line for its proposal card", s.Verb)
		}
		if !s.Risk.Gated() && s.Blast != "" {
			t.Errorf("reversible verb %q declares a blast radius but never reaches a card", s.Verb)
		}
		if s.Class == ClassAct && !s.Risk.Gated() {
			t.Errorf("act verb %q is labeled %q — a mutating verb must never run on one chat line", s.Verb, s.Risk)
		}
	}
}

// Parse threads the grammar's label onto the Result, so the shell reads gating off the
// parse output instead of re-deriving it.
func TestParse_ThreadsRiskAndGating(t *testing.T) {
	cfg := baseCfg()
	cases := []struct {
		text  string
		risk  Risk
		gated bool
	}{
		{"<@UBOT> status", RiskReversible, false},
		{"<@UBOT> ping", RiskReversible, false},
		{"<@UBOT> halt", RiskReversible, false},
		{"<@UBOT> approve abcd1234", RiskReversible, false},
		{"<@UBOT> deny abcd1234", RiskReversible, false},
		{"<@UBOT> dispatch #2265", RiskOutwardFacing, true},
		{"<@UBOT> resume run-42", RiskOutwardFacing, true},
		{"<@UBOT> bench frontierswe", RiskIrreversible, true},
	}
	for _, tc := range cases {
		got := Parse(proposeMsg(tc.text), cfg)
		if got.Refused {
			t.Errorf("%q: unexpected refusal %s", tc.text, got.Reason)
			continue
		}
		if got.Risk != tc.risk || got.Gated() != tc.gated {
			t.Errorf("%q: risk=%q gated=%v; want risk=%q gated=%v",
				tc.text, got.Risk, got.Gated(), tc.risk, tc.gated)
		}
	}
	// A refusal is never "gated" — it already stopped at the parse fence.
	refused := Parse(Message{User: "UX", Channel: "CTRL", Text: "<@UBOT> dispatch #1"}, cfg)
	if !refused.Refused || refused.Gated() {
		t.Fatalf("a refused parse must not report as gated; got %+v", refused)
	}
}

// ---------------------------------------------------------------------------
// Turn one — Propose
// ---------------------------------------------------------------------------

// Propose freezes the command, binds it to the proposing thread, and stamps a deadline.
func TestPropose_BindsTheCommandAndItsDeadline(t *testing.T) {
	p, row := gatedProposal(t)
	if p.Nonce == "" {
		t.Fatal("a proposal must carry an approval nonce")
	}
	if p.Verb != VerbDispatch || p.Operand != "#2265" || p.Risk != RiskOutwardFacing {
		t.Errorf("proposal did not freeze the command: %+v", p)
	}
	if p.Proposer != "UADMIN" || p.Channel != "CTRL" || p.ThreadTS != "1754568000.000100" {
		t.Errorf("proposal not bound to its proposer/thread: %+v", p)
	}
	if !p.ExpiresAt.Equal(t0.Add(DefaultTTL)) {
		t.Errorf("ExpiresAt = %v; want now+%v", p.ExpiresAt, DefaultTTL)
	}
	if p.Resolved || p.Verdict != "" {
		t.Errorf("a fresh proposal must be unburned: %+v", p)
	}
	// The propose row is the journal's first turn: who asked, for what, when — and NO
	// approver, because nobody has approved anything yet.
	if row.Event != EventPropose || row.Nonce != p.Nonce || row.Proposer != "UADMIN" {
		t.Errorf("propose row does not record the first turn: %+v", row)
	}
	if row.Approver != "" || row.Verdict != "" || row.RunID != "" {
		t.Errorf("a propose row must claim no approver, verdict, or run: %+v", row)
	}
}

// A non-positive TTL falls back to DefaultTTL rather than minting an already-dead (or
// unbounded) proposal.
func TestPropose_NonPositiveTTLFallsBackToDefault(t *testing.T) {
	res := Parse(proposeMsg("<@UBOT> bench frontierswe"), baseCfg())
	for _, ttl := range []time.Duration{0, -time.Hour} {
		p, _, ok := Propose(res, t0, ttl)
		if !ok {
			t.Fatalf("ttl=%v: bench must mint a proposal", ttl)
		}
		if !p.ExpiresAt.Equal(t0.Add(DefaultTTL)) {
			t.Errorf("ttl=%v: ExpiresAt = %v; want the default deadline", ttl, p.ExpiresAt)
		}
	}
}

// The nonce is derived from the MESSAGE, not the clock: a re-delivered Slack event mints
// the same nonce and therefore cannot open a second pending approval for one command.
func TestPropose_NonceIsDeterministicAcrossRedelivery(t *testing.T) {
	res := Parse(proposeMsg("<@UBOT> dispatch #2265"), baseCfg())
	first, _, _ := Propose(res, t0, DefaultTTL)
	redelivered, _, _ := Propose(res, t0.Add(90*time.Second), DefaultTTL)
	if first.Nonce != redelivered.Nonce {
		t.Fatalf("a re-delivery must re-mint the same nonce; got %q then %q", first.Nonce, redelivered.Nonce)
	}
}

// Different commands — and the same command from a different operator or thread — get
// different nonces, so one approval can never be spent on another proposal.
func TestPropose_DistinctCommandsGetDistinctNonces(t *testing.T) {
	cfg := Config{BotUserID: "UBOT", ControlChannel: "CTRL", Admins: []string{"UADMIN", "UOTHER"}}
	mk := func(m Message) string {
		p, _, ok := Propose(Parse(m, cfg), t0, DefaultTTL)
		if !ok {
			t.Fatalf("expected a gated proposal for %+v", m)
		}
		return p.Nonce
	}
	base := proposeMsg("<@UBOT> dispatch #2265")
	otherOperand := proposeMsg("<@UBOT> dispatch #2267")
	otherVerb := proposeMsg("<@UBOT> bench frontierswe")
	otherUser := base
	otherUser.User = "UOTHER"
	otherThread := base
	otherThread.TS = "1754569999.000999"

	seen := map[string]string{}
	for name, m := range map[string]Message{
		"base": base, "operand": otherOperand, "verb": otherVerb,
		"user": otherUser, "thread": otherThread,
	} {
		n := mk(m)
		if prev, dup := seen[n]; dup {
			t.Errorf("nonce %q collides between %q and %q", n, prev, name)
		}
		seen[n] = name
	}
}

// Propose refuses to mint a card for anything that must not have one: a reversible verb
// (it just runs) or a refused parse (it already stopped).
func TestPropose_OnlyMintsForGatedCommands(t *testing.T) {
	cfg := baseCfg()
	for _, text := range []string{"<@UBOT> status", "<@UBOT> ping", "<@UBOT> halt"} {
		if _, _, ok := Propose(Parse(proposeMsg(text), cfg), t0, DefaultTTL); ok {
			t.Errorf("%q is reversible and must not mint a proposal", text)
		}
	}
	refused := Parse(Message{User: "UX", Channel: "CTRL", Text: "<@UBOT> dispatch #1"}, cfg)
	if _, _, ok := Propose(refused, t0, DefaultTTL); ok {
		t.Error("a refused parse must never mint a proposal")
	}
}

// ---------------------------------------------------------------------------
// Turn two — Adjudicate: the authz x nonce-replay x TTL matrix
// ---------------------------------------------------------------------------

// The refusal matrix. Each row perturbs exactly ONE dimension of a known-good approval —
// who replied (authz), which nonce and where (replay/binding), or when (TTL) — and pins
// the closed token it must produce. The final row is the control: unperturbed, it executes.
func TestAdjudicate_RefusalMatrix(t *testing.T) {
	p, _ := gatedProposal(t)
	burned := p
	burned.Resolved = true
	burned.Verdict = VerdictApproved
	noDeadline := p
	noDeadline.ExpiresAt = time.Time{}

	twoAdmins := Config{BotUserID: "UBOT", ControlChannel: "CTRL", Admins: []string{"UADMIN", "USECOND"}}

	cases := []struct {
		name    string
		cfg     Config
		reply   Message
		held    Proposal
		now     time.Time
		outcome Outcome
		reason  string
	}{
		// --- authz ---
		{
			name:  "authz: a non-admin reply never reaches the ledger",
			cfg:   baseCfg(),
			reply: replyMsg("UINTRUDER", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  p, now: t0, outcome: OutcomeRefused, reason: ReasonNotAdmin,
		},
		{
			name:  "authz: an empty allowlist refuses fail-closed",
			cfg:   Config{BotUserID: "UBOT", ControlChannel: "CTRL"},
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  p, now: t0, outcome: OutcomeRefused, reason: ReasonNotAdmin,
		},
		{
			name:  "authz: self-approval refused once a second operator exists",
			cfg:   twoAdmins,
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  p, now: t0, outcome: OutcomeRefused, reason: ReasonApprovalSelf,
		},
		// --- nonce / replay ---
		{
			name:  "nonce: no proposal held under it",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve deadbeef"),
			held:  Proposal{}, now: t0, outcome: OutcomeRefused, reason: ReasonApprovalUnknownNonce,
		},
		{
			name:  "nonce: a real proposal but the wrong nonce typed",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve deadbeef"),
			held:  p, now: t0, outcome: OutcomeRefused, reason: ReasonApprovalUnknownNonce,
		},
		{
			name:  "nonce: spent in a foreign channel",
			cfg:   Config{BotUserID: "UBOT", Admins: []string{"UADMIN", "USECOND"}}, // channel gate off
			reply: replyMsg("USECOND", "OTHER", "<@UBOT> approve "+p.Nonce),
			held:  p, now: t0, outcome: OutcomeRefused, reason: ReasonApprovalForeignThread,
		},
		{
			name:  "replay: the nonce was already burned",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  burned, now: t0, outcome: OutcomeRefused, reason: ReasonApprovalReplayed,
		},
		// --- TTL ---
		{
			name:  "ttl: exactly at the deadline is already too late",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  p, now: p.ExpiresAt, outcome: OutcomeRefused, reason: ReasonApprovalExpired,
		},
		{
			name:  "ttl: past the deadline",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  p, now: p.ExpiresAt.Add(time.Second), outcome: OutcomeRefused, reason: ReasonApprovalExpired,
		},
		{
			name:  "ttl: a missing deadline is expired, not unlimited",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  noDeadline, now: t0, outcome: OutcomeRefused, reason: ReasonApprovalExpired,
		},
		// --- shape ---
		{
			name:  "shape: a non-approval verb never adjudicates",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> status"),
			held:  p, now: t0, outcome: OutcomeRefused, reason: ReasonUnknownVerb,
		},
		{
			name:  "shape: a parse refusal is threaded through unchanged",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve"), // no nonce operand
			held:  p, now: t0, outcome: OutcomeRefused, reason: ReasonMissingOperand,
		},
		// --- control: nothing perturbed ---
		{
			name:  "control: one operator, live nonce, right thread => execute",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  p, now: t0.Add(time.Minute), outcome: OutcomeExecute, reason: "",
		},
	}

	for _, tc := range cases {
		v := Adjudicate(Parse(tc.reply, tc.cfg), tc.held, tc.cfg, tc.now)
		if v.Outcome != tc.outcome || v.Reason != tc.reason {
			t.Errorf("%s: outcome=%s reason=%q; want outcome=%s reason=%q",
				tc.name, v.Outcome, v.Reason, tc.outcome, tc.reason)
			continue
		}
		if tc.outcome == OutcomeRefused {
			// A refusal must never burn the nonce and must never yield an execution row.
			if v.Proposal.Resolved && !tc.held.Resolved {
				t.Errorf("%s: a refusal burned a still-valid nonce", tc.name)
			}
			if _, ok := v.ExecuteRow("run-1", tc.now); ok {
				t.Errorf("%s: a refusal must produce no execution row", tc.name)
			}
			if v.Audit.Event != EventRefuse || v.Audit.Reason != tc.reason {
				t.Errorf("%s: refusal row does not carry the token: %+v", tc.name, v.Audit)
			}
		}
	}
}

// The happy path: approve executes, burns the nonce, and writes an approval row naming
// both the proposer and the approver.
func TestAdjudicate_ApproveExecutesAndBurnsTheNonce(t *testing.T) {
	p, _ := gatedProposal(t)
	cfg := baseCfg()
	at := t0.Add(2 * time.Minute)
	v := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg), p, cfg, at)

	if v.Outcome != OutcomeExecute {
		t.Fatalf("outcome=%s reason=%q; want EXECUTE", v.Outcome, v.Reason)
	}
	if !v.Proposal.Resolved || v.Proposal.Verdict != VerdictApproved {
		t.Errorf("the nonce must be burned on approval: %+v", v.Proposal)
	}
	row := v.Audit
	if row.Event != EventApprove || row.Nonce != p.Nonce || row.Verdict != VerdictApproved {
		t.Errorf("approval row: %+v", row)
	}
	if row.Proposer != "UADMIN" || row.Approver != "UADMIN" || !row.At.Equal(at) {
		t.Errorf("approval row must carry proposer/approver/when: %+v", row)
	}
	if row.Verb != VerbDispatch || row.Operand != "#2265" || row.Risk != RiskOutwardFacing {
		t.Errorf("approval row must carry what was approved: %+v", row)
	}
	if row.RunID != "" {
		t.Error("an approval row must not claim a run — execution is a separate row")
	}
	// Re-playing the SAME approval against the burned proposal is refused.
	replay := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg), v.Proposal, cfg, at)
	if replay.Outcome != OutcomeRefused || replay.Reason != ReasonApprovalReplayed {
		t.Fatalf("a burned nonce must refuse on replay; got outcome=%s reason=%q", replay.Outcome, replay.Reason)
	}
}

// Deny burns the nonce exactly like approve, but releases nothing to run.
func TestAdjudicate_DenyBurnsAndExecutesNothing(t *testing.T) {
	p, _ := gatedProposal(t)
	cfg := baseCfg()
	v := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> deny "+p.Nonce), cfg), p, cfg, t0)

	if v.Outcome != OutcomeDenied {
		t.Fatalf("outcome=%s reason=%q; want DENIED", v.Outcome, v.Reason)
	}
	if !v.Proposal.Resolved || v.Proposal.Verdict != VerdictDenied {
		t.Errorf("a denial must burn the nonce: %+v", v.Proposal)
	}
	if v.Audit.Event != EventDeny || v.Audit.Verdict != VerdictDenied {
		t.Errorf("denial row: %+v", v.Audit)
	}
	if _, ok := v.ExecuteRow("run-1", t0); ok {
		t.Fatal("a denial must produce no execution row")
	}
	// A denied nonce cannot be re-approved: deny is final, not a retry prompt.
	retry := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg), v.Proposal, cfg, t0)
	if retry.Outcome != OutcomeRefused || retry.Reason != ReasonApprovalReplayed {
		t.Fatalf("a denied nonce must not be re-approvable; got outcome=%s reason=%q", retry.Outcome, retry.Reason)
	}
}

// The self-approval policy is explicit: allowed while the allowlist names exactly ONE
// operator, refused as soon as a second seat exists, and always recorded in the row.
func TestAdjudicate_SelfApprovalPolicy(t *testing.T) {
	p, _ := gatedProposal(t)
	one := baseCfg() // Admins: {"UADMIN"}
	dup := Config{BotUserID: "UBOT", ControlChannel: "CTRL", Admins: []string{"UADMIN", "UADMIN"}}
	two := Config{BotUserID: "UBOT", ControlChannel: "CTRL", Admins: []string{"UADMIN", "USECOND"}}

	// One operator: the proposer may approve, and the row says so.
	v := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), one), p, one, t0)
	if v.Outcome != OutcomeExecute || !v.SelfApproved || !v.Audit.SelfApproved {
		t.Fatalf("one-operator self-approval must execute AND be recorded; got %+v", v)
	}

	// A duplicated entry is one operator listed twice, not a second seat — it must not
	// lock the single operator out of their own fleet.
	vd := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), dup), p, dup, t0)
	if vd.Outcome != OutcomeExecute || !vd.SelfApproved {
		t.Fatalf("a duplicated allowlist entry is still one operator; got outcome=%s reason=%q", vd.Outcome, vd.Reason)
	}

	// Two operators: the proposer may not approve their own proposal...
	vs := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), two), p, two, t0)
	if vs.Outcome != OutcomeRefused || vs.Reason != ReasonApprovalSelf {
		t.Fatalf("two-operator self-approval must refuse; got outcome=%s reason=%q", vs.Outcome, vs.Reason)
	}
	// ...but the second operator may, and that row is NOT a self-approval.
	vo := Adjudicate(Parse(replyMsg("USECOND", "CTRL", "<@UBOT> approve "+p.Nonce), two), p, two, t0)
	if vo.Outcome != OutcomeExecute || vo.SelfApproved || vo.Audit.SelfApproved {
		t.Fatalf("a second operator must be able to approve, un-flagged; got %+v", vo)
	}
	if vo.Audit.Proposer != "UADMIN" || vo.Audit.Approver != "USECOND" {
		t.Errorf("the row must distinguish proposer from approver: %+v", vo.Audit)
	}
}

// A refusal must NOT consume the nonce: a mistyped channel or a premature reply cannot
// destroy an approval the operator is still entitled to give.
func TestAdjudicate_RefusalLeavesTheNonceSpendable(t *testing.T) {
	p, _ := gatedProposal(t)
	cfg := baseCfg()
	bad := Adjudicate(Parse(replyMsg("UINTRUDER", "CTRL", "<@UBOT> approve "+p.Nonce), cfg), p, cfg, t0)
	if bad.Outcome != OutcomeRefused {
		t.Fatalf("expected a refusal, got %s", bad.Outcome)
	}
	good := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg), bad.Proposal, cfg, t0)
	if good.Outcome != OutcomeExecute {
		t.Fatalf("a refusal must not burn the nonce; the real approval got outcome=%s reason=%q",
			good.Outcome, good.Reason)
	}
}

// The nonce compare is case-insensitive (it is hex, and Slack clients capitalize), while
// authorization stays case-SENSITIVE on the user id — the two must not be confused.
func TestAdjudicate_NonceCompareIsCaseInsensitive(t *testing.T) {
	p, _ := gatedProposal(t)
	cfg := baseCfg()
	shouted := strings.ToUpper(p.Nonce)
	v := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+shouted), cfg), p, cfg, t0)
	if v.Outcome != OutcomeExecute {
		t.Fatalf("an upper-cased nonce must still match; got outcome=%s reason=%q", v.Outcome, v.Reason)
	}
}

// The approval fence runs in its documented order. Each row makes TWO gates fail at once
// and pins which one is reported — without these, swapping a pair of checks survives the
// rest of the suite.
func TestAdjudicate_FenceOrdering(t *testing.T) {
	p, _ := gatedProposal(t)
	burnedAndExpired := p
	burnedAndExpired.Resolved = true
	expired := p.ExpiresAt.Add(time.Hour)
	two := Config{BotUserID: "UBOT", ControlChannel: "CTRL", Admins: []string{"UADMIN", "USECOND"}}

	cases := []struct {
		name   string
		cfg    Config
		reply  Message
		held   Proposal
		now    time.Time
		reason string
	}{
		{
			// NOT_ADMIN (3) precedes APPROVAL_UNKNOWN_NONCE (4): an unauthorized sender
			// learns nothing about which nonces exist.
			name:  "not-admin beats unknown-nonce",
			cfg:   baseCfg(),
			reply: replyMsg("UINTRUDER", "CTRL", "<@UBOT> approve deadbeef"),
			held:  p, now: t0, reason: ReasonNotAdmin,
		},
		{
			// UNKNOWN_NONCE (4) precedes FOREIGN_THREAD (5): a nonce that does not exist
			// is reported as unknown, never as "wrong room", which would confirm it exists.
			name:  "unknown-nonce beats foreign-thread",
			cfg:   Config{BotUserID: "UBOT", Admins: []string{"UADMIN"}},
			reply: replyMsg("UADMIN", "OTHER", "<@UBOT> approve deadbeef"),
			held:  p, now: t0, reason: ReasonApprovalUnknownNonce,
		},
		{
			// REPLAYED (6) precedes EXPIRED (7): an operator who double-taps is told they
			// already answered, which is the more actionable of the two truths.
			name:  "replayed beats expired",
			cfg:   baseCfg(),
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  burnedAndExpired, now: expired, reason: ReasonApprovalReplayed,
		},
		{
			// EXPIRED (7) precedes SELF (8): a dead proposal is dead regardless of who is
			// replying — the self-approval policy never resurrects one.
			name:  "expired beats self",
			cfg:   two,
			reply: replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce),
			held:  p, now: expired, reason: ReasonApprovalExpired,
		},
	}
	for _, tc := range cases {
		v := Adjudicate(Parse(tc.reply, tc.cfg), tc.held, tc.cfg, tc.now)
		if v.Reason != tc.reason {
			t.Errorf("%s: reason=%q; want %q", tc.name, v.Reason, tc.reason)
		}
	}
}

// Every approval refusal token is a member of the package's closed vocabulary, so a
// downstream audit can bind any of them.
func TestReasons_CarriesTheApprovalFamily(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range Reasons() {
		seen[r] = true
	}
	for _, want := range []string{
		ReasonApprovalUnknownNonce, ReasonApprovalForeignThread,
		ReasonApprovalReplayed, ReasonApprovalExpired, ReasonApprovalSelf,
	} {
		if !seen[want] {
			t.Errorf("Reasons() is missing the approval token %q", want)
		}
	}
}

// The zero Outcome is a refusal — a Verdict with a dropped field can never read as an
// accidental execute.
func TestZeroOutcomeIsRefused(t *testing.T) {
	var o Outcome
	if o != OutcomeRefused || o.String() != "REFUSED" {
		t.Fatalf("the zero Outcome must be REFUSED, got %s", o)
	}
	var v Verdict
	if _, ok := v.ExecuteRow("run-1", t0); ok {
		t.Fatal("a zero Verdict must never release an execution row")
	}
}

// ---------------------------------------------------------------------------
// The journal: separate rows, and the pending set rebuilt from them
// ---------------------------------------------------------------------------

// Execution is a SEPARATE row from approval, and only an executing verdict has one. The
// journal must never conflate "an operator approved this" with "this ran".
func TestExecuteRow_IsSeparateAndOnlyOnExecute(t *testing.T) {
	p, _ := gatedProposal(t)
	cfg := baseCfg()
	v := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg), p, cfg, t0)

	ranAt := t0.Add(3 * time.Second)
	row, ok := v.ExecuteRow("run-4242", ranAt)
	if !ok {
		t.Fatal("an approved verdict must yield an execution row")
	}
	if row.Event != EventExecute || row.RunID != "run-4242" || !row.At.Equal(ranAt) {
		t.Errorf("execution row must record what ran and when: %+v", row)
	}
	if row.Nonce != p.Nonce || row.Proposer != "UADMIN" || row.Approver != "UADMIN" {
		t.Errorf("execution row must stay bound to the approval: %+v", row)
	}
	if row.At.Equal(v.Audit.At) {
		t.Error("the execution row is stamped when the run started, not when it was approved")
	}
}

// Pending replays the journal — the source of truth — into the live approvable set.
func TestPending_JournalIsTheSourceOfTruth(t *testing.T) {
	cfg := baseCfg()
	mk := func(text, ts string, at time.Time) (Proposal, AuditRow) {
		m := proposeMsg(text)
		m.TS = ts
		p, row, ok := Propose(Parse(m, cfg), at, DefaultTTL)
		if !ok {
			t.Fatalf("%q must mint a proposal", text)
		}
		return p, row
	}
	pa, rowA := mk("<@UBOT> dispatch #2265", "1754568000.000100", t0)
	pb, rowB := mk("<@UBOT> bench frontierswe", "1754568001.000100", t0.Add(time.Minute))
	_, rowC := mk("<@UBOT> resume run-42", "1754568002.000100", t0.Add(2*time.Minute))

	journal := []AuditRow{rowA, rowB, rowC}

	// All three are live a minute later, oldest first — a slow approval stays visible.
	live := Pending(journal, t0.Add(3*time.Minute))
	if len(live) != 3 {
		t.Fatalf("want 3 pending, got %d: %+v", len(live), live)
	}
	if live[0].Nonce != pa.Nonce || live[0].Verb != VerbDispatch || live[0].Operand != "#2265" {
		t.Errorf("pending set lost the command detail / journal order: %+v", live[0])
	}
	if live[0].Proposer != "UADMIN" || live[0].Channel != "CTRL" || live[0].ThreadTS != "1754568000.000100" {
		t.Errorf("pending set lost the thread binding: %+v", live[0])
	}

	// A re-delivered propose row re-mints the same nonce and must not become a 4th entry.
	if got := len(Pending(append(journal, rowA), t0.Add(3*time.Minute))); got != 3 {
		t.Errorf("a re-delivered proposal must not double the pending set; got %d", got)
	}

	// An approve row burns its proposal out of the set; a refuse row leaves one standing.
	vA := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+pa.Nonce), cfg), pa, cfg, t0.Add(time.Minute))
	refB := Adjudicate(Parse(replyMsg("UINTRUDER", "CTRL", "<@UBOT> deny "+pb.Nonce), cfg), pb, cfg, t0.Add(time.Minute))
	after := Pending(append(journal, vA.Audit, refB.Audit), t0.Add(3*time.Minute))
	if len(after) != 2 {
		t.Fatalf("an approval must burn exactly one entry (a refusal burns none); got %d: %+v", len(after), after)
	}
	for _, p := range after {
		if p.Nonce == pa.Nonce {
			t.Error("the approved proposal is still pending")
		}
	}

	// TTL: everything falls out of the set once its deadline passes.
	if got := Pending(journal, t0.Add(DefaultTTL+3*time.Minute)); len(got) != 0 {
		t.Errorf("expired proposals must drop out of the pending set; got %d", len(got))
	}
	// The deadline is re-derived from the propose row's timestamp, so a hand-edited
	// journal cannot extend a TTL beyond the policy.
	stretched := rowC
	stretched.At = t0
	if got := Pending([]AuditRow{stretched}, t0.Add(DefaultTTL)); len(got) != 0 {
		t.Errorf("a journal row must not be able to buy itself more time; got %d", len(got))
	}
}

// ---------------------------------------------------------------------------
// The witnessed thread transcript
// ---------------------------------------------------------------------------

// The proposal card is an exact-output fixture: what would run, the blast-radius line the
// grammar declares, who asked, the nonce and its deadline, and how to answer. Pinning it
// byte-for-byte is what makes a re-posted card coalesce on the durable outbox — and what
// makes "the operator was told the blast radius" a checkable claim rather than a promise.
func TestCard_ExactTranscript(t *testing.T) {
	p := Proposal{
		Nonce:     "a1b2c3d4",
		Verb:      VerbDispatch,
		Operand:   "#2265",
		Risk:      RiskOutwardFacing,
		Proposer:  "UADMIN",
		Channel:   "CTRL",
		ThreadTS:  "1754568000.000100",
		At:        t0,
		ExpiresAt: t0.Add(DefaultTTL),
	}
	want := "approval required — `dispatch #2265`\n" +
		"blast radius: outward-facing — starts a detached worker that commits and pushes on the shared trunk and comments on the issue\n" +
		"proposed by <@UADMIN> · nonce `a1b2c3d4` · expires 2026-08-07T12:15:00Z\n" +
		"reply `approve a1b2c3d4` or `deny a1b2c3d4` in this thread"
	if got := Card(p); got != want {
		t.Errorf("card transcript drifted.\n got:\n%s\nwant:\n%s", got, want)
	}
}

// A card for an unlabeled verb names the missing label and the missing blast radius out
// loud. Silence would read as "no consequences", which is the one thing it must not mean.
func TestCard_UnknownRadiusIsStatedNotOmitted(t *testing.T) {
	got := Card(Proposal{Nonce: "ffff0000", Verb: Verb("newverb"), Operand: "x", ExpiresAt: t0})
	if !strings.Contains(got, "blast radius: unclassified — unknown") {
		t.Fatalf("an unlabeled verb must say so on its card; got:\n%s", got)
	}
}

// The whole v0 contract, end to end, as it plays in one Slack thread: a gated verb from an
// admin does NOT run, it posts a card; the card's nonce approves it; the approval and the
// execution are two separate journal rows; and the nonce is dead afterwards.
//
// This is the witnessed transcript for the one real gated verb (`dispatch`). v1 (#2267,
// Block Kit buttons) replaces only the last line of the card — every assertion below is
// on the contract, not the input surface, so this test keeps guarding v1 unchanged.
func TestThreadTranscript_ProposeApproveExecute(t *testing.T) {
	cfg := baseCfg()

	// Turn one: the gated verb parses cleanly but must not run.
	cmd := Parse(proposeMsg("<@UBOT> dispatch #2265"), cfg)
	if cmd.Refused {
		t.Fatalf("the command must parse: %s", cmd.Reason)
	}
	if !cmd.Gated() {
		t.Fatal("dispatch is outward-facing — a single chat line must never fire it")
	}
	p, proposeRow, ok := Propose(cmd, t0, DefaultTTL)
	if !ok {
		t.Fatal("a gated command must mint a proposal")
	}

	// The card an operator actually sees carries the nonce they must quote back.
	card := Card(p)
	for _, want := range []string{"dispatch #2265", "blast radius: outward-facing", p.Nonce, "approve " + p.Nonce} {
		if !strings.Contains(card, want) {
			t.Errorf("the proposal card omits %q:\n%s", want, card)
		}
	}

	// Turn two: the operator replies in-thread with the nonce.
	approvedAt := t0.Add(4 * time.Minute)
	v := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg), p, cfg, approvedAt)
	if v.Outcome != OutcomeExecute {
		t.Fatalf("the approval must release the command; got outcome=%s reason=%q", v.Outcome, v.Reason)
	}

	// Only now does anything run — and it is journaled as its own row.
	ranAt := approvedAt.Add(2 * time.Second)
	execRow, ok := v.ExecuteRow("run-2266", ranAt)
	if !ok {
		t.Fatal("an approved command must yield an execution row")
	}

	journal := []AuditRow{proposeRow, v.Audit, execRow}
	wantEvents := []string{EventPropose, EventApprove, EventExecute}
	for i, row := range journal {
		if row.Event != wantEvents[i] {
			t.Fatalf("journal row %d is %q; want %q", i, row.Event, wantEvents[i])
		}
		if row.Nonce != p.Nonce {
			t.Errorf("journal row %q lost the nonce binding: %+v", row.Event, row)
		}
		if row.Proposer != "UADMIN" {
			t.Errorf("journal row %q does not name the proposer: %+v", row.Event, row)
		}
		if row.Verb != VerbDispatch || row.Operand != "#2265" {
			t.Errorf("journal row %q does not name what was asked for: %+v", row.Event, row)
		}
	}
	// Who approved appears on the approval and execution rows, never on the proposal.
	if journal[0].Approver != "" || journal[1].Approver != "UADMIN" || journal[2].Approver != "UADMIN" {
		t.Errorf("the approver must appear only from the approval row onward: %+v", journal)
	}
	// The three rows are stamped in the order they happened.
	if !journal[0].At.Before(journal[1].At) || !journal[1].At.Before(journal[2].At) {
		t.Errorf("journal timestamps must be monotonic across the three turns: %v %v %v",
			journal[0].At, journal[1].At, journal[2].At)
	}
	// What ran is on the execution row alone.
	if journal[2].RunID != "run-2266" {
		t.Errorf("the execution row must name the run: %+v", journal[2])
	}
	// And the pending set — rebuilt from that journal — is empty: nothing is still waiting.
	if got := Pending(journal, ranAt); len(got) != 0 {
		t.Errorf("nothing should still be pending after execution; got %+v", got)
	}
	// The nonce is spent for good.
	replay := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> approve "+p.Nonce), cfg), v.Proposal, cfg, ranAt)
	if replay.Outcome != OutcomeRefused || replay.Reason != ReasonApprovalReplayed {
		t.Fatalf("the nonce must be single-use; replay gave outcome=%s reason=%q", replay.Outcome, replay.Reason)
	}
}

// The deny arm of the same thread: the command is refused by a human and never runs.
func TestThreadTranscript_ProposeDeny(t *testing.T) {
	cfg := baseCfg()
	p, proposeRow := gatedProposal(t)

	v := Adjudicate(Parse(replyMsg("UADMIN", "CTRL", "<@UBOT> deny "+p.Nonce), cfg), p, cfg, t0.Add(time.Minute))
	if v.Outcome != OutcomeDenied {
		t.Fatalf("outcome=%s reason=%q; want DENIED", v.Outcome, v.Reason)
	}
	if _, ok := v.ExecuteRow("run-2266", t0.Add(time.Minute)); ok {
		t.Fatal("a denied command must never produce an execution row")
	}
	journal := []AuditRow{proposeRow, v.Audit}
	if journal[1].Event != EventDeny || journal[1].Verdict != VerdictDenied || journal[1].Approver != "UADMIN" {
		t.Errorf("the denial row must name who said no: %+v", journal[1])
	}
	if got := Pending(journal, t0.Add(2*time.Minute)); len(got) != 0 {
		t.Errorf("a denied proposal must leave the pending set; got %+v", got)
	}
}
