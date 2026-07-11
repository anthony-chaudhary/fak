package chatops

import "testing"

// baseCfg is a fully-configured door: a known bot, one control channel, one admin.
func baseCfg() Config {
	return Config{
		BotUserID:      "UBOT",
		ControlChannel: "CTRL",
		Admins:         []string{"UADMIN"},
	}
}

// admin builds a well-formed message from the seeded admin in the control channel.
func admin(text string) Message {
	return Message{User: "UADMIN", Channel: "CTRL", TS: "1712000000.000100", Text: text}
}

func TestParse_AcceptsClosedVerbs(t *testing.T) {
	cfg := baseCfg()
	cases := []struct {
		text    string
		verb    Verb
		class   Class
		operand string
	}{
		{"<@UBOT> status", VerbStatus, ClassRead, ""},
		{"<@UBOT> fleet", VerbFleet, ClassRead, ""},
		{"<@UBOT> help", VerbHelp, ClassRead, ""},
		{"<@UBOT> ping", VerbPing, ClassRead, ""},
		{"<@UBOT> halt", VerbHalt, ClassControl, ""},
		{"<@UBOT> dispatch #2265", VerbDispatch, ClassAct, "#2265"},
		{"<@UBOT> resume run-42", VerbResume, ClassAct, "run-42"},
		{"<@UBOT> bench frontierswe", VerbBench, ClassAct, "frontierswe"},
	}
	for _, tc := range cases {
		got := Parse(admin(tc.text), cfg)
		if got.Refused {
			t.Fatalf("%q: unexpected refusal %s", tc.text, got.Reason)
		}
		if got.Verb != tc.verb || got.Class != tc.class || got.Operand != tc.operand {
			t.Errorf("%q: got verb=%s class=%s operand=%q; want verb=%s class=%s operand=%q",
				tc.text, got.Verb, got.Class, got.Operand, tc.verb, tc.class, tc.operand)
		}
		if got.Nonce != "1712000000.000100" {
			t.Errorf("%q: nonce=%q; want the message ts", tc.text, got.Nonce)
		}
		if got.Channel != "CTRL" || got.User != "UADMIN" {
			t.Errorf("%q: channel/user not threaded: %+v", tc.text, got)
		}
	}
}

// The multi-word operand is joined and trimmed.
func TestParse_MultiWordOperand(t *testing.T) {
	got := Parse(admin("<@UBOT> resume   loop  alpha "), baseCfg())
	if got.Refused || got.Operand != "loop alpha" {
		t.Fatalf("operand=%q refused=%v; want %q", got.Operand, got.Refused, "loop alpha")
	}
}

// Verbs are case-insensitive on the leading token (clients auto-capitalize).
func TestParse_VerbCaseInsensitive(t *testing.T) {
	got := Parse(admin("<@UBOT> Status"), baseCfg())
	if got.Refused || got.Verb != VerbStatus {
		t.Fatalf("got %+v; want status", got)
	}
}

// The ordered fence: each gate fires in priority order and returns the FIRST failure.
func TestParse_OrderedFence(t *testing.T) {
	cfg := baseCfg()
	cases := []struct {
		name   string
		msg    Message
		reason string
	}{
		{"bot loop via bot_id", Message{User: "UADMIN", BotID: "B1", Channel: "CTRL", Text: "<@UBOT> status"}, ReasonBotLoop},
		{"bot loop via self user", Message{User: "UBOT", Channel: "CTRL", Text: "<@UBOT> status"}, ReasonBotLoop},
		{"wrong channel", Message{User: "UADMIN", Channel: "OTHER", Text: "<@UBOT> status"}, ReasonWrongChannel},
		{"not addressed", Message{User: "UADMIN", Channel: "CTRL", Text: "status now please"}, ReasonNotAddressed},
		{"bare mention is empty", Message{User: "UADMIN", Channel: "CTRL", Text: "<@UBOT>   "}, ReasonEmpty},
		{"non-admin refused before grammar", Message{User: "UINTRUDER", Channel: "CTRL", Text: "<@UBOT> dispatch #1"}, ReasonNotAdmin},
		{"admin unknown verb", admin("<@UBOT> selfdestruct"), ReasonUnknownVerb},
		{"act verb missing operand", admin("<@UBOT> dispatch"), ReasonMissingOperand},
	}
	for _, tc := range cases {
		got := Parse(tc.msg, cfg)
		if !got.Refused || got.Reason != tc.reason {
			t.Errorf("%s: got refused=%v reason=%q; want refusal %q", tc.name, got.Refused, got.Reason, tc.reason)
		}
		if got.Class != ClassRefused {
			t.Errorf("%s: refusal must carry ClassRefused, got %s", tc.name, got.Class)
		}
	}
}

// Bot-loop wins over wrong-channel: a bot message in the wrong channel is BOT_LOOP,
// proving the checks run in the documented order, not by field convenience.
func TestParse_LoopBeatsChannel(t *testing.T) {
	got := Parse(Message{User: "UADMIN", BotID: "B1", Channel: "OTHER", Text: "<@UBOT> status"}, baseCfg())
	if got.Reason != ReasonBotLoop {
		t.Fatalf("reason=%q; want BOT_LOOP to precede WRONG_CHANNEL", got.Reason)
	}
}

// Non-admin never learns the grammar: a valid verb AND garbage both return NOT_ADMIN,
// so the refusal reason cannot be used as a grammar oracle.
func TestParse_NonAdminLeaksNothing(t *testing.T) {
	cfg := baseCfg()
	valid := Parse(Message{User: "UX", Channel: "CTRL", Text: "<@UBOT> status"}, cfg)
	garbage := Parse(Message{User: "UX", Channel: "CTRL", Text: "<@UBOT> zzzzz"}, cfg)
	if valid.Reason != ReasonNotAdmin || garbage.Reason != ReasonNotAdmin {
		t.Fatalf("non-admin must always get NOT_ADMIN; got %q and %q", valid.Reason, garbage.Reason)
	}
}

// Fail-closed authorization: an empty admin allowlist refuses everyone, including a
// syntactically perfect command.
func TestParse_EmptyAdminsFailClosed(t *testing.T) {
	cfg := Config{BotUserID: "UBOT", ControlChannel: "CTRL"} // no admins
	got := Parse(Message{User: "UANYONE", Channel: "CTRL", Text: "<@UBOT> status"}, cfg)
	if !got.Refused || got.Reason != ReasonNotAdmin {
		t.Fatalf("empty allowlist must refuse; got %+v", got)
	}
}

// Authorization is on the immutable user id, not any display text in the body.
func TestParse_AuthzIgnoresBodyText(t *testing.T) {
	// A non-admin who types the admin's id as plain text is still not an admin.
	got := Parse(Message{User: "UX", Channel: "CTRL", Text: "<@UBOT> status UADMIN"}, baseCfg())
	if got.Reason != ReasonNotAdmin {
		t.Fatalf("body text must not grant authority; got %q", got.Reason)
	}
}

// A labeled mention (<@UBOT|fak>) still counts as addressed and is stripped.
func TestParse_LabeledMention(t *testing.T) {
	got := Parse(admin("<@UBOT|fak-bot> ping"), baseCfg())
	if got.Refused || got.Verb != VerbPing {
		t.Fatalf("labeled mention should parse; got %+v", got)
	}
}

// A prefix-collision mention (a different id that starts with the bot id) is NOT the
// bot being addressed.
func TestParse_MentionPrefixCollision(t *testing.T) {
	got := Parse(admin("<@UBOTX> status"), baseCfg())
	if got.Reason != ReasonNotAddressed {
		t.Fatalf("prefix-collision id must not count as addressed; got %q", got.Reason)
	}
}

// With no bot id configured the addressing gate is disabled (degenerate/test config):
// the whole text is the body and parses.
func TestParse_NoBotIDDisablesAddressing(t *testing.T) {
	cfg := Config{ControlChannel: "CTRL", Admins: []string{"UADMIN"}}
	got := Parse(Message{User: "UADMIN", Channel: "CTRL", Text: "status"}, cfg)
	if got.Refused || got.Verb != VerbStatus {
		t.Fatalf("no-bot-id config should parse bare text; got %+v", got)
	}
}

// With no control channel configured the channel gate is disabled: a command from any
// channel is accepted (still subject to the admin gate).
func TestParse_NoControlChannelDisablesChannelGate(t *testing.T) {
	cfg := Config{BotUserID: "UBOT", Admins: []string{"UADMIN"}}
	got := Parse(Message{User: "UADMIN", Channel: "ANYWHERE", Text: "<@UBOT> ping"}, cfg)
	if got.Refused {
		t.Fatalf("no control channel should not gate on channel; got %q", got.Reason)
	}
}

// The zero Result is a refusal — the safe default (a dropped/forgotten field can never
// read as an accidental accept).
func TestZeroResultIsRefusal(t *testing.T) {
	var r Result
	if r.Class != ClassRefused {
		t.Fatalf("zero Result must be ClassRefused, got %s", r.Class)
	}
}

// Grammar() is a defensive copy: mutating the returned slice cannot corrupt the door.
func TestGrammar_DefensiveCopy(t *testing.T) {
	g := Grammar()
	if len(g) == 0 {
		t.Fatal("grammar is empty")
	}
	g[0].Verb = "poisoned"
	if Grammar()[0].Verb == "poisoned" {
		t.Fatal("Grammar() leaked its backing array")
	}
}

// Every grammar verb round-trips through Parse — the closed set has no dead rows and
// every declared operand requirement is enforced.
func TestGrammar_EveryVerbParses(t *testing.T) {
	cfg := baseCfg()
	for _, s := range Grammar() {
		text := "<@UBOT> " + string(s.Verb)
		if s.NeedsOperand {
			text += " x"
		}
		got := Parse(admin(text), cfg)
		if got.Refused {
			t.Errorf("verb %q refused unexpectedly: %s", s.Verb, got.Reason)
			continue
		}
		if got.Verb != s.Verb || got.Class != s.Class {
			t.Errorf("verb %q: parsed as verb=%s class=%s", s.Verb, got.Verb, got.Class)
		}
	}
}

// Reasons() enumerates exactly the tokens Parse can emit — the closed refusal
// vocabulary, with no duplicates.
func TestReasons_ClosedAndUnique(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range Reasons() {
		if r == "" {
			t.Fatal("empty reason token")
		}
		if seen[r] {
			t.Fatalf("duplicate reason token %q", r)
		}
		seen[r] = true
	}
	for _, want := range []string{
		ReasonBotLoop, ReasonWrongChannel, ReasonNotAddressed,
		ReasonEmpty, ReasonNotAdmin, ReasonUnknownVerb, ReasonMissingOperand,
	} {
		if !seen[want] {
			t.Errorf("Reasons() missing %q", want)
		}
	}
}

// EMPTY (step 4) precedes NOT_ADMIN (step 5): a NON-admin bare mention is EMPTY, not
// NOT_ADMIN. This pins the 4-before-5 adjacency the way TestParse_LoopBeatsChannel pins
// 1-before-2 — without it, swapping the empty and admin gates survives the whole suite
// (every other test pairs a non-admin with a non-empty body, or an empty body with an
// admin, so none distinguishes the two orderings).
func TestParse_EmptyBeatsNotAdmin(t *testing.T) {
	got := Parse(Message{User: "UINTRUDER", Channel: "CTRL", Text: "<@UBOT>   "}, baseCfg())
	if got.Reason != ReasonEmpty {
		t.Fatalf("reason=%q; want EMPTY to precede NOT_ADMIN for a non-admin bare mention", got.Reason)
	}
}

// Admin matching is exact: Slack user ids are case-sensitive, so a case-variant of the
// seeded admin id is NOT the admin and is refused, while the exact id is accepted. Locks
// the documented exact compare — a case-insensitive weakening (EqualFold) would widen
// the allowlist and otherwise pass every existing test.
func TestParse_AdminMatchIsCaseSensitive(t *testing.T) {
	cfg := baseCfg() // Admins: {"UADMIN"}
	variant := Parse(Message{User: "uadmin", Channel: "CTRL", TS: "1", Text: "<@UBOT> ping"}, cfg)
	if !variant.Refused || variant.Reason != ReasonNotAdmin {
		t.Fatalf("case-variant id must not match the exact allowlist; got %+v", variant)
	}
	exact := Parse(Message{User: "UADMIN", Channel: "CTRL", TS: "1", Text: "<@UBOT> ping"}, cfg)
	if exact.Refused || exact.Verb != VerbPing {
		t.Fatalf("exact admin id must be accepted; got %+v", exact)
	}
}

// Fail-closed on an empty sender id: even if the allowlist carries a blank entry (a
// mis-seeded config), an empty-User message must never match it — isAdmin rejects the
// empty user before it can equal a blank allowlist entry. Deleting that guard authorizes
// an empty-User sender against a blank entry, and no existing test exercises an empty User.
func TestParse_EmptyUserFailsClosed(t *testing.T) {
	cfg := Config{BotUserID: "UBOT", ControlChannel: "CTRL", Admins: []string{"UADMIN", ""}}
	got := Parse(Message{User: "", Channel: "CTRL", TS: "1", Text: "<@UBOT> ping"}, cfg)
	if !got.Refused || got.Reason != ReasonNotAdmin {
		t.Fatalf("empty user must never authorize, even against a blank allowlist entry; got %+v", got)
	}
}

// An embedded (non-leading) mention still addresses the door: the verb before a trailing
// `<@UBOT>` is the command. Locks the documented "leading-or-embedded" stripping — a
// leading-only narrowing (requiring the mention at index 0) would refuse this as
// NOT_ADDRESSED yet still pass every current test.
func TestParse_EmbeddedMentionAddressed(t *testing.T) {
	got := Parse(admin("halt <@UBOT>"), baseCfg())
	if got.Refused || got.Verb != VerbHalt {
		t.Fatalf("embedded mention should address the door; got %+v", got)
	}
}
