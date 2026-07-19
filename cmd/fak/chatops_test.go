package main

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/chatops"
	"github.com/anthony-chaudhary/fak/internal/chatrelay"
)

// fakeSlack is an in-memory chatrelay.SlackClient: History returns the seeded messages newer
// than oldestTS (mirroring Slack's oldest-exclusive contract via the door's own ts compare),
// and Post records every reply. No network — the whole read-only door is exercised offline.
type fakeSlack struct {
	msgs    []chatrelay.Message
	posts   []fakePost
	postErr error
}

type fakePost struct{ channel, thread, text string }

func (f *fakeSlack) History(_ context.Context, _ string, oldestTS string, _ int) ([]chatrelay.Message, error) {
	var out []chatrelay.Message
	for _, m := range f.msgs {
		if chatopsTSAfter(m.TS, oldestTS) {
			out = append(out, m)
		}
	}
	return out, nil
}

func (f *fakeSlack) Post(_ context.Context, channel, thread, text string) (string, error) {
	if f.postErr != nil {
		return "", f.postErr
	}
	f.posts = append(f.posts, fakePost{channel, thread, text})
	return "9999.0001", nil
}

const (
	ctBot   = "UBOT"
	ctAdmin = "UADMIN"
	ctEve   = "UEVE"
	ctChan  = "CCTRL"
)

func newTestDoor(slack chatrelay.SlackClient, audit *strings.Builder) *chatopsDoor {
	d := &chatopsDoor{
		Slack: slack,
		Cfg: chatops.Config{
			BotUserID:      ctBot,
			ControlChannel: ctChan,
			Admins:         []string{ctAdmin},
		},
		Channel: ctChan,
	}
	if audit != nil {
		d.Audit = audit
	}
	return d
}

// adminMsg is a control-channel message from the seeded admin, addressed to the door.
func adminMsg(ts, text string) chatrelay.Message {
	return chatrelay.Message{User: ctAdmin, TS: ts, Text: "<@" + ctBot + "> " + text}
}

func TestChatopsDoor_AnswersReadVerbs(t *testing.T) {
	slack := &fakeSlack{msgs: []chatrelay.Message{
		adminMsg("1.0", "help"),
		adminMsg("2.0", "ping"),
		adminMsg("3.0", "status"),
		adminMsg("4.0", "fleet"),
	}}
	door := newTestDoor(slack, nil)

	n, err := door.Tick(context.Background())
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if n != 4 {
		t.Fatalf("replied=%d, want 4", n)
	}
	if len(slack.posts) != 4 {
		t.Fatalf("posts=%d, want 4", len(slack.posts))
	}
	// Each reply threads under the message it answers.
	if slack.posts[0].thread != "1.0" {
		t.Errorf("help reply thread=%q, want 1.0", slack.posts[0].thread)
	}
	if !strings.Contains(slack.posts[0].text, "help") {
		t.Errorf("help reply missing grammar: %q", slack.posts[0].text)
	}
	if slack.posts[1].text != "pong" {
		t.Errorf("ping reply=%q, want pong", slack.posts[1].text)
	}
	if !strings.Contains(slack.posts[2].text, "online") {
		t.Errorf("status reply missing liveness: %q", slack.posts[2].text)
	}
	if !strings.Contains(slack.posts[3].text, "not surfaced") {
		t.Errorf("fleet reply should be honest about being unwired: %q", slack.posts[3].text)
	}
}

func TestChatopsDoor_IdempotentRepoll(t *testing.T) {
	slack := &fakeSlack{msgs: []chatrelay.Message{adminMsg("1.0", "ping")}}
	door := newTestDoor(slack, nil)

	if _, err := door.Tick(context.Background()); err != nil {
		t.Fatalf("first Tick: %v", err)
	}
	if len(slack.posts) != 1 {
		t.Fatalf("after first Tick posts=%d, want 1", len(slack.posts))
	}
	// A re-poll over the same history must not answer the already-seen message again.
	if _, err := door.Tick(context.Background()); err != nil {
		t.Fatalf("second Tick: %v", err)
	}
	if len(slack.posts) != 1 {
		t.Fatalf("after re-poll posts=%d, want 1 (idempotent)", len(slack.posts))
	}
}

func TestChatopsDoor_StructuredRefusalReplies(t *testing.T) {
	slack := &fakeSlack{msgs: []chatrelay.Message{
		adminMsg("1.0", "frobnicate"), // UNKNOWN_VERB
		adminMsg("2.0", "dispatch"),   // MISSING_OPERAND (dispatch needs an operand)
	}}
	door := newTestDoor(slack, nil)

	if _, err := door.Tick(context.Background()); err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if len(slack.posts) != 2 {
		t.Fatalf("posts=%d, want 2", len(slack.posts))
	}
	if !strings.Contains(slack.posts[0].text, "unknown command") {
		t.Errorf("unknown-verb reply=%q", slack.posts[0].text)
	}
	if !strings.Contains(slack.posts[1].text, "needs an argument") {
		t.Errorf("missing-operand reply=%q", slack.posts[1].text)
	}
}

func TestChatopsDoor_SilentOnFenceAndAuthz(t *testing.T) {
	slack := &fakeSlack{msgs: []chatrelay.Message{
		{User: ctEve, TS: "1.0", Text: "<@" + ctBot + "> status"},             // NOT_ADMIN
		{User: "UAPP", BotID: "B1", TS: "2.0", Text: "<@" + ctBot + "> ping"}, // BOT_LOOP
		{User: ctAdmin, TS: "3.0", Text: "just chatting, no mention"},         // NOT_ADDRESSED
		adminMsg("4.0", ""), // EMPTY (bare mention)
	}}
	door := newTestDoor(slack, nil)

	n, err := door.Tick(context.Background())
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if n != 0 || len(slack.posts) != 0 {
		t.Fatalf("door replied to a fenced/unauthorized message: n=%d posts=%d", n, len(slack.posts))
	}
	// All four are still marked seen (mark advances) so they are never re-evaluated.
	if _, err := door.Tick(context.Background()); err != nil {
		t.Fatalf("second Tick: %v", err)
	}
	if len(slack.posts) != 0 {
		t.Fatalf("posts=%d after re-poll, want 0", len(slack.posts))
	}
}

func TestChatopsDoor_DeclinesActAndControl(t *testing.T) {
	slack := &fakeSlack{msgs: []chatrelay.Message{
		adminMsg("1.0", "dispatch #2265"), // ClassAct
		adminMsg("2.0", "halt"),           // ClassControl
	}}
	door := newTestDoor(slack, nil)

	if _, err := door.Tick(context.Background()); err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if len(slack.posts) != 2 {
		t.Fatalf("posts=%d, want 2", len(slack.posts))
	}
	if !strings.Contains(slack.posts[0].text, "does not execute act verbs") {
		t.Errorf("dispatch decline=%q", slack.posts[0].text)
	}
	if !strings.Contains(slack.posts[1].text, "halt") {
		t.Errorf("halt decline=%q", slack.posts[1].text)
	}
}

func TestChatopsDoor_AuditRows(t *testing.T) {
	var audit strings.Builder
	slack := &fakeSlack{msgs: []chatrelay.Message{
		adminMsg("1.0", "ping"),                                   // answered read
		adminMsg("2.0", "dispatch #7"),                            // declined act
		{User: ctEve, TS: "3.0", Text: "<@" + ctBot + "> status"}, // silent NOT_ADMIN
	}}
	door := newTestDoor(slack, &audit)

	if _, err := door.Tick(context.Background()); err != nil {
		t.Fatalf("Tick: %v", err)
	}
	rows := decodeAudit(t, audit.String())
	if len(rows) != 3 {
		t.Fatalf("audit rows=%d, want 3 (one per inbound message)", len(rows))
	}
	// ping: read, replied.
	if rows[0].Verb != "ping" || rows[0].Refused || !rows[0].Replied {
		t.Errorf("ping row=%+v", rows[0])
	}
	// dispatch: act, replied (declined), operand preserved.
	if rows[1].Verb != "dispatch" || rows[1].Operand != "#7" || !rows[1].Replied {
		t.Errorf("dispatch row=%+v", rows[1])
	}
	// non-admin status: refused NOT_ADMIN, not replied, verb NOT leaked.
	if !rows[2].Refused || rows[2].Reason != chatops.ReasonNotAdmin || rows[2].Replied {
		t.Errorf("not-admin row=%+v", rows[2])
	}
	if rows[2].Verb != "" {
		t.Errorf("refused row leaked a verb: %+v", rows[2])
	}
}

func TestChatopsDoor_PrimeSkipsBacklog(t *testing.T) {
	slack := &fakeSlack{msgs: []chatrelay.Message{
		adminMsg("1.0", "ping"),
		adminMsg("2.0", "status"),
	}}
	door := newTestDoor(slack, nil)

	if err := door.Prime(context.Background()); err != nil {
		t.Fatalf("Prime: %v", err)
	}
	if _, err := door.Tick(context.Background()); err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if len(slack.posts) != 0 {
		t.Fatalf("primed door answered backlog: posts=%d, want 0", len(slack.posts))
	}
}

func TestChatopsReply_Pure(t *testing.T) {
	env := chatopsEnv{Channel: ctChan, Admins: 1, Grammar: len(chatops.Grammar())}
	cases := []struct {
		name     string
		res      chatops.Result
		wantSub  string
		wantPost bool
	}{
		{"help", chatops.Result{Class: chatops.ClassRead, Verb: chatops.VerbHelp}, "verbs", true},
		{"ping", chatops.Result{Class: chatops.ClassRead, Verb: chatops.VerbPing}, "pong", true},
		{"status", chatops.Result{Class: chatops.ClassRead, Verb: chatops.VerbStatus}, "online", true},
		{"fleet", chatops.Result{Class: chatops.ClassRead, Verb: chatops.VerbFleet}, "not surfaced", true},
		{"act-dispatch", chatops.Result{Class: chatops.ClassAct, Verb: chatops.VerbDispatch}, "act verbs", true},
		{"control-halt", chatops.Result{Class: chatops.ClassControl, Verb: chatops.VerbHalt}, "halt", true},
		{"unknown", chatops.Result{Refused: true, Reason: chatops.ReasonUnknownVerb}, "unknown command", true},
		{"missing-operand", chatops.Result{Refused: true, Reason: chatops.ReasonMissingOperand}, "needs an argument", true},
		{"not-admin", chatops.Result{Refused: true, Reason: chatops.ReasonNotAdmin}, "", false},
		{"bot-loop", chatops.Result{Refused: true, Reason: chatops.ReasonBotLoop}, "", false},
		{"not-addressed", chatops.Result{Refused: true, Reason: chatops.ReasonNotAddressed}, "", false},
		{"empty", chatops.Result{Refused: true, Reason: chatops.ReasonEmpty}, "", false},
		{"wrong-channel", chatops.Result{Refused: true, Reason: chatops.ReasonWrongChannel}, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			text, post := chatopsReply(tc.res, env)
			if post != tc.wantPost {
				t.Fatalf("post=%v, want %v (text=%q)", post, tc.wantPost, text)
			}
			if tc.wantSub != "" && !strings.Contains(text, tc.wantSub) {
				t.Errorf("text=%q, want substring %q", text, tc.wantSub)
			}
			if !post && text != "" {
				t.Errorf("silent case returned text=%q", text)
			}
		})
	}
}

func TestParseAdminList(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"", nil},
		{"U1", []string{"U1"}},
		{"U1,U2,U3", []string{"U1", "U2", "U3"}},
		{" U1 , U2 ,, U1 , U3 ", []string{"U1", "U2", "U3"}}, // trims, drops empties + dups
	}
	for _, tc := range cases {
		got := parseAdminList(tc.in)
		if strings.Join(got, ",") != strings.Join(tc.want, ",") {
			t.Errorf("parseAdminList(%q)=%v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestChatopsTSAfter(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"2.0", "1.0", true},
		{"1.0", "2.0", false},
		{"1.0", "1.0", false},
		{"1.0", "", true},  // everything is after the zero mark
		{"", "1.0", false}, // the zero ts is after nothing
		{"1699999999.000200", "1699999999.000100", true},
	}
	for _, tc := range cases {
		if got := chatopsTSAfter(tc.a, tc.b); got != tc.want {
			t.Errorf("chatopsTSAfter(%q,%q)=%v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
}

func decodeAudit(t *testing.T, s string) []chatopsAuditRow {
	t.Helper()
	var rows []chatopsAuditRow
	for _, line := range strings.Split(strings.TrimSpace(s), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var r chatopsAuditRow
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("bad audit row %q: %v", line, err)
		}
		rows = append(rows, r)
	}
	return rows
}
