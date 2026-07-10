package query

// query_test.go — the witnessed floor for C2 (#4193). The tests assert the three
// done-conditions: every query kind answers against a fixture session with a taint-filtered
// projection; the raw-bytes (full) disclosure is separately gated from metadata and a query
// never returns a quarantined span; and a scoped query never streams the whole transcript
// (the projection is bounded).

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/resume/transcript"
	"github.com/anthony-chaudhary/fak/internal/sessionread"
	"github.com/anthony-chaudhary/fak/internal/sessionread/screen"
)

// secret is content that lives only inside a quarantined (sealed) turn. Every taint
// assertion checks this substring never crosses the boundary in any projection.
const secret = "hunter2 LAUNCH CODES the dropped originating task"

// fixtureSession is a representative projected session: a user ask, an assistant decision,
// a failed tool-terminal, a successful tool-terminal that touched a file, a QUARANTINED
// assistant turn carrying the secret, and a clean span-matchable turn.
func fixtureSession() []Turn {
	return []Turn{
		{Index: 0, Role: "user", Text: "please add auth login to the gateway"},
		{Index: 1, Role: "assistant", Text: "implementing the auth login decision now"},
		{Index: 2, Role: "assistant", Tool: "Bash", ToolTerm: true, ToolFailed: true, Text: "go build failed"},
		{Index: 3, Role: "assistant", Tool: "Edit", ToolTerm: true, Files: []string{"cmd/fak/auth.go"}, Text: "edited the login handler file"},
		{Index: 4, Role: "assistant", Text: secret, Bytes: []byte(secret), Sealed: true},
		{Index: 5, Role: "user", Text: "look for the zzz-marker token", Bytes: []byte("verbatim zzz-marker payload")},
	}
}

// assertNoSecret fails if any item in a result leaked the quarantined content through any
// field.
func assertNoSecret(t *testing.T, res Result) {
	t.Helper()
	for _, it := range res.Items {
		if strings.Contains(it.Text, "hunter2") || strings.Contains(string(it.Bytes), "hunter2") {
			t.Fatalf("quarantined content leaked in item %+v", it)
		}
	}
}

// TestParseQueryGrammarClosed pins the closed grammar: each kind parses with its argument,
// and everything outside the grammar is a closed error (never an open-ended filter).
func TestParseQueryGrammarClosed(t *testing.T) {
	ok := []struct {
		in   string
		want Query
	}{
		{"last-n-turns 5", Query{Kind: KindLastNTurns, N: 5}},
		{"tool-failures", Query{Kind: KindToolFailures}},
		{"files-touched", Query{Kind: KindFilesTouched}},
		{"decisions-about auth login", Query{Kind: KindDecisionsAbout, Term: "auth login"}},
		{"spans-matching zzz-marker", Query{Kind: KindSpansMatching, Term: "zzz-marker"}},
	}
	for _, tc := range ok {
		got, err := ParseQuery(tc.in)
		if err != nil {
			t.Fatalf("ParseQuery(%q) errored: %v", tc.in, err)
		}
		if got != tc.want {
			t.Fatalf("ParseQuery(%q) = %+v, want %+v", tc.in, got, tc.want)
		}
	}
	bad := []struct {
		in      string
		wantErr error
	}{
		{"teleport-to-turn 3", ErrUnknownQueryKind},
		{"", ErrMalformedQuery},
		{"last-n-turns", ErrMalformedQuery},
		{"last-n-turns 0", ErrMalformedQuery},
		{"last-n-turns -4", ErrMalformedQuery},
		{"last-n-turns notanint", ErrMalformedQuery},
		{"tool-failures extra", ErrMalformedQuery},
		{"decisions-about", ErrMalformedQuery},
		{"spans-matching", ErrMalformedQuery},
	}
	for _, tc := range bad {
		_, err := ParseQuery(tc.in)
		if !errors.Is(err, tc.wantErr) {
			t.Fatalf("ParseQuery(%q) err = %v, want errors.Is %v", tc.in, err, tc.wantErr)
		}
	}
}

// TestEachKindAnswersProjection is done-condition bullet 1: every kind answers against the
// fixture session, returning the expected bounded projection.
func TestEachKindAnswersProjection(t *testing.T) {
	sess := fixtureSession()

	// last-n-turns 2 → the last two turns (index 4 withheld because sealed, index 5 clean).
	res, err := Answer(Query{Kind: KindLastNTurns, N: 2}, sess, sessionread.DisclosureRedacted)
	if err != nil {
		t.Fatalf("last-n-turns: %v", err)
	}
	if len(res.Items) != 2 {
		t.Fatalf("last-n-turns 2 returned %d items, want 2", len(res.Items))
	}
	if res.Items[0].Index != 4 || !res.Items[0].Withheld {
		t.Fatalf("expected the sealed turn 4 as a withheld marker, got %+v", res.Items[0])
	}
	if res.Items[1].Index != 5 || res.Items[1].Withheld {
		t.Fatalf("expected clean turn 5 disclosed, got %+v", res.Items[1])
	}
	assertNoSecret(t, res)

	// tool-failures → only the failed tool-terminal (turn 2).
	res, err = Answer(Query{Kind: KindToolFailures}, sess, sessionread.DisclosureMetadata)
	if err != nil {
		t.Fatalf("tool-failures: %v", err)
	}
	if len(res.Items) != 1 || res.Items[0].Index != 2 || res.Items[0].Tool != "Bash" {
		t.Fatalf("tool-failures returned %+v, want just turn 2 (Bash)", res.Items)
	}

	// files-touched → only the turn that touched a file (turn 3).
	res, err = Answer(Query{Kind: KindFilesTouched}, sess, sessionread.DisclosureMetadata)
	if err != nil {
		t.Fatalf("files-touched: %v", err)
	}
	if len(res.Items) != 1 || res.Items[0].Index != 3 || len(res.Items[0].Files) != 1 || res.Items[0].Files[0] != "cmd/fak/auth.go" {
		t.Fatalf("files-touched returned %+v, want turn 3 with cmd/fak/auth.go", res.Items)
	}

	// decisions-about "auth" → the assistant decision (turn 1), NOT the sealed turn 4.
	res, err = Answer(Query{Kind: KindDecisionsAbout, Term: "auth"}, sess, sessionread.DisclosureRedacted)
	if err != nil {
		t.Fatalf("decisions-about: %v", err)
	}
	if len(res.Items) != 1 || res.Items[0].Index != 1 || !strings.Contains(res.Items[0].Text, "auth") {
		t.Fatalf("decisions-about auth returned %+v, want turn 1 mentioning auth", res.Items)
	}
	assertNoSecret(t, res)
}

// TestFullDisclosureSeparatelyGated is done-condition bullet 2a: spans-matching returns raw
// bytes and is DisclosureFull; requesting it with only a metadata grant refuses
// READ_SCOPE_DENIED, and with a full grant it returns the verbatim (screened) bytes.
func TestFullDisclosureSeparatelyGated(t *testing.T) {
	sess := fixtureSession()
	q := Query{Kind: KindSpansMatching, Term: "zzz-marker"}

	if got := q.Kind.Disclosure(); got != sessionread.DisclosureFull {
		t.Fatalf("spans-matching disclosure = %q, want %q", got, sessionread.DisclosureFull)
	}

	// metadata grant is insufficient for a full-disclosure query.
	_, err := Answer(q, sess, sessionread.DisclosureMetadata)
	if err == nil {
		t.Fatal("spans-matching answered at a metadata grant — full disclosure was not separately gated")
	}
	if reason := screen.RefusalReason(err); reason != sessionread.ReasonReadScopeDenied {
		t.Fatalf("gate refusal = %q, want %q", reason, sessionread.ReasonReadScopeDenied)
	}

	// redacted grant is still insufficient (full > redacted).
	if _, err := Answer(q, sess, sessionread.DisclosureRedacted); err == nil {
		t.Fatal("spans-matching answered at a redacted grant — full disclosure escalated")
	}

	// full grant discloses the verbatim bytes of the clean matching span (turn 5).
	res, err := Answer(q, sess, sessionread.DisclosureFull)
	if err != nil {
		t.Fatalf("spans-matching at full grant refused: %v", err)
	}
	if len(res.Items) != 1 || res.Items[0].Index != 5 {
		t.Fatalf("spans-matching returned %+v, want turn 5", res.Items)
	}
	if !bytes.Equal(res.Items[0].Bytes, []byte("verbatim zzz-marker payload")) {
		t.Fatalf("full disclosure did not return verbatim bytes: %q", res.Items[0].Bytes)
	}
	// A metadata query (files-touched) is fine at a metadata grant — the gate only blocks up.
	if _, err := Answer(Query{Kind: KindFilesTouched}, sess, sessionread.DisclosureMetadata); err != nil {
		t.Fatalf("metadata query refused at a metadata grant: %v", err)
	}
}

// TestQuarantinedSpanNeverReturned is done-condition bullet 2b: no query kind, at any
// disclosure, ever returns the quarantined turn's content. The sealed turn 4 carries the
// secret; we probe it through every kind that could surface it.
func TestQuarantinedSpanNeverReturned(t *testing.T) {
	sess := fixtureSession()

	// spans-matching on the secret at FULL disclosure — the strongest probe. The sealed
	// span is skipped silently; the secret never appears.
	res, err := Answer(Query{Kind: KindSpansMatching, Term: "hunter2"}, sess, sessionread.DisclosureFull)
	if err != nil {
		t.Fatalf("spans-matching hunter2: %v", err)
	}
	if len(res.Items) != 0 {
		t.Fatalf("spans-matching hunter2 returned %d items, want 0 (sealed span withheld)", len(res.Items))
	}
	assertNoSecret(t, res)

	// decisions-about the secret term — the sealed assistant turn is not disclosed.
	res, err = Answer(Query{Kind: KindDecisionsAbout, Term: "LAUNCH"}, sess, sessionread.DisclosureRedacted)
	if err != nil {
		t.Fatalf("decisions-about LAUNCH: %v", err)
	}
	assertNoSecret(t, res)
	for _, it := range res.Items {
		if it.Index == 4 && !it.Withheld {
			t.Fatal("sealed decision turn 4 disclosed content in decisions-about")
		}
	}

	// last-n-turns covering the sealed turn — it appears only as a bytes-free withheld
	// marker carrying the closed taint reason, never its content.
	res, err = Answer(Query{Kind: KindLastNTurns, N: 6}, sess, sessionread.DisclosureFull)
	if err != nil {
		t.Fatalf("last-n-turns 6: %v", err)
	}
	assertNoSecret(t, res)
	var sawWithheldSeal bool
	for _, it := range res.Items {
		if it.Index == 4 {
			sawWithheldSeal = it.Withheld && it.Reason == sessionread.ReasonReadTaintWithheld && it.Text == "" && len(it.Bytes) == 0
		}
	}
	if !sawWithheldSeal {
		t.Fatal("sealed turn 4 was not surfaced as a bytes-free READ_TAINT_WITHHELD marker")
	}
}

// TestProjectionBounded is done-condition bullet 3: a scoped query never streams the whole
// transcript. Over a 1000-turn session, last-n-turns 3 returns exactly 3, and a broad
// spans-matching is capped at maxProjectionItems with Truncated set.
func TestProjectionBounded(t *testing.T) {
	big := make([]Turn, 1000)
	for i := range big {
		big[i] = Turn{Index: i, Role: "assistant", Text: "common-token decision", Bytes: []byte("common-token payload")}
	}

	res, err := Answer(Query{Kind: KindLastNTurns, N: 3}, big, sessionread.DisclosureRedacted)
	if err != nil {
		t.Fatalf("last-n-turns 3: %v", err)
	}
	if len(res.Items) != 3 {
		t.Fatalf("last-n-turns 3 over 1000 turns returned %d items — the whole transcript was projected", len(res.Items))
	}

	res, err = Answer(Query{Kind: KindSpansMatching, Term: "common-token"}, big, sessionread.DisclosureFull)
	if err != nil {
		t.Fatalf("spans-matching: %v", err)
	}
	if len(res.Items) > maxProjectionItems {
		t.Fatalf("spans-matching returned %d items, exceeds the %d cap", len(res.Items), maxProjectionItems)
	}
	if !res.Truncated {
		t.Fatal("a 1000-match query was not marked Truncated — a silent cap reads as full coverage")
	}

	// last-n-turns with an absurd N is clamped, not honored literally.
	res, err = Answer(Query{Kind: KindLastNTurns, N: 100000}, big, sessionread.DisclosureRedacted)
	if err != nil {
		t.Fatalf("last-n-turns huge: %v", err)
	}
	if len(res.Items) > maxProjectionItems {
		t.Fatalf("last-n-turns 100000 returned %d items, exceeds the %d cap", len(res.Items), maxProjectionItems)
	}
	if !res.Truncated {
		t.Fatal("an over-cap N was not marked Truncated")
	}
}

// TestTurnsFromRealTranscript witnesses the fixture-session bullet against the REAL parser:
// a JSONL transcript is parsed and projected, and a query answers over it.
func TestTurnsFromRealTranscript(t *testing.T) {
	jsonl := strings.Join([]string{
		`{"type":"user","message":{"role":"user","content":"please add auth login"}}`,
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"implementing the auth decision"}]}}`,
		`{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash"},{"type":"tool_result","content":"ok"}]}}`,
	}, "\n")

	recs := transcript.Parse(strings.NewReader(jsonl))
	turns := TurnsFromRecords(recs)
	if len(turns) != 3 {
		t.Fatalf("TurnsFromRecords projected %d turns, want 3", len(turns))
	}
	if turns[0].Role != "user" || !strings.Contains(turns[0].Text, "auth") {
		t.Fatalf("turn 0 = %+v, want a user turn mentioning auth", turns[0])
	}
	if turns[2].Tool != "Bash" || !turns[2].ToolTerm {
		t.Fatalf("turn 2 = %+v, want a Bash tool-terminal", turns[2])
	}

	// A query answers over the projected transcript.
	res, err := Answer(Query{Kind: KindDecisionsAbout, Term: "auth"}, turns, sessionread.DisclosureRedacted)
	if err != nil {
		t.Fatalf("decisions-about over transcript: %v", err)
	}
	if len(res.Items) != 1 || res.Items[0].Index != 1 {
		t.Fatalf("decisions-about auth over transcript returned %+v, want the assistant decision", res.Items)
	}
}
