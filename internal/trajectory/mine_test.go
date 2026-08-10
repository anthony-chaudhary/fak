package trajectory

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func toolTurn(trace string, seq int, tool, verdict, query string) Turn {
	return Turn{TraceID: trace, Seq: seq, Tool: tool, Verdict: verdict, Query: query}
}
func TestMineTurnsSegmentsAggregatesAndIsDeterministic(t *testing.T) {
	turns := []Turn{toolTurn("b", 1, "Read", "ALLOW", "secret-one"), toolTurn("b", 2, "Edit", "ALLOW", "secret-two"), toolTurn("b", 3, "Bash", "ALLOW", "secret-three"), toolTurn("a", 1, "Read", "DENY", "x"), toolTurn("a", 2, "Read", "ALLOW", "y")}
	a := MineTurns(turns, MineOptions{})
	b := MineTurns(turns, MineOptions{})
	var ab, bb bytes.Buffer
	a.Render(&ab)
	b.Render(&bb)
	if ab.String() != bb.String() {
		t.Fatal("render is nondeterministic")
	}
	if len(a.Segments) < 2 || len(a.Aggregates) < 2 {
		t.Fatalf("segments=%d aggregates=%d", len(a.Segments), len(a.Aggregates))
	}
	if strings.Contains(ab.String(), "secret-") {
		t.Fatal("default render leaked query content")
	}
}
func TestMineTurnsOverlapAndNoMatch(t *testing.T) {
	r := MineTurns([]Turn{toolTurn("x", 1, "Read", "ALLOW", ""), toolTurn("x", 2, "Edit", "ALLOW", ""), toolTurn("x", 3, "Bash", "ALLOW", "")}, MineOptions{})
	if len(r.Segments) != 1 {
		t.Fatalf("overlap policy not applied: %#v", r)
	}
	empty := MineTurns([]Turn{toolTurn("x", 1, "UnknownTool", "ALLOW", "")}, MineOptions{})
	if len(empty.Segments) != 0 {
		t.Fatalf("wanted abstention: %#v", empty)
	}
}
func TestImportScrubbedChatPrivacyAndErrors(t *testing.T) {
	secret := "DO_NOT_EMIT_SECRET"
	doc := `{"format":"fak.scrubbed-chat/1","raw_message":"` + secret + `","conversations":[{"id":"c","entries":[{"tool":"Read","status":"ok"},{"tool":"Edit","status":"ok"},{"tool":"Bash","status":"ok"}]}]}`
	r, n, err := ImportScrubbedChat(strings.NewReader(doc))
	if err != nil || n != 3 {
		t.Fatalf("n=%d err=%v", n, err)
	}
	var out bytes.Buffer
	r.Mine(MineOptions{}).Render(&out)
	if strings.Contains(out.String(), secret) {
		t.Fatal("secret leaked")
	}
	if _, _, err := ImportScrubbedChat(strings.NewReader(`{"format":"other","conversations":[]}`)); !errors.Is(err, ErrUnsupportedFormat) {
		t.Fatalf("unsupported err=%v", err)
	}
	if _, _, err := ImportScrubbedChat(strings.NewReader(`{`)); !errors.Is(err, ErrMalformedExport) {
		t.Fatalf("malformed err=%v", err)
	}
}
