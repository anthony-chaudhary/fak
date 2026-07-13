package sessionledger

import (
	"encoding/json"
	"testing"
)

func TestLedgerChainAndFork(t *testing.T) {
	l := Memory()
	for _, k := range []string{"user", "planner", "verdict"} {
		if _, e := l.Append("a", k, json.RawMessage(`{"v":1}`)); e != nil {
			t.Fatal(e)
		}
	}
	c, e := l.Chain("a")
	if e != nil {
		t.Fatal(e)
	}
	if e = Verify(c); e != nil {
		t.Fatal(e)
	}
	n := l.NodeCount()
	h, e := l.Fork("a", "b")
	if e != nil {
		t.Fatal(e)
	}
	if h != l.Head("a") || l.NodeCount() != n {
		t.Fatal("fork copied entries")
	}
	if _, e = l.Append("b", "user", json.RawMessage(`{"v":2}`)); e != nil {
		t.Fatal(e)
	}
	if l.Head("a") == l.Head("b") {
		t.Fatal("fork did not diverge")
	}
}
func TestDecodePreservesTypedBlocks(t *testing.T) {
	raw := json.RawMessage(`{"content":[{"type":"tool_result","tool_use_id":"x","content":[{"type":"text","text":"ok"}]}]}`)
	l := Memory()
	if _, e := l.Append("x", "user", raw); e != nil {
		t.Fatal(e)
	}
	c, _ := l.Chain("x")
	if string(c[0].Content) != string(raw) {
		t.Fatalf("got %s", c[0].Content)
	}
}
