package slackoutbox

import (
	"context"
	"testing"
	"time"
)

// TestCallStatsDerivesPerSourceFromLog drives a realistic mix through a real drain — a
// coalesced burst of card edits, a clean post, and a fence-refused post across three
// sources — then asserts CallStats reconstructs each surface's spent-vs-saved footprint
// purely from the durable log. This is the measuring stick behind `fak slack outbox calls`.
func TestCallStatsDerivesPerSourceFromLog(t *testing.T) {
	o := testOutbox(t)

	// A guard-session card: three edits of one card key. The drain coalesces to the newest,
	// so ONE chat.update reaches the wire and TWO are saved — exactly the noise the outbox
	// suppresses, and exactly what CallStats must attribute to this source.
	for _, v := range []string{"turns=1", "turns=1", "turns=1"} {
		if _, err := o.Enqueue(Row{Channel: "C1", Text: v, UpdateTS: "7.7", Source: "guard-session:status"}); err != nil {
			t.Fatal(err)
		}
	}
	// A clean one-shot post.
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "shipped", Source: "runcard"}); err != nil {
		t.Fatal(err)
	}
	// A leaky post the fence refuses — a call the outbox avoided entirely.
	needle := "node-" + "windows-a" // base PUBLIC_LEAK needle, assembled at runtime
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "ran on " + needle, Source: "logvault"}); err != nil {
		t.Fatal(err)
	}

	if _, err := o.Drain(context.Background(), newFakeWire(), drainOpts(nil)); err != nil {
		t.Fatal(err)
	}

	cs, err := o.CallStats(time.Now())
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]SourceCalls{}
	for _, sc := range cs.Sources {
		got[sc.Source] = sc
	}
	if g := got["guard-session:status"]; g.Updates != 1 || g.Coalesced != 2 || g.Sent() != 1 || g.Saved() != 2 {
		t.Fatalf("guard-session:status footprint wrong: %+v", g)
	}
	if g := got["runcard"]; g.Posts != 1 || g.Sent() != 1 {
		t.Fatalf("runcard footprint wrong: %+v", g)
	}
	if g := got["logvault"]; g.Refused != 1 || g.Sent() != 0 || g.Saved() != 1 {
		t.Fatalf("logvault footprint wrong: %+v", g)
	}
	if cs.TotalSent != 2 || cs.TotalSaved != 3 {
		t.Fatalf("totals wrong: sent=%d saved=%d (want 2/3)", cs.TotalSent, cs.TotalSaved)
	}
	if cs.LastCompactAgeS != -1 {
		t.Fatalf("uncompacted spool must report window floor -1, got %d", cs.LastCompactAgeS)
	}
}

// TestCallStatsSortsLoudestFirst pins the ordering contract: surfaces sort by calls SENT
// (descending), ties broken by name, so the biggest rate-limit spender floats to the top.
func TestCallStatsSortsLoudestFirst(t *testing.T) {
	o := testOutbox(t)
	// "beta" sends 2 posts, "alpha" sends 1, "zeta" sends 0 (only a coalesced-away edit).
	for _, v := range []string{"b1", "b2"} {
		if _, err := o.Enqueue(Row{Channel: "C1", Text: v, Source: "beta"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := o.Enqueue(Row{Channel: "C1", Text: "a1", Source: "alpha"}); err != nil {
		t.Fatal(err)
	}
	for _, v := range []string{"z1", "z2"} {
		if _, err := o.Enqueue(Row{Channel: "C2", Text: v, UpdateTS: "9.9", Source: "zeta"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := o.Drain(context.Background(), newFakeWire(), drainOpts(nil)); err != nil {
		t.Fatal(err)
	}
	cs, err := o.CallStats(time.Now())
	if err != nil {
		t.Fatal(err)
	}
	order := make([]string, len(cs.Sources))
	for i, sc := range cs.Sources {
		order[i] = sc.Source
	}
	// beta(2) > alpha(1) > zeta(0).
	if len(order) != 3 || order[0] != "beta" || order[1] != "alpha" || order[2] != "zeta" {
		t.Fatalf("sort order = %v, want [beta alpha zeta]", order)
	}
	if line := CallStatsReportLine(cs); line == "" {
		t.Fatal("report line is empty")
	}
}
