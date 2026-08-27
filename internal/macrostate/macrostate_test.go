package macrostate

import (
	"testing"
	"time"
)

func TestRestartCompactionCorrectionDeletionAndRetirement(t *testing.T) {
	now := time.Unix(100, 0)
	expiry := now.Add(-time.Second)
	s := &Store{}
	events := []Event{{Schema: Schema, ID: "1", At: now, Kind: Promote, Key: "commitment", Value: "ship", Provenance: "child:r1"}, {Schema: Schema, ID: "2", At: now, Kind: Promote, Key: "preference", Value: "stale", Provenance: "session:r2", ExpiresAt: &expiry}, {Schema: Schema, ID: "3", At: now, Kind: Correct, Key: "commitment", Value: "hold", Provenance: "operator:r3", Replaces: "1"}, {Schema: Schema, ID: "4", At: now, Kind: Delete, Key: "preference", Provenance: "policy:r4"}}
	for _, e := range events {
		r, err := s.Apply(e)
		if err != nil || !r.Applied || r.Digest == "" {
			t.Fatalf("apply receipt=%+v err=%v", r, err)
		}
	}
	restarted, err := Replay(s.Events())
	if err != nil {
		t.Fatal(err)
	}
	state := restarted.Compact(now)
	if len(state) != 1 || state["commitment"] != "hold" {
		t.Fatalf("state=%v", state)
	}
	if _, err := restarted.Apply(Event{Schema: Schema, ID: "5", At: now, Kind: Retire, Provenance: "operator:r5"}); err != nil {
		t.Fatal(err)
	}
	if len(restarted.Compact(now)) != 0 {
		t.Fatal("retirement retained state")
	}
	if _, err := restarted.Apply(events[0]); err == nil {
		t.Fatal("retired store accepted update")
	}
}
