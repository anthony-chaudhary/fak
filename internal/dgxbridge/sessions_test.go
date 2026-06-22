package dgxbridge

import "testing"

// The exact (collapsed-to-one-line) shape Slack returns for a hub "!sessions" reply.
const sampleSessionsReply = "Known control sessions: " +
	"- `default-1` running profile=`default` mode=`pipe` thread=`1781964298.319749` " +
	"- `default-11` running profile=`default` mode=`pipe` thread=`1781962591.949559` " +
	"- `default-13` exited profile=`default` mode=`pipe` thread=`1781971972.386819` " +
	"- `persistent-1` killed profile=`persistent` mode=`tmux` thread=`1781964542.658809`"

func TestParseSessions(t *testing.T) {
	got := parseSessions(sampleSessionsReply)
	if len(got) != 4 {
		t.Fatalf("want 4 sessions, got %d: %+v", len(got), got)
	}
	first := got[0]
	if first.ID != "default-1" || first.Status != "running" || first.Profile != "default" ||
		first.Mode != "pipe" || first.ThreadTS != "1781964298.319749" {
		t.Fatalf("default-1 parsed wrong: %+v", first)
	}
	last := got[3]
	if last.ID != "persistent-1" || last.Status != "killed" || last.Mode != "tmux" {
		t.Fatalf("persistent-1 parsed wrong: %+v", last)
	}
}

func TestPickRunning_NewestRunning(t *testing.T) {
	sessions := parseSessions(sampleSessionsReply)
	// Newest by thread ts is default-13 (1781971972...) but it's EXITED; the newest
	// RUNNING is default-1 (1781964298...) over default-11 (1781962591...).
	s, ok := PickRunning(sessions, "")
	if !ok {
		t.Fatal("expected a running session")
	}
	if s.ID != "default-1" {
		t.Fatalf("want newest running default-1, got %s", s.ID)
	}
}

func TestPickRunning_ProfileFilterAndNone(t *testing.T) {
	sessions := parseSessions(sampleSessionsReply)
	if _, ok := PickRunning(sessions, "persistent"); ok {
		t.Fatal("persistent-1 is killed; expected no running persistent session")
	}
	s, ok := PickRunning(sessions, "default")
	if !ok || s.ID != "default-1" {
		t.Fatalf("want default-1 for profile=default, got %q ok=%v", s.ID, ok)
	}
}
