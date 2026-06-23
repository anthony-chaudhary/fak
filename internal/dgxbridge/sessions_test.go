package dgxbridge

import (
	"strings"
	"testing"
)

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

// sampleCompactReply is the NEW "*Sessions:*" format the hub switched to: a header line,
// a profiles line, a section title, then `id status profile/mode | age | thread=...`
// entries (live first). This is the exact shape captured live from the control channel.
const sampleCompactReply = "*Sessions:* 56 total, 6 live (4 bridges) — running=4 stopped=2 | exited=34 terminated=16\n" +
	"*Profiles:* default=51 (pipe) | persistent=5 (tmux)\n\n" +
	"*All sessions* — live first\n" +
	"- `persistent-3` running persistent/tmux | 2d03h | thread=`1782049840.825709`\n" +
	"- `persistent-6` running persistent/tmux | 2d02h | thread=`1782052065.672089`\n" +
	"- `default-4` stopped default/pipe | 3d02h | thread=`1781967135.188639`\n" +
	"- `default-55` terminated default/pipe | 2d04h | thread=`1782045432.768949`\n" +
	"- `default-102` exited default/pipe | 2d02h | thread=`1782053089.399039`"

func TestParseSessions_CompactFormat(t *testing.T) {
	got := parseSessions(sampleCompactReply)
	if len(got) != 5 {
		t.Fatalf("want 5 sessions, got %d: %+v", len(got), got)
	}
	first := got[0]
	if first.ID != "persistent-3" || first.Status != "running" ||
		first.Profile != "persistent" || first.Mode != "tmux" ||
		first.ThreadTS != "1782049840.825709" {
		t.Fatalf("persistent-3 parsed wrong: %+v", first)
	}
	// The age field must not be mistaken for the thread ts.
	for _, s := range got {
		if s.ThreadTS == "" || strings.Contains(s.ThreadTS, "d") {
			t.Fatalf("thread ts looks wrong (age leaked in?): %+v", s)
		}
	}
}

func TestPickRunning_CompactNewestRunning(t *testing.T) {
	got := parseSessions(sampleCompactReply)
	s, ok := PickRunning(got, "")
	if !ok {
		t.Fatal("expected a running session in the compact reply")
	}
	// persistent-6 (1782052065...) is the newest running over persistent-3 (1782049840...).
	if s.ID != "persistent-6" {
		t.Fatalf("want newest running persistent-6, got %s", s.ID)
	}
}

func TestParseSessions_MixedNoDoubleCount(t *testing.T) {
	// A reply that somehow carries both grammars for the same id must not double-count.
	mixed := sampleSessionsReply + "\n" +
		"- `default-1` running default/pipe | 1d | thread=`1781964298.319749`"
	got := parseSessions(mixed)
	seen := map[string]int{}
	for _, s := range got {
		seen[s.ID]++
	}
	if seen["default-1"] != 1 {
		t.Fatalf("default-1 counted %d times, want 1: %+v", seen["default-1"], got)
	}
}

func TestIsSessionsListing_BothHeaders(t *testing.T) {
	if !isSessionsListing("Known control sessions: ...") {
		t.Fatal("verbose header should be recognized")
	}
	if !isSessionsListing("*Sessions:* 56 total ...") {
		t.Fatal("compact header should be recognized")
	}
	if isSessionsListing("just some chatter") {
		t.Fatal("unrelated text should not be a listing")
	}
}
