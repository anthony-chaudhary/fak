package scoreboard

import (
	"strings"
	"testing"
)

// An ACTION verdict must never render the green glyph, even with a grade-A headline:
// the industry map grades A on honesty while reporting weak_standing/ACTION, and a
// green circle on that post would read as "all good" — the map-vs-standing conflation
// standing_score exists to surface. The verdict, not the letter grade, decides the glyph.
func TestGradeEmojiActionOverridesGradeA(t *testing.T) {
	got := gradeEmoji("A", "ACTION")
	if got != ":red_circle:" {
		t.Fatalf("A-grade ACTION must show the action glyph, got %q", got)
	}
}

func TestGradeEmojiOKGradeAStaysGreen(t *testing.T) {
	got := gradeEmoji("A", "OK")
	if got != ":large_green_circle:" {
		t.Fatalf("A-grade OK should stay green, got %q", got)
	}
}

func TestGradeEmojiActionNoGradeIsRed(t *testing.T) {
	if got := gradeEmoji("", "ACTION"); got != ":red_circle:" {
		t.Fatalf("ACTION with no grade should be red, got %q", got)
	}
}

// The end-to-end render: an A-graded ACTION update must surface ACTION in its text and
// must not lead with the green glyph.
func TestUpdateTextActionNotGreen(t *testing.T) {
	u := Update{Title: "fak-industry-scorecard/2", Grade: "A", Score: "98.8", Verdict: "ACTION"}
	txt := u.Text()
	if strings.Contains(txt, ":large_green_circle:") {
		t.Fatalf("A-graded ACTION post must not render green: %q", txt)
	}
	if !strings.Contains(txt, "ACTION") {
		t.Fatalf("verdict ACTION must appear in the text: %q", txt)
	}
}
