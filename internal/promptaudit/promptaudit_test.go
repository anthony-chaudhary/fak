package promptaudit

import (
	"strings"
	"testing"
)

// sentence builds a "Today's date is ..." carrier sentence with a chosen
// apostrophe rune and a chosen date separator, so a fixture can exercise either
// channel independently or both together.
func sentence(apostrophe string, dateSep string) string {
	date := "2026" + dateSep + "06" + dateSep + "30"
	return "You are a helpful assistant. Today" + apostrophe + "s date is " + date + "."
}

// findOne returns the first finding of the given kind, or fails the test.
func findOne(t *testing.T, fs []Finding, kind Kind) Finding {
	t.Helper()
	for _, f := range fs {
		if f.Kind == kind {
			return f
		}
	}
	t.Fatalf("expected a finding of kind %q, got %d findings: %v", kind, len(fs), fs)
	return Finding{}
}

func hasKind(fs []Finding, kind Kind) bool {
	for _, f := range fs {
		if f.Kind == kind {
			return true
		}
	}
	return false
}

func hasChannel(fs []Finding, ch Channel) bool {
	for _, f := range fs {
		if f.Channel == ch {
			return true
		}
	}
	return false
}

// TestApostropheVariants covers the four documented apostrophe-position values:
// ASCII (benign) must NOT fire; the three non-ASCII marker runes MUST each fire
// a structured hostname-channel finding.
func TestApostropheVariants(t *testing.T) {
	cases := []struct {
		name     string
		apos     string
		wantFire bool
		wantCP   string
	}{
		{"ascii-apostrophe-benign", "'", false, ""},
		{"right-single-quote", "’", true, "U+2019"},
		{"modifier-letter-apostrophe", "ʼ", true, "U+02BC"},
		{"modifier-letter-prime", "ʹ", true, "U+02B9"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			// Use the benign '-' separator so only the apostrophe channel is
			// under test here.
			text := sentence(c.apos, "-")
			fs := Scan(text)
			fired := hasKind(fs, KindLookalikeApostrophe)
			if fired != c.wantFire {
				t.Fatalf("apostrophe %q: fired=%v want=%v (findings=%v)", c.apos, fired, c.wantFire, fs)
			}
			if !c.wantFire {
				return
			}
			f := findOne(t, fs, KindLookalikeApostrophe)
			if f.Channel != ChannelHostname {
				t.Errorf("channel = %q, want %q", f.Channel, ChannelHostname)
			}
			if len(f.Codepoints) != 1 || f.Codepoints[0] != c.wantCP {
				t.Errorf("codepoints = %v, want [%s]", f.Codepoints, c.wantCP)
			}
			// The finding must be explainable: location, raw, normalized.
			if f.ByteOffset <= 0 {
				t.Errorf("ByteOffset = %d, want > 0", f.ByteOffset)
			}
			if f.RuneOffset <= 0 {
				t.Errorf("RuneOffset = %d, want > 0", f.RuneOffset)
			}
			if f.Raw != c.apos {
				t.Errorf("Raw = %q, want %q", f.Raw, c.apos)
			}
			if f.Normalized != "'" {
				t.Errorf("Normalized = %q, want ASCII apostrophe", f.Normalized)
			}
			// The raw offending rune must NOT have been mutated out of the
			// original input we passed (evidence preserved).
			if !strings.Contains(text, c.apos) {
				t.Errorf("input lost the marker rune — evidence should be preserved")
			}
			// String() must render without panicking and mention the codepoint.
			if !strings.Contains(f.String(), c.wantCP) {
				t.Errorf("String() = %q, want it to contain %s", f.String(), c.wantCP)
			}
		})
	}
}

// TestDateSeparatorChannel covers BOTH date separators around the carrier
// sentence: '-' (default) must NOT fire; '/' (off-channel) MUST fire a
// structured timezone-channel finding.
func TestDateSeparatorChannel(t *testing.T) {
	t.Run("dash-default-benign", func(t *testing.T) {
		fs := Scan(sentence("'", "-"))
		if hasKind(fs, KindDateSeparator) {
			t.Fatalf("default '-' separator should not fire, got %v", fs)
		}
	})
	t.Run("slash-offchannel", func(t *testing.T) {
		text := sentence("'", "/")
		fs := Scan(text)
		f := findOne(t, fs, KindDateSeparator)
		if f.Channel != ChannelTimezone {
			t.Errorf("channel = %q, want %q", f.Channel, ChannelTimezone)
		}
		if f.Raw != "2026/06/30" {
			t.Errorf("Raw = %q, want the slash date token", f.Raw)
		}
		if f.Normalized != "2026-06-30" {
			t.Errorf("Normalized = %q, want dash-normalized token", f.Normalized)
		}
		if len(f.Codepoints) != 1 || f.Codepoints[0] != "U+002F" {
			t.Errorf("codepoints = %v, want [U+002F]", f.Codepoints)
		}
		if f.ByteOffset <= 0 {
			t.Errorf("ByteOffset = %d, want > 0", f.ByteOffset)
		}
	})
}

// TestBenignProseDoesNotFire is the fail-closed guard: ordinary curly-quoted
// prose (a right single quote used as a normal contraction) WITHOUT the date
// carrier sentence must NOT produce any finding.
func TestBenignProseDoesNotFire(t *testing.T) {
	benign := []string{
		"It’s a lovely day and the model’s job is to help. Don’t worry.",
		"Use ASCII apostrophes here: it's fine, that's normal, we're good.",
		"The weather report says highs of 2026/06 percent humidity in the lab.", // slash digits but no "date is" carrier
		"Plain prose with no markers at all, just words and a date 2026-06-30 mentioned casually.",
	}
	for _, b := range benign {
		fs := Scan(b)
		if len(fs) != 0 {
			t.Errorf("benign text fired %d findings (should be 0): input=%q findings=%v", len(fs), b, fs)
		}
	}
}

// TestMixedChannels is the regression matrix the ticket requires: hostname
// marker only, timezone marker only, and both together.
func TestMixedChannels(t *testing.T) {
	t.Run("hostname-only", func(t *testing.T) {
		fs := Scan(sentence("ʼ", "-"))
		if !hasChannel(fs, ChannelHostname) {
			t.Errorf("expected hostname-marker channel, got %v", fs)
		}
		if hasChannel(fs, ChannelTimezone) {
			t.Errorf("did not expect timezone-marker channel, got %v", fs)
		}
	})
	t.Run("timezone-only", func(t *testing.T) {
		fs := Scan(sentence("'", "/"))
		if !hasChannel(fs, ChannelTimezone) {
			t.Errorf("expected timezone-marker channel, got %v", fs)
		}
		if hasChannel(fs, ChannelHostname) {
			t.Errorf("did not expect hostname-marker channel, got %v", fs)
		}
	})
	t.Run("both-together", func(t *testing.T) {
		fs := Scan(sentence("’", "/"))
		if !hasChannel(fs, ChannelHostname) {
			t.Errorf("expected hostname-marker channel, got %v", fs)
		}
		if !hasChannel(fs, ChannelTimezone) {
			t.Errorf("expected timezone-marker channel, got %v", fs)
		}
		if len(fs) < 2 {
			t.Errorf("expected at least 2 findings for both channels, got %d: %v", len(fs), fs)
		}
		// Findings must be ordered by byte offset.
		for i := 1; i < len(fs); i++ {
			if fs[i-1].ByteOffset > fs[i].ByteOffset {
				t.Errorf("findings not ordered by ByteOffset: %v", fs)
			}
		}
	})
}

// TestInvisibleRunes covers the principled, category-driven invisible-rune
// detection (zero-width / format / control), independent of the date sentence.
func TestInvisibleRunes(t *testing.T) {
	cases := []struct {
		name string
		text string
		cp   string
	}{
		{"zero-width-space", "hello​world", "U+200B"},
		{"zero-width-joiner", "a‍b", "U+200D"},
		{"zero-width-non-joiner", "a‌b", "U+200C"},
		{"bom-word-joiner", "x⁠y", "U+2060"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs := Scan(c.text)
			f := findOne(t, fs, KindInvisibleRune)
			if f.Channel != ChannelUnicodeControl {
				t.Errorf("channel = %q, want %q", f.Channel, ChannelUnicodeControl)
			}
			if len(f.Codepoints) != 1 || f.Codepoints[0] != c.cp {
				t.Errorf("codepoints = %v, want [%s]", f.Codepoints, c.cp)
			}
		})
	}
}

// TestEmptyAndPlain confirm the no-finding paths.
func TestEmptyAndPlain(t *testing.T) {
	if fs := Scan(""); len(fs) != 0 {
		t.Errorf("empty input fired %d findings", len(fs))
	}
	if fs := Scan("just some perfectly ordinary text"); len(fs) != 0 {
		t.Errorf("plain input fired %d findings: %v", len(fs), fs)
	}
}

// TestStringStable ensures String() is non-empty and stable in shape.
func TestStringStable(t *testing.T) {
	fs := Scan(sentence("’", "/"))
	if len(fs) == 0 {
		t.Fatal("expected findings")
	}
	for _, f := range fs {
		s := f.String()
		if !strings.HasPrefix(s, "[") {
			t.Errorf("String() = %q, want it to start with the kind in brackets", s)
		}
		if !strings.Contains(s, "channel=") {
			t.Errorf("String() = %q, want it to name the channel", s)
		}
	}
}
