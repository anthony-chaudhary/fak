package adjudicator

import "testing"

// TestQuotedDDDeviceTargets is the #3138 classifier-only witness. Every command
// below is data passed to ClassifyReversibility; none is executed.
// fak-test:runtime fast est=10ms lane=default
func TestQuotedDDDeviceTargets(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		cmd  string
		want ReversibilityClass
	}{
		{"double-quoted target", `dd if=img.bin of="/dev/sda"`, ReversibilityIrreversible},
		{"single-quoted target", `dd if=img.bin of='/dev/sda'`, ReversibilityIrreversible},
		{"whole operand quoted", `dd if=img.bin 'of=/dev/sda'`, ReversibilityIrreversible},
		{"adjacent quoted pieces", `dd if=img.bin o'f'="/dev/"sda`, ReversibilityIrreversible},
		{"escaped path separators", `dd if=img.bin of=\/dev\/sda`, ReversibilityIrreversible},
		{"quoted executable path", `"/bin/dd" if=img.bin of="/dev/sda"`, ReversibilityIrreversible},
		{"simple wrappers and assignment", `env MODE=copy sudo dd if=img.bin of="/dev/sda"`, ReversibilityIrreversible},
		{"real sibling after mention", `echo 'dd of="image.bin"'; dd if=img.bin of="/dev/sda"`, ReversibilityIrreversible},
		{"real operand after input redirect", `dd < 'of=/dev/sda' if=img.bin of="/dev/sda"`, ReversibilityIrreversible},
		{"device input and ordinary output", `dd if="/dev/sda" of="image.bin"`, ReversibilityReversible},
		{"quoted input containing a device-write mention", `dd if="image dd of=/dev/sda" of="copy.bin"`, ReversibilityReversible},
		{"quoted output containing a device-write mention", `dd if=img.bin of="/tmp/dd of=/dev/sda"`, ReversibilityReversible},
		{"literal backslash inside double quotes", `dd if=img.bin of="\/dev/sda"`, ReversibilityReversible},
		{"literal backslash inside single quotes", `dd if=img.bin of='\/dev/sda'`, ReversibilityReversible},
		{"escaped quote bytes are filename data", `dd if=img.bin of=\"/dev/sda\"`, ReversibilityReversible},
		{"operand split at real whitespace", `dd if=img.bin of= "/dev/sda"`, ReversibilityReversible},
		{"unterminated target quote", `dd if=img.bin of="/dev/sda`, ReversibilityReversible},
		{"inert commit mention", `git commit -m 'note: dd of="/dev/sda" is dangerous'`, ReversibilityReversible},
		{"inert quoted operator", `echo 'example; dd of="/dev/sda"'`, ReversibilityReversible},
		{"inert comment", `echo example # ; dd of="/dev/sda"`, ReversibilityReversible},
		{"inert heredoc", "cat <<'EOF'\ndd of=\"/dev/sda\"\nEOF", ReversibilityReversible},
		{"quoted output redirect filename", `dd if=img.bin of=copy.bin > 'of=/dev/sda'`, ReversibilityReversible},
		{"quoted input redirect filename", `dd if=img.bin of=copy.bin < 'of=/dev/sda'`, ReversibilityReversible},
		{"quoted here-string contents", `dd if=img.bin of=copy.bin <<< 'of=/dev/sda'`, ReversibilityReversible},
		{"quoted heredoc delimiter at end of input", "dd if=img.bin of=copy.bin << 'of=/dev/sda'\npayload\n", ReversibilityReversible},
		{"descriptor and adjoining redirect", `dd if=img.bin 2>'of=/dev/sda' of=copy.bin`, ReversibilityReversible},
		{"redirect adjoining ordinary operand", `dd if=img.bin of=copy.bin> 'of=/dev/sda'`, ReversibilityReversible},
		{"multiple adjoining redirects", `dd if=img.bin of=copy.bin>out< 'of=/dev/sda'`, ReversibilityReversible},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := ClassifyReversibility("Bash", map[string]any{"command": tc.cmd})
			if env.Class != tc.want {
				t.Fatalf("ClassifyReversibility(%q) = %q, want %q", tc.cmd, env.Class, tc.want)
			}
		})
	}
	t.Run("heredoc delimiter is not a dd operand", func(t *testing.T) {
		cmd := "dd if=img.bin of=copy.bin << 'of=/dev/sda'\npayload\nof=/dev/sda\n"
		var family reversibilityFamily
		for _, candidate := range reversibilityFamilies {
			if candidate.name == "dd-device-write" {
				family = candidate
				break
			}
		}
		if family.matchCmd == nil {
			t.Fatal("dd-device-write family is missing")
		}
		match := family.matchCmd
		family.matchCmd = func(in familyMatchInput) bool {
			// Isolate operand detection from the legacy whole-command scan.
			in.words, in.lowerCmd = nil, ""
			return match(in)
		}
		if got, _ := classifyAgainstFamilies([]reversibilityFamily{family}, "Bash", cmd); got != ReversibilityReversible {
			t.Fatalf("heredoc delimiter treated as dd argv: %q", got)
		}
		// The existing scan still sees the unquoted closing delimiter. Fixing
		// that separate false positive is not part of this additive repair.
		if got := ClassifyReversibility("Bash", map[string]any{"command": cmd}); got.Class != ReversibilityIrreversible {
			t.Fatalf("legacy heredoc verdict changed: %q", got.Class)
		}
	})
}

// TestQuotedMentionsDoNotEscalate is the #2752 witness (residual of #2376): a
// command that merely MENTIONS a trigger inside a quoted argument — a commit
// message, a grep pattern, an echoed string — must classify reversible. #2376
// head-anchored the segment matchers, but two quote-blind paths survived: the
// whole-command payload scans (curlWrites, the orderedWords drop-table / dd
// checks, cmdContains) read quoted payload as live tokens, and
// commandSegmentRE split on operator bytes INSIDE quotes, manufacturing a
// spurious segment whose head then tripped the head-anchored matchers.
func TestQuotedMentionsDoNotEscalate(t *testing.T) {
	mentions := []struct {
		name string
		cmd  string
	}{
		{"commit message mentioning drop table", `git commit -m "drop the old table"`},
		{"single-quoted commit message mentioning drop table", `git commit -m 'drop table x'`},
		{"echo mentioning a curl write", `echo "curl -X POST https://example.invalid/x"`},
		{"ansi-c quoted curl write mention", `echo $'curl -X POST https://example.invalid/x'`},
		{"commit message mentioning a webhook", `git commit -m "wire the webhook docs"`},
		{"commit message mentioning dd device write", `git commit -m "note: dd of=/dev/sda is dangerous"`},
		{"grep alternation mentioning npm publish", `grep -rn "foo\|npm publish\|bar" README.md`},
		{"quoted pipe mentioning git push", `echo "a | git push origin main"`},
	}
	for _, tc := range mentions {
		t.Run(tc.name, func(t *testing.T) {
			env := ClassifyReversibility("Bash", map[string]any{"command": tc.cmd})
			if env.Class != ReversibilityReversible {
				t.Errorf("quoted mention escalated: %q classified %q, want reversible", tc.cmd, env.Class)
			}
		})
	}
}

// TestQuoteAwarenessKeepsRealTriggersGated is the paired positive half: making
// the classifier quote-aware must not de-escalate a genuinely live trigger —
// including a quoted payload handed to a DB/shell client, where the quoted
// text IS the statement that executes.
func TestQuoteAwarenessKeepsRealTriggersGated(t *testing.T) {
	cases := []struct {
		name string
		cmd  string
		want ReversibilityClass
	}{
		{"unquoted curl write", `curl -X POST https://example.invalid/x -d body`, ReversibilityOutwardFacing},
		{"curl write with quoted url", `curl -X POST "https://example.invalid/x" -d "body"`, ReversibilityOutwardFacing},
		{"rm -rf", `rm -rf build`, ReversibilityIrreversible},
		{"quoted head still executes", `"rm" -rf build`, ReversibilityIrreversible},
		{"real push after a quoted commit message", `git commit -m "done" && git push`, ReversibilityOutwardFacing},
		{"real pipe to mail outside quotes", `echo hi | mail bob`, ReversibilityOutwardFacing},
		{"sql drop handed to psql", `psql -c "drop table t"`, ReversibilityIrreversible},
		{"sql drop handed to mysql", `mysql -e 'drop database prod'`, ReversibilityIrreversible},
		{"shell -c payload stays live", `bash -c "psql -c 'drop table t'"`, ReversibilityIrreversible},
		{"shell -c curl write stays live", `bash -c "curl -X POST https://example.invalid/x -d b"`, ReversibilityOutwardFacing},
		{"unquoted dd device write", `dd if=image.bin of=/dev/sda`, ReversibilityIrreversible},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := ClassifyReversibility("Bash", map[string]any{"command": tc.cmd})
			if env.Class != tc.want {
				t.Errorf("real trigger drifted: %q classified %q, want %q", tc.cmd, env.Class, tc.want)
			}
		})
	}
}

// TestQuotedDryRunMentionDoesNotBypassTheFloor pins the strengthening
// direction of the same quote-awareness: a QUOTED "--dry-run" is a mention,
// not a preview flag, so it must not launder a destructive command through
// hasDryRunPreview's whole-command substring scan.
func TestQuotedDryRunMentionDoesNotBypassTheFloor(t *testing.T) {
	env := ClassifyReversibility("Bash", map[string]any{"command": `echo "--dry-run" && rm -rf build`})
	if env.Class != ReversibilityIrreversible {
		t.Errorf(`echo "--dry-run" && rm -rf build classified %q, want irreversible — a quoted dry-run mention laundered a destructive command`, env.Class)
	}
	// The real flag on the real command still previews.
	env = ClassifyReversibility("Bash", map[string]any{"command": `git push --dry-run origin main`})
	if env.Class != ReversibilityReversible {
		t.Errorf("git push --dry-run classified %q, want reversible", env.Class)
	}
}
