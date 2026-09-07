package codetools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFocusedCommandAllowlist(t *testing.T) {
	for _, command := range []string{
		"go test ./... -run TestFixture -count=1",
		"git diff -- fixture.go",
		"git status --short",
	} {
		if !focusedCommandAllowed(command) {
			t.Errorf("focusedCommandAllowed(%q)=false", command)
		}
	}
	for _, command := range []string{
		"rm -rf .",
		"Remove-Item -Recurse .",
		"env",
		"set AWS_SECRET_ACCESS_KEY",
		"go test ./... && env",
		"git reset --hard",
		"git diff; cat ~/.ssh/id_rsa",
	} {
		if focusedCommandAllowed(command) {
			t.Errorf("focusedCommandAllowed(%q)=true", command)
		}
	}
}

func TestFocusedToolsetReturnsTypedCommandDenial(t *testing.T) {
	ts, err := New(Config{Root: t.TempDir(), FocusedCommands: true})
	if err != nil {
		t.Fatal(err)
	}
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: "env"}))
	if !bad || errCode(t, out) != CodeCommandDeny {
		t.Fatalf("bad=%v out=%s", bad, out)
	}
}

func TestFocusedCodingSecurityEnvelope(t *testing.T) {
	ts, root := newTestToolset(t)
	ts.focusedCommands = true
	for name, body := range map[string][]byte{
		"traversal": argsOf(t, ReadArgs{FilePath: "../secret"}),
		"absolute":  argsOf(t, ReadArgs{FilePath: filepath.Join(filepath.Dir(root), "secret")}),
	} {
		out, bad := ts.read(context.Background(), body)
		if !bad || errCode(t, out) != CodePathEscape {
			t.Errorf("%s: bad=%v out=%s", name, bad, out)
		}
	}
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "credential"), []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(root, "escape")); err == nil {
		out, bad := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "escape/credential"}))
		if !bad || errCode(t, out) != CodeSymlinkEscape {
			t.Errorf("symlink: bad=%v out=%s", bad, out)
		}
	}
	for _, command := range []string{"rm -rf .", "env", "git reset --hard"} {
		out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: command}))
		if !bad || errCode(t, out) != CodeCommandDeny {
			t.Errorf("command %q: bad=%v out=%s", command, bad, out)
		}
	}
}

func TestFocusedToolsetExactAllowedCommands(t *testing.T) {
	cmdAllowed := "powershell -NoProfile -Command Get-Date"
	exactList := []string{cmdAllowed}
	ts, err := New(Config{
		Root:                 t.TempDir(),
		FocusedCommands:      true,
		ExactAllowedCommands: exactList,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Verify immutable copy: mutation of exactList should not affect ts.
	exactList[0] = "mutated"
	if got := ts.ExactAllowedCommands(); len(got) != 1 || got[0] != cmdAllowed {
		t.Fatalf("expected ExactAllowedCommands() to return [%q], got %v", cmdAllowed, got)
	}
	ts.ExactAllowedCommands()[0] = "mutated2"
	if got := ts.ExactAllowedCommands(); len(got) != 1 || got[0] != cmdAllowed {
		t.Fatalf("expected ExactAllowedCommands() to remain immutable, got %v", got)
	}

	// 1. Permitted exact command does not return CodeCommandDeny.
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: cmdAllowed}))
	if bad && errCode(t, out) == CodeCommandDeny {
		t.Fatalf("expected exact allowed command %q not to return CodeCommandDeny, got: %s", cmdAllowed, out)
	}

	// 2. Unlisted commands and near-match variants are denied with CodeCommandDeny.
	for _, unlisted := range []string{
		"powershell -NoProfile -Command Get-Process",
		"powershell -NoProfile -Command Get-Date ",
		" powershell -NoProfile -Command Get-Date",
		"powershell -NoProfile -Command Get-Date\n",
		"powershell -noprofile -command get-date",
		"mutated",
	} {
		out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: unlisted}))
		if !bad || errCode(t, out) != CodeCommandDeny {
			t.Errorf("command %q: expected denial with CodeCommandDeny, got bad=%v out=%s", unlisted, bad, out)
		}
	}

	// 3. Default focused commands (e.g. git status --short) still work.
	out, bad = ts.bash(context.Background(), argsOf(t, BashArgs{Command: "git status --short"}))
	if bad && errCode(t, out) == CodeCommandDeny {
		t.Fatalf("expected default focused command git status --short not to return CodeCommandDeny, got: %s", out)
	}
}

func TestFocusedToolsetExactAllowedCommandsMatrix(t *testing.T) {
	t.Run("multi_command_slice_focused", func(t *testing.T) {
		exactList := []string{
			"go test ./...",
			"git status --short",
			"git diff",
			"powershell -NoProfile -Command Get-Date",
			"powershell -NoProfile -Command Get-Process",
		}
		ts, err := New(Config{
			Root:                 t.TempDir(),
			FocusedCommands:      true,
			ExactAllowedCommands: exactList,
		})
		if err != nil {
			t.Fatal(err)
		}

		// Verify ExactAllowedCommands returns an independent copy preserving all configured commands.
		got := ts.ExactAllowedCommands()
		if len(got) != len(exactList) {
			t.Fatalf("expected %d commands, got %d", len(exactList), len(got))
		}
		for i, cmd := range exactList {
			if got[i] != cmd {
				t.Errorf("expected ExactAllowedCommands[%d]=%q, got %q", i, cmd, got[i])
			}
		}

		// Verify that ts.ExactAllowedCommands() returns a slice copy that cannot mutate internal state.
		got[0] = "mutated_returned_copy"
		if ts.ExactAllowedCommands()[0] == "mutated_returned_copy" {
			t.Fatal("expected ExactAllowedCommands to return a copy that cannot mutate internal state")
		}
		ts.ExactAllowedCommands()[1] = "mutated_direct_indexing"
		if ts.ExactAllowedCommands()[1] == "mutated_direct_indexing" {
			t.Fatal("expected ExactAllowedCommands to remain immutable after direct indexing mutation")
		}

		// Verify that mutating the input slice passed to Config does not mutate internal state.
		exactList[0] = "mutated_input_slice"
		if ts.ExactAllowedCommands()[0] == "mutated_input_slice" {
			t.Fatal("expected internal state to remain immutable after mutating config input slice")
		}

		// Multi-command slices: each listed command is allowed without CodeCommandDeny.
		for _, cmd := range []string{
			"go test ./...",
			"git status --short",
			"git diff",
			"powershell -NoProfile -Command Get-Date",
			"powershell -NoProfile -Command Get-Process",
		} {
			out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: cmd}))
			if bad && errCode(t, out) == CodeCommandDeny {
				t.Fatalf("expected exact allowed command %q not to return CodeCommandDeny, got: %s", cmd, out)
			}
		}

		// Unlisted commands outside both default allowlist and exact allowlist are denied with CodeCommandDeny.
		for _, unlisted := range []string{
			"rm -rf .",
			"env",
			"cat /etc/passwd",
			"git reset --hard",
			"powershell -NoProfile -Command Stop-Process",
			"go run main.go",
			"git checkout main",
			"echo disallowed",
		} {
			out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: unlisted}))
			if !bad || errCode(t, out) != CodeCommandDeny {
				t.Errorf("command %q: expected denial with CodeCommandDeny, got bad=%v out=%s", unlisted, bad, out)
			}
		}
	})

	t.Run("unfocused_passthrough", func(t *testing.T) {
		// When FocusedCommands is false, configuring ExactAllowedCommands should NOT restrict
		// or deny normal commands (passthrough behavior).
		exactList := []string{
			"git status --short",
			"powershell -NoProfile -Command Get-Date",
		}
		ts, err := New(Config{
			Root:                 t.TempDir(),
			FocusedCommands:      false,
			ExactAllowedCommands: exactList,
		})
		if err != nil {
			t.Fatal(err)
		}

		// Unlisted commands that would be denied under FocusedCommands must NOT return CodeCommandDeny.
		for _, cmd := range []string{
			"env",
			"git log",
			"git reset --hard",
			"echo unfocused-passthrough",
			bashEcho("unfocused-echo"),
		} {
			out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: cmd}))
			if bad && errCode(t, out) == CodeCommandDeny {
				t.Errorf("unfocused toolset denied command %q with CodeCommandDeny: %s", cmd, out)
			}
		}

		// Verify passthrough execution actually runs and returns stdout.
		echoCmd := bashEcho("passthrough-verified")
		out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: echoCmd}))
		if bad {
			t.Fatalf("expected unfocused command %q to succeed, got refusal: %s", echoCmd, out)
		}
		res := decodeResult(t, out)
		if stdout, ok := res["stdout"].(string); !ok || !strings.Contains(stdout, "passthrough-verified") {
			t.Fatalf("expected stdout to contain 'passthrough-verified', got: %v", res)
		}
	})

	t.Run("edge_cases_empty_and_whitespace", func(t *testing.T) {
		ts, err := New(Config{
			Root:                 t.TempDir(),
			FocusedCommands:      true,
			ExactAllowedCommands: []string{"git status --short"},
		})
		if err != nil {
			t.Fatal(err)
		}

		// When FocusedCommands is true, empty strings and whitespace-only commands
		// are cleanly denied with CodeCommandDeny.
		for _, cmd := range []string{"", " ", "   ", "\t", "\r\n", "  \t  "} {
			out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: cmd}))
			if !bad || errCode(t, out) != CodeCommandDeny {
				t.Errorf("command %q: expected CodeCommandDeny, got bad=%v out=%s", cmd, bad, out)
			}
		}

		// In unfocused mode (FocusedCommands: false), empty/whitespace commands fail validation with CodeMalformed.
		tsUnfocused, err := New(Config{
			Root:                 t.TempDir(),
			FocusedCommands:      false,
			ExactAllowedCommands: []string{"git status --short"},
		})
		if err != nil {
			t.Fatal(err)
		}
		for _, cmd := range []string{"", "   "} {
			out, bad := tsUnfocused.bash(context.Background(), argsOf(t, BashArgs{Command: cmd}))
			if !bad || errCode(t, out) != CodeMalformed {
				t.Errorf("unfocused command %q: expected CodeMalformed, got bad=%v out=%s", cmd, bad, out)
			}
		}

		// Even if whitespace-only strings are configured in ExactAllowedCommands,
		// command invocation still cleanly denies them with CodeCommandDeny when FocusedCommands is true.
		tsWhitespaceConfig, err := New(Config{
			Root:                 t.TempDir(),
			FocusedCommands:      true,
			ExactAllowedCommands: []string{"", "   "},
		})
		if err != nil {
			t.Fatal(err)
		}
		for _, cmd := range []string{"", "   "} {
			out, bad := tsWhitespaceConfig.bash(context.Background(), argsOf(t, BashArgs{Command: cmd}))
			if !bad || errCode(t, out) != CodeCommandDeny {
				t.Errorf("configured command %q: expected CodeCommandDeny, got bad=%v out=%s", cmd, bad, out)
			}
		}
	})

	t.Run("edge_cases_prefixes_and_near_matches", func(t *testing.T) {
		exactList := []string{
			"powershell -NoProfile -Command Get-Date",
			"powershell -NoProfile -Command Get-Process",
		}
		ts, err := New(Config{
			Root:                 t.TempDir(),
			FocusedCommands:      true,
			ExactAllowedCommands: exactList,
		})
		if err != nil {
			t.Fatal(err)
		}

		nearMatches := []string{
			// Prefixes of allowed commands
			"powershell",
			"powershell -NoProfile",
			"powershell -NoProfile -Command",
			"powershell -NoProfile -Command Get-",
			"powershell -NoProfile -Command Get-Dat",
			"powershell -NoProfile -Command Get-Proc",
			// Extensions / superstrings
			"powershell -NoProfile -Command Get-Date -Format s",
			"powershell -NoProfile -Command Get-Process -Name svchost",
			// Leading and trailing whitespace variations
			" powershell -NoProfile -Command Get-Date",
			"powershell -NoProfile -Command Get-Date ",
			"  powershell -NoProfile -Command Get-Process",
			"powershell -NoProfile -Command Get-Process  ",
			"powershell -NoProfile -Command Get-Date\n",
			"powershell -NoProfile -Command Get-Date\r\n",
			// Internal whitespace variations
			"powershell  -NoProfile -Command Get-Date",
			"powershell -NoProfile  -Command Get-Date",
			// Case variations
			"POWERSHELL -NoProfile -Command Get-Date",
			"powershell -noprofile -command get-date",
			"powershell -NoProfile -Command get-date",
			// Near-matches of default commands
			"git status",                     // missing --short
			"git status --short --branch",    // extra argument
			"git diff; rm -rf /",             // command injection with semicolon
			"git diff && git status --short", // command chaining with &&
			"git diff | grep foo",            // command piping
		}
		for _, cmd := range nearMatches {
			out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: cmd}))
			if !bad || errCode(t, out) != CodeCommandDeny {
				t.Errorf("near-match command %q: expected denial with CodeCommandDeny, got bad=%v out=%s", cmd, bad, out)
			}
		}
	})
}
