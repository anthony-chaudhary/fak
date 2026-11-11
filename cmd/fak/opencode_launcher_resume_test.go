package main

// opencode_launcher_resume_test.go — #14330: the launcher-side half of OpenCode
// exact-resume. Pins the --continue/--session/--resume flag mapping onto the child argv,
// the `fak opencode resume` one-touch subcommand (fail-closed preflight through the
// stubbed opencoderesume.Verify seam), and the open/close session-journal rows the
// launcher writes so guarded opencode launches become visible to `fak session recover`'s
// journal crash merge. No real opencode spawn: opencodeLaunchRun and
// opencodeLauncherVerify are stubbed throughout.

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/opencoderesume"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

// opencodeResumeTestSessionID matches ^ses_[A-Za-z0-9]{16,64}$ (26 alphanumeric chars).
const opencodeResumeTestSessionID = "ses_e3861b794ffeKK5UOg25IFE1mR"

// stubOpencodeLaunchRun replaces the exec seam with a recorder returning childCode.
func stubOpencodeLaunchRun(t *testing.T, childCode int, argvOut *[]string) {
	t.Helper()
	orig := opencodeLaunchRun
	t.Cleanup(func() { opencodeLaunchRun = orig })
	opencodeLaunchRun = func(_, _ io.Writer, argv, _ []string) int {
		*argvOut = append([]string(nil), argv...)
		return childCode
	}
}

// fak-test:runtime fast est=5ms
func TestOpencodeLauncherResumeFlagMapping(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts opencodeLaunchOptions
		want []string
	}{
		{
			name: "session flag maps to child argv",
			opts: opencodeLaunchOptions{sessionID: opencodeResumeTestSessionID},
			want: []string{"opencode", "--session", opencodeResumeTestSessionID},
		},
		{
			name: "continue flag passes through",
			opts: opencodeLaunchOptions{continueFlag: true},
			want: []string{"opencode", "--continue"},
		},
		{
			name: "both flags",
			opts: opencodeLaunchOptions{continueFlag: true, sessionID: opencodeResumeTestSessionID},
			want: []string{"opencode", "--continue", "--session", opencodeResumeTestSessionID},
		},
		{
			name: "run passthrough keeps resume flags right after run",
			opts: opencodeLaunchOptions{sessionID: opencodeResumeTestSessionID, passthrough: []string{"run", "status"}},
			want: []string{"opencode", "run", "--session", opencodeResumeTestSessionID, "status"},
		},
		{
			name: "no resume flags leaves argv untouched",
			opts: opencodeLaunchOptions{passthrough: []string{"run", "status"}},
			want: []string{"opencode", "run", "status"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := buildOpencodeDirectArgv(tc.opts); !slices.Equal(got, tc.want) {
				t.Fatalf("direct child argv = %q, want %q", got, tc.want)
			}
			guarded := buildOpencodeLaunchArgv("fak", tc.opts)
			sep := slices.Index(guarded, "--")
			if sep < 0 || sep+1 >= len(guarded) || guarded[sep+1] != "opencode" {
				t.Fatalf("guarded argv has no `-- opencode` child boundary: %q", guarded)
			}
			wantChild := append([]string{"opencode"}, tc.want[1:]...)
			if !slices.Equal(guarded[sep+1:], wantChild) {
				t.Fatalf("guarded child argv = %q, want %q", guarded[sep+1:], wantChild)
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestRunOpencodeSessionFlags(t *testing.T) {
	t.Run("session flag reaches the child argv", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "test-key")
		var argv []string
		stubOpencodeLaunchRun(t, 0, &argv)
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"--session", opencodeResumeTestSessionID, "--split", "off", "--quiet"})
		if code != 0 {
			t.Fatalf("runOpencode exit = %d, stderr = %q", code, stderr.String())
		}
		i := slices.Index(argv, "--session")
		if i < 0 || i+1 >= len(argv) || argv[i+1] != opencodeResumeTestSessionID {
			t.Fatalf("child argv lacks `--session %s`: %q", opencodeResumeTestSessionID, argv)
		}
	})

	t.Run("resume alias maps identically", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "test-key")
		var argv []string
		stubOpencodeLaunchRun(t, 0, &argv)
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"--resume", opencodeResumeTestSessionID, "--split", "off", "--quiet"})
		if code != 0 {
			t.Fatalf("runOpencode exit = %d, stderr = %q", code, stderr.String())
		}
		i := slices.Index(argv, "--session")
		if i < 0 || i+1 >= len(argv) || argv[i+1] != opencodeResumeTestSessionID {
			t.Fatalf("--resume did not map to `--session %s`: %q", opencodeResumeTestSessionID, argv)
		}
	})

	t.Run("continue passes through", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "test-key")
		var argv []string
		stubOpencodeLaunchRun(t, 0, &argv)
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"--continue", "--split", "off", "--quiet"})
		if code != 0 {
			t.Fatalf("runOpencode exit = %d, stderr = %q", code, stderr.String())
		}
		if !slices.Contains(argv, "--continue") {
			t.Fatalf("child argv lacks --continue: %q", argv)
		}
	})

	t.Run("invalid session id refuses before exec", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "test-key")
		ran := false
		orig := opencodeLaunchRun
		t.Cleanup(func() { opencodeLaunchRun = orig })
		opencodeLaunchRun = func(_, _ io.Writer, _, _ []string) int { ran = true; return 0 }
		var stdout, stderr bytes.Buffer
		badIDs := []string{
			"ses_short",                      // too short
			"ses_!!bad",                      // non-alphanumeric
			"not-a-session",                  // missing ses_ prefix
			"ses_" + strings.Repeat("a", 65), // too long
			"ses_e3861b794ffeKK5UOg25IFE1mR --session x", // injection attempt
		}
		for _, bad := range badIDs {
			stderr.Reset()
			code := runOpencode(&stdout, &stderr, []string{"--session", bad, "--split", "off"})
			if code != 2 {
				t.Fatalf("--session %q: exit = %d, want 2 (stderr %q)", bad, code, stderr.String())
			}
			if ran {
				t.Fatalf("--session %q: child executed despite invalid shape", bad)
			}
			if !strings.Contains(stderr.String(), "CREATE") {
				t.Fatalf("--session %q: refusal must name the create-if-missing hazard: %q", bad, stderr.String())
			}
		}
	})

	t.Run("session and resume disagree refuse", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "test-key")
		ran := false
		orig := opencodeLaunchRun
		t.Cleanup(func() { opencodeLaunchRun = orig })
		opencodeLaunchRun = func(_, _ io.Writer, _, _ []string) int { ran = true; return 0 }
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{
			"--session", opencodeResumeTestSessionID,
			"--resume", "ses_000000000000000000000000X",
			"--split", "off",
		})
		if code != 2 || ran {
			t.Fatalf("disagreeing --session/--resume: exit = %d ran = %v, want 2/no-exec", code, ran)
		}
		if !strings.Contains(stderr.String(), "disagree") {
			t.Fatalf("refusal must name the disagreement: %q", stderr.String())
		}
	})

	t.Run("trailing passthrough after -- is unchanged", func(t *testing.T) {
		t.Setenv("OPENAI_API_KEY", "test-key")
		var argv []string
		stubOpencodeLaunchRun(t, 0, &argv)
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{
			"--session", opencodeResumeTestSessionID, "--split", "off", "--quiet",
			"--", "run", "status",
		})
		if code != 0 {
			t.Fatalf("runOpencode exit = %d, stderr = %q", code, stderr.String())
		}
		want := []string{"opencode", "run", "--session", opencodeResumeTestSessionID, "status"}
		if !slices.Equal(argv, want) {
			t.Fatalf("child argv = %q, want %q", argv, want)
		}
	})

	t.Run("config subcommand unaffected", func(t *testing.T) {
		tmp := t.TempDir()
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"config", "--write", "--dir", tmp})
		if code != 0 {
			t.Fatalf("runOpencode config exit = %d, stderr = %q", code, stderr.String())
		}
		if _, err := os.Stat(filepath.Join(tmp, "opencode.json")); err != nil {
			t.Fatalf("config --write did not land opencode.json: %v", err)
		}
	})

	t.Run("free-models subcommand still dispatches", func(t *testing.T) {
		// A bogus --opencode-bin proves the call reached runOpencodeFreeModels (exec
		// failure -> exit 1) instead of being swallowed by the launcher flag parser
		// (unknown flag -> exit 2).
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"free-models", "--opencode-bin", "definitely-not-a-real-binary-xyz"})
		if code != 1 {
			t.Fatalf("runOpencode free-models exit = %d, want 1 (dispatch intact); stderr = %q", code, stderr.String())
		}
	})
}

// fak-test:runtime fast est=10ms
func TestRunOpencodeResumeSubcommand(t *testing.T) {
	const id = opencodeResumeTestSessionID

	t.Run("resumable_guarded_argv_shape", func(t *testing.T) {
		dir := t.TempDir()
		originalVerify := opencodeLauncherVerify
		t.Cleanup(func() { opencodeLauncherVerify = originalVerify })
		var seenDir string
		opencodeLauncherVerify = func(_ context.Context, opts opencoderesume.Options) (string, error) {
			if opts.SessionID != id {
				t.Fatalf("verifier saw session id %q, want %q", opts.SessionID, id)
			}
			seenDir = opts.Dir
			return opencoderesume.VerdictResumable, nil
		}
		var argv []string
		stubOpencodeLaunchRun(t, 0, &argv)
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"resume", "--session", id, "--guard", "--dir", dir})
		if code != 0 {
			t.Fatalf("runOpencode resume exit = %d, stderr = %q", code, stderr.String())
		}
		want := []string{tuiExecutable(), "guard", "--", "opencode", "run", "--session", id}
		if !slices.Equal(argv, want) {
			t.Fatalf("guarded resume argv = %q, want %q", argv, want)
		}
		if seenDir != filepath.Clean(dir) {
			t.Fatalf("verifier dir = %q, want %q", seenDir, filepath.Clean(dir))
		}
	})

	t.Run("resumable_unguarded_with_prompt", func(t *testing.T) {
		originalVerify := opencodeLauncherVerify
		t.Cleanup(func() { opencodeLauncherVerify = originalVerify })
		opencodeLauncherVerify = func(_ context.Context, _ opencoderesume.Options) (string, error) {
			return opencoderesume.VerdictResumable, nil
		}
		var argv []string
		stubOpencodeLaunchRun(t, 0, &argv)
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"resume", "--session", id, "--prompt", "keep going"})
		if code != 0 {
			t.Fatalf("runOpencode resume exit = %d, stderr = %q", code, stderr.String())
		}
		want := []string{"opencode", "run", "--session", id, "keep going"}
		if !slices.Equal(argv, want) {
			t.Fatalf("resume argv = %q, want %q", argv, want)
		}
	})

	t.Run("not_found_refuses_without_exec", func(t *testing.T) {
		originalVerify := opencodeLauncherVerify
		t.Cleanup(func() { opencodeLauncherVerify = originalVerify })
		opencodeLauncherVerify = func(_ context.Context, _ opencoderesume.Options) (string, error) {
			return opencoderesume.VerdictNotFound, nil
		}
		ran := false
		orig := opencodeLaunchRun
		t.Cleanup(func() { opencodeLaunchRun = orig })
		opencodeLaunchRun = func(_, _ io.Writer, _, _ []string) int { ran = true; return 0 }
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"resume", "--session", id})
		if code != 2 || ran {
			t.Fatalf("not-found verdict: exit = %d ran = %v, want 2/no-exec", code, ran)
		}
		msg := stderr.String()
		if !strings.Contains(msg, "not resumable in") || !strings.Contains(msg, opencoderesume.VerdictNotFound) {
			t.Fatalf("refusal must name the verdict: %q", msg)
		}
		if !strings.Contains(msg, "CREATES") {
			t.Fatalf("refusal must name the create-if-missing hazard: %q", msg)
		}
	})

	t.Run("store_unreadable_refuses_with_error", func(t *testing.T) {
		originalVerify := opencodeLauncherVerify
		t.Cleanup(func() { opencodeLauncherVerify = originalVerify })
		opencodeLauncherVerify = func(_ context.Context, _ opencoderesume.Options) (string, error) {
			return opencoderesume.VerdictStoreUnreadable, errors.New("opencode CLI missing")
		}
		ran := false
		orig := opencodeLaunchRun
		t.Cleanup(func() { opencodeLaunchRun = orig })
		opencodeLaunchRun = func(_, _ io.Writer, _, _ []string) int { ran = true; return 0 }
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"resume", "--session", id})
		if code != 2 || ran {
			t.Fatalf("store-unreadable verdict: exit = %d ran = %v, want 2/no-exec", code, ran)
		}
		if !strings.Contains(stderr.String(), opencoderesume.VerdictStoreUnreadable) {
			t.Fatalf("refusal must name the store_unreadable verdict: %q", stderr.String())
		}
	})

	t.Run("invalid_id_refused_before_verify", func(t *testing.T) {
		originalVerify := opencodeLauncherVerify
		t.Cleanup(func() { opencodeLauncherVerify = originalVerify })
		verifyCalls := 0
		opencodeLauncherVerify = func(_ context.Context, _ opencoderesume.Options) (string, error) {
			verifyCalls++
			return opencoderesume.VerdictResumable, nil
		}
		ran := false
		orig := opencodeLaunchRun
		t.Cleanup(func() { opencodeLaunchRun = orig })
		opencodeLaunchRun = func(_, _ io.Writer, _, _ []string) int { ran = true; return 0 }
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"resume", "--session", "bogus"})
		if code != 2 || verifyCalls != 0 || ran {
			t.Fatalf("invalid id: exit = %d verifyCalls = %d ran = %v, want 2/0/no-exec", code, verifyCalls, ran)
		}
		if !strings.Contains(stderr.String(), "CREATE") {
			t.Fatalf("refusal must name the create-if-missing hazard: %q", stderr.String())
		}
	})

	t.Run("missing_session_flag_refused", func(t *testing.T) {
		ran := false
		orig := opencodeLaunchRun
		t.Cleanup(func() { opencodeLaunchRun = orig })
		opencodeLaunchRun = func(_, _ io.Writer, _, _ []string) int { ran = true; return 0 }
		var stdout, stderr bytes.Buffer
		code := runOpencode(&stdout, &stderr, []string{"resume"})
		if code != 2 || ran {
			t.Fatalf("missing --session: exit = %d ran = %v, want 2/no-exec", code, ran)
		}
		if !strings.Contains(stderr.String(), "--session is required") {
			t.Fatalf("refusal must say --session is required: %q", stderr.String())
		}
	})
}

// runOpencodeWithJournalStub runs the launcher with a capturing journal seam and returns
// the exit code, captured journal events, child argv, and stderr.
func runOpencodeWithJournalStub(t *testing.T, args []string, childCode int) (int, []sessionjournal.Event, []string, string) {
	t.Helper()
	t.Setenv("OPENAI_API_KEY", "test-key")
	var mu sync.Mutex
	var events []sessionjournal.Event
	origAppend := opencodeLauncherAppendJournal
	t.Cleanup(func() { opencodeLauncherAppendJournal = origAppend })
	opencodeLauncherAppendJournal = func(_ string, ev sessionjournal.Event) error {
		mu.Lock()
		defer mu.Unlock()
		events = append(events, ev)
		return nil
	}
	var argv []string
	stubOpencodeLaunchRun(t, childCode, &argv)
	var stdout, stderr bytes.Buffer
	code := runOpencode(&stdout, &stderr, args)
	mu.Lock()
	defer mu.Unlock()
	return code, append([]sessionjournal.Event(nil), events...), argv, stderr.String()
}

// fak-test:runtime fast est=10ms
func TestRunOpencodeWritesSessionJournalRows(t *testing.T) {
	t.Run("explicit session id keys both rows", func(t *testing.T) {
		id := opencodeResumeTestSessionID
		code, events, argv, stderr := runOpencodeWithJournalStub(t, []string{"--session", id, "--split", "off", "--quiet"}, 7)
		if code != 7 {
			t.Fatalf("runOpencode exit = %d, want child code 7 (stderr %q)", code, stderr)
		}
		if len(events) != 2 {
			t.Fatalf("journal events = %d, want open+close: %+v", len(events), events)
		}
		open, closeEv := events[0], events[1]
		if open.Kind != sessionjournal.KindOpen || closeEv.Kind != sessionjournal.KindClose {
			t.Fatalf("journal kinds = (%q, %q), want (open, close)", open.Kind, closeEv.Kind)
		}
		if open.ID != id || closeEv.ID != open.ID {
			t.Fatalf("journal ids = (%q, %q), want the explicit session id on both rows", open.ID, closeEv.ID)
		}
		if open.Agent != "opencode" {
			t.Fatalf("open row agent = %q, want opencode", open.Agent)
		}
		if open.CWD == "" {
			t.Fatal("open row carries no CWD")
		}
		if !slices.Equal(open.Argv, argv) {
			t.Fatalf("open row argv = %q, want the child argv %q", open.Argv, argv)
		}
		if closeEv.Reason != "exit 7" {
			t.Fatalf("close row reason = %q, want exit 7", closeEv.Reason)
		}
	})

	t.Run("fallback launch id when no session flag", func(t *testing.T) {
		code, events, _, stderr := runOpencodeWithJournalStub(t, []string{"--split", "off", "--quiet"}, 0)
		if code != 0 {
			t.Fatalf("runOpencode exit = %d, want 0 (stderr %q)", code, stderr)
		}
		if len(events) != 2 {
			t.Fatalf("journal events = %d, want open+close: %+v", len(events), events)
		}
		open, closeEv := events[0], events[1]
		if !strings.HasPrefix(open.ID, "opencode-") {
			t.Fatalf("open row id = %q, want the fak launch id (opencode-...)", open.ID)
		}
		if closeEv.ID != open.ID {
			t.Fatalf("close row id = %q does not match open row id %q", closeEv.ID, open.ID)
		}
		if closeEv.Reason != "exit 0" {
			t.Fatalf("close row reason = %q, want exit 0", closeEv.Reason)
		}
	})
}

// fak-test:runtime fast est=10ms
func TestRunOpencodeJournalWriteFailureDoesNotFailLaunch(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")
	origAppend := opencodeLauncherAppendJournal
	t.Cleanup(func() { opencodeLauncherAppendJournal = origAppend })
	opencodeLauncherAppendJournal = func(_ string, _ sessionjournal.Event) error {
		return errors.New("journal disk full")
	}
	var argv []string
	stubOpencodeLaunchRun(t, 0, &argv)
	var stdout, stderr bytes.Buffer
	code := runOpencode(&stdout, &stderr, []string{"--split", "off"})
	if code != 0 {
		t.Fatalf("a journal write failure failed the launch: exit = %d", code)
	}
	if len(argv) == 0 {
		t.Fatal("the child never launched despite the journal failure")
	}
	if !strings.Contains(stderr.String(), "session journal") {
		t.Fatalf("journal failure must log to stderr: %q", stderr.String())
	}
}

// fak-test:runtime fast est=5ms
func TestOpencodeLaunchJournalEventShapes(t *testing.T) {
	argv := []string{"opencode", "run", "--session", opencodeResumeTestSessionID}
	open := opencodeLaunchJournalEvent(sessionjournal.KindOpen, "sid-1", "/workspace", argv, "")
	if open.Kind != sessionjournal.KindOpen || open.ID != "sid-1" || open.Agent != "opencode" {
		t.Fatalf("open row base fields wrong: %+v", open)
	}
	if open.CWD != "/workspace" || !slices.Equal(open.Argv, argv) {
		t.Fatalf("open row must carry CWD + Argv: %+v", open)
	}
	if open.Reason != "" {
		t.Fatalf("open row must not carry a Reason: %+v", open)
	}
	closeEv := opencodeLaunchJournalEvent(sessionjournal.KindClose, "sid-1", "", nil, "exit 3")
	if closeEv.Kind != sessionjournal.KindClose || closeEv.ID != "sid-1" || closeEv.Reason != "exit 3" {
		t.Fatalf("close row wrong: %+v", closeEv)
	}
	if closeEv.CWD != "" || len(closeEv.Argv) != 0 {
		t.Fatalf("close row must carry only Reason among the extra fields: %+v", closeEv)
	}
	if open.TS == "" || closeEv.TS == "" {
		t.Fatal("rows must carry RFC3339 timestamps")
	}
}

// fak-test:runtime fast est=5ms
func TestOpencodeLaunchCaptureBounded(t *testing.T) {
	c := &opencodeLaunchCapture{}
	total := 0
	for i := 0; i < 3; i++ {
		n, err := c.Write(make([]byte, opencodeLaunchCaptureLimit))
		if err != nil || n != opencodeLaunchCaptureLimit {
			t.Fatalf("write %d: n=%d err=%v", i, n, err)
		}
		total += n
	}
	if got := len(c.String()); got != opencodeLaunchCaptureLimit {
		t.Fatalf("capture retained %d bytes, want the %d-byte cap", got, opencodeLaunchCaptureLimit)
	}
	if total != 3*opencodeLaunchCaptureLimit {
		t.Fatalf("tee reported %d bytes written, want all %d (the terminal stream sees everything)", total, 3*opencodeLaunchCaptureLimit)
	}
}
