package servelaunch

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// TestBuildSessionLaunchShape proves the launch string matches the pinned
// shape: the log dir is created, the session is detached with a Logfile, and
// the inner command is wrapped in `bash -c '...'` carrying the exit marker and
// the keep-alive tail.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSessionLaunchShape(t *testing.T) {
	got, err := BuildSessionLaunch(SessionPlan{
		Name:      "bench-s1-sglang-1109",
		LogDir:    "/tmp/bench-logs",
		LogPath:   "/tmp/bench-logs/bench-s1-sglang-1109.log",
		Inner:     "python -m sglang.launch_server --port 30000",
		KeepAlive: true,
	})
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	for _, want := range []string{
		"mkdir -p /tmp/bench-logs",
		"screen -dmS bench-s1-sglang-1109",
		"-L -Logfile /tmp/bench-logs/bench-s1-sglang-1109.log",
		`bash -c 'python -m sglang.launch_server --port 30000; echo "[Exited $?]"; exec bash'`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("launch = %q\nmissing %q", got, want)
		}
	}
}

// TestBuildSessionLaunchKeepAliveVariant proves the keep-alive flag changes only
// the tail: the exit marker is emitted in BOTH variants (a caller that watches
// the session for completion still needs it), and only the keep-alive form adds
// `exec bash` — the wrapper that masks a crash.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSessionLaunchKeepAliveVariant(t *testing.T) {
	base := SessionPlan{Name: "s", LogPath: "/tmp/s.log", Inner: "run"}
	alive, err := BuildSessionLaunch(withKeepAlive(base, true))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	dead, err := BuildSessionLaunch(withKeepAlive(base, false))
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(alive, "; exec bash") {
		t.Errorf("keep-alive launch = %q, want exec bash tail", alive)
	}
	if strings.Contains(dead, "exec bash") {
		t.Errorf("non-keep-alive launch = %q, must NOT carry exec bash", dead)
	}
	for _, got := range []string{alive, dead} {
		if !strings.Contains(got, `echo "[Exited $?]"`) {
			t.Errorf("launch = %q, want the exit marker in both variants", got)
		}
	}
}

// TestBuildSessionLaunchQuotesInnerSingleQuotes proves the inner command's single
// quotes survive the outer `bash -c '...'` wrapper. A naive wrap turns
// `echo 'x'` into a broken command on line 2; the escaping keeps it one token.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSessionLaunchQuotesInnerSingleQuotes(t *testing.T) {
	got, err := BuildSessionLaunch(SessionPlan{
		Name:    "s",
		LogPath: "/tmp/s.log",
		Inner:   `echo 'hello world' && test -f '/a b'`,
	})
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	// The escaped form of a single quote inside a single-quoted string.
	if !strings.Contains(got, `'\''hello world'\''`) {
		t.Errorf("launch = %q, want embedded single quotes escaped as '\\''", got)
	}
}

// TestBuildSessionLaunchRefusesIncompletePlan proves the fail-closed plan check:
// a session with no name, no log path, or no inner command is a typed refusal,
// not a launch string that silently logs nowhere.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSessionLaunchRefusesIncompletePlan(t *testing.T) {
	for _, plan := range []SessionPlan{
		{LogPath: "/tmp/s.log", Inner: "run"},
		{Name: "s", Inner: "run"},
		{Name: "s", LogPath: "/tmp/s.log"},
	} {
		if _, err := BuildSessionLaunch(plan); !errors.Is(err, ErrScreenPlanInvalid) {
			t.Errorf("plan %+v: err = %v, want errors.Is(err, ErrScreenPlanInvalid)", plan, err)
		}
	}
}

// TestParseScreenLS proves the roster parse: real session lines become
// PID/name/full entries, and headers, blanks, and the "No Sockets" message are
// skipped rather than guessed at.
// fak-test:runtime fast est=1ms lane=default
func TestParseScreenLS(t *testing.T) {
	out := strings.Join([]string{
		"There are screens on:",
		"\t12345.bench-s1-sglang-1109\t(Detached)",
		"\t67890.bench-s1-hwmon-1109\t(Attached)",
		"2 Sockets in /run/screen/S-root.",
	}, "\n")
	got := ParseScreenLS(out)
	want := []ScreenSession{
		{PID: 12345, Name: "bench-s1-sglang-1109", Full: "12345.bench-s1-sglang-1109", State: "Detached"},
		{PID: 67890, Name: "bench-s1-hwmon-1109", Full: "67890.bench-s1-hwmon-1109", State: "Attached"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("ParseScreenLS =\n%+v\nwant\n%+v", got, want)
	}
	if s := ParseScreenLS("No Sockets found in /run/screen/S-root.\n"); len(s) != 0 {
		t.Errorf("ParseScreenLS(no sockets) = %+v, want empty", s)
	}
}

// TestFindSessionExactMatchNotSubstring proves the collision guard: a lookup for
// `bench` must NOT resolve when the host only has `bench-hwmon`. A naive
// substring grep reports a false alive here, which is the exact failure that
// would have a caller kill or wait on the wrong session.
// fak-test:runtime fast est=1ms lane=default
func TestFindSessionExactMatchNotSubstring(t *testing.T) {
	out := "\t67890.bench-hwmon-1109\t(Detached)\n"
	if _, ok := FindSession(out, "bench"); ok {
		t.Errorf("FindSession(out, %q) = found, want not-found (exact match only)", "bench")
	}
	if _, ok := FindSession(out, "bench-hwmon-1109"); !ok {
		t.Errorf("FindSession exact name not resolved")
	}
	// The full identifier is what a reattach uses.
	got, ok := FindSession(out, "bench-hwmon-1109")
	if !ok || got.Full != "67890.bench-hwmon-1109" {
		t.Errorf("Full = %q ok=%v, want 67890.bench-hwmon-1109", got.Full, ok)
	}
}

// TestScanExitMarker proves the crash witness the keep-alive forces: the exit
// marker yields the status code, an `error: argument` line is flagged as an
// argparse crash, and the hint keeps the last suspicious lines.
// fak-test:runtime fast est=1ms lane=default
func TestScanExitMarker(t *testing.T) {
	ok := ScanExitMarker([]string{"loading weights", "server up", "some line", "x", "y", "[Exited 0]"})
	if !ok.Exited || ok.Code != 0 {
		t.Errorf("obs = %+v, want Exited code 0", ok)
	}
	bad := ScanExitMarker([]string{"usage: launcher", "launcher: error: argument --port: invalid int value: 'x'", "[Exited 2]"})
	if !bad.Exited || bad.Code != 2 {
		t.Errorf("obs = %+v, want Exited code 2", bad)
	}
	if !bad.ArgparseCrash {
		t.Errorf("obs = %+v, want ArgparseCrash", bad)
	}
	if len(bad.Hint) == 0 || !strings.Contains(bad.Hint[len(bad.Hint)-1], "[Exited 2]") {
		t.Errorf("Hint = %v, want the exit marker as the last hint", bad.Hint)
	}
	// No marker, no exit: the session may still be up (keep-alive), but the
	// server's fate is unproven.
	none := ScanExitMarker([]string{"loading", "still loading"})
	if none.Exited {
		t.Errorf("obs = %+v, want Exited=false with no marker", none)
	}
}

// TestBuildCrashLogTailAndLogTail proves the log-reader commands: the crash tail
// is bounded and shell-quotes the path, the follow tail uses -F, and an empty
// path is a typed refusal.
// fak-test:runtime fast est=1ms lane=default
func TestBuildCrashLogTailAndLogTail(t *testing.T) {
	got, err := BuildCrashLogTail("/tmp/bench-logs/s1.log", 0)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if got != "tail -n 150 /tmp/bench-logs/s1.log 2>/dev/null" {
		t.Errorf("crash tail = %q, want the 150-line default", got)
	}
	n, err := BuildCrashLogTail("/tmp/a b/c.log", 25)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if n != "tail -n 25 '/tmp/a b/c.log' 2>/dev/null" {
		t.Errorf("crash tail = %q, want a shell-quoted path", n)
	}
	follow, err := BuildLogTail("/tmp/s.log")
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if follow != "tail -F /tmp/s.log 2>/dev/null" {
		t.Errorf("log tail = %q, want -F", follow)
	}
	if _, err := BuildCrashLogTail("  ", 0); !errors.Is(err, ErrScreenPlanInvalid) {
		t.Errorf("err = %v, want errors.Is(err, ErrScreenPlanInvalid)", err)
	}
}

// TestBuildHwMonitorLaunch proves the hw-monitor loop is a screen session whose
// log carries a timestamped telemetry block at the configured cadence.
// fak-test:runtime fast est=1ms lane=default
func TestBuildHwMonitorLaunch(t *testing.T) {
	got, err := BuildHwMonitorLaunch(SessionPlan{
		Name:    "bench-s1-hwmon",
		LogDir:  "/tmp/bench-logs",
		LogPath: "/tmp/bench-logs/bench-s1-hwmon.log",
	}, 7)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	for _, want := range []string{"screen -dmS bench-s1-hwmon", "nvidia-smi --query-gpu", "free -h", "iostat", "sleep 7"} {
		if !strings.Contains(got, want) {
			t.Errorf("hwmon launch = %q\nmissing %q", got, want)
		}
	}
}

// TestBuildSessionAliveProbeAndKill proves the exact-match probe and kill
// commands fill from a name and refuse an empty one.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSessionAliveProbeAndKill(t *testing.T) {
	probe, err := BuildSessionAliveProbe("bench-s1")
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if probe != "screen -ls | grep -qF bench-s1" {
		t.Errorf("probe = %q", probe)
	}
	kill, err := BuildSessionKill("bench-s1")
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if kill != "screen -XS bench-s1 quit 2>/dev/null" {
		t.Errorf("kill = %q", kill)
	}
	if _, err := BuildSessionAliveProbe(" "); !errors.Is(err, ErrScreenPlanInvalid) {
		t.Errorf("err = %v, want errors.Is(err, ErrScreenPlanInvalid)", err)
	}
}

// TestShellQuote proves the quoting helper: safe strings pass through, unsafe
// strings are single-quoted, and an embedded single quote uses the '\” form.
// fak-test:runtime fast est=1ms lane=default
func TestShellQuote(t *testing.T) {
	cases := map[string]string{
		"/tmp/bench-logs/s1.log": "/tmp/bench-logs/s1.log",
		"/tmp/a b/c.log":         "'/tmp/a b/c.log'",
		"it's":                   `'it'\''s'`,
		"":                       "''",
		"a; rm -rf /":            "'a; rm -rf /'",
	}
	for in, want := range cases {
		if got := ShellQuote(in); got != want {
			t.Errorf("ShellQuote(%q) = %q, want %q", in, got, want)
		}
	}
}

func withKeepAlive(p SessionPlan, alive bool) SessionPlan {
	p.KeepAlive = alive
	return p
}
