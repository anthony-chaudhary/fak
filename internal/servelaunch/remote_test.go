package servelaunch

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// fakeRunner answers remote commands from a table keyed on a substring of the
// remote command (the last argv element). Every call is recorded so tests can
// assert exactly which ssh argv was issued.
type fakeRunner struct {
	answers []fa
	calls   [][]string
}

// fa is one canned answer: the substring to match, the output, and the exit.
type fa struct {
	match string
	out   string
	code  int
	err   error
}

func (r *fakeRunner) run(_ context.Context, argv []string, _ string) (string, int, error) {
	r.calls = append(r.calls, argv)
	if len(argv) == 0 {
		return "", -1, errors.New("empty argv")
	}
	remote := argv[len(argv)-1]
	for _, a := range r.answers {
		if strings.Contains(remote, a.match) {
			return a.out, a.code, a.err
		}
	}
	return "", 0, nil
}

func newTransport(r *fakeRunner) Transport {
	return Transport{Target: SSHTarget{Host: "gpu-box-a", User: "bench"}, Run: r.run}
}

// TestTransportScreenAvailable proves the availability probe maps exit status
// to the boolean and a runner error surfaces (never a silent true).
// fak-test:runtime fast est=1ms lane=default
func TestTransportScreenAvailable(t *testing.T) {
	r := &fakeRunner{answers: []fa{{match: "which screen", out: "/usr/bin/screen", code: 0}}}
	ok, err := newTransport(r).ScreenAvailable(context.Background())
	if err != nil || !ok {
		t.Fatalf("ScreenAvailable = %v, %v; want true, nil", ok, err)
	}
	// The issued argv must be the hardened ssh argv ending in the probe.
	got := r.calls[0]
	if got[0] != "ssh" || !containsPair(got, "-o", "BatchMode=yes") {
		t.Errorf("argv = %v, want hardened ssh argv", got)
	}
	if got[len(got)-1] != ProbeScreenAvailable {
		t.Errorf("remote = %q, want %q", got[len(got)-1], ProbeScreenAvailable)
	}

	notHere := &fakeRunner{answers: []fa{{match: "which screen", out: "", code: 1}}}
	if ok, err := newTransport(notHere).ScreenAvailable(context.Background()); ok || err != nil {
		t.Errorf("ScreenAvailable = %v, %v; want false, nil", ok, err)
	}

	boom := &fakeRunner{answers: []fa{{match: "which screen", err: errors.New("no route to host")}}}
	if _, err := newTransport(boom).ScreenAvailable(context.Background()); err == nil {
		t.Errorf("ScreenAvailable err = nil, want a transport error surfaced")
	}
}

// TestTransportListSessions proves the roster parse over a real ssh round-trip,
// and that the benign "No Sockets" empty answer is an empty list, not an error.
// fak-test:runtime fast est=1ms lane=default
func TestTransportListSessions(t *testing.T) {
	r := &fakeRunner{answers: []fa{{match: "screen -ls",
		out: "There are screens on:\n\t12345.bench-s1-sglang-1109\t(Detached)\n", code: 0}}}
	got, err := newTransport(r).ListSessions(context.Background())
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if len(got) != 1 || got[0].Full != "12345.bench-s1-sglang-1109" {
		t.Errorf("sessions = %+v, want one resolved session", got)
	}

	empty := &fakeRunner{answers: []fa{{match: "screen -ls", out: "No Sockets found in /run/screen/S-root.\n", code: 1}}}
	got, err = newTransport(empty).ListSessions(context.Background())
	if err != nil || len(got) != 0 {
		t.Errorf("ListSessions(no sockets) = %+v, %v; want empty, nil", got, err)
	}

	// A genuine failure (non-zero with no "no sockets" marker) is an error.
	fail := &fakeRunner{answers: []fa{{match: "screen -ls", out: "screen: command not found", code: 127}}}
	if _, err := newTransport(fail).ListSessions(context.Background()); err == nil {
		t.Errorf("ListSessions err = nil, want a failure surfaced")
	}
}

// TestTransportFetchLogTail proves the crash-log read issues the bounded tail
// command and returns the output.
// fak-test:runtime fast est=1ms lane=default
func TestTransportFetchLogTail(t *testing.T) {
	r := &fakeRunner{answers: []fa{{match: "tail -n", out: "last line\n", code: 0}}}
	got, err := newTransport(r).FetchLogTail(context.Background(), "/tmp/s.log", 0)
	if err != nil {
		t.Fatalf("FetchLogTail: %v", err)
	}
	if got != "last line\n" {
		t.Errorf("tail = %q, want the remote output", got)
	}
	if remote := r.calls[0][len(r.calls[0])-1]; !strings.HasPrefix(remote, "tail -n 150 ") {
		t.Errorf("remote = %q, want the bounded 150-line tail", remote)
	}
}

// TestTransportLaunchSession proves the launch issues the screen command and
// resolves the session from the follow-up listing; a session absent after
// launch is a typed error, not a silent success.
// fak-test:runtime fast est=1ms lane=default
func TestTransportLaunchSession(t *testing.T) {
	r := &fakeRunner{answers: []fa{
		{match: "screen -dmS", out: "", code: 0},
		{match: "screen -ls", out: "\t999.bench-s1-sglang-1109\t(Detached)\n", code: 0},
	}}
	got, err := newTransport(r).LaunchSession(context.Background(), SessionPlan{
		Name:    "bench-s1-sglang-1109",
		LogDir:  "/tmp/bench-logs",
		LogPath: "/tmp/bench-logs/bench-s1-sglang-1109.log",
		Inner:   "python -m sglang.launch_server",
	})
	if err != nil {
		t.Fatalf("LaunchSession: %v", err)
	}
	if got.Full != "999.bench-s1-sglang-1109" {
		t.Errorf("resolved = %+v, want the full PID.name", got)
	}

	missing := &fakeRunner{answers: []fa{
		{match: "screen -dmS", out: "", code: 0},
		{match: "screen -ls", out: "No Sockets found.\n", code: 1},
	}}
	if _, err := newTransport(missing).LaunchSession(context.Background(), SessionPlan{Name: "x", LogPath: "/tmp/x.log", Inner: "run"}); err == nil {
		t.Errorf("LaunchSession err = nil, want a not-present error")
	}
}

// TestTransportNoRunner proves the fail-closed seam: a Transport with no runner
// refuses every operation with the typed sentinel rather than a silent no-op.
// fak-test:runtime fast est=1ms lane=default
func TestTransportNoRunner(t *testing.T) {
	t0 := Transport{Target: SSHTarget{Host: "h"}}
	if _, err := t0.ScreenAvailable(context.Background()); !errors.Is(err, ErrNoRunner) {
		t.Errorf("ScreenAvailable err = %v, want errors.Is(err, ErrNoRunner)", err)
	}
	if _, err := t0.ListSessions(context.Background()); !errors.Is(err, ErrNoRunner) {
		t.Errorf("ListSessions err = %v, want errors.Is(err, ErrNoRunner)", err)
	}
}

// TestInspectResolvesAndSaves proves the operator's one call: it probes, lists,
// resolves the named session, fetches its log, and returns a receipt that
// renders the reattach banner and saves as JSON.
// fak-test:runtime fast est=1ms lane=default
func TestInspectResolvesAndSaves(t *testing.T) {
	r := &fakeRunner{answers: []fa{
		{match: "which screen", out: "/usr/bin/screen", code: 0},
		{match: "screen -ls", out: "\t999.bench-s1-sglang-1109\t(Detached)\n", code: 0},
		{match: "tail -n", out: "server ready\n", code: 0},
	}}
	rc, err := Inspect(context.Background(), newTransport(r), "s1", "bench-s1-sglang-1109", 0)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if rc.ScreenSession != "999.bench-s1-sglang-1109" {
		t.Errorf("ScreenSession = %q, want the resolved name", rc.ScreenSession)
	}
	if rc.CrashLog != "server ready\n" {
		t.Errorf("CrashLog = %q, want the log tail", rc.CrashLog)
	}
	if rc.ScreenAvailable == nil || !*rc.ScreenAvailable {
		t.Errorf("ScreenAvailable = %v, want true", rc.ScreenAvailable)
	}
	banner := rc.Render()
	if !strings.Contains(banner, "attach: screen -r 999.bench-s1-sglang-1109") {
		t.Errorf("banner =\n%s\nwant the reattach line", banner)
	}
	if rc.ObservedAt == nil {
		t.Errorf("ObservedAt = nil, want an inspection timestamp")
	}
	if rc.Launched {
		t.Errorf("Launched = true, want false (Inspect observed, did not launch)")
	}
	path := t.TempDir() + "/inspect.json"
	if _, err := rc.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	back, err := LoadReceipt(path)
	if err != nil {
		t.Fatalf("LoadReceipt: %v", err)
	}
	if back.ScreenSession != rc.ScreenSession || back.CrashLog != rc.CrashLog {
		t.Errorf("saved receipt lost fields: %+v", back)
	}
}

// TestInspectNoSessionListsRoster proves a list-only inspection: with no session
// named, the receipt records the roster instead of erroring.
// fak-test:runtime fast est=1ms lane=default
func TestInspectNoSessionListsRoster(t *testing.T) {
	r := &fakeRunner{answers: []fa{
		{match: "which screen", out: "/usr/bin/screen", code: 0},
		{match: "screen -ls", out: "\t1.a\t(Detached)\n\t2.b\t(Detached)\n", code: 0},
	}}
	rc, err := Inspect(context.Background(), newTransport(r), "s1", "", 0)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if rc.ScreenSession != "1.a; 2.b" {
		t.Errorf("ScreenSession = %q, want the roster", rc.ScreenSession)
	}
}

// TestInspectRecordsProbeFailures proves a partial failure still yields a
// useful receipt: a missing session and a failed log fetch are recorded as
// errors, and the receipt is returned rather than abandoned.
// fak-test:runtime fast est=1ms lane=default
func TestInspectRecordsProbeFailures(t *testing.T) {
	r := &fakeRunner{answers: []fa{
		{match: "which screen", out: "/usr/bin/screen", code: 0},
		{match: "screen -ls", out: "\t1.other\t(Detached)\n", code: 0},
	}}
	rc, err := Inspect(context.Background(), newTransport(r), "s1", "bench-absent", 0)
	if err != nil {
		t.Fatalf("Inspect: %v", err)
	}
	if len(rc.Errors) == 0 || !strings.Contains(rc.Errors[0], "not found") {
		t.Errorf("Errors = %v, want a not-found record", rc.Errors)
	}
	// No runner is the one hard refusal.
	if _, err := Inspect(context.Background(), Transport{Target: SSHTarget{Host: "h"}}, "s1", "", 0); !errors.Is(err, ErrNoRunner) {
		t.Errorf("Inspect(no runner) err = %v, want errors.Is(err, ErrNoRunner)", err)
	}
}

// TestResolveSessionName proves the session-name derivation matches the
// launcher's shape so a receipt can reconstruct the log path and reattach
// target, and refuses a missing server.
// fak-test:runtime fast est=1ms lane=default
func TestResolveSessionName(t *testing.T) {
	got, err := ResolveSessionName("bench", "s1", "sglang", "1109-1200")
	if err != nil {
		t.Fatalf("ResolveSessionName: %v", err)
	}
	if got != "bench-s1-sglang-1109-1200" {
		t.Errorf("name = %q, want bench-s1-sglang-1109-1200", got)
	}
	// Engine and prefix default when blank.
	def, err := ResolveSessionName("", "s1", "", "t")
	if err != nil {
		t.Fatalf("ResolveSessionName: %v", err)
	}
	if def != "bench-s1-sglang-t" {
		t.Errorf("name = %q, want bench-s1-sglang-t", def)
	}
	if _, err := ResolveSessionName("bench", "  ", "sglang", "t"); !errors.Is(err, ErrScreenPlanInvalid) {
		t.Errorf("err = %v, want errors.Is(err, ErrScreenPlanInvalid)", err)
	}
}
