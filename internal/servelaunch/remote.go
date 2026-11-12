package servelaunch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Remote inspection is the use case this whole package exists for: the machine
// that runs the server is not the machine you are sitting at, and when the
// server misbehaves you need to get *into* it — check it is even running, find
// its screen session, read its log, and keep a record of what you saw and how
// to get back in. That is an ssh round-trip, and this file is where the built
// commands become real.
//
// Everything here runs through an injected ExecRunner, so a caller (and its
// tests) supply the process seam. The production runner shells out to ssh with
// the hardened argv BuildSSHArgv produces; a test runner answers from a table.
// Nothing in this file spawns a process on its own.

// ExecRunner runs one local command and returns its combined output and exit
// code. It is the single process seam for remote inspection; a nil runner in a
// Transport is a typed refusal, never a silent no-op.
type ExecRunner func(ctx context.Context, argv []string, stdin string) (stdout string, exit int, err error)

// ExecCommandRunner is the production runner: it runs argv as a local process
// (for ssh, that means running the ssh client), returning combined output.
func ExecCommandRunner(ctx context.Context, argv []string, stdin string) (string, int, error) {
	if len(argv) == 0 {
		return "", -1, errors.New("servelaunch: empty argv")
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	out, err := cmd.CombinedOutput()
	code := 0
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			code = exitErr.ExitCode()
		} else {
			return string(out), -1, err
		}
	}
	return string(out), code, nil
}

// DefaultExecTimeout bounds one remote command when the caller names none: an
// ssh probe that has not answered in this long has told you something already.
const DefaultExecTimeout = 30 * time.Second

// Transport runs the built remote commands over one ssh target. Timeout bounds
// each command; Run is the injectable process seam.
type Transport struct {
	Target  SSHTarget
	Run     ExecRunner
	Timeout time.Duration
}

// ErrNoRunner reports a Transport with no process seam.
var ErrNoRunner = errors.New("servelaunch: transport has no ExecRunner wired")

// exec builds the ssh argv for this target and runs `remote` as the command,
// returning stdout and the exit code. It applies Timeout (default when zero).
func (t Transport) exec(ctx context.Context, remote string) (string, int, error) {
	if t.Run == nil {
		return "", -1, ErrNoRunner
	}
	argv, err := BuildSSHArgv(t.Target)
	if err != nil {
		return "", -1, err
	}
	argv = append(argv, remote)
	timeout := t.Timeout
	if timeout <= 0 {
		timeout = DefaultExecTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return t.Run(ctx, argv, "")
}

// SSHTarget exposes the connection target so a caller can build sibling argv.
func (t Transport) SSHTarget() SSHTarget { return t.Target }

// ScreenAvailable reports whether GNU screen is on the host's PATH. A non-zero
// exit (or a runner error) is reported as false with the error, never a silent
// true: a caller that cannot prove screen exists must not assume it does.
func (t Transport) ScreenAvailable(ctx context.Context) (bool, error) {
	_, code, err := t.exec(ctx, ProbeScreenAvailable)
	if err != nil {
		return false, err
	}
	return code == 0, nil
}

// ListSessions returns the host's screen sessions parsed from `screen -ls`. A
// non-zero exit with the "No Sockets" shape is an empty roster, not an error;
// a transport failure is an error.
func (t Transport) ListSessions(ctx context.Context) ([]ScreenSession, error) {
	out, code, err := t.exec(ctx, ProbeScreenList)
	if err != nil {
		return nil, err
	}
	sessions := ParseScreenLS(out)
	if len(sessions) == 0 && code != 0 && !looksLikeNoSessions(out) {
		return nil, fmt.Errorf("servelaunch: screen -ls exited %d: %s", code, strings.TrimSpace(out))
	}
	return sessions, nil
}

// looksLikeNoSessions reports the benign empty-roster output `screen` prints
// when it has no sessions, so an empty list is not mistaken for a failure.
func looksLikeNoSessions(out string) bool {
	s := strings.ToLower(out)
	return strings.Contains(s, "no sockets") || strings.TrimSpace(out) == ""
}

// FetchLogTail fetches the last n lines of a remote screen log. It is the
// crash-time read: the file on disk, not the streaming tail.
func (t Transport) FetchLogTail(ctx context.Context, logPath string, n int) (string, error) {
	cmd, err := BuildCrashLogTail(logPath, n)
	if err != nil {
		return "", err
	}
	out, code, err := t.exec(ctx, cmd)
	if err != nil {
		return "", err
	}
	if code != 0 {
		return "", fmt.Errorf("servelaunch: crash-log fetch exited %d for %s", code, logPath)
	}
	return out, nil
}

// LaunchSession launches plan inside a detached screen session on the host and
// returns the resolved `PID.name` once the session is observed. It is the one
// command here with a side effect: it starts the server. The keep-alive wrapper
// means success is "the session exists", so this verifies by listing, not by
// waiting on the inner command.
func (t Transport) LaunchSession(ctx context.Context, plan SessionPlan) (ScreenSession, error) {
	cmd, err := BuildSessionLaunch(plan)
	if err != nil {
		return ScreenSession{}, err
	}
	out, code, err := t.exec(ctx, cmd)
	if err != nil {
		return ScreenSession{}, err
	}
	if code != 0 {
		return ScreenSession{}, fmt.Errorf("servelaunch: screen launch exited %d: %s", code, strings.TrimSpace(out))
	}
	sessions, err := t.ListSessions(ctx)
	if err != nil {
		return ScreenSession{}, err
	}
	full, ok := findFull(sessions, plan.Name)
	if !ok {
		return ScreenSession{}, fmt.Errorf("servelaunch: session %q not present after launch", plan.Name)
	}
	for _, s := range sessions {
		if s.Full == full {
			return s, nil
		}
	}
	return ScreenSession{Name: plan.Name, Full: full}, nil
}

// KillSession asks a named session to quit (best-effort teardown).
func (t Transport) KillSession(ctx context.Context, name string) error {
	cmd, err := BuildSessionKill(name)
	if err != nil {
		return err
	}
	_, _, err = t.exec(ctx, cmd)
	return err
}

// LaunchSessionHwMonitor launches the paired hardware-monitor session in plan,
// verifying it by listing exactly like LaunchSession. intervalSec <= 0 takes
// DefaultHwMonitorInterval.
func (t Transport) LaunchSessionHwMonitor(ctx context.Context, plan SessionPlan, intervalSec int) (ScreenSession, error) {
	cmd, err := BuildHwMonitorLaunch(plan, intervalSec)
	if err != nil {
		return ScreenSession{}, err
	}
	out, code, err := t.exec(ctx, cmd)
	if err != nil {
		return ScreenSession{}, err
	}
	if code != 0 {
		return ScreenSession{}, fmt.Errorf("servelaunch: hw-monitor launch exited %d: %s", code, strings.TrimSpace(out))
	}
	sessions, err := t.ListSessions(ctx)
	if err != nil {
		return ScreenSession{}, err
	}
	full, ok := findFull(sessions, plan.Name)
	if !ok {
		return ScreenSession{}, fmt.Errorf("servelaunch: hw-monitor session %q not present after launch", plan.Name)
	}
	for _, s := range sessions {
		if s.Full == full {
			return s, nil
		}
	}
	return ScreenSession{Name: plan.Name, Full: full}, nil
}

// Inspect is the operator's single call: it probes the host, resolves the named
// session (or lists what is there), fetches that session's log tail, and
// returns a Receipt capturing exactly what it saw — the artifact a human saves
// and later reads instead of retyping a session name out of scrollback.
//
// A probe failure never aborts the whole inspection: each step records what it
// learned into the receipt and continues, so a host that answers `screen -ls`
// but refuses the log read still yields a useful receipt. Only a refusal to
// even build the transport (no runner) short-circuits.
func Inspect(ctx context.Context, t Transport, serverName, sessionName string, logTailLines int) (*Receipt, error) {
	if t.Run == nil {
		return nil, ErrNoRunner
	}
	at := time.Now()
	rc := &Receipt{
		ServerName: serverName,
		SSHHost:    t.Target.Host,
	}

	if avail, err := t.ScreenAvailable(ctx); err != nil {
		rc.AddError(fmt.Sprintf("screen availability probe failed: %v", err))
	} else {
		rc.ScreenAvailable = &avail
	}

	sessions, err := t.ListSessions(ctx)
	if err != nil {
		rc.AddError(fmt.Sprintf("screen -ls failed: %v", err))
	}

	if sessionName == "" {
		// No session named: report the roster in the receipt's screen field so
		// the operator sees what is running.
		if len(sessions) > 0 {
			rc.ScreenSession = renderRoster(sessions)
		}
		return rc, nil
	}

	full, ok := findFull(sessions, sessionName)
	if !ok {
		rc.AddError(fmt.Sprintf("session %q not found on host", sessionName))
		return rc, nil
	}
	rc.ScreenSession = full
	// The log path is derived from the session name the same way the launcher
	// built it: <logdir>/<session>.log, with logdir defaulting to the launcher's.
	logPath := DefaultScreenLogDir + "/" + full + ".log"
	rc.ScreenLogPath = logPath
	if tail, err := t.FetchLogTail(ctx, logPath, logTailLines); err != nil {
		rc.AddError(fmt.Sprintf("log tail fetch failed: %v", err))
	} else {
		rc.CrashLog = tail
	}
	rc.ObservedAt = &at
	return rc, nil
}

// DefaultScreenLogDir is the remote directory the launcher writes screen logs
// to when a caller names none (the pinned transport's /tmp/bench-logs).
const DefaultScreenLogDir = "/tmp/bench-logs"

// findFull resolves a short name to its full `PID.name` from a parsed roster,
// exact-match on the name after the first dot.
func findFull(sessions []ScreenSession, name string) (string, bool) {
	for _, s := range sessions {
		if s.Name == name {
			if s.Full != "" {
				return s.Full, true
			}
			return strconv.Itoa(s.PID) + "." + s.Name, true
		}
	}
	return "", false
}

// renderRoster renders a session roster as one indented line per session, the
// text form the receipt's screen field carries when Inspect lists rather than
// resolves.
func renderRoster(sessions []ScreenSession) string {
	var b strings.Builder
	for i, s := range sessions {
		if i > 0 {
			b.WriteString("; ")
		}
		full := s.Full
		if full == "" {
			full = strconv.Itoa(s.PID) + "." + s.Name
		}
		b.WriteString(full)
	}
	return b.String()
}

// ResolveSessionName derives the launcher's session name for a server, engine,
// and timestamp: `<prefix>-<server>-<engine>-<ts>`, the exact shape the pinned
// transport built so a log path and a reattach target can be reconstructed from
// the receipt without a live probe.
func ResolveSessionName(prefix, server, engine, ts string) (string, error) {
	if strings.TrimSpace(server) == "" {
		return "", fmt.Errorf("%w: server name is required", ErrScreenPlanInvalid)
	}
	if strings.TrimSpace(engine) == "" {
		engine = "sglang"
	}
	if strings.TrimSpace(prefix) == "" {
		prefix = "bench"
	}
	if strings.TrimSpace(ts) == "" {
		ts = time.Now().Format("0102-1504")
	}
	return fmt.Sprintf("%s-%s-%s-%s", prefix, server, engine, ts), nil
}

// SaveReceipt is a convenience for the CLI: render and save in one call.
func SaveReceipt(r *Receipt, path string) ([]byte, error) { return r.Save(path) }

// WriteBanner writes the receipt's operator banner to w (usually stdout).
func WriteBanner(w io.Writer, r *Receipt) error {
	_, err := io.WriteString(w, r.Render())
	return err
}
