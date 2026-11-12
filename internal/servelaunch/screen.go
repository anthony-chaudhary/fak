package servelaunch

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// ErrScreenPlanInvalid reports a screen plan or probe request that cannot be
// built (a missing name, log path, or inner command).
var ErrScreenPlanInvalid = errors.New("servelaunch: invalid screen plan")

// Screen sessions are how the pinned study's launcher kept a long-running
// server alive across the ssh disconnect that would otherwise kill it, and how
// it kept a readable log of everything the server printed. A detached GNU
// screen owns the server process on the remote host; ssh can come and go
// freely; the operator can `screen -r` in later and look at a live terminal.
//
// This file is the pure half of that: the command builders and the output
// parsers. Nothing here execs anything — the caller runs the returned strings
// over whatever transport it owns.
//
// The two things worth knowing before reading the builders:
//
//   - The session deliberately wraps the server in `exec bash` (keep-alive) so
//     the screen stays up for inspection after the server exits. That is
//     exactly why a crashed server is NOT visible as "session gone": the
//     session is still there, the server is not. The pinned transport learned
//     this the hard way, and the fix is the `[Exited $?]` marker the builder
//     appends and ScanExitMarker looks for.
//   - The session name travels through an ssh command string and then a remote
//     shell, so the inner command is single-quote escaped for the outer
//     `bash -c '...'`; getting that wrong is the difference between a launch
//     and a syntax error on line 2.

const (
	// DefaultCrashLogLines bounds the crash-log tail this builder asks for: the
	// last 150 lines, the pinned transport's number. A crash's cause is almost
	// always in the last screenful; a whole-file dump is noise.
	DefaultCrashLogLines = 150

	// ExitMarkerPrefix and ExitMarkerSuffix bracket the status code a finished
	// inner command leaves behind: the literal `[Exited 0]` shape. The
	// keep-alive wrapper means the session outlives the server, so this marker
	// — not the session's existence — is how a caller proves the inner command
	// actually returned.
	ExitMarkerPrefix = "[Exited "
	ExitMarkerSuffix = "]"
)

// SessionPlan names one screen session launch: the session name, the remote
// log directory and file, and the inner command the session runs. Builders
// turn it into a remote shell string; parsers read the host's answer back.
type SessionPlan struct {
	// Name is the screen session name (screen -S / -dmS).
	Name string
	// LogDir is the remote directory that will hold the screen log.
	LogDir string
	// LogPath is the remote screen log file (screen -Logfile).
	LogPath string
	// Inner is the command the session runs. Trailing whitespace is stripped
	// before wrapping so a YAML-folded trailing newline cannot push the exit
	// marker onto a second line and break the `bash -c` quoting.
	Inner string
	// KeepAlive, when true, appends `; exec bash` so the session survives the
	// inner command's exit for operator inspection. When false the session ends
	// with the command, which lets a caller wait on `screen -ls` for completion.
	KeepAlive bool
}

// ShellQuote single-quotes a string for a POSIX shell, escaping embedded single
// quotes in the standard `'\”` form. Safe strings pass through unquoted so the
// emitted command stays readable.
func ShellQuote(s string) string {
	if s == "" {
		return "''"
	}
	const unsafe = "'\"\\ `$;&|<>(){}[]!*?~#\t\r\n"
	if !strings.ContainsAny(s, unsafe) {
		return s
	}
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

// BuildSessionLaunch builds the remote command that (re)creates a detached
// screen session running plan.Inner, logging to plan.LogPath, with the exit
// marker appended. Shape:
//
//	mkdir -p LOGDIR && screen -dmS NAME -L -Logfile LOGPATH \
//	  bash -c 'INNER; echo "[Exited $?]"<keep-alive>'
//
// A zero plan.LogDir is omitted from the mkdir (the caller logs into an
// existing dir). An empty Name or Inner is a typed refusal via
// SessionPlan.Validate.
func BuildSessionLaunch(plan SessionPlan) (string, error) {
	if err := plan.Validate(); err != nil {
		return "", err
	}
	inner := strings.TrimRight(plan.Inner, " \t\r\n")
	// Inner runs inside `bash -c '...'`, so its single quotes need the '\''
	// treatment; double quotes inside are already safe there.
	escaped := strings.ReplaceAll(inner, "'", `'\''`)
	tail := ""
	if plan.KeepAlive {
		tail = "; exec bash"
	}
	wrapped := fmt.Sprintf(`bash -c '%s; echo "%s$?%s"%s'`,
		escaped, ExitMarkerPrefix, ExitMarkerSuffix, tail)

	fields := []string{}
	if dir := strings.TrimSpace(plan.LogDir); dir != "" {
		fields = append(fields, "mkdir -p "+ShellQuote(dir), "&&")
	}
	fields = append(fields,
		"screen", "-dmS", ShellQuote(plan.Name),
		"-L", "-Logfile", ShellQuote(plan.LogPath),
		wrapped)
	return strings.Join(fields, " "), nil
}

// BuildHwMonitorLaunch builds the looping hardware-monitor screen session the
// pinned transport paired with every server launch: a timestamped block of
// nvidia-smi, cpu, memory, and disk-io every intervalSec, so a later look at
// the log shows what the box was doing while the server ran. A non-positive
// interval takes DefaultHwMonitorInterval.
func BuildHwMonitorLaunch(plan SessionPlan, intervalSec int) (string, error) {
	if intervalSec <= 0 {
		intervalSec = DefaultHwMonitorInterval
	}
	script := fmt.Sprintf(
		`while true; do `+
			`echo '=== '$(date '+%%Y-%%m-%%d %%H:%%M:%%S')' ==='; `+
			`nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || echo 'nvidia-smi unavailable'; `+
			`echo '--- CPU ---'; top -bn1 | head -5; `+
			`echo '--- Memory ---'; free -h; `+
			`echo '--- Disk I/O ---'; iostat -x 1 1 2>/dev/null | tail -n +4 || echo 'iostat unavailable'; `+
			`echo; sleep %d; `+
			`done`, intervalSec)
	plan.Inner = script
	return BuildSessionLaunch(plan)
}

// DefaultHwMonitorInterval is the hw-monitor loop cadence in seconds when the
// caller names none (the pinned transport's 5s).
const DefaultHwMonitorInterval = 5

// Validate refuses a plan that cannot produce a launch: a session needs a name
// and an inner command. LogPath is required too — a session with no log file
// is a session no one can debug after the fact, which defeats the purpose.
func (p SessionPlan) Validate() error {
	if strings.TrimSpace(p.Name) == "" {
		return fmt.Errorf("%w: session name is required", ErrScreenPlanInvalid)
	}
	if strings.TrimSpace(p.LogPath) == "" {
		return fmt.Errorf("%w: log path is required", ErrScreenPlanInvalid)
	}
	if strings.TrimSpace(p.Inner) == "" {
		return fmt.Errorf("%w: inner command is required", ErrScreenPlanInvalid)
	}
	return nil
}

// ScreenProbeCommands are the pure command strings a caller runs to observe the
// host's screen state. They are constants, not builders, because their shape is
// fixed and identical to the pinned transport's probes.
const (
	// ProbeScreenAvailable is the availability check (exit 0 iff screen is on
	// PATH on the remote host).
	ProbeScreenAvailable = "which screen"
	// ProbeScreenList lists the host's sessions, one per line, as `PID.name`.
	ProbeScreenList = "screen -ls"
	// ProbeSessionAlive lists sessions and greps for an exact name match; the
	// exit status is the answer (0 = alive). -F keeps the name out of regex
	// metacharacter territory.
	ProbeSessionAlive = "screen -ls | grep -qF %s"
	// CommandSessionKill asks a named session to quit; best-effort.
	CommandSessionKill = "screen -XS %s quit 2>/dev/null"
	// CommandScreenAttachPrefix is the operator reattach instruction printed by
	// the receipt banner.
	CommandScreenAttachPrefix = "screen -r "
)

// ScreenSession is one `PID.name` entry from a `screen -ls` listing, resolved.
type ScreenSession struct {
	// PID is the screen server's pid.
	PID int
	// Name is the session name (the part after the first dot).
	Name string
	// Full is the canonical `PID.name` identifier a reattach uses.
	Full string
	// State is the trailing state word (`Detached`, `Attached`, ...), if any.
	State string
}

// ParseScreenLS parses `screen -ls` output and returns every session it can
// identify. Lines that are not session entries (headers, the "No Sockets"
// message, blank lines) are skipped, never guessed at. An exact-name lookup is
// FindSession's job; this returns the whole roster.
func ParseScreenLS(output string) []ScreenSession {
	var out []ScreenSession
	for _, raw := range strings.Split(output, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || !strings.Contains(line, ".") {
			continue
		}
		// A session line looks like: `12345.bench-name\t(Detached)`.
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		id := fields[0]
		pidStr, name, ok := strings.Cut(id, ".")
		if !ok || name == "" {
			continue
		}
		pid, err := strconv.Atoi(pidStr)
		if err != nil {
			continue
		}
		state := ""
		if len(fields) > 1 {
			state = strings.Trim(fields[1], "()")
		}
		out = append(out, ScreenSession{PID: pid, Name: name, Full: id, State: state})
	}
	return out
}

// FindSession resolves a short session name to its full `PID.name` entry from
// `screen -ls` output. Matching is EXACT on the name after the first dot, never
// a substring: a session named `bench` must not be reported as present when the
// host only has `bench-hwmon`, which a naive `grep bench` would happily match.
func FindSession(output, name string) (ScreenSession, bool) {
	for _, s := range ParseScreenLS(output) {
		if s.Name == name {
			return s, true
		}
	}
	return ScreenSession{}, false
}

// ExitObservation is what ScanExitMarker found in a session's captured output.
type ExitObservation struct {
	// Exited is true when an exit marker line was seen — the inner command
	// returned, whatever the code.
	Exited bool
	// Code is the exit status parsed from the marker (valid only when Exited).
	Code int
	// ArgparseCrash is true when an `error: argument` line appears, the
	// signature of a server that rejected its own argv and died before it could
	// even reach the exit marker.
	ArgparseCrash bool
	// Hint carries the up-to-5 last suspicious lines (an error, a traceback, or
	// the marker), the context an operator wants in a crash banner.
	Hint []string
}

// ScanExitMarker scans captured session output for the exit marker the launch
// wrapper appends and for the argparse-crash signature. This is the crash
// witness that the keep-alive wrapper forces on a caller: the session staying
// up proves nothing about the server, so the marker (or a missing one) does.
func ScanExitMarker(lines []string) ExitObservation {
	var obs ExitObservation
	var hint []string
	for _, l := range lines {
		if idx := strings.Index(l, ExitMarkerPrefix); idx >= 0 {
			rest := l[idx+len(ExitMarkerPrefix):]
			if end := strings.Index(rest, ExitMarkerSuffix); end >= 0 {
				if code, err := strconv.Atoi(strings.TrimSpace(rest[:end])); err == nil {
					obs.Exited = true
					obs.Code = code
				}
			}
		}
		lower := strings.ToLower(l)
		if strings.Contains(lower, "error: argument") {
			obs.ArgparseCrash = true
		}
		if obs.Exited || strings.Contains(lower, "error") || strings.Contains(l, "Traceback") {
			hint = append(hint, l)
		}
	}
	if n := len(hint); n > 5 {
		hint = hint[n-5:]
	}
	obs.Hint = hint
	return obs
}

// BuildCrashLogTail builds the command that fetches the last n lines of a
// remote screen log: `tail -n N PATH 2>/dev/null`. n <= 0 takes
// DefaultCrashLogLines. This is the reader the pinned transport used precisely
// because the in-memory tail cannot be trusted to have caught a crash that
// happened in the first second: read the file on disk instead of the stream.
func BuildCrashLogTail(logPath string, n int) (string, error) {
	if strings.TrimSpace(logPath) == "" {
		return "", fmt.Errorf("%w: log path is required", ErrScreenPlanInvalid)
	}
	if n <= 0 {
		n = DefaultCrashLogLines
	}
	return fmt.Sprintf("tail -n %d %s 2>/dev/null", n, ShellQuote(logPath)), nil
}

// BuildLogTail builds the follow-command a caller uses to stream a live screen
// log: `tail -F PATH 2>/dev/null`. The pinned transport opened this as a
// long-lived ssh channel so output streamed while the server ran.
func BuildLogTail(logPath string) (string, error) {
	if strings.TrimSpace(logPath) == "" {
		return "", fmt.Errorf("%w: log path is required", ErrScreenPlanInvalid)
	}
	return fmt.Sprintf("tail -F %s 2>/dev/null", ShellQuote(logPath)), nil
}

// BuildSessionAliveProbe fills ProbeSessionAlive with an exact-match name.
func BuildSessionAliveProbe(name string) (string, error) {
	if strings.TrimSpace(name) == "" {
		return "", fmt.Errorf("%w: session name is required", ErrScreenPlanInvalid)
	}
	return fmt.Sprintf(ProbeSessionAlive, ShellQuote(name)), nil
}

// BuildSessionKill fills CommandSessionKill with an exact-match name.
func BuildSessionKill(name string) (string, error) {
	if strings.TrimSpace(name) == "" {
		return "", fmt.Errorf("%w: session name is required", ErrScreenPlanInvalid)
	}
	return fmt.Sprintf(CommandSessionKill, ShellQuote(name)), nil
}
