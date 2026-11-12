package servelaunch

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// A launch receipt is the record that makes a remote launch debuggable after
// the fact. The operator who asks "what happened to that server, and how do I
// look at it now?" gets three things from one artifact: the exact command that
// ran, the screen session and log path that are still on the host, and the
// crash log if it died. The pinned study recorded exactly this shape as a
// serializable launch event and printed a banner with the reattach line; this
// file is the distilled, testable port of that.
//
// Two write paths matter, and they answer two different questions:
//
//   - Render is what a human reads at launch time: the session name, the log
//     path, and the literal `screen -r <name>` they type to get back in.
//   - Save is what survives the session: the whole receipt as JSON, so a later
//     tool can read the screen names and log paths without a human retyping
//     them from scrollback.

// Receipt is the serializable record of one launch attempt and its lifecycle.
// Its JSON shape is the artifact; every field is written so a reader six months
// on knows what the launch was and where its debris lives. Timestamps are
// RFC3339 strings (not epoch floats) so the file is legible without a decoder.
type Receipt struct {
	// ServerName identifies the launched unit within a fleet.
	ServerName string `json:"server_name"`
	// SSHHost is the host the launch connected to.
	SSHHost string `json:"ssh_host"`
	// LaunchCommand is the exact remote command, post-auto-selection, so the
	// banner cannot lie about what ran.
	LaunchCommand string `json:"launch_command"`
	// Env carries the launch-relevant environment.
	Env map[string]string `json:"env,omitempty"`

	// ScreenSession is the resolved `PID.name` (or the short name before
	// resolution); the reattach target.
	ScreenSession string `json:"screen_session,omitempty"`
	// ScreenLogPath is the remote screen log the server writes to.
	ScreenLogPath string `json:"screen_log_path,omitempty"`
	// ScreenAvailable is tri-state: nil = never probed, else the probe answer.
	// A pointer is deliberate — "we never asked" is different from "no".
	ScreenAvailable *bool `json:"screen_available,omitempty"`

	// HwScreenSession / HwScreenLogPath are the paired hardware-monitor session.
	HwScreenSession string `json:"hw_screen_session,omitempty"`
	HwScreenLogPath string `json:"hw_screen_log_path,omitempty"`

	// AutoSelectedPort and AutoSelectedGPUs record the fail-closed picks.
	AutoSelectedPort int   `json:"auto_selected_port,omitempty"`
	AutoSelectedGPUs []int `json:"auto_selected_gpus,omitempty"`

	// LaunchedAt / ReadyAt / StoppedAt bound the lifecycle. Pointers so an
	// unreached phase is omitted from the JSON rather than serializing Go's
	// zero time as year 1.
	LaunchedAt *time.Time `json:"launched_at,omitempty"`
	ReadyAt    *time.Time `json:"ready_at,omitempty"`
	StoppedAt  *time.Time `json:"stopped_at,omitempty"`
	// ObservedAt records when an inspection (as opposed to a launch) looked at
	// the host; it is how a read-only receipt timestamps itself without
	// claiming it launched anything.
	ObservedAt *time.Time `json:"observed_at,omitempty"`
	// LaunchDurationSec is ReadyAt-LaunchedAt, the number an operator compares.
	LaunchDurationSec float64 `json:"launch_duration_sec,omitempty"`

	// Launched / Ready / Stopped are the lifecycle booleans.
	Launched bool `json:"launched"`
	Ready    bool `json:"ready"`
	Stopped  bool `json:"stopped"`

	// Errors collects the typed failure lines in order.
	Errors []string `json:"errors,omitempty"`
	// CrashLog is the fetched on-disk tail when the launch failed.
	CrashLog string `json:"crash_log,omitempty"`
}

// MarkLaunched records a successful screen launch: the session name and log
// path, the launch command, and the launched timestamp. It is the single place
// the receipt's screen fields are set, so the launch and the receipt cannot
// drift.
func (r *Receipt) MarkLaunched(at time.Time, session, logPath, command string) {
	r.ScreenSession = session
	r.ScreenLogPath = logPath
	r.LaunchCommand = command
	r.LaunchedAt = &at
	r.Launched = true
}

// MarkReady records readiness and derives the launch duration from the launch
// timestamp (zero-safe: with no launch time the duration stays zero).
func (r *Receipt) MarkReady(at time.Time) {
	r.ReadyAt = &at
	r.Ready = true
	if r.LaunchedAt != nil {
		r.LaunchDurationSec = at.Sub(*r.LaunchedAt).Seconds()
	}
}

// MarkStopped records teardown.
func (r *Receipt) MarkStopped(at time.Time) {
	r.StoppedAt = &at
	r.Stopped = true
}

// AddError appends one failure line, preserving order.
func (r *Receipt) AddError(msg string) {
	if strings.TrimSpace(msg) == "" {
		return
	}
	r.Errors = append(r.Errors, msg)
}

// Render returns the human-facing banner: what is running, where its log is,
// and the exact command that gets the operator back in. It is deliberately
// plain text (no color, no width assumptions) so it reads the same in a
// terminal, a pipe, or a saved log. A receipt with no session renders the
// no-session line rather than an empty banner.
func (r *Receipt) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "launch: %s on %s\n", r.ServerName, r.SSHHost)
	if cmd := strings.TrimSpace(r.LaunchCommand); cmd != "" {
		fmt.Fprintf(&b, "  command: %s\n", cmd)
	}
	if r.ScreenSession == "" {
		b.WriteString("  screen: (no session — launch did not reach the host's screen)\n")
	} else {
		fmt.Fprintf(&b, "  screen: %s\n", r.ScreenSession)
		if r.ScreenLogPath != "" {
			fmt.Fprintf(&b, "    log:    %s\n", r.ScreenLogPath)
		}
		fmt.Fprintf(&b, "    attach: %s%s\n", CommandScreenAttachPrefix, r.ScreenSession)
	}
	if r.HwScreenSession != "" {
		fmt.Fprintf(&b, "  hw mon: %s\n", r.HwScreenSession)
		if r.HwScreenLogPath != "" {
			fmt.Fprintf(&b, "    log:    %s\n", r.HwScreenLogPath)
		}
	}
	if r.Launched || r.Ready || r.Stopped {
		fmt.Fprintf(&b, "  state:  launched=%t ready=%t stopped=%t", r.Launched, r.Ready, r.Stopped)
		if r.LaunchDurationSec > 0 {
			fmt.Fprintf(&b, " (%.1fs to ready)", r.LaunchDurationSec)
		}
		b.WriteString("\n")
	}
	if len(r.Errors) > 0 {
		b.WriteString("  errors:\n")
		for _, e := range r.Errors {
			fmt.Fprintf(&b, "    - %s\n", e)
		}
	}
	if strings.TrimSpace(r.CrashLog) != "" {
		b.WriteString("  crash log:\n")
		for _, l := range strings.Split(strings.TrimRight(r.CrashLog, "\n"), "\n") {
			fmt.Fprintf(&b, "    %s\n", l)
		}
	}
	return b.String()
}

// Marshal renders the receipt as indented JSON, the shape Save writes.
func (r *Receipt) Marshal() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// Save writes the receipt as JSON to path. The write is atomic — a temp file in
// the same directory is renamed over the target — so a reader never sees a
// half-written receipt and a crash mid-write cannot destroy the previous one.
// It returns the bytes written so a caller can hash or echo them.
func (r *Receipt) Save(path string) ([]byte, error) {
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("%w: receipt path is required", ErrReceiptInvalid)
	}
	data, err := r.Marshal()
	if err != nil {
		return nil, fmt.Errorf("servelaunch: marshal receipt: %w", err)
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("servelaunch: create receipt dir %q: %w", dir, err)
	}
	tmp, err := os.CreateTemp(dir, ".receipt-*.json.tmp")
	if err != nil {
		return nil, fmt.Errorf("servelaunch: create temp receipt: %w", err)
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return nil, fmt.Errorf("servelaunch: write temp receipt: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return nil, fmt.Errorf("servelaunch: close temp receipt: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return nil, fmt.Errorf("servelaunch: install receipt %q: %w", path, err)
	}
	return data, nil
}

// LoadReceipt reads a receipt back from path. The round-trip through Save is
// the durability contract: what a later tool reads is what the launch wrote.
func LoadReceipt(path string) (*Receipt, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Receipt
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("servelaunch: parse receipt %q: %w", path, err)
	}
	return &r, nil
}

// ErrReceiptInvalid reports a receipt request that cannot be satisfied (a
// missing save path).
var ErrReceiptInvalid = errors.New("servelaunch: invalid receipt request")

// LaunchProbes is the injected observation surface a Collector fills a receipt
// from. Every field is a function so the collector has no transport of its own:
// production wires these to ssh exec, tests wire them to fakes, and neither the
// collector nor its tests need a network.
type LaunchProbes struct {
	// ScreenAvailable reports whether GNU screen is on the host's PATH. Nil
	// leaves the receipt's tri-state at "never probed".
	ScreenAvailable func() (bool, error)
	// ResolveSession resolves a short session name to its full `PID.name`.
	// Nil keeps the short name.
	ResolveSession func(name string) (string, error)
	// CrashLogTail fetches the last N lines of the screen log. Nil leaves the
	// crash log empty.
	CrashLogTail func(logPath string, n int) (string, error)
}

// Collector fills a Receipt from injected probes, so recording a launch is
// testable without a transport. It never panics on a nil probe: a missing probe
// is simply a field the receipt does not learn.
type Collector struct {
	Receipt *Receipt
	Probes  LaunchProbes
}

// NewCollector starts a receipt for namedServer on host.
func NewCollector(serverName, host string, probes LaunchProbes) *Collector {
	return &Collector{
		Receipt: &Receipt{ServerName: serverName, SSHHost: host},
		Probes:  probes,
	}
}

// RecordScreenProbe folds the screen-availability probe into the receipt's
// tri-state. A probe error is recorded as an error line and leaves the
// tri-state untouched (still "unknown"), never falsely "unavailable".
func (c *Collector) RecordScreenProbe() {
	if c.Probes.ScreenAvailable == nil {
		return
	}
	ok, err := c.Probes.ScreenAvailable()
	if err != nil {
		c.Receipt.AddError(fmt.Sprintf("screen availability probe failed: %v", err))
		return
	}
	c.Receipt.ScreenAvailable = &ok
}

// RecordLaunch records a successful launch and resolves the session's full
// `PID.name` through the probe when one is wired. A resolution error is
// recorded but the short name is kept — a failed lookup must not lose the
// session field the operator needs.
func (c *Collector) RecordLaunch(at time.Time, session, logPath, command string) {
	c.Receipt.MarkLaunched(at, session, logPath, command)
	if c.Probes.ResolveSession == nil || session == "" {
		return
	}
	full, err := c.Probes.ResolveSession(session)
	if err != nil {
		c.Receipt.AddError(fmt.Sprintf("resolve screen session %q: %v", session, err))
		return
	}
	if full != "" {
		c.Receipt.ScreenSession = full
	}
}

// RecordCrash fetches the crash log through the probe and stores it. Nil probe
// or empty log path is a no-op; a fetch error is recorded.
func (c *Collector) RecordCrash(logPath string, n int) {
	if c.Probes.CrashLogTail == nil || strings.TrimSpace(logPath) == "" {
		return
	}
	tail, err := c.Probes.CrashLogTail(logPath, n)
	if err != nil {
		c.Receipt.AddError(fmt.Sprintf("fetch crash log %q: %v", logPath, err))
		return
	}
	c.Receipt.CrashLog = tail
}
