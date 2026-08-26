package servicewatchdog

import (
	"errors"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

// SystemdState is the stable, machine-oriented read-back returned by
// `systemctl show`; it deliberately excludes human `systemctl status` prose.
type SystemdState struct {
	Unit                  string `json:"unit"`
	LoadState             string `json:"load_state"`
	ActiveState           string `json:"active_state"`
	SubState              string `json:"sub_state"`
	Result                string `json:"result"`
	MainPID               int    `json:"main_pid"`
	InvocationID          string `json:"invocation_id"`
	NRestarts             int    `json:"restart_count"`
	WatchdogTimestampUSec uint64 `json:"watchdog_timestamp_usec"`
	BootID                string `json:"boot_id"`
}

var systemdShowProperties = []string{"LoadState", "ActiveState", "SubState", "Result", "MainPID", "InvocationID", "NRestarts", "WatchdogTimestampUSec", "BootID"}

// CommandBoundary is the narrow, fixtureable systemd command seam.
type CommandBoundary interface {
	Run(args ...string) error
	Output(args ...string) ([]byte, error)
}

// Manager performs idempotent lifecycle operations without implementing a
// second supervisor. Stop disables the unit so desired-stop survives reboot.
type Manager struct {
	Command CommandBoundary
	Unit    string
}

func (m Manager) Install() error {
	if err := m.Command.Run("daemon-reload"); err != nil {
		return err
	}
	return m.Command.Run("enable", "--now", m.Unit)
}
func (m Manager) Remove() error {
	if err := m.Command.Run("disable", "--now", m.Unit); err != nil {
		return err
	}
	return m.Command.Run("daemon-reload")
}
func (m Manager) Start() error   { return m.Command.Run("enable", "--now", m.Unit) }
func (m Manager) Stop() error    { return m.Command.Run("disable", "--now", m.Unit) }
func (m Manager) Restart() error { return m.Command.Run("restart", m.Unit) }
func (m Manager) Status() (SystemdState, error) {
	out, err := m.Command.Output(SystemdShowArgs(m.Unit)...)
	if err != nil {
		return SystemdState{}, err
	}
	state, err := ParseSystemdShow(out)
	state.Unit = m.Unit
	return state, err
}

// SystemdShowArgs returns the exact stable-property query used for read-back.
func SystemdShowArgs(unit string) []string {
	args := []string{"show", unit, "--no-pager"}
	for _, p := range systemdShowProperties {
		args = append(args, "--property="+p)
	}
	return args
}

// ParseSystemdShow parses systemctl's documented KEY=VALUE output and fails
// closed when a required property disappears or changes type.
func ParseSystemdShow(b []byte) (SystemdState, error) {
	values := map[string]string{}
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		k, v, ok := strings.Cut(strings.TrimSuffix(line, "\r"), "=")
		if !ok {
			return SystemdState{}, fmt.Errorf("systemd read-back: malformed property %q", line)
		}
		values[k] = v
	}
	for _, p := range systemdShowProperties {
		if _, ok := values[p]; !ok {
			return SystemdState{}, fmt.Errorf("systemd read-back: missing property %s", p)
		}
	}
	pid, err := strconv.Atoi(values["MainPID"])
	if err != nil {
		return SystemdState{}, fmt.Errorf("systemd read-back: MainPID: %w", err)
	}
	restarts, err := strconv.Atoi(values["NRestarts"])
	if err != nil {
		return SystemdState{}, fmt.Errorf("systemd read-back: NRestarts: %w", err)
	}
	watchdog, err := strconv.ParseUint(values["WatchdogTimestampUSec"], 10, 64)
	if err != nil {
		return SystemdState{}, fmt.Errorf("systemd read-back: WatchdogTimestampUSec: %w", err)
	}
	return SystemdState{LoadState: values["LoadState"], ActiveState: values["ActiveState"], SubState: values["SubState"], Result: values["Result"], MainPID: pid, InvocationID: values["InvocationID"], NRestarts: restarts, WatchdogTimestampUSec: watchdog, BootID: values["BootID"]}, nil
}

// Notifier sends systemd readiness and watchdog messages. Progress is called
// by the guarded event loop itself, so a wedged loop cannot emit a heartbeat.
type Notifier struct {
	socket   string
	interval time.Duration
	last     time.Time
	now      func() time.Time
	send     func(string, string) error
}

func NewNotifierFromEnv(expected bool) (*Notifier, error) {
	socket := os.Getenv("NOTIFY_SOCKET")
	raw := os.Getenv("WATCHDOG_USEC")
	if !expected {
		if socket != "" || raw != "" {
			return nil, errors.New("systemd notify environment present for a non-notify service")
		}
		return nil, nil
	}
	if socket == "" || raw == "" {
		return nil, errors.New("systemd notify/watchdog declaration was not negotiated")
	}
	usec, err := strconv.ParseUint(raw, 10, 64)
	if err != nil || usec == 0 {
		return nil, fmt.Errorf("invalid WATCHDOG_USEC %q", raw)
	}
	if pid := os.Getenv("WATCHDOG_PID"); pid != "" && pid != strconv.Itoa(os.Getpid()) {
		return nil, errors.New("WATCHDOG_PID does not name this process")
	}
	return &Notifier{socket: socket, interval: time.Duration(usec) * time.Microsecond / 2, now: time.Now, send: sendNotify}, nil
}

func sendNotify(socket, state string) error {
	address := socket
	if strings.HasPrefix(address, "@") {
		address = "\x00" + address[1:]
	}
	c, err := net.DialUnix("unixgram", nil, &net.UnixAddr{Name: address, Net: "unixgram"})
	if err != nil {
		return err
	}
	defer c.Close()
	_, err = c.Write([]byte(state))
	return err
}

func (n *Notifier) Ready() error {
	if n == nil {
		return nil
	}
	n.last = n.now()
	return n.send(n.socket, "READY=1")
}
func (n *Notifier) Progress() error {
	if n == nil {
		return nil
	}
	now := n.now()
	if !n.last.IsZero() && now.Sub(n.last) < n.interval {
		return nil
	}
	if err := n.send(n.socket, "WATCHDOG=1"); err != nil {
		return err
	}
	n.last = now
	return nil
}
func (n *Notifier) Stopping() error {
	if n == nil {
		return nil
	}
	return n.send(n.socket, "STOPPING=1")
}
