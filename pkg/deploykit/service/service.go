package service

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Scope selects which manager owns the registration: system (machine-wide) or
// user (the invoking user's session).
type Scope string

const (
	// ScopeSystem is a machine-owned registration: SCM service on Windows, a
	// PID-1 systemd unit on Linux, a LaunchDaemon on macOS.
	ScopeSystem Scope = "system"
	// ScopeUser is a per-user registration: Scheduled Task on Windows,
	// systemctl --user on Linux, a LaunchAgent on macOS.
	ScopeUser Scope = "user"
)

// ParseScope maps operator input to a Scope, refusing anything else.
func ParseScope(s string) (Scope, error) {
	switch s {
	case "system":
		return ScopeSystem, nil
	case "user":
		return ScopeUser, nil
	}
	return "", fmt.Errorf("deploykit/service: unknown scope %q (want \"system\" or \"user\")", s)
}

// Command is the injectable command boundary every non-SCM driver shells
// through (systemctl / launchctl / schtasks). Tests substitute a fake; the
// default is a real exec.Command runner.
type Command interface {
	Run(args ...string) error
	Output(args ...string) ([]byte, error)
}

type execCommand struct{}

func (execCommand) Run(args ...string) error {
	if len(args) == 0 {
		return errors.New("deploykit/service: empty command")
	}
	return exec.Command(args[0], args[1:]...).Run()
}

func (execCommand) Output(args ...string) ([]byte, error) {
	if len(args) == 0 {
		return nil, errors.New("deploykit/service: empty command")
	}
	return exec.Command(args[0], args[1:]...).Output()
}

// Definition is what an Update/Install applies: where the registration document
// lives (unit file, plist, task XML), its rendered text, and — for the Windows
// SCM driver — the executable and arguments the service will run.
type Definition struct {
	// Path is the target registration document path. The Windows SCM driver
	// ignores it (its registration is the service record itself); the other
	// drivers stage Text beside it and swap it in atomically.
	Path string
	// Text is the rendered registration document. Empty only for the SCM
	// driver, whose configuration comes from Executable/Args.
	Text string
	// Executable is the program the service runs (required by the SCM driver).
	Executable string
	// Args are the command-line arguments after Executable.
	Args []string
}

// Snapshot captures the prior registration so Restore can put it back: the
// document bytes (file-based drivers) or the platform state (SCM driver), plus
// whether the service was running and enabled before the update began.
type Snapshot struct {
	// Path is the prior registration location: the document path for file
	// drivers, the prior image path for the SCM driver.
	Path string
	// Prior is the prior registration document bytes (nil when nothing was
	// registered).
	Prior []byte
	// WasRunning reports whether the service was active at snapshot time.
	WasRunning bool
	// WasEnabled reports whether the registration persisted across reboots at
	// snapshot time (enabled unit / booted job / registered task).
	WasEnabled bool
	// Platform carries driver-specific prior state (the SCM config, recovery
	// ladder, and flags). Nil for file-based drivers.
	Platform any
}

// exists reports whether the snapshot captured a live prior registration.
func (s Snapshot) exists() bool { return s.Prior != nil || s.Platform != nil }

// Readback is the observed registration state after control.
type Readback struct {
	// Command is the command line the registration currently runs ("" when the
	// platform does not expose one, e.g. scheduled tasks).
	Command string
	// State is Running, Stopped, or Unknown.
	State string
	// PID is the main process id (0 when not running or unreported).
	PID int
}

// Read-back state vocabulary. Deliberately distinct strings from the numeric
// SCM State* constants above: these describe observation, those describe the
// wire protocol.
const (
	Running = "running"
	Stopped = "stopped"
	Unknown = "unknown"
)

// driverKind names the four control planes the facade drives.
type driverKind int

const (
	driverSystemd driverKind = iota
	driverLaunchd
	driverScheduledTask
	driverSCM
)

// Manager is one deployable's service registration: a scope plus a name, with
// injectable command/SCM boundaries and directory overrides for tests.
type Manager struct {
	scope     Scope
	name      string
	kind      driverKind
	cmd       Command
	scm       SCMBoundary
	unitDir   string
	plistDir  string
	binaryDir string
	uid       int
}

// Option customizes a Manager.
type Option func(*Manager)

// WithCommand injects the command boundary (systemctl/launchctl/schtasks).
// Nil keeps the platform default (real exec).
func WithCommand(c Command) Option {
	return func(m *Manager) {
		if c != nil {
			m.cmd = c
		}
	}
}

// WithSCM injects the Windows SCM boundary. Nil keeps the platform default
// (a live SCM connection on Windows; an error elsewhere).
func WithSCM(b SCMBoundary) Option {
	return func(m *Manager) {
		if b != nil {
			m.scm = b
		}
	}
}

// WithUnitDir overrides the systemd unit directory (default:
// /etc/systemd/system for system scope, $HOME/.config/systemd/user for user).
func WithUnitDir(dir string) Option { return func(m *Manager) { m.unitDir = dir } }

// WithPlistDir overrides the launchd plist directory (default:
// /Library/LaunchDaemons for system scope, $HOME/Library/LaunchAgents for user).
func WithPlistDir(dir string) Option { return func(m *Manager) { m.plistDir = dir } }

// WithBinaryDir overrides the staged-executable directory the Windows SCM
// driver publishes into (default: %ProgramData%\fak\bin).
func WithBinaryDir(dir string) Option { return func(m *Manager) { m.binaryDir = dir } }

// WithUID overrides the uid used for the darwin user-scope gui/<uid> domain
// (default: os.Getuid()).
func WithUID(uid int) Option { return func(m *Manager) { m.uid = uid } }

// New builds a Manager for (scope, name) on the current OS. The name is the
// registration identity: the launchd label, the systemd unit stem, the
// scheduled-task name, or the SCM service name.
func New(scope Scope, name string, opts ...Option) (*Manager, error) {
	if name == "" {
		return nil, errors.New("deploykit/service: service name is required")
	}
	kind, err := kindFor(runtime.GOOS, scope)
	if err != nil {
		return nil, err
	}
	m := &Manager{
		scope: scope,
		name:  name,
		kind:  kind,
		cmd:   execCommand{},
		uid:   os.Getuid(),
	}
	for _, o := range opts {
		o(m)
	}
	return m, nil
}

func kindFor(goos string, scope Scope) (driverKind, error) {
	switch goos {
	case "linux":
		return driverSystemd, nil
	case "darwin":
		return driverLaunchd, nil
	case "windows":
		if scope == ScopeUser {
			return driverScheduledTask, nil
		}
		return driverSCM, nil
	}
	return 0, fmt.Errorf("deploykit/service: unsupported GOOS %q", goos)
}

func (m *Manager) driver() driver {
	switch m.kind {
	case driverLaunchd:
		return launchdDriver{}
	case driverScheduledTask:
		return scheduledTaskDriver{}
	case driverSCM:
		return scmDriver{}
	default:
		return systemdDriver{}
	}
}

// command returns the effective command boundary with the user-scope prefix
// applied where the platform needs one (systemctl --user).
func (m *Manager) command() Command {
	inner := m.cmd
	if inner == nil {
		inner = execCommand{}
	}
	if m.kind == driverSystemd && m.scope == ScopeUser {
		return userPrefixedCommand{inner: inner, user: true}
	}
	return inner
}

// userPrefixedCommand prepends "--user" to every invocation for user-scope
// systemd units.
type userPrefixedCommand struct {
	inner Command
	user  bool
}

func (p userPrefixedCommand) Run(args ...string) error {
	return p.inner.Run(p.prefix(args)...)
}

func (p userPrefixedCommand) Output(args ...string) ([]byte, error) {
	return p.inner.Output(p.prefix(args)...)
}

func (p userPrefixedCommand) prefix(args []string) []string {
	if !p.user {
		return args
	}
	return append(append([]string(nil), "--user"), args...)
}

// phaseError tags a failure with the named phase it happened in, so Update can
// shape the returned error without re-parsing messages.
type phaseError struct {
	phase string
	err   error
}

func (e *phaseError) Error() string { return e.phase + ": " + e.err.Error() }

func (e *phaseError) Unwrap() error { return e.err }

func phaseErr(phase string, err error) error {
	if err == nil {
		return nil
	}
	if pe, ok := err.(*phaseError); ok {
		return pe // already tagged with a more specific phase
	}
	return &phaseError{phase: phase, err: err}
}

// updateFailure shapes an Update failure exactly like cmd/fak's
// windowsInstallFailure: the failed phase and cause first, then either the
// preserved prior registration or the rollback failure with the retained
// state.
func updateFailure(phase string, cause, rollbackErr error, oldPath string) error {
	if rollbackErr != nil {
		return fmt.Errorf("%s: %v; rollback failed: %v; previous registration retained at [%s]", phase, cause, rollbackErr, oldPath)
	}
	return fmt.Errorf("%s: %v; previous service configuration preserved at [%s]", phase, cause, oldPath)
}

// Render renders the platform-native registration document for this Manager's
// (GOOS, scope) from def's executable and arguments. It is pure: no command
// runs, no file is touched. Callers that own their templates (e.g. fak node
// install) pass their rendered text as Definition.Text instead.
func (m *Manager) Render(def Definition) (string, error) {
	if def.Executable == "" {
		return "", errors.New("deploykit/service: Definition.Executable is required for Render")
	}
	switch m.kind {
	case driverSystemd:
		return renderSystemdUnit(m.name, m.scope, def)
	case driverLaunchd:
		return renderLaunchdPlist(m.name, m.scope, def)
	case driverScheduledTask:
		return renderScheduledTaskXML(m.name, def)
	case driverSCM:
		return renderSCMDefinition(m.name, def)
	}
	return "", fmt.Errorf("deploykit/service: unsupported driver %d", m.kind)
}

// Install registers def when nothing is registered yet. If a registration
// already exists it behaves exactly like Update. On a fresh-install failure the
// partial registration is removed best-effort.
func (m *Manager) Install(def Definition) error {
	d := m.driver()
	snap, err := d.snapshot(m, def)
	if err != nil {
		return fmt.Errorf("snapshot service %s: %w", m.name, err)
	}
	if snap.exists() {
		return m.Update(def)
	}
	if err := d.apply(m, snap, def); err != nil {
		var pe *phaseError
		if !errors.As(err, &pe) {
			pe = &phaseError{phase: "apply", err: err}
		}
		if rerr := d.restore(m, snap); rerr != nil {
			return updateFailure(pe.phase, pe.err, rerr, snap.Path)
		}
		return updateFailure(pe.phase, pe.err, nil, snap.Path)
	}
	return nil
}

// Update snapshots the prior registration, applies def (stop → publish →
// register/update → start → read back), and on any failure after the snapshot
// restores the prior registration AND the prior running state, returning an
// error shaped by updateFailure.
func (m *Manager) Update(def Definition) error {
	d := m.driver()
	snap, err := d.snapshot(m, def)
	if err != nil {
		return fmt.Errorf("snapshot service %s: %w", m.name, err)
	}
	if err := d.apply(m, snap, def); err != nil {
		var pe *phaseError
		if !errors.As(err, &pe) {
			pe = &phaseError{phase: "apply", err: err}
		}
		if rerr := m.Restore(snap); rerr != nil {
			return updateFailure(pe.phase, pe.err, rerr, snap.Path)
		}
		return updateFailure(pe.phase, pe.err, nil, snap.Path)
	}
	return nil
}

// Restore puts back snap's prior registration and prior running state. It is
// safe to call when snap captured nothing (no-op).
func (m *Manager) Restore(snap Snapshot) error {
	return m.driver().restore(m, snap)
}

// Snapshot captures the current prior-registration state without changing
// anything.
func (m *Manager) Snapshot() (Snapshot, error) {
	return m.driver().snapshot(m, Definition{})
}

// Start brings the registration up.
func (m *Manager) Start() error { return m.driver().start(m) }

// Stop takes the registration down.
func (m *Manager) Stop() error { return m.driver().stop(m) }

// Restart cycles the registration.
func (m *Manager) Restart() error { return m.driver().restart(m) }

// Readback reports the observed registration state.
func (m *Manager) Readback() (Readback, error) { return m.driver().readback(m) }

// driver is the internal contract every control plane implements.
type driver interface {
	snapshot(m *Manager, def Definition) (Snapshot, error)
	apply(m *Manager, snap Snapshot, def Definition) error
	restore(m *Manager, snap Snapshot) error
	readback(m *Manager) (Readback, error)
	start(m *Manager) error
	stop(m *Manager) error
	restart(m *Manager) error
}

// ── pure renderers (Render dispatch) ────────────────────────────────────────
//
// Each renderer is a deterministic function of (name, scope, def): no command,
// no filesystem. Values containing control characters or double quotes are
// refused rather than escaped — the same refuse-not-escape rule the platform
// projections use — so every accepted input renders byte-for-byte.

func renderRefuse(name, v string) error {
	if v == "" {
		return fmt.Errorf("deploykit/service: %s is empty", name)
	}
	if containsControl(v) {
		return fmt.Errorf("deploykit/service: %s contains a control character", name)
	}
	if strings.ContainsRune(v, '"') {
		return fmt.Errorf("deploykit/service: %s contains a double quote; rendering would be ambiguous", name)
	}
	return nil
}

func containsControl(v string) bool {
	for _, r := range v {
		if r < 0x20 || r == 0x7f {
			return true
		}
	}
	return false
}

// systemdQuote wraps a value in double quotes when it carries a space or tab;
// embedded quotes were already refused by renderRefuse.
func systemdQuote(v string) string {
	if strings.ContainsAny(v, " \t") {
		return `"` + v + `"`
	}
	return v
}

// renderSystemdUnit renders the minimal always-on unit for both scopes; the
// scope picks only the [Install] target.
func renderSystemdUnit(name string, scope Scope, def Definition) (string, error) {
	if err := renderRefuse("service name", name); err != nil {
		return "", err
	}
	if err := renderRefuse("executable", def.Executable); err != nil {
		return "", err
	}
	for _, a := range def.Args {
		if err := renderRefuse("argument", a); err != nil {
			return "", err
		}
	}
	want := "default.target"
	if scope == ScopeSystem {
		want = "multi-user.target"
	}
	args := make([]string, 0, len(def.Args)+1)
	args = append(args, systemdQuote(def.Executable))
	for _, a := range def.Args {
		args = append(args, systemdQuote(a))
	}
	var b strings.Builder
	fmt.Fprintf(&b, "[Unit]\nDescription=fak service %s\nAfter=network.target\n\n[Service]\nExecStart=%s\nRestart=always\n\n[Install]\nWantedBy=%s\n",
		name, strings.Join(args, " "), want)
	return b.String(), nil
}

// renderLaunchdPlist renders the minimal always-on job; the scope picks only
// the label domain convention (system jobs live under LaunchDaemons, user
// jobs under LaunchAgents — the label text is identical).
func renderLaunchdPlist(name string, scope Scope, def Definition) (string, error) {
	if err := renderRefuse("label", name); err != nil {
		return "", err
	}
	if err := renderRefuse("executable", def.Executable); err != nil {
		return "", err
	}
	for _, a := range def.Args {
		if err := renderRefuse("argument", a); err != nil {
			return "", err
		}
	}
	var b bytes.Buffer
	b.WriteString(`<?xml version="1.0" encoding="UTF-8"?>` + "\n")
	b.WriteString(`<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">` + "\n")
	b.WriteString(`<plist version="1.0"><dict>` + "\n")
	b.WriteString(`<key>Label</key><string>`)
	_ = xml.EscapeText(&b, []byte(name))
	b.WriteString(`</string>` + "\n")
	b.WriteString(`<key>ProgramArguments</key><array><string>`)
	_ = xml.EscapeText(&b, []byte(def.Executable))
	b.WriteString(`</string>`)
	for _, a := range def.Args {
		b.WriteString(`<string>`)
		_ = xml.EscapeText(&b, []byte(a))
		b.WriteString(`</string>`)
	}
	b.WriteString(`</array>` + "\n")
	b.WriteString(`<key>RunAtLoad</key><true/>` + "\n")
	b.WriteString(`<key>KeepAlive</key><true/>` + "\n")
	b.WriteString(`</dict></plist>` + "\n")
	return b.String(), nil
}

// renderScheduledTaskXML renders the boot-triggered, restart-on-failure task
// definition that runs def's executable with its arguments.
func renderScheduledTaskXML(name string, def Definition) (string, error) {
	if err := renderRefuse("task name", name); err != nil {
		return "", err
	}
	if err := renderRefuse("executable", def.Executable); err != nil {
		return "", err
	}
	for _, a := range def.Args {
		if err := renderRefuse("argument", a); err != nil {
			return "", err
		}
	}
	esc := func(s string) string {
		var b bytes.Buffer
		_ = xml.EscapeText(&b, []byte(s))
		return b.String()
	}
	args := strings.Join(def.Args, " ")
	var out strings.Builder
	out.WriteString(`<?xml version="1.0" encoding="UTF-16"?>` + "\n")
	out.WriteString(`<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">` + "\n")
	out.WriteString(`  <RegistrationInfo>` + "\n")
	out.WriteString(`    <Description>fak service ` + esc(name) + ` — written by deploykit</Description>` + "\n")
	out.WriteString(`  </RegistrationInfo>` + "\n")
	out.WriteString(`  <Triggers>` + "\n")
	out.WriteString(`    <BootTrigger><Enabled>true</Enabled></BootTrigger>` + "\n")
	out.WriteString(`  </Triggers>` + "\n")
	out.WriteString(`  <Principals>` + "\n")
	out.WriteString(`    <Principal id="Author">` + "\n")
	out.WriteString(`      <LogonType>InteractiveToken</LogonType>` + "\n")
	out.WriteString(`      <RunLevel>LeastPrivilege</RunLevel>` + "\n")
	out.WriteString(`    </Principal>` + "\n")
	out.WriteString(`  </Principals>` + "\n")
	out.WriteString(`  <Settings>` + "\n")
	out.WriteString(`    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>` + "\n")
	out.WriteString(`    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>` + "\n")
	out.WriteString(`    <RestartOnFailure>` + "\n")
	out.WriteString(`      <Interval>PT1M</Interval>` + "\n")
	out.WriteString(`      <Count>999</Count>` + "\n")
	out.WriteString(`    </RestartOnFailure>` + "\n")
	out.WriteString(`  </Settings>` + "\n")
	out.WriteString(`  <Actions Context="Author">` + "\n")
	out.WriteString(`    <Exec>` + "\n")
	out.WriteString(`      <Command>` + esc(def.Executable) + `</Command>` + "\n")
	out.WriteString(`      <Arguments>` + esc(args) + `</Arguments>` + "\n")
	out.WriteString(`    </Exec>` + "\n")
	out.WriteString(`  </Actions>` + "\n")
	out.WriteString(`</Task>` + "\n")
	return out.String(), nil
}

// renderSCMDefinition renders the deterministic sc.exe sequence expressing the
// system-scope registration: auto start, restricted SID, least-privilege
// account, and the fak recovery ladder (3s/10s/30s, reset 86400, non-crash on).
func renderSCMDefinition(name string, def Definition) (string, error) {
	if err := renderRefuse("service name", name); err != nil {
		return "", err
	}
	if err := renderRefuse("executable", def.Executable); err != nil {
		return "", err
	}
	for _, a := range def.Args {
		if err := renderRefuse("argument", a); err != nil {
			return "", err
		}
	}
	binPath := scmCommandLine(def.Executable, def.Args...)
	var b strings.Builder
	fmt.Fprintf(&b, "sc.exe create %s binPath= \"%s\" start= auto obj= \"NT AUTHORITY\\LocalService\" DisplayName= \"fak service %s\"\n",
		name, strings.ReplaceAll(binPath, `"`, `\"`), name)
	fmt.Fprintf(&b, "sc.exe failure %s reset= 86400 actions= restart/3000/restart/10000/restart/30000\n", name)
	fmt.Fprintf(&b, "sc.exe failureflag %s 1\n", name)
	return b.String(), nil
}

// scmCommandLine joins executable + args with Windows quoting: an argument
// with spaces or tabs is wrapped in double quotes (embedded quotes were
// refused upstream).
func scmCommandLine(executable string, args ...string) string {
	quoted := make([]string, 0, len(args)+1)
	quoted = append(quoted, winQuote(executable))
	for _, a := range args {
		quoted = append(quoted, winQuote(a))
	}
	return strings.Join(quoted, " ")
}

func winQuote(a string) string {
	if a == "" || strings.ContainsAny(a, " \t") {
		return `"` + a + `"`
	}
	return a
}

// binaryDirPath resolves the staged-executable directory for the SCM driver.
func (m *Manager) binaryDirPath() string {
	if m.binaryDir != "" {
		return m.binaryDir
	}
	pd := os.Getenv("ProgramData")
	if pd == "" {
		pd = filepath.Join(`C:`, `ProgramData`)
	}
	return filepath.Join(pd, "fak", "bin")
}

// scmBoundary resolves the effective SCM boundary: injected, or the platform
// default (live connection on Windows, refusal elsewhere).
func (m *Manager) scmBoundary() (SCMBoundary, error) {
	if m.scm != nil {
		return m.scm, nil
	}
	return defaultSCMBoundary()
}
