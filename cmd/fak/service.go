package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"os/user"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/serviceledger"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
	"github.com/anthony-chaudhary/fak/internal/servicewatchdog"
	"github.com/anthony-chaudhary/fak/internal/systemservice"
)

type serviceResult struct {
	Manager    string                        `json:"manager"`
	Unit       string                        `json:"unit"`
	Path       string                        `json:"path,omitempty"`
	Active     bool                          `json:"active,omitempty"`
	Executable string                        `json:"executable,omitempty"`
	PIDBefore  uint32                        `json:"pid_before,omitempty"`
	PIDAfter   uint32                        `json:"pid_after,omitempty"`
	StateKept  bool                          `json:"state_kept,omitempty"`
	Systemd    *servicewatchdog.SystemdState `json:"systemd,omitempty"`
}

var serviceCommand = exec.Command
var serviceCommandOutput = exec.Command
var serviceTick = func(stdout, stderr io.Writer) int {
	return runResumeWatchdog(stdout, stderr, []string{"--live", "--json"})
}

// defaultRootedPath fills an unset path flag with a default anchored at the filesystem
// root, joining `segments` under the platform separator so the same call yields the
// systemd/launchd location on each host instead of a hand-spelled literal per platform.
// A flag the operator already set is left exactly as given.
func defaultRootedPath(flagValue *string, segments ...string) {
	if *flagValue != "" {
		return
	}
	*flagValue = filepath.Join(append([]string{string(filepath.Separator)}, segments...)...)
}

type systemctlBoundary struct{ stdout, stderr io.Writer }

func (b systemctlBoundary) Run(args ...string) error {
	c := serviceCommand("systemctl", args...)
	c.Stdout = b.stdout
	c.Stderr = b.stderr
	return c.Run()
}
func (b systemctlBoundary) Output(args ...string) ([]byte, error) {
	return serviceCommandOutput("systemctl", args...).Output()
}

func cmdService(args []string) { os.Exit(runService(os.Stdout, os.Stderr, args)) }
func runService(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		serviceUsage(stderr)
		return 2
	}
	if args[0] == "windows-run" {
		return runWindowsServiceDispatcher(stdout, stderr)
	}
	if args[0] == "run" {
		return runServiceLoop(stdout, stderr, args[1:])
	}
	if args[0] == "events" {
		return runServiceEvents(stdout, stderr, args[1:])
	}
	if args[0] == "bridge" {
		return runServiceBridge(stdout, stderr, args[1:])
	}
	// The bare `status` verb keeps its platform-manager meaning; the observed-
	// event ledger rollup (#4753) answers when the operator names a ledger,
	// either explicitly (--ledger-dir) or via FAK_SERVICE_LEDGER_DIR.
	if args[0] == "status" && (hasServiceLedgerFlag(args[1:]) || os.Getenv("FAK_SERVICE_LEDGER_DIR") != "") {
		return runServiceStatus(stdout, stderr, args[1:])
	}
	action := args[0]
	fs := flag.NewFlagSet("service "+action, flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "machine-readable result")
	dry := fs.Bool("dry-run", false, "render/validate without changing service manager")
	unitDir := fs.String("unit-dir", "", "override service definition directory")
	stateDir := fs.String("state-dir", "", "durable control-plane state directory")
	principal := fs.String("principal", "", "unprivileged account used by the system service")
	execPath := fs.String("exec-path", "", "stable OS-owned executable path")
	if err := fs.Parse(args[1:]); err != nil || fs.NArg() != 0 {
		return 2
	}
	exe, err := os.Executable()
	if err != nil {
		return 1
	}
	var manager, name, path, definition, stagedExecutable string
	var install, status, uninstall func() error
	var systemdStatus func() (servicewatchdog.SystemdState, error)
	switch runtime.GOOS {
	case "windows":
		result, rc := windowsServiceAction(action, stdout, stderr, *dry)
		if *asJSON {
			_ = json.NewEncoder(stdout).Encode(result)
		} else if rc == 0 {
			fmt.Fprintf(stdout, "%s %s %s\n", result.Manager, action, result.Path)
		}
		return rc
	case "linux":
		manager = "systemd-system"
		name = systemservice.SystemdUnitName
		defaultRootedPath(unitDir, "etc", "systemd", "system")
		defaultRootedPath(stateDir, "var", "lib", "fak")
		defaultRootedPath(execPath, "usr", "local", "libexec", "fak-guard-control")
		stagedExecutable = *execPath
		path = filepath.Join(*unitDir, name)
		definition, err = systemservice.RenderSystemdSystemUnit(systemservice.SystemdConfig{Executable: stagedExecutable, StateDir: *stateDir})
		boundary := systemctlBoundary{stdout: stdout, stderr: stderr}
		manager := servicewatchdog.Manager{Command: boundary, Unit: name}
		install = manager.Install
		systemdStatus = manager.Status
		uninstall = manager.Remove
		status = func() error { _, e := manager.Status(); return e }
	case "darwin":
		manager = "launchd-system"
		name = systemservice.LaunchdLabel
		defaultRootedPath(unitDir, "Library", "LaunchDaemons")
		defaultRootedPath(stateDir, "var", "db", "fak")
		if *principal == "" {
			*principal = "nobody"
		}
		defaultRootedPath(execPath, "usr", "local", "libexec", "fak-guard-control")
		stagedExecutable = *execPath
		logDir := filepath.Join(*stateDir, "logs")
		path = filepath.Join(*unitDir, name+".plist")
		definition, err = systemservice.RenderLaunchDaemon(systemservice.LaunchdConfig{Executable: stagedExecutable, StateDir: *stateDir, StdoutPath: filepath.Join(logDir, "guard-control.out.log"), StderrPath: filepath.Join(logDir, "guard-control.err.log"), UserName: *principal})
		domain := "system"
		target := domain + "/" + name
		cmd := func(argv ...string) error {
			c := serviceCommand("launchctl", argv...)
			c.Stdout = stdout
			c.Stderr = stderr
			return c.Run()
		}
		install = func() error {
			_ = cmd("bootout", target)
			if e := cmd("bootstrap", domain, path); e != nil {
				return e
			}
			return cmd("kickstart", "-k", target)
		}
		status = func() error { return cmd("print", target) }
		uninstall = func() error { return cmd("bootout", target) }
	default:
		fmt.Fprintln(stderr, "fak service: supported on Linux systemd and macOS launchd")
		return 2
	}
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	result := serviceResult{Manager: manager, Unit: name, Path: path, Executable: stagedExecutable}
	if *dry {
		if action == "install" {
			fmt.Fprint(stdout, definition)
		}
		if *asJSON {
			_ = json.NewEncoder(stdout).Encode(result)
		}
		return 0
	}
	switch action {
	case "install":
		registryDir := filepath.Join(*stateDir, "registry")
		if os.MkdirAll(*unitDir, 0o755) != nil || os.MkdirAll(*stateDir, 0o700) != nil || os.MkdirAll(filepath.Join(*stateDir, "logs"), 0o700) != nil {
			return 1
		}
		if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
			if err := os.Chmod(*stateDir, 0o711); err != nil {
				fmt.Fprintln(stderr, "fak service: prepare state traversal:", err)
				return 1
			}
		}
		if err := prepareSharedGuardRegistry(registryDir); err != nil {
			fmt.Fprintln(stderr, "fak service: prepare shared registry:", err)
			return 1
		}
		if stagedExecutable != "" {
			if err := installServiceExecutable(exe, stagedExecutable); err != nil {
				fmt.Fprintln(stderr, "fak service: stage executable:", err)
				return 1
			}
		}
		if runtime.GOOS == "darwin" {
			if err := chownServiceStateExcept(*stateDir, *principal, registryDir); err != nil {
				fmt.Fprintln(stderr, "fak service: prepare launchd state:", err)
				return 1
			}
			if err := os.Chmod(*stateDir, 0o711); err != nil {
				fmt.Fprintln(stderr, "fak service: preserve registry traversal:", err)
				return 1
			}
		}
		if writeFileAtomic(path, []byte(definition), 0o644) != nil || install() != nil {
			return 1
		}
	case "status":
		if systemdStatus != nil {
			s, e := systemdStatus()
			if e != nil {
				return 3
			}
			result.Systemd = &s
			result.Active = s.ActiveState == "active"
		} else if status() == nil {
			result.Active = true
		} else {
			return 3
		}
	case "uninstall", "remove":
		if e := uninstall(); e != nil {
			return 1
		}
		if e := os.Remove(path); e != nil && !os.IsNotExist(e) {
			return 1
		}
	case "start", "stop", "restart":
		if runtime.GOOS != "linux" {
			serviceUsage(stderr)
			return 2
		}
		manager := servicewatchdog.Manager{Command: systemctlBoundary{stdout: stdout, stderr: stderr}, Unit: name}
		var e error
		switch action {
		case "start":
			e = manager.Start()
		case "stop":
			e = manager.Stop()
		case "restart":
			e = manager.Restart()
		}
		if e != nil {
			return 1
		}
	default:
		serviceUsage(stderr)
		return 2
	}
	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(result)
	} else {
		fmt.Fprintf(stdout, "%s %s %s\n", manager, action, path)
	}
	return 0
}
func runServiceLoopContext(ctx context.Context, stdout, stderr io.Writer, interval time.Duration, notifyModeOpt ...string) int {
	notifyMode := "none"
	if len(notifyModeOpt) > 0 {
		notifyMode = notifyModeOpt[0]
	}
	notify, err := servicewatchdog.NewNotifierFromEnv(notifyMode == "systemd")
	if err != nil {
		fmt.Fprintln(stderr, "fak service run:", err)
		return 1
	}
	if err := notify.Ready(); err != nil {
		fmt.Fprintln(stderr, "fak service run: ready:", err)
		return 1
	}
	defer func() { _ = notify.Stopping() }()
	for {
		if rc := serviceTick(stdout, stderr); rc != 0 {
			return rc
		}
		if err := notify.Progress(); err != nil {
			fmt.Fprintln(stderr, "fak service run: watchdog:", err)
			return 1
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return 0
		case <-timer.C:
		}
	}
}
func runServiceLoop(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("service run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	interval := fs.Duration("interval", 15*time.Second, "control-plane tick interval")
	once := fs.Bool("once", false, "run one tick and exit")
	notifyMode := fs.String("notify", "none", "service notification protocol: none or systemd")
	if fs.Parse(args) != nil || fs.NArg() != 0 || *interval <= 0 {
		return 2
	}
	if *notifyMode != "none" && *notifyMode != "systemd" {
		fmt.Fprintln(stderr, "fak service run: --notify must be none or systemd")
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	if *once {
		return serviceTick(stdout, stderr)
	}
	return runServiceLoopContext(ctx, stdout, stderr, *interval, *notifyMode)
}
func serviceUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak service install|remove|start|stop|restart|status|run [--dry-run] [--json]")
	fmt.Fprintln(w, "       fak service run [--interval D] [--once] [--notify none|systemd]")
	fmt.Fprintln(w, "       fak service events [--json] [--ledger-dir D] [--service S]")
	fmt.Fprintln(w, "       fak service events --ingest windows-xml|journald-json|launchd-ndjson --file F --node N --service S [--workload W] [--unit U] [--json]")
	fmt.Fprintln(w, "       fak service status --ledger-dir D [--json]    (observed-event rollup; also picked when FAK_SERVICE_LEDGER_DIR is set)")
	fmt.Fprintln(w, "       fak service bridge --spec F --role machine|watchdog|broker [--sha256 HEX] [--principal U] [--observed F | --live [--unit S]] [--ledger-dir D] [--json]")
	fmt.Fprintln(w, "       fak service bridge --judge terminal-kill|termservice-reset|host-reboot|scm-process-kill --ledger-dir D [--service S] [--json]")
}

// hasServiceLedgerFlag reports whether the operator explicitly named the
// observed-event ledger, which routes `status` to the ledger rollup instead
// of the platform service manager.
func hasServiceLedgerFlag(args []string) bool {
	for _, a := range args {
		if a == "-ledger-dir" || a == "--ledger-dir" ||
			strings.HasPrefix(a, "-ledger-dir=") || strings.HasPrefix(a, "--ledger-dir=") {
			return true
		}
	}
	return false
}

func openServiceLedger(stderr io.Writer, dir string) (*serviceledger.Ledger, int) {
	if dir == "" {
		dir = serviceledger.DefaultDir()
	}
	led, err := serviceledger.Open(dir)
	if err != nil {
		fmt.Fprintln(stderr, "fak service: open ledger:", err)
		return nil, 1
	}
	return led, 0
}

func filterServiceEvents(events []serviceledger.Event, service string) []serviceledger.Event {
	if service == "" {
		return events
	}
	out := events[:0]
	for _, e := range events {
		if e.Identity.Service == service {
			out = append(out, e)
		}
	}
	return out
}

// runServiceEvents implements `fak service events` (#4753): list the
// correlated observed-event timeline, or ingest a native manager export
// through the fixture-tested adapters (windows-xml | journald-json |
// launchd-ndjson; `--file -` reads stdin, e.g. piped from `wevtutil qe
// System /f:xml`).
func runServiceEvents(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("service events", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "machine-readable output (JSONL events / ingest summary)")
	ledgerDir := fs.String("ledger-dir", "", "observed-event ledger directory (default FAK_SERVICE_LEDGER_DIR, else user config dir)")
	ingest := fs.String("ingest", "", "ingest a native export: windows-xml|journald-json|launchd-ndjson")
	file := fs.String("file", "-", "native export file for --ingest (- reads stdin)")
	node := fs.String("node", "", "identity node for --ingest")
	service := fs.String("service", "", "identity service (--ingest) / list filter")
	workload := fs.String("workload", "", "identity workload for --ingest")
	unit := fs.String("unit", "", "native unit/service-name/label filter for --ingest")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		return 2
	}
	led, rc := openServiceLedger(stderr, *ledgerDir)
	if rc != 0 {
		return rc
	}
	if *ingest != "" {
		var parse func(io.Reader, serviceledger.AdapterConfig) ([]serviceledger.Event, error)
		switch *ingest {
		case "windows-xml":
			parse = serviceledger.AdaptWindowsEventXML
		case "journald-json":
			parse = serviceledger.AdaptJournaldExport
		case "launchd-ndjson":
			parse = serviceledger.AdaptLaunchdNDJSON
		default:
			fmt.Fprintf(stderr, "fak service events: unknown ingest format %q\n", *ingest)
			return 2
		}
		var r io.Reader = os.Stdin
		if *file != "-" {
			f, err := os.Open(*file)
			if err != nil {
				fmt.Fprintln(stderr, "fak service events:", err)
				return 1
			}
			defer f.Close()
			r = f
		}
		cfg := serviceledger.AdapterConfig{
			Identity: servicespec.Identity{Node: *node, Service: *service, Workload: *workload},
			Unit:     *unit,
		}
		evs, err := parse(r, cfg)
		if err != nil {
			fmt.Fprintln(stderr, "fak service events:", err)
			return 1
		}
		ingested, duplicates, err := led.AppendAll(evs)
		if err != nil {
			fmt.Fprintln(stderr, "fak service events:", err)
			return 1
		}
		if *asJSON {
			_ = json.NewEncoder(stdout).Encode(map[string]int{
				"ingested": ingested, "duplicates": duplicates, "events": len(led.Events()),
			})
		} else {
			fmt.Fprintf(stdout, "ingested %d event(s), skipped %d duplicate(s)\n", ingested, duplicates)
		}
		return 0
	}
	events := filterServiceEvents(led.Events(), *service)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		for _, e := range events {
			_ = enc.Encode(e)
		}
		return 0
	}
	serviceledger.WriteTimeline(stdout, events)
	return 0
}

// runServiceStatus implements the ledger-backed `fak service status` rollup
// (#4753): per-workload phase, correlation high-water marks, restart-storm
// verdict, and stale-but-still-running owners. Exit code 4 flags a detected
// restart storm or stale owner so scripts can alert without parsing output.
func runServiceStatus(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("service status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "machine-readable result")
	ledgerDir := fs.String("ledger-dir", "", "observed-event ledger directory (default FAK_SERVICE_LEDGER_DIR, else user config dir)")
	service := fs.String("service", "", "only this identity service")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 {
		return 2
	}
	led, rc := openServiceLedger(stderr, *ledgerDir)
	if rc != 0 {
		return rc
	}
	events := filterServiceEvents(led.Events(), *service)
	sts := serviceledger.Status(events, serviceledger.StatusOptions{})
	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(sts)
	} else {
		serviceledger.WriteStatus(stdout, sts)
	}
	for _, st := range sts {
		if st.RestartStorm || len(st.StaleOwners) > 0 {
			return 4
		}
	}
	return 0
}
func prepareSharedGuardRegistry(registryDir string) error {
	if err := os.MkdirAll(registryDir, 0o733); err != nil {
		return err
	}
	if err := os.Chmod(registryDir, os.ModeSticky|0o733); err != nil {
		return err
	}
	index := guardsessions.IndexPath(registryDir)
	f, err := os.OpenFile(index, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o666)
	if err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Chmod(index, 0o666)
}

func installServiceExecutable(source, destination string) error {
	if filepath.Clean(source) == filepath.Clean(destination) {
		return nil
	}
	b, err := os.ReadFile(source)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		return err
	}
	return writeFileAtomic(destination, b, 0o755)
}

func chownServiceStateExcept(path, principal, excluded string) error {
	u, err := user.Lookup(principal)
	if err != nil {
		return fmt.Errorf("lookup principal %q: %w", principal, err)
	}
	uid64, err := strconv.ParseInt(u.Uid, 10, 32)
	if err != nil {
		return fmt.Errorf("parse uid for %q: %w", principal, err)
	}
	gid64, err := strconv.ParseInt(u.Gid, 10, 32)
	if err != nil {
		return fmt.Errorf("parse gid for %q: %w", principal, err)
	}
	excluded = filepath.Clean(excluded)
	return filepath.Walk(path, func(p string, _ os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if filepath.Clean(p) == excluded {
			return filepath.SkipDir
		}
		return os.Chown(p, int(uid64), int(gid64))
	})
}

func writeFileAtomic(path string, b []byte, mode os.FileMode) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".fak-service-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	ok := false
	defer func() {
		_ = tmp.Close()
		if !ok {
			_ = os.Remove(name)
		}
	}()
	if err = tmp.Chmod(mode); err != nil {
		return err
	}
	if _, err = tmp.Write(b); err != nil {
		return err
	}
	if err = tmp.Sync(); err != nil {
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	if err = os.Rename(name, path); err != nil {
		return err
	}
	ok = true
	return nil
}

var _ = strings.TrimSpace
