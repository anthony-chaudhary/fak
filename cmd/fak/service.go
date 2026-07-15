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
	"github.com/anthony-chaudhary/fak/internal/systemservice"
)

type serviceResult struct {
	Manager    string `json:"manager"`
	Unit       string `json:"unit"`
	Path       string `json:"path,omitempty"`
	Active     bool   `json:"active,omitempty"`
	Executable string `json:"executable,omitempty"`
	PIDBefore  uint32 `json:"pid_before,omitempty"`
	PIDAfter   uint32 `json:"pid_after,omitempty"`
	StateKept  bool   `json:"state_kept,omitempty"`
}

var serviceCommand = exec.Command
var serviceTick = func(stdout, stderr io.Writer) int {
	return runResumeWatchdog(stdout, stderr, []string{"--live", "--json"})
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
		if *unitDir == "" {
			*unitDir = filepath.Join(string(filepath.Separator), "etc", "systemd", "system")
		}
		if *stateDir == "" {
			*stateDir = filepath.Join(string(filepath.Separator), "var", "lib", "fak")
		}
		if *execPath == "" {
			*execPath = filepath.Join(string(filepath.Separator), "usr", "local", "libexec", "fak-guard-control")
		}
		stagedExecutable = *execPath
		path = filepath.Join(*unitDir, name)
		definition, err = systemservice.RenderSystemdSystemUnit(systemservice.SystemdConfig{Executable: stagedExecutable, StateDir: *stateDir})
		cmd := func(argv ...string) error {
			c := serviceCommand("systemctl", argv...)
			c.Stdout = stdout
			c.Stderr = stderr
			return c.Run()
		}
		install = func() error {
			if e := cmd("daemon-reload"); e != nil {
				return e
			}
			return cmd("enable", "--now", name)
		}
		status = func() error { return cmd("is-active", name) }
		uninstall = func() error { _ = cmd("disable", "--now", name); return cmd("daemon-reload") }
	case "darwin":
		manager = "launchd-system"
		name = systemservice.LaunchdLabel
		if *unitDir == "" {
			*unitDir = filepath.Join(string(filepath.Separator), "Library", "LaunchDaemons")
		}
		if *stateDir == "" {
			*stateDir = filepath.Join(string(filepath.Separator), "var", "db", "fak")
		}
		if *principal == "" {
			*principal = "nobody"
		}
		if *execPath == "" {
			*execPath = filepath.Join(string(filepath.Separator), "usr", "local", "libexec", "fak-guard-control")
		}
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
		if status() == nil {
			result.Active = true
		} else {
			return 3
		}
	case "uninstall":
		_ = uninstall()
		if e := os.Remove(path); e != nil && !os.IsNotExist(e) {
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
func runServiceLoopContext(ctx context.Context, stdout, stderr io.Writer, interval time.Duration) int {
	for {
		if rc := serviceTick(stdout, stderr); rc != 0 {
			return rc
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
	if fs.Parse(args) != nil || fs.NArg() != 0 || *interval <= 0 {
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	for {
		if rc := serviceTick(stdout, stderr); rc != 0 {
			return rc
		}
		if *once {
			return 0
		}
		timer := time.NewTimer(*interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return 0
		case <-timer.C:
		}
	}
}
func serviceUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak service install|status|uninstall|run [--dry-run] [--json]")
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
