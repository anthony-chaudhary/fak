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
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/systemservice"
)

type serviceResult struct {
	Manager string `json:"manager"`
	Unit    string `json:"unit"`
	Path    string `json:"path,omitempty"`
	Active  bool   `json:"active,omitempty"`
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
	if err := fs.Parse(args[1:]); err != nil || fs.NArg() != 0 {
		return 2
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return 1
	}
	exe, err := os.Executable()
	if err != nil {
		return 1
	}
	var manager, name, path, definition string
	var install, status, uninstall func() error
	switch runtime.GOOS {
	case "linux":
		manager = "systemd-user"
		name = systemservice.SystemdUnitName
		if *unitDir == "" {
			*unitDir = filepath.Join(home, ".config", "systemd", "user")
		}
		if *stateDir == "" {
			*stateDir = filepath.Join(home, ".local", "state", "fak")
		}
		path = filepath.Join(*unitDir, name)
		definition, err = systemservice.RenderSystemdUserUnit(systemservice.SystemdConfig{Executable: exe, StateDir: *stateDir})
		cmd := func(argv ...string) error {
			c := serviceCommand("systemctl", append([]string{"--user"}, argv...)...)
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
		manager = "launchd-user"
		name = systemservice.LaunchdLabel
		if *unitDir == "" {
			*unitDir = filepath.Join(home, "Library", "LaunchAgents")
		}
		if *stateDir == "" {
			*stateDir = filepath.Join(home, "Library", "Application Support", "fak")
		}
		logDir := filepath.Join(*stateDir, "logs")
		path = filepath.Join(*unitDir, name+".plist")
		definition, err = systemservice.RenderLaunchAgent(systemservice.LaunchdConfig{Executable: exe, StateDir: *stateDir, StdoutPath: filepath.Join(logDir, "guard-control.out.log"), StderrPath: filepath.Join(logDir, "guard-control.err.log")})
		domain := "gui/" + strconv.Itoa(os.Getuid())
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
	result := serviceResult{Manager: manager, Unit: name, Path: path}
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
		if os.MkdirAll(*unitDir, 0o755) != nil || os.MkdirAll(*stateDir, 0o700) != nil || os.MkdirAll(filepath.Join(*stateDir, "logs"), 0o700) != nil {
			return 1
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
