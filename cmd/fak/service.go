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
	unitDir := fs.String("unit-dir", "", "override systemd user unit directory")
	stateDir := fs.String("state-dir", "", "durable control-plane state directory")
	if err := fs.Parse(args[1:]); err != nil || fs.NArg() != 0 {
		return 2
	}
	if runtime.GOOS != "linux" {
		fmt.Fprintln(stderr, "fak service: systemd user service is available on Linux")
		return 2
	}
	home, err := os.UserHomeDir()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *unitDir == "" {
		*unitDir = filepath.Join(home, ".config", "systemd", "user")
	}
	if *stateDir == "" {
		*stateDir = filepath.Join(home, ".local", "state", "fak")
	}
	path := filepath.Join(*unitDir, systemservice.SystemdUnitName)
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	unit, err := systemservice.RenderSystemdUserUnit(systemservice.SystemdConfig{Executable: exe, StateDir: *stateDir})
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	result := serviceResult{Manager: "systemd-user", Unit: systemservice.SystemdUnitName, Path: path}
	run := func(argv ...string) error {
		if *dry {
			return nil
		}
		cmd := serviceCommand("systemctl", append([]string{"--user"}, argv...)...)
		cmd.Stdout = stdout
		cmd.Stderr = stderr
		return cmd.Run()
	}
	switch action {
	case "install":
		if *dry {
			fmt.Fprint(stdout, unit)
		} else {
			if err := os.MkdirAll(*unitDir, 0o755); err != nil {
				return 1
			}
			if err := os.MkdirAll(*stateDir, 0o700); err != nil {
				return 1
			}
			if err := writeFileAtomic(path, []byte(unit), 0o644); err != nil {
				return 1
			}
			if run("daemon-reload") != nil || run("enable", "--now", systemservice.SystemdUnitName) != nil {
				return 1
			}
		}
	case "status":
		if err := run("is-active", systemservice.SystemdUnitName); err == nil {
			result.Active = true
		} else if !*dry {
			return 3
		}
	case "uninstall":
		if !*dry {
			_ = run("disable", "--now", systemservice.SystemdUnitName)
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				return 1
			}
			if run("daemon-reload") != nil {
				return 1
			}
		}
	default:
		serviceUsage(stderr)
		return 2
	}
	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(result)
	} else if action != "install" || !*dry {
		fmt.Fprintf(stdout, "%s %s %s\n", result.Manager, action, strings.TrimSpace(path))
	}
	return 0
}
func runServiceLoop(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("service run", flag.ContinueOnError)
	fs.SetOutput(stderr)
	interval := fs.Duration("interval", 15*time.Second, "control-plane tick interval")
	once := fs.Bool("once", false, "run one tick and exit")
	if err := fs.Parse(args); err != nil || fs.NArg() != 0 || *interval <= 0 {
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
	if err := tmp.Chmod(mode); err != nil {
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		return err
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, path); err != nil {
		return err
	}
	ok = true
	return nil
}
