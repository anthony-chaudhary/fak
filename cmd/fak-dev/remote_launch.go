// remote-launch - drive the servelaunch remote-inspection surface from the
// command line, so an operator who has to get into a benchmark host over ssh
// gets a saved, readable record instead of retyped scrollback.
//
// Two modes:
//
//	inspect (default)  probe the host, list its screen sessions, resolve the
//	                   named session (or report the roster), fetch that
//	                   session's log tail, print the reattach banner, and save
//	                   the receipt as JSON;
//	start              launch a command in a detached screen session, then run
//	                   the same inspect path against the new session.
//
// The knowledge this exposes is the port of the legacy Benchmark Python SSH
// transport (hardened argv, GNU-screen launch/log capture, crash-log tail, the
// ServerLaunchEvent receipt) into internal/servelaunch. The verb is the point
// of contact: when a remote server misbehaves and the only way in is ssh, this
// is the command that leaves behind a log you can read later.
//
// Exit codes: 0 success, 1 runtime/probe failure, 2 usage.

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/servelaunch"
)

const remoteLaunchUsage = "usage: fak-dev remote-launch [inspect|start] --host H [--server S] [--session NAME] [--log-dir D] [--command CMD] [--tail N] [--save PATH] [--user U] [--port P] [--key K] [--timeout SEC] [--json]"

// remoteLaunchExecRunner is the process seam the verb uses; a test swaps it to
// drive the whole inspect path against canned ssh output without a network.
var remoteLaunchExecRunner servelaunch.ExecRunner = servelaunch.ExecCommandRunner

func runRemoteLaunch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("remote-launch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	mode := "inspect"
	rest := argv
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		mode = argv[0]
		rest = argv[1:]
	}

	host := fs.String("host", "", "ssh host (IP or hostname) to reach")
	user := fs.String("user", "", "ssh user (default "+servelaunch.DefaultSSHUser+")")
	port := fs.Int("port", 0, "ssh port (default 22)")
	key := fs.String("key", "", "ssh private key path for -i")
	server := fs.String("server", "", "logical server name recorded in the receipt")
	session := fs.String("session", "", "screen session name to resolve (empty lists the roster)")
	logDir := fs.String("log-dir", servelaunch.DefaultScreenLogDir, "remote directory holding screen logs")
	command := fs.String("command", "", "command to launch under screen (start mode); ignored in inspect mode")
	keepAlive := fs.Bool("keep-alive", true, "start mode: keep the screen session up after the command exits")
	engine := fs.String("engine", "sglang", "engine tag used in the derived session name (start mode)")
	hwMonitor := fs.Bool("hw-monitor", false, "start mode: also start the paired hardware-monitor session")
	tail := fs.Int("tail", 0, "log-tail line count (0 = servelaunch default)")
	timeout := fs.Int("timeout", int(servelaunch.DefaultExecTimeout/time.Second), "per-command ssh timeout in seconds")
	save := fs.String("save", "", "write the receipt JSON here")
	asJSON := fs.Bool("json", false, "emit the receipt as JSON to stdout")
	if err := fs.Parse(rest); err != nil {
		return 2
	}
	switch mode {
	case "inspect", "start":
	default:
		fmt.Fprintf(stderr, "fak-dev remote-launch: unknown mode %q\n", mode)
		fmt.Fprintln(stderr, remoteLaunchUsage)
		return 2
	}
	if strings.TrimSpace(*host) == "" {
		fmt.Fprintln(stderr, "fak-dev remote-launch: --host is required")
		fmt.Fprintln(stderr, remoteLaunchUsage)
		return 2
	}
	if mode == "start" && strings.TrimSpace(*command) == "" {
		fmt.Fprintln(stderr, "fak-dev remote-launch start: --command is required")
		fmt.Fprintln(stderr, remoteLaunchUsage)
		return 2
	}

	t := servelaunch.Transport{
		Target: servelaunch.SSHTarget{
			Host: *host, User: *user, Port: *port, Key: *key,
		},
		Run:     remoteLaunchExecRunner,
		Timeout: time.Duration(*timeout) * time.Second,
	}
	ctx := context.Background()
	serverName := *server
	if serverName == "" {
		serverName = *host
	}

	if mode == "start" {
		plan, rc, code := launchRemoteSession(ctx, t, stderr, serverName, *session, *engine, *logDir, *command, *keepAlive, *hwMonitor)
		if code != 0 {
			return code
		}
		*session = plan.Name
		_ = rc
	}

	rc, err := servelaunch.Inspect(ctx, t, serverName, *session, *tail)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev remote-launch: %v\n", err)
		return 1
	}
	if err := emitReceipt(stdout, stderr, rc, *save, *asJSON); err != nil {
		fmt.Fprintf(stderr, "fak-dev remote-launch: %v\n", err)
		return 1
	}
	// A receipt that recorded probe failures is a runtime failure: the operator
	// needs a non-zero exit to notice the host did not answer cleanly.
	if len(rc.Errors) > 0 {
		return 1
	}
	return 0
}

// launchRemoteSession builds the plan, launches it, and (optionally) the paired
// hw-monitor session. It returns the resolved plan and a non-zero code when the
// launch itself failed.
func launchRemoteSession(ctx context.Context, t servelaunch.Transport, stderr io.Writer,
	server, session, engine, logDir, command string, keepAlive, hwMonitor bool) (servelaunch.SessionPlan, *servelaunch.Receipt, int) {
	ts := time.Now().Format("0102-1504")
	name := session
	if name == "" {
		derived, err := servelaunch.ResolveSessionName("bench", server, engine, ts)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev remote-launch: %v\n", err)
			return servelaunch.SessionPlan{}, nil, 2
		}
		name = derived
	}
	if strings.TrimSpace(logDir) == "" {
		logDir = servelaunch.DefaultScreenLogDir
	}
	plan := servelaunch.SessionPlan{
		Name:      name,
		LogDir:    logDir,
		LogPath:   logDir + "/" + name + ".log",
		Inner:     command,
		KeepAlive: keepAlive,
	}
	if _, err := t.LaunchSession(ctx, plan); err != nil {
		fmt.Fprintf(stderr, "fak-dev remote-launch: launch failed: %v\n", err)
		return plan, nil, 1
	}
	fmt.Fprintf(stderr, "fak-dev remote-launch: launched %s (log %s)\n", plan.Name, plan.LogPath)

	if hwMonitor {
		hwName, err := servelaunch.ResolveSessionName("bench", server, "hwmon", ts)
		if err != nil {
			fmt.Fprintf(stderr, "fak-dev remote-launch: hw-monitor name: %v\n", err)
		} else {
			hwPlan := servelaunch.SessionPlan{
				Name:    hwName,
				LogDir:  logDir,
				LogPath: logDir + "/" + hwName + ".log",
			}
			if _, err := t.LaunchSessionHwMonitor(ctx, hwPlan, 0); err != nil {
				fmt.Fprintf(stderr, "fak-dev remote-launch: hw-monitor launch failed: %v\n", err)
			} else {
				fmt.Fprintf(stderr, "fak-dev remote-launch: launched hw-monitor %s (log %s)\n", hwName, hwPlan.LogPath)
			}
		}
	}
	return plan, nil, 0
}

// emitReceipt prints the human banner (default) or the JSON document, and saves
// the receipt when a path is given. The save happens in both modes so the
// artifact exists regardless of how the operator read it.
func emitReceipt(stdout, stderr io.Writer, rc *servelaunch.Receipt, save string, asJSON bool) error {
	if asJSON {
		data, err := rc.Marshal()
		if err != nil {
			return err
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		if err := servelaunch.WriteBanner(stdout, rc); err != nil {
			return err
		}
	}
	if strings.TrimSpace(save) != "" {
		path := save
		if isDirLike(save) {
			path = filepath.Join(save, sanitizeName(rc.ServerName)+"-launch-receipt.json")
		}
		if _, err := rc.Save(path); err != nil {
			return err
		}
		fmt.Fprintf(stderr, "fak-dev remote-launch: receipt saved to %s\n", path)
	}
	return nil
}

// isDirLike reports whether a --save value names a directory (trailing
// separator, or an existing directory) rather than a file.
func isDirLike(path string) bool {
	if strings.HasSuffix(path, "/") || strings.HasSuffix(path, string(os.PathSeparator)) {
		return true
	}
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

// sanitizeName makes a server name safe for a filename.
func sanitizeName(s string) string {
	if strings.TrimSpace(s) == "" {
		return "server"
	}
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_', r == '.':
			return r
		default:
			return '-'
		}
	}, s)
}
