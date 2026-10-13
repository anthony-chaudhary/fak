package main

// fak router up (fak-private#3348): one verb that starts the always-on routing
// gateway from the ~/.fak/router layout (%LOCALAPPDATA%\fak\router on
// Windows). It derives the `fak-server route` launch line, creates a missing
// gateway.key (0600), asks fak-server to write a default routing.json only when
// none exists, exits 0 when a healthy router already answers on the address,
// and otherwise runs fak-server in the foreground with signals forwarded, so a
// service manager (fak-private#3347) can supervise it directly. The manual
// local-engine server (`fak serve`, `fak-server serve --local-engine`) is a
// separate path and is not touched.

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

const (
	routerDefaultAddr = "127.0.0.1:18095"
	// exitRouterConfig is EX_CONFIG: a config error a service should not
	// restart into. fak-server route uses the same code.
	exitRouterConfig = 78
)

// routerUpDeps are the side effects runRouterUp performs, injectable for tests.
type routerUpDeps struct {
	getenv   func(string) string
	homeDir  func() (string, error)
	goos     string
	lookPath func(string) (string, error)
	healthy  func(addr string) bool
	// run executes argv in the foreground with stdout/stderr attached and
	// returns its exit code.
	run func(argv []string, stdout, stderr io.Writer) (int, error)
}

func defaultRouterUpDeps() routerUpDeps {
	return routerUpDeps{getenv: os.Getenv, homeDir: os.UserHomeDir, goos: runtime.GOOS,
		lookPath: exec.LookPath, healthy: routerHealthy, run: runForegroundForwarding}
}

func cmdRouter(argv []string) { os.Exit(runRouter(os.Stdout, os.Stderr, argv, defaultRouterUpDeps())) }

func runRouter(stdout, stderr io.Writer, argv []string, d routerUpDeps) int {
	if len(argv) == 0 || argv[0] == "-h" || argv[0] == "--help" || argv[0] == "help" {
		fmt.Fprintln(stderr, "usage: fak router up [--addr 127.0.0.1:18095] [--home DIR] [--log-file PATH] [--dry-run]")
		if len(argv) == 0 {
			return 2
		}
		return 0
	}
	switch argv[0] {
	case "up":
		return runRouterUp(stdout, stderr, argv[1:], d)
	default:
		fmt.Fprintf(stderr, "fak router: unknown subcommand %q (want: up)\n", argv[0])
		return 2
	}
}

// routerHome is the router's state directory: %LOCALAPPDATA%\fak\router on
// Windows, ~/.fak/router elsewhere.
func routerHome(d routerUpDeps) (string, error) {
	if d.goos == "windows" {
		if la := strings.TrimSpace(d.getenv("LOCALAPPDATA")); la != "" {
			return filepath.Join(la, "fak", "router"), nil
		}
	}
	h, err := d.homeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(h, ".fak", "router"), nil
}

// resolveFakServerBin prefers the router's pinned binary under <home>/bin,
// then fak-server on PATH.
func resolveFakServerBin(home string, d routerUpDeps) (string, error) {
	name := "fak-server"
	if d.goos == "windows" {
		name += ".exe"
	}
	pinned := filepath.Join(home, "bin", name)
	if info, err := os.Stat(pinned); err == nil && !info.IsDir() {
		return pinned, nil
	}
	if p, err := d.lookPath("fak-server"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("fak-server not found at %s or on PATH", pinned)
}

// ensureGatewayKey creates a random gateway key with mode 0600 when the file
// is missing. An existing key is never rewritten. created reports a write.
func ensureGatewayKey(path string) (created bool, err error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return false, err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return false, err
	}
	if _, err := io.WriteString(f, hex.EncodeToString(buf)+"\n"); err != nil {
		_ = f.Close()
		return true, err
	}
	return true, f.Close()
}

// routerServerArgs is the derived `fak-server route` launch line.
func routerServerArgs(bin, home, addr string) []string {
	return []string{bin, "route",
		"--gateway-key-file", filepath.Join(home, "gateway.key"),
		"--usage-receipts", filepath.Join(home, "usage.jsonl"),
		"--addr", addr,
		"--routing-config", filepath.Join(home, "routing.json"),
		"--routing-journal", filepath.Join(home, "journal.jsonl"),
		"--init-routing-config",
	}
}

// routerHealthy reports whether a router already answers /healthz with 200.
func routerHealthy(addr string) bool {
	c := &http.Client{Timeout: 2 * time.Second}
	resp, err := c.Get("http://" + addr + "/healthz")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode == http.StatusOK
}

func runRouterUp(stdout, stderr io.Writer, argv []string, d routerUpDeps) int {
	fs := flag.NewFlagSet("router up", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", routerDefaultAddr, "router listen address")
	homeFlag := fs.String("home", "", "router state directory (default ~/.fak/router; %LOCALAPPDATA%\\fak\\router on Windows)")
	logFile := fs.String("log-file", "", "append fak-server stdout/stderr to this file instead of the terminal")
	dryRun := fs.Bool("dry-run", false, "print the derived fak-server launch line and exit without starting or writing anything")
	if err := fs.Parse(argv); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	home := *homeFlag
	if home == "" {
		h, err := routerHome(d)
		if err != nil {
			fmt.Fprintf(stderr, "fak router up: config error: %v\n", err)
			return exitRouterConfig
		}
		home = h
	}
	bin, err := resolveFakServerBin(home, d)
	if err != nil {
		if !*dryRun {
			fmt.Fprintf(stderr, "fak router up: config error: %v\n", err)
			return exitRouterConfig
		}
		bin = "fak-server"
	}
	args := routerServerArgs(bin, home, *addr)
	if *dryRun {
		fmt.Fprintln(stdout, shellQuoteArgs(args))
		return 0
	}
	if d.healthy(*addr) {
		fmt.Fprintf(stdout, "fak router up: a healthy router is already serving on %s\n", *addr)
		return 0
	}
	keyPath := filepath.Join(home, "gateway.key")
	if created, err := ensureGatewayKey(keyPath); err != nil {
		fmt.Fprintf(stderr, "fak router up: config error: gateway key %s: %v\n", keyPath, err)
		return exitRouterConfig
	} else if created {
		fmt.Fprintf(stderr, "fak router up: created %s (0600)\n", keyPath)
	}
	out, errw := stdout, stderr
	if *logFile != "" {
		f, err := os.OpenFile(*logFile, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
		if err != nil {
			fmt.Fprintf(stderr, "fak router up: config error: log file: %v\n", err)
			return exitRouterConfig
		}
		defer f.Close()
		out, errw = f, f
	}
	code, err := d.run(args, out, errw)
	if err != nil {
		fmt.Fprintf(stderr, "fak router up: %v\n", err)
		if code == 0 {
			code = 1
		}
	}
	return code
}

func shellQuoteArgs(args []string) string {
	q := make([]string, len(args))
	for i, a := range args {
		if a == "" || strings.ContainsAny(a, " \t'\"\\$") {
			a = "'" + strings.ReplaceAll(a, "'", `'\''`) + "'"
		}
		q[i] = a
	}
	return strings.Join(q, " ")
}

// runForegroundForwarding runs argv as a child in the foreground and forwards
// interrupt/terminate to it, returning the child's own exit code so a 78
// config error reaches the service manager unchanged.
func runForegroundForwarding(argv []string, stdout, stderr io.Writer) (int, error) {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = nil, stdout, stderr
	if err := cmd.Start(); err != nil {
		return exitRouterConfig, fmt.Errorf("start %s: %w", argv[0], err)
	}
	sigs := make(chan os.Signal, 2)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigs)
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	for {
		select {
		case s := <-sigs:
			_ = cmd.Process.Signal(s)
		case err := <-done:
			var ee *exec.ExitError
			if errors.As(err, &ee) {
				return ee.ExitCode(), nil
			}
			if err != nil {
				return 1, err
			}
			return 0, nil
		}
	}
}
