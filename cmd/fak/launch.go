package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

func cmdLaunch(argv []string) { os.Exit(runLaunch(os.Stdout, os.Stderr, argv)) }

var (
	osExecutableForLaunch           = os.Executable
	launchInput           io.Reader = os.Stdin
	stableLaunchResolve             = launchshim.ResolveExecutable
)

func maybeLaunchDefault() bool {
	c, err := launchshim.Load()
	if err != nil || strings.TrimSpace(c.Default) == "" {
		return false
	}
	os.Exit(runLaunch(os.Stdout, os.Stderr, nil))
	return true
}

func runLaunch(stdout, stderr io.Writer, argv []string) int {
	if delegated, code := delegateStableLaunch(stdout, stderr, argv); delegated {
		return code
	}
	if len(argv) > 0 {
		switch argv[0] {
		case "install":
			return runLaunchInstall(stdout, stderr, argv[1:])
		case "add":
			return runLaunchAdd(stdout, stderr, argv[1:])
		case "remove":
			return runLaunchRemove(stdout, stderr, argv[1:])
		case "list":
			return runLaunchList(stdout, stderr, argv[1:])
		case "uninstall":
			return runLaunchUninstall(stdout, stderr, argv[1:])
		case "default":
			return runLaunchDefault(stdout, stderr, argv[1:])
		case "enable", "disable":
			return runLaunchToggle(stdout, stderr, argv[0] == "disable")
		case "status":
			return runLaunchStatus(stdout, stderr, argv[1:])
		case "doctor":
			return runLaunchDoctor(stdout, stderr, argv[1:])
		case "stats":
			return runLaunchStats(stdout, stderr, argv[1:])
		case "help", "-h", "--help":
			fmt.Fprint(stdout, launchHelpText)
			return 0
		}
	}
	fs := flag.NewFlagSet("launch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	direct := fs.Bool("direct", false, "run the underlying provider without fak")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	args := fs.Args()
	c, err := launchshim.Load()
	if err != nil {
		fmt.Fprintf(stderr, "fak launch: %v\n", err)
		return 1
	}
	provider := ""
	if len(args) > 0 {
		if p, e := launchshim.NormalizeProvider(args[0]); e == nil {
			provider, args = p, args[1:]
		}
	}
	if provider == "" {
		provider = c.Default
	}
	if len(args) > 0 && args[0] == "--" {
		args = args[1:]
	}
	if provider == "" {
		fmt.Fprintln(stderr, "fak launch: no provider selected; run 'fak launch default claude|codex'")
		return 2
	}
	p, err := launchshim.NormalizeProvider(provider)
	if err != nil {
		fmt.Fprintf(stderr, "fak launch: %v\n", err)
		return 2
	}
	if len(args) > 0 && args[0] == "--fak-direct" {
		*direct, args = true, args[1:]
	}
	pconf := c.Providers[p]
	command := strings.TrimSpace(pconf.Command)
	if command == "" {
		command, err = exec.LookPath(p)
		if err != nil {
			fmt.Fprintf(stderr, "fak launch: underlying %s executable is not recorded; rerun 'fak launch install --provider %s'\n", p, p)
			return 1
		}
	}
	args = append(append([]string(nil), pconf.Args...), args...)
	directMode := launchshim.EffectiveDirect(c, *direct)
	surface := "explicit"
	if len(argv) == 0 {
		surface = "bare"
	}
	started := time.Now()
	if directMode {
		code := launchChildRunner(launchInput, stdout, stderr, command, args)
		_ = launchshim.Record(surface, p, "direct", launchOutcome(code), time.Since(started))
		return code
	}
	fak, err := os.Executable()
	if err != nil {
		fmt.Fprintf(stderr, "fak launch: resolve fak executable: %v\n", err)
		return 1
	}
	guardArgs := []string{"guard"}
	// The launch front door already knows whether the provider invocation is an
	// attended UI. Preserve that fact across shims/wrappers: those can make fak's
	// stdin fail the terminal probe even though Claude/Codex is about to own the
	// terminal, which made guard's auto mode print the full startup wall. Animate
	// degrades to the compact three-line render when stderr is not a color TTY.
	if guardChildInteractive(append([]string{command}, args...)) {
		guardArgs = append(guardArgs, "--banner=animate")
	}
	guardArgs = append(guardArgs, "--", command)
	guardArgs = append(guardArgs, args...)
	code := launchChildRunner(launchInput, stdout, stderr, fak, guardArgs)
	_ = launchshim.Record(surface, p, "guarded", launchOutcome(code), time.Since(started))
	return code
}

// delegateStableLaunch keeps provider shims pinned to an executable that is
// never itself replaced. It selects the deployed/prior binary at the last
// possible moment, so ordinary launches participate in the update transaction.
func delegateStableLaunch(stdout, stderr io.Writer, argv []string) (bool, int) {
	exe, err := osExecutableForLaunch()
	if err != nil || !strings.HasPrefix(strings.ToLower(filepath.Base(exe)), "fak-launch") {
		return false, 0
	}
	c, err := launchshim.Load()
	if err != nil {
		fmt.Fprintln(stderr, "fak launch:", err)
		return true, 1
	}
	target, bindErr := launchshim.StableExecutable(exe)
	if errors.Is(bindErr, os.ErrNotExist) {
		target = strings.TrimSpace(c.Executable)
		bindErr = nil
	}
	if bindErr != nil {
		fmt.Fprintln(stderr, "fak launch:", bindErr)
		return true, 1
	}
	if strings.TrimSpace(target) == "" || samePath(exe, target) {
		return false, 0
	}
	forward, policyFlag, waitFlag, err := stableLaunchFlags(argv)
	if err != nil {
		fmt.Fprintln(stderr, "fak launch:", err)
		return true, 2
	}
	policy, wait, err := launchshim.UpdatePolicy(c, policyFlag, waitFlag)
	if err != nil {
		fmt.Fprintln(stderr, "fak launch:", err)
		return true, 2
	}
	selected, err := stableLaunchResolve(target, policy, wait)
	if err != nil {
		fmt.Fprintln(stderr, "fak launch:", err)
		return true, 75
	}
	forward = append([]string{"launch"}, forward...)
	return true, launchChildRunner(launchInput, stdout, stderr, selected, forward)
}

func stableLaunchFlags(argv []string) (forward []string, policy, wait string, err error) {
	forward = make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "--" || !strings.HasPrefix(arg, "-") {
			forward = append(forward, argv[i:]...)
			return forward, policy, wait, nil
		}
		switch {
		case arg == "--update-launch-policy":
			if i+1 >= len(argv) {
				return nil, "", "", errors.New("--update-launch-policy requires prior, wait, or fail")
			}
			policy, i = argv[i+1], i+1
		case strings.HasPrefix(arg, "--update-launch-policy="):
			policy = strings.TrimPrefix(arg, "--update-launch-policy=")
		case arg == "--update-launch-wait":
			if i+1 >= len(argv) {
				return nil, "", "", errors.New("--update-launch-wait requires a positive Go duration")
			}
			wait, i = argv[i+1], i+1
		case strings.HasPrefix(arg, "--update-launch-wait="):
			wait = strings.TrimPrefix(arg, "--update-launch-wait=")
		default:
			// Existing launch flags such as --direct remain byte-for-byte in
			// the child argv while update flags may follow them.
			forward = append(forward, arg)
		}
	}
	return forward, policy, wait, nil
}

var launchChildRunner = runLaunchChild

func runLaunchChild(stdin io.Reader, stdout, stderr io.Writer, command string, args []string) int {
	cmd := exec.Command(command, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = stdin, stdout, stderr
	if err := cmd.Run(); err != nil {
		if e, ok := err.(*exec.ExitError); ok {
			return e.ExitCode()
		}
		fmt.Fprintf(stderr, "fak launch: %v\n", err)
		return 1
	}
	return 0
}

func launchBinDir() (string, error) { return launchshim.BinDir() }

func mustLaunchBinDir() string {
	dir, _ := launchBinDir()
	return dir
}

func runLaunchInstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("launch install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	providerFlag := fs.String("provider", "all", "provider shim to install: claude, codex, or all")
	makeDefault := fs.String("default", "", "also make claude or codex the default for bare fak")
	noPath := fs.Bool("no-path", false, "do not edit supported shell startup files")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	providers := []string{"claude", "codex"}
	if *providerFlag != "all" {
		p, err := launchshim.NormalizeProvider(*providerFlag)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		providers = []string{p}
	}
	c, err := launchshim.Load()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	dir, err := launchBinDir()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fak, err := os.Executable()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	installed := 0
	stable, err := installStableLaunchTarget(dir, fak)
	if err != nil {
		fmt.Fprintln(stderr, "fak launch install:", err)
		return 1
	}
	if !samePath(stable, fak) {
		c.Executable = fak
	} else if strings.TrimSpace(c.Executable) == "" {
		fmt.Fprintln(stderr, "fak launch install: stable launcher has no deployed fak target; run install from the deployed fak executable")
		return 1
	}
	for _, p := range providers {
		underlying, err := exec.LookPath(p)
		if err != nil {
			fmt.Fprintf(stderr, "fak launch install: skip %s: not found on PATH\n", p)
			continue
		}
		if sameLaunchDir(underlying, dir) {
			if old := c.Providers[p].Command; old != "" {
				underlying = old
			} else {
				fmt.Fprintf(stderr, "fak launch install: %s already resolves to shim and no underlying command is recorded\n", p)
				continue
			}
		}
		c.Providers[p] = launchshim.Provider{Command: underlying, InstallShim: true}
		if err := writeLaunchShim(dir, p, stable); err != nil {
			fmt.Fprintf(stderr, "fak launch install: %v\n", err)
			return 1
		}
		installed++
		fmt.Fprintf(stdout, "installed %s shim -> fak launch %s\n", filepath.Join(dir, shimName(p)), p)
	}
	if *makeDefault != "" {
		p, err := launchshim.NormalizeProvider(*makeDefault)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		c.Default = p
	}
	if err := launchshim.Save(c); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if installed == 0 {
		return 1
	}
	if !*noPath {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Fprintln(stderr, "fak launch install PATH:", err)
			return 1
		}
		for _, profile := range launchshim.Profiles(home, runtime.GOOS) {
			if _, err := launchshim.Activate(profile, dir); err != nil {
				fmt.Fprintf(stderr, "fak launch install PATH %s: %v\n", launchshim.ProfileSummary(profile), err)
				return 1
			}
			fmt.Fprintf(stdout, "activated PATH in %s\n", profile.Path)
			fmt.Fprintf(stdout, "current shell: %s\n", launchshim.CurrentShellCommand(profile.Shell, dir))
		}
	} else {
		fmt.Fprintf(stdout, "prepend %s to PATH (the provider binaries were not overwritten)\n", dir)
	}
	return 0
}

func sameLaunchDir(path, dir string) bool {
	a, _ := filepath.Abs(filepath.Dir(path))
	b, _ := filepath.Abs(dir)
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
}
func stableLaunchTarget(dir, fak string) string {
	if filepath.Clean(filepath.Dir(fak)) == filepath.Clean(dir) && strings.HasPrefix(strings.ToLower(filepath.Base(fak)), "fak-launch") {
		return fak
	}
	name := "fak-launch"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(dir, name)
}

func installStableLaunchTarget(dir, fak string) (string, error) {
	target := stableLaunchTarget(dir, fak)
	if samePath(target, fak) {
		return target, nil
	}
	raw, err := os.ReadFile(fak)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(target, raw, 0o755); err != nil {
		return "", err
	}
	if err := launchshim.BindStableExecutable(target, fak); err != nil {
		return "", err
	}
	return target, nil
}

func shimName(provider string) string {
	if runtime.GOOS == "windows" {
		return provider + ".cmd"
	}
	return provider
}
func writeLaunchShim(dir, provider, fak string) error {
	path := filepath.Join(dir, shimName(provider))
	var body string
	if runtime.GOOS == "windows" {
		body = "@echo off\r\n\"" + fak + "\" launch " + provider + " -- %*\r\n"
	} else {
		body = "#!/bin/sh\nexec \"" + strings.ReplaceAll(fak, "\"", "\\\"") + "\" launch " + provider + " -- \"$@\"\n"
	}
	return os.WriteFile(path, []byte(body), 0o755)
}

func runLaunchUninstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("launch uninstall", flag.ContinueOnError)
	fs.SetOutput(stderr)
	pflag := fs.String("provider", "all", "claude, codex, or all")
	if fs.Parse(argv) != nil {
		return 2
	}
	ps := []string{"claude", "codex"}
	if *pflag != "all" {
		p, e := launchshim.NormalizeProvider(*pflag)
		if e != nil {
			fmt.Fprintln(stderr, e)
			return 2
		}
		ps = []string{p}
	}
	dir, e := launchBinDir()
	if e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	c, e := launchshim.Load()
	if e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	for _, p := range ps {
		_ = os.Remove(filepath.Join(dir, shimName(p)))
		delete(c.Providers, p)
		if c.Default == p {
			c.Default = ""
		}
		fmt.Fprintf(stdout, "removed %s shim\n", p)
	}
	if e = launchshim.Save(c); e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	home, err := os.UserHomeDir()
	if err == nil {
		for _, profile := range launchshim.Profiles(home, runtime.GOOS) {
			if _, err := launchshim.Deactivate(profile); err != nil {
				fmt.Fprintf(stderr, "fak launch uninstall PATH %s: %v\n", launchshim.ProfileSummary(profile), err)
				return 1
			}
		}
	}
	return 0
}
func runLaunchDefault(stdout, stderr io.Writer, argv []string) int {
	if len(argv) != 1 {
		fmt.Fprintln(stderr, "usage: fak launch default claude|codex")
		return 2
	}
	p, e := launchshim.NormalizeProvider(argv[0])
	if e != nil {
		fmt.Fprintln(stderr, e)
		return 2
	}
	c, e := launchshim.Load()
	if e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	if _, ok := c.Providers[p]; !ok {
		fmt.Fprintf(stderr, "fak launch default: provider %q is not configured; run 'fak launch add %s --command PATH' first\n", p, p)
		return 1
	}
	c.Default = p
	if e = launchshim.Save(c); e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	fmt.Fprintf(stdout, "default provider: %s\n", p)
	return 0
}
func runLaunchToggle(stdout, stderr io.Writer, disabled bool) int {
	c, e := launchshim.Load()
	if e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	c.Disabled = disabled
	if e = launchshim.Save(c); e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	if disabled {
		fmt.Fprintln(stdout, "fak provider interception disabled; shims now pass through directly")
	} else {
		fmt.Fprintln(stdout, "fak provider interception enabled")
	}
	return 0
}
func runLaunchStatus(stdout, stderr io.Writer, argv []string) int {
	if len(argv) != 0 {
		fmt.Fprintf(stderr, "fak launch status: unexpected argument %q\n", argv[0])
		return 2
	}
	c, e := launchshim.Load()
	if e != nil {
		fmt.Fprintln(stderr, e)
		return 1
	}
	ks := make([]string, 0, len(c.Providers))
	for k := range c.Providers {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	interception := "active"
	if c.Disabled {
		interception = "disabled"
	} else if len(c.Providers) == 0 {
		interception = "inactive (no configured providers)"
	}
	fmt.Fprintf(stdout, "default: %s\ninterception: %s\nbuild: %s\n", firstNonEmpty(c.Default, "(unset)"), interception, firstNonEmpty(buildIdentityFromRuntime().CommitShort, "unknown"))
	for _, k := range ks {
		fmt.Fprintf(stdout, "%s: %s\n", k, c.Providers[k].Command)
	}
	return 0
}

func launchOutcome(code int) string {
	if code == 0 {
		return "success"
	}
	if code > 1 {
		return "provider_exit"
	}
	return "launch_error"
}
func runLaunchStats(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("launch stats", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit versioned JSON")
	reset := fs.Bool("reset", false, "reset aggregate counters")
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 {
		return 2
	}
	if *reset {
		if err := launchshim.ResetStats(); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, "launch stats reset")
		return 0
	}
	stats, err := launchshim.ReadStats()
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, stats, "fak launch stats")
	}
	fmt.Fprintf(stdout, "LAUNCH STATS schema=%s counters=%d\n", stats.Schema, len(stats.Counters))
	for _, c := range stats.Counters {
		fmt.Fprintf(stdout, "  %s %s %s %s %s %d\n", c.Surface, c.Provider, c.Posture, c.Outcome, c.Latency, c.Count)
	}
	return 0
}
