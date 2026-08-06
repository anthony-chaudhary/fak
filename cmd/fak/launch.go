package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

func cmdLaunch(argv []string) { os.Exit(runLaunch(os.Stdout, os.Stderr, argv)) }
func maybeLaunchDefault() bool {
	c, err := launchshim.Load()
	if err != nil || strings.TrimSpace(c.Default) == "" {
		return false
	}
	os.Exit(runLaunch(os.Stdout, os.Stderr, nil))
	return true
}

func runLaunch(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 {
		switch argv[0] {
		case "install":
			return runLaunchInstall(stdout, stderr, argv[1:])
		case "uninstall":
			return runLaunchUninstall(stdout, stderr, argv[1:])
		case "default":
			return runLaunchDefault(stdout, stderr, argv[1:])
		case "enable", "disable":
			return runLaunchToggle(stdout, stderr, argv[0] == "disable")
		case "status":
			return runLaunchStatus(stdout, stderr)
		case "doctor":
			return runLaunchDoctor(stdout, stderr, argv[1:])
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
	command := strings.TrimSpace(c.Providers[p].Command)
	if command == "" {
		command, err = exec.LookPath(p)
		if err != nil {
			fmt.Fprintf(stderr, "fak launch: underlying %s executable is not recorded; rerun 'fak launch install --provider %s'\n", p, p)
			return 1
		}
	}
	if launchshim.EffectiveDirect(c, *direct) {
		return runLaunchChild(stdout, stderr, command, args)
	}
	fak, err := os.Executable()
	if err != nil {
		fmt.Fprintf(stderr, "fak launch: resolve fak executable: %v\n", err)
		return 1
	}
	guardArgs := append([]string{"guard", "--"}, append([]string{command}, args...)...)
	return runLaunchChild(stdout, stderr, fak, guardArgs)
}

func runLaunchChild(stdout, stderr io.Writer, command string, args []string) int {
	cmd := exec.Command(command, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, stdout, stderr
	if err := cmd.Run(); err != nil {
		if e, ok := err.(*exec.ExitError); ok {
			return e.ExitCode()
		}
		fmt.Fprintf(stderr, "fak launch: %v\n", err)
		return 1
	}
	return 0
}

func launchBinDir() (string, error) {
	if p := strings.TrimSpace(os.Getenv("FAK_LAUNCH_BIN")); p != "" {
		return p, nil
	}
	d, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(d, "fak", "bin"), nil
}

func runLaunchInstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("launch install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	providerFlag := fs.String("provider", "all", "provider shim to install: claude, codex, or all")
	makeDefault := fs.String("default", "", "also make claude or codex the default for bare fak")
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
		c.Providers[p] = launchshim.Provider{Command: underlying}
		if err := writeLaunchShim(dir, p, fak); err != nil {
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
	fmt.Fprintf(stdout, "prepend %s to PATH (the provider binaries were not overwritten)\n", dir)
	return 0
}

func sameLaunchDir(path, dir string) bool {
	a, _ := filepath.Abs(filepath.Dir(path))
	b, _ := filepath.Abs(dir)
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
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
func runLaunchStatus(stdout, stderr io.Writer) int {
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
	fmt.Fprintf(stdout, "default: %s\ninterception: %s\n", firstNonEmpty(c.Default, "(unset)"), map[bool]string{true: "disabled", false: "enabled"}[c.Disabled])
	for _, k := range ks {
		fmt.Fprintf(stdout, "%s: %s\n", k, c.Providers[k].Command)
	}
	return 0
}
