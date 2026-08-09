package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/launchshim"
)

type repeatedLaunchArg []string

func (a *repeatedLaunchArg) String() string     { return strings.Join(*a, " ") }
func (a *repeatedLaunchArg) Set(v string) error { *a = append(*a, v); return nil }

var reservedLaunchAliases = map[string]bool{
	"agent": true, "bench": true, "boundary": true, "buildcheck": true,
	"commit": true, "concept": true, "dev": true, "doctor": true,
	"fleet": true, "go": true, "guard": true, "help": true,
	"hook": true, "launch": true, "merge": true, "model": true,
	"policy": true, "preflight": true, "replay": true, "route": true,
	"run": true, "serve": true, "session": true, "stats": true,
	"sweep": true, "sync": true, "version": true, "worktree": true,
}

func validateLaunchAlias(name string) (string, error) {
	name, err := launchshim.NormalizeProvider(name)
	if err != nil {
		return "", err
	}
	if reservedLaunchAliases[name] || name == "fak" {
		return "", fmt.Errorf("provider alias %q is reserved by fak", name)
	}
	return name, nil
}

func runLaunchAdd(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("launch add", flag.ContinueOnError)
	fs.SetOutput(stderr)
	command := fs.String("command", "", "provider executable path")
	makeDefault := fs.Bool("default", false, "make this alias the bare-fak default")
	installShim := fs.Bool("shim", false, "install a same-name shim")
	var template repeatedLaunchArg
	fs.Var(&template, "arg", "fixed provider argument (repeatable; no shell parsing)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 1 || strings.TrimSpace(*command) == "" {
		fmt.Fprintln(stderr, "fak launch add: NAME and --command PATH are required")
		return 2
	}
	name, err := validateLaunchAlias(fs.Arg(0))
	if err != nil {
		fmt.Fprintln(stderr, "fak launch add:", err)
		return 2
	}
	canonical, err := launchshim.CanonicalCommand(*command)
	if err != nil {
		fmt.Fprintln(stderr, "fak launch add:", err)
		return 1
	}
	c, err := launchshim.Load()
	if err != nil {
		fmt.Fprintln(stderr, "fak launch add:", err)
		return 1
	}
	if c.Providers == nil {
		c.Providers = map[string]launchshim.Provider{}
	}
	c.Providers[name] = launchshim.Provider{Command: canonical, Canonical: canonical, Args: append([]string(nil), template...), InstallShim: *installShim}
	if *makeDefault {
		c.Default = name
	}
	if err := launchshim.Save(c); err != nil {
		fmt.Fprintln(stderr, "fak launch add:", err)
		return 1
	}
	if *installShim {
		if err := os.MkdirAll(mustLaunchBinDir(), 0o755); err != nil {
			fmt.Fprintln(stderr, "fak launch add:", err)
			return 1
		}
		exe, err := os.Executable()
		if err != nil {
			fmt.Fprintln(stderr, "fak launch add:", err)
			return 1
		}
		stable, err := installStableLaunchTarget(mustLaunchBinDir(), exe)
		if err != nil {
			fmt.Fprintln(stderr, "fak launch add:", err)
			return 1
		}
		if err := writeLaunchShim(mustLaunchBinDir(), name, stable); err != nil {
			fmt.Fprintln(stderr, "fak launch add:", err)
			return 1
		}
	}
	fmt.Fprintf(stdout, "launch alias %s added\n", name)
	return 0
}

func runLaunchRemove(stdout, stderr io.Writer, argv []string) int {
	if len(argv) != 1 {
		fmt.Fprintln(stderr, "fak launch remove: NAME is required")
		return 2
	}
	name, err := validateLaunchAlias(argv[0])
	if err != nil {
		fmt.Fprintln(stderr, "fak launch remove:", err)
		return 2
	}
	c, err := launchshim.Load()
	if err != nil {
		fmt.Fprintln(stderr, "fak launch remove:", err)
		return 1
	}
	provider, ok := c.Providers[name]
	if !ok {
		fmt.Fprintf(stderr, "fak launch remove: alias %q is not configured\n", name)
		return 1
	}
	delete(c.Providers, name)
	if c.Default == name {
		c.Default = ""
	}
	if err := launchshim.Save(c); err != nil {
		fmt.Fprintln(stderr, "fak launch remove:", err)
		return 1
	}
	if provider.InstallShim {
		_ = os.Remove(filepath.Join(mustLaunchBinDir(), shimName(name)))
	}
	fmt.Fprintf(stdout, "launch alias %s removed\n", name)
	return 0
}

func runLaunchList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("launch list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit redacted JSON")
	if err := fs.Parse(argv); err != nil || fs.NArg() != 0 {
		return 2
	}
	c, err := launchshim.Load()
	if err != nil {
		fmt.Fprintln(stderr, "fak launch list:", err)
		return 1
	}
	names := make([]string, 0, len(c.Providers))
	for name := range c.Providers {
		names = append(names, name)
	}
	sort.Strings(names)
	if *jsonOut {
		type row struct {
			Name    string `json:"name"`
			Args    int    `json:"template_args"`
			Default bool   `json:"default"`
			Shim    bool   `json:"shim"`
		}
		rows := make([]row, 0, len(names))
		for _, name := range names {
			p := c.Providers[name]
			rows = append(rows, row{Name: name, Args: len(p.Args), Default: c.Default == name, Shim: p.InstallShim})
		}
		return encodeLaunchJSON(stdout, rows)
	}
	for _, name := range names {
		p := c.Providers[name]
		fmt.Fprintf(stdout, "%s\tdefault=%t\tshim=%t\ttemplate_args=%d\n", name, c.Default == name, p.InstallShim, len(p.Args))
	}
	return 0
}

func encodeLaunchJSON(w io.Writer, v any) int {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return 1
	}
	return 0
}
