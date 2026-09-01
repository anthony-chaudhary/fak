package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/launchguard"
)

func cmdLaunchguard(args []string) { os.Exit(runLaunchguard(os.Stdout, os.Stderr, args)) }

func runLaunchguard(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "usage: fak launchguard <status|reset> --identity ID [--json]")
		return 2
	}
	fs := flag.NewFlagSet("launchguard "+args[0], flag.ContinueOnError)
	fs.SetOutput(stderr)
	identity := fs.String("identity", "", "stable launch identity")
	jsonOut := fs.Bool("json", false, "emit JSON")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	if *identity == "" {
		fmt.Fprintln(stderr, "launchguard: --identity is required")
		return 2
	}
	g, err := defaultLaunchguard()
	if err != nil {
		fmt.Fprintf(stderr, "launchguard: %v\n", err)
		return 1
	}
	switch args[0] {
	case "status":
		status, err := g.Inspect(*identity)
		if err != nil {
			fmt.Fprintf(stderr, "launchguard status: %v\n", err)
			return 1
		}
		if *jsonOut {
			if err := json.NewEncoder(stdout).Encode(status); err != nil {
				fmt.Fprintf(stderr, "launchguard status: %v\n", err)
				return 1
			}
			return 0
		}
		fmt.Fprintf(stdout, "identity=%s attempts=%d/%d active=%t quarantined=%t", status.Identity, status.Attempts, status.MaxAttempts, status.Active, status.Quarantined)
		if !status.LastFailure.IsZero() {
			fmt.Fprintf(stdout, " last_failure=%s", status.LastFailure.UTC().Format(time.RFC3339))
		}
		fmt.Fprintln(stdout)
		return 0
	case "reset":
		if err := g.Reset(*identity); err != nil {
			fmt.Fprintf(stderr, "launchguard reset: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "reset identity=%s\n", launchguard.StableIdentity(*identity))
		return 0
	default:
		fmt.Fprintf(stderr, "launchguard: unknown subcommand %q\n", args[0])
		return 2
	}
}

func defaultLaunchguard() (*launchguard.Guard, error) {
	root := os.Getenv("FAK_LAUNCHGUARD_DIR")
	if root == "" {
		cache, err := os.UserCacheDir()
		if err != nil {
			return nil, err
		}
		root = filepath.Join(cache, "fak", "launchguard")
	}
	return launchguard.New(launchguard.Config{Dir: root, MaxAttempts: 3, Window: 10 * time.Minute, BaseBackoff: 5 * time.Second, MaxBackoff: time.Minute, StaleAfter: 15 * time.Minute})
}
