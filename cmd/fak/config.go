package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/configguide"
)

func cmdConfig(argv []string) {
	os.Exit(runConfigGuide(os.Stdout, os.Stderr, argv))
}

func runConfigGuide(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "guide" {
		fmt.Fprintf(stderr, "usage: fak config guide [--posture %s] [--json|--write FILE] [intent overrides]\n", strings.Join(configguide.Names(), "|"))
		return 2
	}
	fs := flag.NewFlagSet("config guide", flag.ContinueOnError)
	fs.SetOutput(stderr)
	posture := fs.String("posture", "default", "intent posture: "+strings.Join(configguide.Names(), ", "))
	policy := fs.String("policy", "", "reviewed policy path for the hardened posture")
	keyEnv := fs.String("key-env", "", "environment-variable name containing the gateway token (never the secret value)")
	budget := fs.Int("budget", 0, "context-token budget for the long-session posture")
	bind := fs.String("bind", "", "listener address for team-gateway or hardened posture")
	asJSON := fs.Bool("json", false, "emit deterministic machine-readable guide result")
	writePath := fs.String("write", "", "write the generated minimal manifest (refuses an existing file unless --force)")
	force := fs.Bool("force", false, "replace an existing --write path")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak config guide: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}
	result, err := configguide.Guide(configguide.Options{
		Posture: *posture, PolicyPath: *policy, KeyEnv: *keyEnv, Budget: *budget, Bind: *bind,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak config guide: %v\n", err)
		return 2
	}
	if *writePath != "" {
		if !result.NeedsConfig {
			fmt.Fprintln(stderr, "fak config guide: the default posture needs no config file; nothing written")
			return 0
		}
		if !*force {
			if _, err := os.Stat(*writePath); err == nil {
				fmt.Fprintf(stderr, "fak config guide: %s already exists (use --force to overwrite)\n", *writePath)
				return 1
			}
		}
		if err := os.WriteFile(*writePath, []byte(result.Manifest), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak config guide: write %s: %v\n", *writePath, err)
			return 1
		}
		fmt.Fprintf(stdout, "wrote %s (%s posture, %d explained delta(s))\n", *writePath, result.Posture, len(result.Changes))
		return 0
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak config guide: encode: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "posture: %s\n%s\n", result.Posture, result.Summary)
	if !result.NeedsConfig {
		fmt.Fprintf(stdout, "\nNo config file required. Run: %s\n", result.Run)
		return 0
	}
	fmt.Fprintln(stdout, "\nMinimal fak.toml delta:")
	fmt.Fprint(stdout, result.Manifest)
	fmt.Fprintln(stdout, "\nWhy these values:")
	for _, change := range result.Changes {
		fmt.Fprintf(stdout, "- %s=%v — %s Equivalent flag: %s\n", change.Field, change.Value, change.Why, change.EquivalentFlag)
	}
	fmt.Fprintf(stdout, "\nRun: %s\n", result.Run)
	return 0
}
