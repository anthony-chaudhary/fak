package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

type routeInitReceipt struct {
	Profile      string `json:"profile"`
	OutDir       string `json:"out_dir"`
	AccountsPath string `json:"accounts_path"`
	ManifestPath string `json:"manifest_path"`
	NextCommand  string `json:"next_command"`
	StatusCheck  string `json:"status_check"`
}

func runRouteInit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("route init", flag.ContinueOnError)
	fs.SetOutput(stderr)
	profile := fs.String("profile", "hybrid", "starter routing profile: hybrid|local-only|multi-cloud")
	outDir := fs.String("out-dir", ".fak-router", "directory where accounts.json and manifest.json are written")
	force := fs.Bool("force", false, "overwrite existing accounts.json and manifest.json")
	asJSON := fs.Bool("json", false, "emit initialization receipt as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fak route init: unexpected argument(s): %s\n", strings.Join(fs.Args(), " "))
		return 2
	}

	dir := strings.TrimSpace(*outDir)
	if dir == "" {
		fmt.Fprintln(stderr, "fak route init: --out-dir must not be empty")
		return 2
	}

	roster, manifest, err := routeInitProfile(strings.TrimSpace(*profile))
	if err != nil {
		fmt.Fprintf(stderr, "fak route init: %v\n", err)
		return 2
	}

	accountsPath := filepath.Join(dir, "accounts.json")
	manifestPath := filepath.Join(dir, "manifest.json")
	if !*force {
		for _, p := range []string{accountsPath, manifestPath} {
			if _, statErr := os.Stat(p); statErr == nil {
				fmt.Fprintf(stderr, "fak route init: %s already exists (pass --force to overwrite)\n", p)
				return 1
			} else if !os.IsNotExist(statErr) {
				fmt.Fprintf(stderr, "fak route init: stat %s: %v\n", p, statErr)
				return 1
			}
		}
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintf(stderr, "fak route init: create %s: %v\n", dir, err)
		return 1
	}
	if err := os.WriteFile(accountsPath, roster.JSON(), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak route init: write %s: %v\n", accountsPath, err)
		return 1
	}
	if err := os.WriteFile(manifestPath, manifest.JSON(), 0o644); err != nil {
		fmt.Fprintf(stderr, "fak route init: write %s: %v\n", manifestPath, err)
		return 1
	}

	if _, err := modelroute.LoadRoster(accountsPath); err != nil {
		fmt.Fprintf(stderr, "fak route init: validate %s: %v\n", accountsPath, err)
		return 1
	}
	if _, err := modelroute.LoadManifest(manifestPath); err != nil {
		fmt.Fprintf(stderr, "fak route init: validate %s: %v\n", manifestPath, err)
		return 1
	}

	accountsDisplay := filepath.ToSlash(accountsPath)
	manifestDisplay := filepath.ToSlash(manifestPath)
	nextCmd := fmt.Sprintf("fak serve --route-accounts %s --route-manifest %s --guard", accountsDisplay, manifestDisplay)
	statusCheck := "curl -s http://127.0.0.1:8080/v1/router/status"

	if *asJSON {
		b, _ := json.MarshalIndent(routeInitReceipt{
			Profile:      strings.TrimSpace(*profile),
			OutDir:       filepath.ToSlash(dir),
			AccountsPath: accountsDisplay,
			ManifestPath: manifestDisplay,
			NextCommand:  nextCmd,
			StatusCheck:  statusCheck,
		}, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return 0
	}

	fmt.Fprintf(stdout, "Initialized %s router config in %s\n", strings.TrimSpace(*profile), filepath.ToSlash(dir))
	fmt.Fprintf(stdout, "  accounts: %s\n", accountsDisplay)
	fmt.Fprintf(stdout, "  manifest: %s\n\n", manifestDisplay)
	fmt.Fprintf(stdout, "Next step:\n  %s\n\n", nextCmd)
	fmt.Fprintf(stdout, "Check live routing status:\n  %s\n", statusCheck)
	return 0
}

func routeInitProfile(profile string) (modelroute.Roster, modelroute.Manifest, error) {
	switch profile {
	case "hybrid":
		return modelroute.Roster{
				Version: modelroute.RosterVersion,
				Accounts: []modelroute.Account{
					{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:8080/v1"},
					{ID: "openai", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_API_KEY"},
					{ID: "anthropic", Kind: modelroute.KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
				},
				Bindings: []modelroute.Binding{
					{
						Model:   "default",
						Account: "local-gpu",
						Fallbacks: []modelroute.FallbackTarget{
							{Account: "openai", UpstreamModel: "gpt-4o-mini"},
						},
					},
					{
						Model:          "private",
						Account:        "local-gpu",
						ResidencyFloor: "local_only",
					},
					{
						Model:         "frontier",
						Account:       "anthropic",
						UpstreamModel: "claude-sonnet-4-20250514",
						Fallbacks: []modelroute.FallbackTarget{
							{Account: "openai", UpstreamModel: "gpt-4o"},
						},
					},
				},
				Default: "local-gpu",
			}, modelroute.Manifest{
				Version: modelroute.Version,
				Default: modelroute.Plan{
					Members: []modelroute.Member{{Model: "default", Role: "primary"}},
					Reason:  "hybrid default: local-gpu primary with openai fallback",
				},
				Rules: []modelroute.Rule{
					{
						Name:  "private-local-only",
						Match: modelroute.Match{Labels: map[string]string{"sensitivity": "tenant"}},
						Plan: modelroute.Plan{
							Members: []modelroute.Member{{Model: "private", Role: "primary"}},
							Reason:  "tenant/private traffic stays local_only",
						},
					},
					{
						Name:  "hard-reasoning",
						Match: modelroute.Match{MinComplexity: modelroute.ComplexityHigh},
						Plan: modelroute.Plan{
							Members: []modelroute.Member{{Model: "frontier", Role: "primary"}},
							Reason:  "high complexity routes to frontier with openai fallback",
						},
					},
				},
			}, nil

	case "local-only":
		return modelroute.Roster{
				Version: modelroute.RosterVersion,
				Accounts: []modelroute.Account{
					{ID: "local-gpu", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:8080/v1"},
					{ID: "local-backup", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
				},
				Bindings: []modelroute.Binding{
					{
						Model:          "default",
						Account:        "local-gpu",
						ResidencyFloor: "local_only",
						Fallbacks: []modelroute.FallbackTarget{
							{Account: "local-backup"},
						},
					},
					{
						Model:          "private",
						Account:        "local-gpu",
						ResidencyFloor: "local_only",
					},
				},
				Default: "local-gpu",
			}, modelroute.Manifest{
				Version: modelroute.Version,
				Default: modelroute.Plan{
					Members: []modelroute.Member{{Model: "default", Role: "primary"}},
					Reason:  "local-only default: local-gpu with local-backup fallback",
				},
				Rules: []modelroute.Rule{
					{
						Name:  "private-local-only",
						Match: modelroute.Match{Labels: map[string]string{"sensitivity": "tenant"}},
						Plan: modelroute.Plan{
							Members: []modelroute.Member{{Model: "private", Role: "primary"}},
							Reason:  "tenant/private traffic stays local_only",
						},
					},
				},
			}, nil

	case "multi-cloud":
		return modelroute.Roster{
				Version: modelroute.RosterVersion,
				Accounts: []modelroute.Account{
					{ID: "anthropic", Kind: modelroute.KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
					{ID: "openai", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_API_KEY"},
					{ID: "openrouter", Kind: modelroute.KindOpenRouter, CredEnv: "OPENROUTER_API_KEY"},
				},
				Bindings: []modelroute.Binding{
					{
						Model:         "default",
						Account:       "anthropic",
						UpstreamModel: "claude-sonnet-4-20250514",
						Fallbacks: []modelroute.FallbackTarget{
							{Account: "openai", UpstreamModel: "gpt-4o-mini"},
							{Account: "openrouter", UpstreamModel: "anthropic/claude-sonnet-4"},
						},
					},
					{
						Model:         "frontier",
						Account:       "anthropic",
						UpstreamModel: "claude-sonnet-4-20250514",
						Fallbacks: []modelroute.FallbackTarget{
							{Account: "openai", UpstreamModel: "gpt-4o"},
							{Account: "openrouter", UpstreamModel: "anthropic/claude-opus-4"},
						},
					},
				},
				Default: "anthropic",
			}, modelroute.Manifest{
				Version: modelroute.Version,
				Default: modelroute.Plan{
					Members: []modelroute.Member{{Model: "default", Role: "primary"}},
					Reason:  "multi-cloud default: anthropic -> openai -> openrouter",
				},
				Rules: []modelroute.Rule{
					{
						Name:  "hard-reasoning",
						Match: modelroute.Match{MinComplexity: modelroute.ComplexityHigh},
						Plan: modelroute.Plan{
							Members: []modelroute.Member{{Model: "frontier", Role: "primary"}},
							Reason:  "high complexity routes to frontier multi-cloud chain",
						},
					},
				},
			}, nil

	default:
		return modelroute.Roster{}, modelroute.Manifest{}, fmt.Errorf("unknown profile %q (want hybrid, local-only, or multi-cloud)", profile)
	}
}
