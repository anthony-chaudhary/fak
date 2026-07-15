package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

const (
	guardDenyOverlayVersion = "fak-guard-deny/v1"
	guardDenyOverlayEnv     = "FAK_GUARD_DENY_OVERLAY"
)

type guardDenyOverlay struct {
	Version string   `json:"version"`
	Deny    []string `json:"deny,omitempty"`
}

func guardDenyOverlayPath() string {
	if p := strings.TrimSpace(os.Getenv(guardDenyOverlayEnv)); p != "" {
		return p
	}
	return filepath.Join(findRepoRoot("."), ".fak", "guard", "deny.json")
}

func loadGuardDenyOverlay(path string) (guardDenyOverlay, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return guardDenyOverlay{Version: guardDenyOverlayVersion}, nil
		}
		return guardDenyOverlay{}, fmt.Errorf("guard deny overlay %s: %w", path, err)
	}
	var ov guardDenyOverlay
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&ov); err != nil {
		return guardDenyOverlay{}, fmt.Errorf("guard deny overlay %s: invalid: %w", path, err)
	}
	if v := strings.TrimSpace(ov.Version); v != "" && v != guardDenyOverlayVersion {
		return guardDenyOverlay{}, fmt.Errorf("guard deny overlay %s: unsupported version %q (want %s)", path, ov.Version, guardDenyOverlayVersion)
	}
	ov.Version = guardDenyOverlayVersion
	ov.Deny = guardAllowNormalize(ov.Deny)
	return ov, nil
}

func saveGuardDenyOverlay(path string, ov guardDenyOverlay) error {
	ov.Version = guardDenyOverlayVersion
	ov.Deny = guardAllowNormalize(ov.Deny)
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("guard deny overlay: mkdir %s: %w", dir, err)
		}
	}
	b, err := json.MarshalIndent(ov, "", "  ")
	if err != nil {
		return err
	}
	return writeGuardAllowOverlayAtomic(path, append(b, 10))
}

func guardApplyDenyOverlay(rt *policy.Runtime, ov guardDenyOverlay) int {
	if rt == nil || len(ov.Deny) == 0 {
		return 0
	}
	if rt.Adjudicator.Deny == nil {
		rt.Adjudicator.Deny = map[string]abi.ReasonCode{}
	}
	added := 0
	for _, tool := range ov.Deny {
		if _, exists := rt.Adjudicator.Deny[tool]; !exists {
			added++
		}
		rt.Adjudicator.Deny[tool] = abi.ReasonPolicyBlock
	}
	return added
}

func printGuardDenyOverlay(w io.Writer, path string, ov guardDenyOverlay) {
	fmt.Fprintf(w, "repo-local deny overlay: %s\n", path)
	if len(ov.Deny) == 0 {
		fmt.Fprintln(w, "  (empty � no tools denied beyond the capability floor)")
		return
	}
	fmt.Fprintf(w, "  deny (exact): %s\n", strings.Join(ov.Deny, ", "))
}

func cmdGuardDeny(argv []string) {
	fs := flag.NewFlagSet("guard deny", flag.ExitOnError)
	list := fs.Bool("list", false, "print the repo-local deny overlay and its path")
	remove := fs.Bool("remove", false, "remove exact tool names")
	_ = fs.Parse(argv)
	path := guardDenyOverlayPath()
	ov, err := loadGuardDenyOverlay(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak guard deny:", err)
		os.Exit(1)
	}
	if *list {
		printGuardDenyOverlay(os.Stdout, path, ov)
		return
	}
	names := guardAllowNormalize(fs.Args())
	if len(names) == 0 {
		fmt.Fprintln(os.Stderr, "usage: fak guard deny [--list|--remove] <tool>...")
		os.Exit(2)
	}
	if *remove {
		ov.Deny = guardAllowSubtract(ov.Deny, names)
	} else {
		ov.Deny = append(ov.Deny, names...)
	}
	if err := saveGuardDenyOverlay(path, ov); err != nil {
		fmt.Fprintln(os.Stderr, "fak guard deny:", err)
		os.Exit(1)
	}
	printGuardDenyOverlay(os.Stdout, path, ov)
}
