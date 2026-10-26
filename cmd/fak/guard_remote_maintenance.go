package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

const guardRemoteMaintenanceMaxTTL = 15 * time.Minute

var (
	guardRemoteMaintenanceNow            = time.Now
	guardRemoteMaintenanceDefaultPath    = remotemaint.DefaultConfigPath
	guardRemoteMaintenanceLoadAuthority  = remotemaint.LoadAuthority
	guardRemoteMaintenanceProtectedPaths = remotemaint.ProtectedPaths
)

// runGuardRemoteMaintenance is the operator-only installation/revocation path.
// It never executes a maintenance operation. The reviewed config fixes the host,
// operation, paths, units, leases, native adapter and its immutable digest; the
// caller can only shorten that reviewed authority with the bounded TTL.
func runGuardRemoteMaintenance(stdout, stderr io.Writer, reviewedOrProfile string, ttl time.Duration, remove bool) int {
	if guardAllowSessionID() != "" {
		fmt.Fprintln(stderr, "fak guard allow --remote-maintenance: refused inside a guarded agent session; the attending operator must run this from an independent shell")
		return 1
	}
	destination, err := guardRemoteMaintenanceDefaultPath()
	if err != nil {
		fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: %v\n", err)
		return 1
	}
	return runGuardRemoteMaintenanceAt(stdout, stderr, destination, reviewedOrProfile, ttl, remove, guardRemoteMaintenanceNow().UTC())
}

func runGuardRemoteMaintenanceAt(stdout, stderr io.Writer, destination, reviewedOrProfile string, ttl time.Duration, remove bool, now time.Time) int {
	value := strings.TrimSpace(reviewedOrProfile)
	if value == "" {
		fmt.Fprintln(stderr, "fak guard allow --remote-maintenance: name a reviewed config file, or use --remove with one profile id")
		return 2
	}
	if remove {
		if ttl != 0 {
			fmt.Fprintln(stderr, "fak guard allow --remote-maintenance: --ttl cannot combine with --remove")
			return 2
		}
		return revokeGuardRemoteMaintenance(stdout, stderr, destination, value)
	}
	if ttl <= 0 || ttl > guardRemoteMaintenanceMaxTTL {
		fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: --ttl is required and must be in (0,%s]\n", guardRemoteMaintenanceMaxTTL)
		return 2
	}
	return installGuardRemoteMaintenance(stdout, stderr, destination, value, ttl, now)
}

func installGuardRemoteMaintenance(stdout, stderr io.Writer, destination, reviewedPath string, ttl time.Duration, now time.Time) int {
	// LoadAuthority is the package's strict, executable-hashing validator. Validate
	// the operator-reviewed source before decoding it for the sole permitted rewrite.
	if _, err := guardRemoteMaintenanceLoadAuthority(reviewedPath); err != nil {
		fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: reviewed config refused: %v\n", err)
		return 1
	}
	cfg, err := readGuardRemoteMaintenanceConfig(reviewedPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: reviewed config refused: %v\n", err)
		return 1
	}
	deadline := now.Add(ttl).UTC()
	for i := range cfg.Profiles {
		if cfg.Profiles[i].ExpiresAt.After(deadline) {
			cfg.Profiles[i].ExpiresAt = deadline
		}
	}
	if err := writeGuardRemoteMaintenanceConfigAtomic(destination, cfg, true); err != nil {
		fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: install: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak guard allow: installed %d remote-maintenance profile(s) until no later than %s.\n", len(cfg.Profiles), deadline.Format(time.RFC3339))
	fmt.Fprintf(stdout, "  Context: %s\n", destination)
	fmt.Fprintln(stdout, "  The next fak guard/fak serve launch loads this authority; callers select only profile_id and operation_digest.")
	return 0
}

func revokeGuardRemoteMaintenance(stdout, stderr io.Writer, destination, profileID string) int {
	if profileID == "*" {
		if err := os.Remove(destination); err != nil && !errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: clear all: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, "fak guard allow: cleared all remote-maintenance authority.")
		return 0
	}
	cfg, err := readGuardRemoteMaintenanceConfigUnchecked(destination)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: current context refused: %v\n", err)
		return 1
	}
	kept := cfg.Profiles[:0]
	found := false
	for _, profile := range cfg.Profiles {
		if profile.ProfileID == profileID {
			found = true
			continue
		}
		kept = append(kept, profile)
	}
	if !found {
		fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: profile %q is not installed\n", profileID)
		return 1
	}
	cfg.Profiles = kept
	if len(cfg.Profiles) == 0 {
		if err := os.Remove(destination); err != nil && !errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: revoke: %v\n", err)
			return 1
		}
	} else {
		// Revocation only narrows authority. Preserve other profiles byte-semantically
		// even when they are expired or their adapter is unavailable; those profiles
		// remain denied by the runtime loader and must not prevent removing this id.
		if err := writeGuardRemoteMaintenanceConfigAtomic(destination, cfg, false); err != nil {
			fmt.Fprintf(stderr, "fak guard allow --remote-maintenance: revoke: %v\n", err)
			return 1
		}
	}
	fmt.Fprintf(stdout, "fak guard allow: revoked remote-maintenance profile %s.\n", profileID)
	return 0
}

func readGuardRemoteMaintenanceConfigUnchecked(path string) (remotemaint.Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return remotemaint.Config{}, err
	}
	if len(raw) == 0 || len(raw) > 1<<20 {
		return remotemaint.Config{}, errors.New("invalid context size")
	}
	var cfg remotemaint.Config
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return remotemaint.Config{}, errors.New("malformed context")
	}
	if dec.Decode(&struct{}{}) != io.EOF || cfg.Schema != remotemaint.ConfigSchema {
		return remotemaint.Config{}, errors.New("malformed context")
	}
	seen := make(map[string]bool, len(cfg.Profiles))
	for _, profile := range cfg.Profiles {
		if strings.TrimSpace(profile.ProfileID) == "" || seen[profile.ProfileID] {
			return remotemaint.Config{}, errors.New("ambiguous profile identity")
		}
		seen[profile.ProfileID] = true
	}
	return cfg, nil
}

func readGuardRemoteMaintenanceConfig(path string) (remotemaint.Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return remotemaint.Config{}, err
	}
	return remotemaint.ParseConfig(raw)
}

func writeGuardRemoteMaintenanceConfigAtomic(path string, cfg remotemaint.Config, validateTemp bool) error {
	body, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	body = append(body, '\n')
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".remote-maintenance-*.json.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(body); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if validateTemp {
		if _, err := guardRemoteMaintenanceLoadAuthority(tmpPath); err != nil {
			return fmt.Errorf("candidate refused: %w", err)
		}
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	return nil
}

// protectRemoteMaintenanceAuthority wires the actual normal guard/serve floor to
// the trusted host authority. The default context path is protected even before it
// exists; a present context is loaded now and its pinned executable paths are added.
func protectRemoteMaintenanceAuthority(rt policy.Runtime) policy.Runtime {
	configPath, err := guardRemoteMaintenanceDefaultPath()
	if err != nil {
		warnRemoteMaintenanceDisabled(err)
		return rt
	}
	// Protect the fixed sidecar path before any parse/load result. A malformed or
	// expired optional authority disables only that authority, never the base floor.
	rt = protectGuardPolicyConfig(rt, configPath)
	paths, err := guardRemoteMaintenanceProtectedPaths(configPath)
	if err != nil {
		warnRemoteMaintenanceDisabled(err)
		return rt
	}
	rt = protectGuardPolicyConfig(rt, paths...)
	if _, statErr := os.Stat(configPath); statErr == nil {
		if _, err := guardRemoteMaintenanceLoadAuthority(configPath); err != nil {
			warnRemoteMaintenanceDisabled(err)
			return rt
		}
	} else if !errors.Is(statErr, os.ErrNotExist) {
		warnRemoteMaintenanceDisabled(statErr)
	}
	return rt
}

func warnRemoteMaintenanceDisabled(err error) {
	reason := "unavailable"
	switch {
	case errors.Is(err, remotemaint.ErrExpired):
		reason = "expired"
	case errors.Is(err, remotemaint.ErrChanged):
		reason = "changed"
	case errors.Is(err, remotemaint.ErrAuthority):
		reason = "invalid"
	}
	fmt.Fprintf(os.Stderr, "fak: remote maintenance authority disabled (%s); base capability floor unchanged\n", reason)
}
