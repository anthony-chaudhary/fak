package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"

	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

// applyPolicy loads a capability-floor manifest and swaps it into the registered
// adjudicator before the kernel runs. Empty path = keep the built-in
// DefaultPolicy. A bad manifest is fatal: a misconfigured floor must fail loudly
// at startup, never silently fall back to a more permissive default.
// policyReloadMu serializes the complete multi-singleton floor swap.
var policyReloadMu sync.Mutex

func applyPolicy(path string) {
	applyFloorWithProfile(path, os.Getenv("FAK_PROFILE"))
}

func applyFloorWithProfile(path string, profile string) {
	profile = strings.TrimSpace(profile)
	if profile == "" {
		profile = strings.TrimSpace(os.Getenv("FAK_PROFILE"))
	}
	implicitDefault := false
	if path == "" && profile == "" {
		profile = string(policy.ProfileStandard)
		implicitDefault = true
	}
	policyReloadMu.Lock()
	defer policyReloadMu.Unlock()
	if path == "" && profile != "" {
		prof, err := policy.ParseProfile(profile)
		must(err)
		rt, err := policy.RuntimeForProfile(prof)
		must(err)
		_, err = applyPolicyRuntimeLocked(rt, "profile:"+string(prof), "", "", false)
		must(err)
		if !implicitDefault {
			fmt.Fprintf(os.Stderr, "fak: applied permission profile %s\n", prof)
		}
		return
	}
	_, _, err := loadAndApplyPolicyWithProfileLocked(path, profile, false)
	must(err)
	if profile != "" {
		fmt.Fprintf(os.Stderr, "fak: loaded capability floor from %s with profile %s\n", path, profile)
	} else {
		fmt.Fprintf(os.Stderr, "fak: loaded capability floor from %s\n", path)
	}
}

func applyPolicyWithProfile(path string, profile string) {
	if path == "" && profile == "" {
		profile = string(policy.ProfileStandard)
	}
	policyReloadMu.Lock()
	defer policyReloadMu.Unlock()
	rt, floorSource, policyDigest, _ := loadGuardCapabilityFloor(path, "", profile)
	_, err := applyPolicyRuntimeLocked(rt, floorSource, policyDigest, "", false)
	must(err)
	fmt.Fprintf(os.Stderr, "fak: loaded capability floor from %s (digest %s)\n", floorSource, policyDigest)
}

func reloadPolicy(path string) (policy.Runtime, string, error) {
	policyReloadMu.Lock()
	defer policyReloadMu.Unlock()
	return loadAndApplyPolicyWithProfileLocked(path, "", true)
}

func reloadPolicyWithPrior(path string) (policy.Runtime, adjudicator.Policy, string, error) {
	policyReloadMu.Lock()
	defer policyReloadMu.Unlock()
	prior := adjudicator.Default.PolicySnapshot()
	rt, warning, err := loadAndApplyPolicyWithProfileLocked(path, "", true)
	return rt, prior, warning, err
}

func loadAndApplyPolicyWithProfileLocked(path string, profile string, enforceWideningGate bool) (policy.Runtime, string, error) {
	if path == "" {
		return policy.Runtime{}, "", errors.New("policy reload requires --policy FILE")
	}
	rt, err := policy.LoadRuntime(path)
	if err != nil {
		// A rejected floor swap is exactly what an auditor asks about (an operator
		// trying to widen or break the capability floor with a malformed edit), so
		// record the refused attempt — with the digest of whatever bytes we could
		// read — before returning. journal.Active() is nil on an unjournaled run and
		// AppendConfigSwap no-ops, keeping that run byte-identical.
		journal.Active().AppendConfigSwap(journal.ConfigSwapFloor, path, configFileDigest(path), journal.ConfigSwapRejected, err.Error())
		return policy.Runtime{}, "", err
	}
	if profile != "" {
		prof, err := policy.ParseProfile(profile)
		if err != nil {
			return policy.Runtime{}, "", err
		}
		prof.Apply(&rt)
	}
	// Re-apply the operator overlays before comparing effective floors. This keeps
	// the gate from mistaking a persisted always-allow entry for a manifest widening.
	denyPath := guardDenyOverlayPath()
	overlayWarning := ""
	if ov, _, ovErr := loadGuardAllowOverlayLayers(); ovErr == nil {
		guardApplyAllowOverlay(&rt, ov)
	} else {
		overlayWarning = "overlay_error: " + ovErr.Error()
	}
	applyLaunchToolGrant(&rt)
	if ov, ovErr := loadGuardDenyOverlay(denyPath); ovErr == nil {
		guardApplyDenyOverlay(&rt, ov)
	} else if overlayWarning == "" {
		overlayWarning = "deny_overlay_error: " + ovErr.Error()
	} else {
		overlayWarning += "\ndeny_overlay_error: " + ovErr.Error()
	}
	rt = protectGuardPolicyConfig(rt, append(guardAllowOverlayLayerPaths(), denyPath, path)...)
	digest := configFileDigest(path)
	overlayWarning, err = applyPolicyRuntimeLocked(rt, path, digest, overlayWarning, enforceWideningGate)
	if err != nil {
		return policy.Runtime{}, "", err
	}
	return rt, overlayWarning, nil
}

// configFileDigest returns the sha256 (as "sha256:<hex>") of a config manifest's
// bytes, for the over-which-bytes field of a CONFIG_SWAP audit row. It reads the
// file fresh — the swap installs whatever is on disk now — and yields "" on an
// unreadable file, so a rejected swap still records the attempt, just without a
// digest.
func configFileDigest(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func policyReloader(path string) gateway.PolicyReloadFunc {
	if path == "" {
		return nil
	}
	return func(context.Context) (gateway.PolicyReloadResponse, error) {
		rt, prior, overlayWarning, err := reloadPolicyWithPrior(path)
		if err != nil {
			return gateway.PolicyReloadResponse{}, err
		}
		summary := policy.SummaryRuntime(rt)
		if overlayWarning != "" {
			fmt.Fprintln(os.Stderr, "fak policy reload warning:", overlayWarning)
			summary += "\n" + overlayWarning
		}
		policyBytes, readErr := os.ReadFile(path)
		if readErr != nil {
			return gateway.PolicyReloadResponse{}, readErr
		}
		rollback := func() {
			policyReloadMu.Lock()
			defer policyReloadMu.Unlock()
			adjudicator.Default.SetPolicy(prior)
		}
		return gateway.PolicyReloadResponse{Reloaded: true, Source: path, Summary: summary, EffectiveDigest: effectiveDigestWithLaunchGrant(policyBytes), Rollback: rollback}, nil
	}
}

// guardPolicyReloader is the reloader `fak guard` wires (guard.go). For an explicit
// --policy FILE it is exactly policyReloader (the file-reload path, unchanged). For the
// default built-in floor (empty path) it returns a working reloader that re-derives the
// embedded floor + operator allow overlay — instead of policyReloader's nil, which had
// disabled POST /v1/fak/policy/reload (404) on the most common guard launch (#3957).
// `fak serve` keeps calling policyReloader directly, so its non-guard no-`--policy`
// default stays byte-identical, as does `--policy` everywhere.
func guardPolicyReloader(policyPath string) gateway.PolicyReloadFunc {
	if strings.TrimSpace(policyPath) != "" {
		return policyReloader(policyPath)
	}
	return func(context.Context) (gateway.PolicyReloadResponse, error) {
		rt, overlayWarning, err := guardReloadDefaultFloor()
		if err != nil {
			return gateway.PolicyReloadResponse{}, err
		}
		summary := policy.SummaryRuntime(rt)
		if overlayWarning != "" {
			fmt.Fprintln(os.Stderr, "fak guard reload warning:", overlayWarning)
			summary += "\n" + overlayWarning
		}
		return gateway.PolicyReloadResponse{Reloaded: true, Source: launchGrantSource("built-in guard floor + operator allow overlay"), Summary: summary, EffectiveDigest: effectiveDigestWithLaunchGrant(guardDefaultPolicyJSON)}, nil
	}
}
