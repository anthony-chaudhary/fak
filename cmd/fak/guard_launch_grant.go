package main

import (
	"errors"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/policy"
)

// launchToolFlag collects exact capabilities the operator grants to this guard
// process. It is intentionally narrower than the durable allow overlay: no prefixes,
// persistence, or agent-writable amendment channel.
var exactLaunchToolName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:/-]*$`)

type launchToolFlag []string

func (f *launchToolFlag) String() string {
	if f == nil {
		return ""
	}
	return strings.Join(*f, ",")
}

func (f *launchToolFlag) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return errors.New("--allow-tool requires a non-empty exact tool name")
	}
	if !exactLaunchToolName.MatchString(value) || strings.Contains(value, "..") {
		return errors.New("--allow-tool requires one literal tool name without wildcard, pattern, or traversal syntax")
	}
	*f = append(*f, value)
	return nil
}

var launchGrantState struct {
	sync.RWMutex
	allow guardAllowOverlay
}

func setLaunchToolGrant(names []string) {
	launchGrantState.Lock()
	defer launchGrantState.Unlock()
	launchGrantState.allow = guardAllowOverlay{
		Version: guardAllowOverlayVersion,
		Allow:   guardAllowNormalize(names),
	}
}

func launchToolGrant() guardAllowOverlay {
	launchGrantState.RLock()
	defer launchGrantState.RUnlock()
	return guardAllowOverlay{
		Version: guardAllowOverlayVersion,
		Allow:   append([]string(nil), launchGrantState.allow.Allow...),
	}
}

func applyLaunchToolGrant(rt *policy.Runtime) (guardAllowOverlay, int) {
	grant := launchToolGrant()
	return grant, guardApplyAllowOverlay(rt, grant)
}

func launchGrantSource(base string) string {
	if len(launchToolGrant().Allow) == 0 {
		return base
	}
	return base + " + launch-scoped grant"
}

func mergeLaunchAllowOverlays(base, extra guardAllowOverlay) guardAllowOverlay {
	merged := guardAllowOverlay{
		Version:     guardAllowOverlayVersion,
		Allow:       append(append([]string(nil), base.Allow...), extra.Allow...),
		AllowPrefix: append(append([]string(nil), base.AllowPrefix...), extra.AllowPrefix...),
	}
	if len(base.Expiry) > 0 {
		merged.Expiry = make(map[string]string, len(base.Expiry))
		for name, expiry := range base.Expiry {
			merged.Expiry[name] = expiry
		}
	}
	for name, expiry := range extra.Expiry {
		if merged.Expiry == nil {
			merged.Expiry = make(map[string]string)
		}
		merged.Expiry[name] = expiry
	}
	merged.Allow = guardAllowNormalize(merged.Allow)
	merged.AllowPrefix = guardAllowNormalize(merged.AllowPrefix)
	return merged
}

// effectiveDigestWithLaunchGrant attests the same effective floor
// a live guard reload installs, including its process-scoped exact grants.
func effectiveDigestWithLaunchGrant(policyBytes []byte) string {
	allow, _, err := loadGuardAllowOverlayLayers()
	if err != nil {
		allow = guardAllowOverlay{Version: guardAllowOverlayVersion}
	}
	allow = mergeLaunchAllowOverlays(allow, launchToolGrant())
	deny, err := loadGuardDenyOverlay(guardDenyOverlayPath())
	if err != nil {
		deny = guardDenyOverlay{Version: guardDenyOverlayVersion}
	}
	return guardEffectivePolicyDigest(policyBytes, allow, deny)
}
