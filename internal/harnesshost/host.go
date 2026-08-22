package harnesshost

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

const primitiveVersion = "1.0.0"

type Artifacts struct {
	Host     string
	Manifest string
	Lock     string
}

func Build(host, contractVersion string) (Artifacts, error) {
	host = strings.ToLower(strings.TrimSpace(host))
	if host == "" {
		return Artifacts{}, nil
	}
	if host != "codex" && host != "claude" {
		return Artifacts{}, fmt.Errorf("unsupported --host %q (want codex|claude)", host)
	}
	if strings.TrimSpace(contractVersion) == "" {
		return Artifacts{}, fmt.Errorf("contract version is required for host %q", host)
	}
	profile, ok := builtinProfile(host)
	if !ok {
		return Artifacts{}, fmt.Errorf("built-in host profile %q is unavailable", host)
	}
	binding, err := harnessprofile.Bind(profile)
	if err != nil {
		return Artifacts{}, err
	}
	authority := "fak:internal/harnessprofile"
	source := authority + "/builtin/" + profile.Name + "@" + profile.AdapterVersion
	return BuildResolved(binding, authority, source, contractVersion)
}

// BuildResolved projects an explicitly resolved descriptor binding through the same graph
// and lock path used by the built-in wrapper. The caller supplies provenance; this function
// does not consult mutable process-global profile state.
func BuildResolved(binding harnessprofile.Binding, authority, source, contractVersion string) (Artifacts, error) {
	if binding.Schema != harnessprofile.BindingSchema || strings.TrimSpace(binding.Host) == "" {
		return Artifacts{}, fmt.Errorf("resolved harness binding is required")
	}
	if binding.AdapterVersion == "" {
		return Artifacts{}, fmt.Errorf("resolved host profile %q has no adapter version", binding.Host)
	}
	if strings.TrimSpace(authority) == "" || strings.TrimSpace(source) == "" {
		return Artifacts{}, fmt.Errorf("resolved host profile %q requires source authority and provenance", binding.Host)
	}
	if !binding.Wire.Valid() {
		return Artifacts{}, fmt.Errorf("resolved host profile %q has unknown wire %q", binding.Host, binding.Wire)
	}
	for _, mechanism := range binding.Repoint {
		if !mechanism.Valid() {
			return Artifacts{}, fmt.Errorf("resolved host profile %q has unknown repoint mechanism %q", binding.Host, mechanism)
		}
	}

	hostID := "host:" + binding.Host
	requirements := []harnessresolve.Requirement{{Capability: "wire:" + string(binding.Wire), Range: ">=1.0.0"}}
	components := []harnessresolve.Component{{
		ID: hostID, Version: binding.AdapterVersion, Digest: binding.AdapterDigest, Source: source,
		Provides: []string{"harness:" + binding.Host}, Compatibility: harnessresolve.Compatibility{Contract: contractVersion},
		Adapters: []string{"harnessprofile"}, Evidence: stackresolve.Evidence{Authority: authority, Source: source, Tier: "shipped"},
	}}
	components = append(components, primitiveComponent("wire:"+string(binding.Wire), "gateway:"+string(binding.Wire), authority, contractVersion))
	for _, mechanism := range binding.Repoint {
		capability := "repoint:" + string(mechanism)
		requirements = append(requirements, harnessresolve.Requirement{Capability: capability, Range: ">=1.0.0"})
		components = append(components, primitiveComponent(capability, "guard:"+string(mechanism), authority, contractVersion))
	}
	components[0].Requires = requirements
	manifest := harnessresolve.Manifest{
		Schema: harnessresolve.Schema, Roots: []string{hostID}, Components: components,
		Compatibility: harnessresolve.Compatibility{Contract: contractVersion},
		Assets:        harnesscompose.Manifest{Schema: harnesscompose.Schema},
	}
	manifestRaw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return Artifacts{}, err
	}
	parsed, err := harnessresolve.Parse(manifestRaw)
	if err != nil {
		return Artifacts{}, fmt.Errorf("validate generated host manifest: %w", err)
	}
	resolved, err := harnessresolve.Resolve(context.Background(), parsed, nil, harnessresolve.Environment{OS: "portable", Arch: "portable", Contract: contractVersion})
	if err != nil {
		return Artifacts{}, fmt.Errorf("resolve generated host manifest: %w", err)
	}
	if err := harnessresolve.VerifyLock(resolved.Lock); err != nil {
		return Artifacts{}, fmt.Errorf("verify generated host lock: %w", err)
	}
	if err := harnessresolve.Mixable(resolved.Lock); err != nil {
		return Artifacts{}, fmt.Errorf("validate generated host lock evidence: %w", err)
	}
	lockRaw, err := json.MarshalIndent(resolved.Lock, "", "  ")
	if err != nil {
		return Artifacts{}, err
	}
	return Artifacts{Host: binding.Host, Manifest: string(manifestRaw) + "\n", Lock: string(lockRaw) + "\n"}, nil
}

// VerifyResolved binds a graph and lock back to the descriptor identity that authored
// them. Canonical lock self-hashing alone cannot detect a newer descriptor using an old,
// internally consistent lock.
func VerifyResolved(binding harnessprofile.Binding, artifacts Artifacts) error {
	manifest, err := harnessresolve.Parse([]byte(artifacts.Manifest))
	if err != nil {
		return fmt.Errorf("stale harness artifacts: manifest: %w", err)
	}
	var lock harnessresolve.Lock
	if err := json.Unmarshal([]byte(artifacts.Lock), &lock); err != nil {
		return fmt.Errorf("stale harness artifacts: lock: %w", err)
	}
	if err := harnessresolve.VerifyLock(lock); err != nil {
		return fmt.Errorf("stale harness artifacts: lock: %w", err)
	}
	if artifacts.Host != binding.Host {
		return fmt.Errorf("stale harness artifacts: host %q does not match descriptor %q", artifacts.Host, binding.Host)
	}
	if err := verifyComponents(binding, manifest.Components); err != nil {
		return err
	}
	locked := make([]harnessresolve.Component, 0, len(lock.Components))
	for _, component := range lock.Components {
		locked = append(locked, harnessresolve.Component{ID: component.ID, Version: component.Version, Digest: component.Digest})
	}
	return verifyComponents(binding, locked)
}

func verifyComponents(binding harnessprofile.Binding, components []harnessresolve.Component) error {
	wantIDs := []string{"host:" + binding.Host, "wire:" + string(binding.Wire)}
	for _, mechanism := range binding.Repoint {
		wantIDs = append(wantIDs, "repoint:"+string(mechanism))
	}
	sort.Strings(wantIDs)
	gotIDs := make([]string, 0, len(wantIDs))
	foundHost := false
	for _, component := range components {
		if component.ID == "host:"+binding.Host {
			foundHost = component.Version == binding.AdapterVersion && component.Digest == binding.AdapterDigest
		}
		if strings.HasPrefix(component.ID, "host:") || strings.HasPrefix(component.ID, "wire:") || strings.HasPrefix(component.ID, "repoint:") {
			gotIDs = append(gotIDs, component.ID)
		}
	}
	sort.Strings(gotIDs)
	if !foundHost || !slices.Equal(gotIDs, wantIDs) {
		return fmt.Errorf("stale harness artifacts for %q: components %v do not match descriptor %v", binding.Host, gotIDs, wantIDs)
	}
	return nil
}

func builtinProfile(name string) (harnessprofile.HarnessProfile, bool) {
	for _, profile := range harnessprofile.Builtins() {
		if profile.Name == name {
			return profile, true
		}
	}
	return harnessprofile.HarnessProfile{}, false
}

func primitiveComponent(capability, adapter, authority, contractVersion string) harnessresolve.Component {
	source := authority + "/primitive/" + capability + "@" + primitiveVersion
	return harnessresolve.Component{
		ID: capability, Version: primitiveVersion, Digest: digestText(source), Source: source,
		Provides: []string{capability}, Compatibility: harnessresolve.Compatibility{Contract: contractVersion},
		Adapters: []string{adapter}, Evidence: stackresolve.Evidence{Authority: authority, Source: source, Tier: "shipped"},
	}
}

func digestText(text string) string {
	sum := sha256.Sum256([]byte(text))
	return "sha256:" + hex.EncodeToString(sum[:])
}
