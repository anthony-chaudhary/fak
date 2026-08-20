package harnessinit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/harnessresolve"
	"github.com/anthony-chaudhary/fak/internal/stackresolve"
)

const (
	HostManifestPath = "product.json"
	HostLockPath     = "product.lock.json"
	primitiveVersion = "1.0.0"
)

type hostArtifacts struct {
	host     string
	manifest string
	lock     string
}

func hostArtifactsFor(host string) (hostArtifacts, error) {
	host = strings.ToLower(strings.TrimSpace(host))
	if host == "" {
		return hostArtifacts{}, nil
	}
	if host != "codex" && host != "claude" {
		return hostArtifacts{}, fmt.Errorf("unsupported --host %q (want codex|claude)", host)
	}
	profile, ok := builtinHostProfile(host)
	if !ok {
		return hostArtifacts{}, fmt.Errorf("built-in host profile %q is unavailable", host)
	}
	if profile.AdapterVersion == "" {
		return hostArtifacts{}, fmt.Errorf("built-in host profile %q has no adapter version", host)
	}
	profileDigest, err := harnessprofile.SemanticDigest(profile)
	if err != nil {
		return hostArtifacts{}, err
	}

	hostID := "host:" + profile.Name
	authority := "fak:internal/harnessprofile"
	source := authority + "/builtin/" + profile.Name + "@" + profile.AdapterVersion
	requirements := []harnessresolve.Requirement{{Capability: "wire:" + string(profile.Wire), Range: ">=1.0.0"}}
	components := []harnessresolve.Component{{
		ID: hostID, Version: profile.AdapterVersion, Digest: profileDigest, Source: source,
		Provides: []string{"harness:" + profile.Name}, Compatibility: harnessresolve.Compatibility{Contract: ContractVersion},
		Adapters: []string{"harnessprofile"}, Evidence: stackresolve.Evidence{Authority: authority, Source: source, Tier: "shipped"},
	}}
	components = append(components, primitiveComponent("wire:"+string(profile.Wire), "gateway:"+string(profile.Wire), authority))
	for _, mechanism := range profile.Repoint {
		capability := "repoint:" + string(mechanism)
		requirements = append(requirements, harnessresolve.Requirement{Capability: capability, Range: ">=1.0.0"})
		components = append(components, primitiveComponent(capability, "guard:"+string(mechanism), authority))
	}
	components[0].Requires = requirements
	manifest := harnessresolve.Manifest{
		Schema: harnessresolve.Schema, Roots: []string{hostID}, Components: components,
		Compatibility: harnessresolve.Compatibility{Contract: ContractVersion},
		Assets:        harnesscompose.Manifest{Schema: harnesscompose.Schema},
	}
	manifestRaw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return hostArtifacts{}, err
	}
	parsed, err := harnessresolve.Parse(manifestRaw)
	if err != nil {
		return hostArtifacts{}, fmt.Errorf("validate generated host manifest: %w", err)
	}
	resolved, err := harnessresolve.Resolve(context.Background(), parsed, nil, harnessresolve.Environment{OS: "portable", Arch: "portable", Contract: ContractVersion})
	if err != nil {
		return hostArtifacts{}, fmt.Errorf("resolve generated host manifest: %w", err)
	}
	if err := harnessresolve.VerifyLock(resolved.Lock); err != nil {
		return hostArtifacts{}, fmt.Errorf("verify generated host lock: %w", err)
	}
	if err := harnessresolve.Mixable(resolved.Lock); err != nil {
		return hostArtifacts{}, fmt.Errorf("validate generated host lock evidence: %w", err)
	}
	lockRaw, err := json.MarshalIndent(resolved.Lock, "", "  ")
	if err != nil {
		return hostArtifacts{}, err
	}
	return hostArtifacts{host: host, manifest: string(manifestRaw) + "\n", lock: string(lockRaw) + "\n"}, nil
}

func builtinHostProfile(name string) (harnessprofile.HarnessProfile, bool) {
	for _, profile := range harnessprofile.Builtins() {
		if profile.Name == name {
			return profile, true
		}
	}
	return harnessprofile.HarnessProfile{}, false
}

func primitiveComponent(capability, adapter, authority string) harnessresolve.Component {
	source := authority + "/primitive/" + capability + "@" + primitiveVersion
	return harnessresolve.Component{
		ID: capability, Version: primitiveVersion, Digest: digestText(source), Source: source,
		Provides: []string{capability}, Compatibility: harnessresolve.Compatibility{Contract: ContractVersion},
		Adapters: []string{adapter}, Evidence: stackresolve.Evidence{Authority: authority, Source: source, Tier: "shipped"},
	}
}

func digestText(text string) string {
	sum := sha256.Sum256([]byte(text))
	return "sha256:" + hex.EncodeToString(sum[:])
}
