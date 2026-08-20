package harnesshost

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
	if profile.AdapterVersion == "" {
		return Artifacts{}, fmt.Errorf("built-in host profile %q has no adapter version", host)
	}
	profileDigest, err := harnessprofile.SemanticDigest(profile)
	if err != nil {
		return Artifacts{}, err
	}

	hostID := "host:" + profile.Name
	authority := "fak:internal/harnessprofile"
	source := authority + "/builtin/" + profile.Name + "@" + profile.AdapterVersion
	requirements := []harnessresolve.Requirement{{Capability: "wire:" + string(profile.Wire), Range: ">=1.0.0"}}
	components := []harnessresolve.Component{{
		ID: hostID, Version: profile.AdapterVersion, Digest: profileDigest, Source: source,
		Provides: []string{"harness:" + profile.Name}, Compatibility: harnessresolve.Compatibility{Contract: contractVersion},
		Adapters: []string{"harnessprofile"}, Evidence: stackresolve.Evidence{Authority: authority, Source: source, Tier: "shipped"},
	}}
	components = append(components, primitiveComponent("wire:"+string(profile.Wire), "gateway:"+string(profile.Wire), authority, contractVersion))
	for _, mechanism := range profile.Repoint {
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
	return Artifacts{Host: host, Manifest: string(manifestRaw) + "\n", Lock: string(lockRaw) + "\n"}, nil
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
