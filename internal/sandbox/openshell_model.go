// Conceptual INSPIRE-ONLY seam derived from NVIDIA/OpenShell
// No Rust runtime is ported and no isolation guarantee is claimed without a
// native escape test.
package sandbox

import (
	"fmt"
	"path/filepath"
	"strings"
)

// OpenShellNetworkMode governs outbound network posture for an agent tool run.
type OpenShellNetworkMode string

const (
	OpenShellNetDeny      OpenShellNetworkMode = "deny"
	OpenShellNetAllowlist OpenShellNetworkMode = "allowlist"
)

// OpenShellPermissions declares the fail-closed capability surface for a run.
type OpenShellPermissions struct {
	Network        OpenShellNetworkMode `json:"network"`
	AllowedHosts   []string             `json:"allowed_hosts,omitempty"`
	ReadPaths      []string             `json:"read_paths,omitempty"`
	WritePaths     []string             `json:"write_paths,omitempty"`
	InheritHostEnv bool                 `json:"inherit_host_env"`
	AllowSubproc   bool                 `json:"allow_subproc"`
}

// OpenShellBoundary defines the minimum isolation tier and permission envelope.
type OpenShellBoundary struct {
	MinTier     Tier                 `json:"min_tier"`
	Permissions OpenShellPermissions `json:"permissions"`
}

// DefaultOpenShellBoundary returns the safe-default execution posture:
// L1 native OS minimum, network denied, no host env inheritance, and explicit
// workspace-only read/write paths.
func DefaultOpenShellBoundary(workspace string) OpenShellBoundary {
	var readPaths, writePaths []string
	if strings.TrimSpace(workspace) != "" {
		clean := filepath.Clean(workspace)
		readPaths = []string{clean}
		writePaths = []string{clean}
	}
	return OpenShellBoundary{
		MinTier: TierL1NativeOS,
		Permissions: OpenShellPermissions{
			Network:        OpenShellNetDeny,
			ReadPaths:      readPaths,
			WritePaths:     writePaths,
			InheritHostEnv: false,
			AllowSubproc:   false,
		},
	}
}

// OpenShellExecRequest describes a proposed tool execution against a boundary.
type OpenShellExecRequest struct {
	Tier         Tier     `json:"tier"`
	NeedsNetwork bool     `json:"needs_network"`
	TargetHost   string   `json:"target_host,omitempty"`
	WritePaths   []string `json:"write_paths,omitempty"`
	InheritEnv   bool     `json:"inherit_env"`
	SpawnsChild  bool     `json:"spawns_child"`
}

// OpenShellDecision is the fail-closed admission verdict for OpenShellExecRequest.
type OpenShellDecision struct {
	Allowed      bool   `json:"allowed"`
	Reason       string `json:"reason,omitempty"`
	RequiredTier Tier   `json:"required_tier"`
}

// Admit evaluates req against b, refusing any request that weakens isolation
// or exceeds the declared permission surface.
func (b OpenShellBoundary) Admit(req OpenShellExecRequest) OpenShellDecision {
	if !b.MinTier.Valid() || !req.Tier.Valid() {
		return OpenShellDecision{Allowed: false, Reason: "invalid sandbox tier", RequiredTier: b.MinTier}
	}
	if req.Tier.IsolationLevel() < b.MinTier.IsolationLevel() {
		return OpenShellDecision{
			Allowed:      false,
			Reason:       fmt.Sprintf("tier %s below required %s", req.Tier, b.MinTier),
			RequiredTier: b.MinTier,
		}
	}
	if req.InheritEnv && !b.Permissions.InheritHostEnv {
		return OpenShellDecision{Allowed: false, Reason: "host env inheritance denied", RequiredTier: b.MinTier}
	}
	if req.SpawnsChild && !b.Permissions.AllowSubproc {
		return OpenShellDecision{Allowed: false, Reason: "subprocess spawn denied", RequiredTier: b.MinTier}
	}
	if req.NeedsNetwork {
		if b.Permissions.Network == OpenShellNetDeny || b.Permissions.Network == "" {
			return OpenShellDecision{Allowed: false, Reason: "network access denied", RequiredTier: b.MinTier}
		}
		if !hostAllowed(req.TargetHost, b.Permissions.AllowedHosts) {
			return OpenShellDecision{Allowed: false, Reason: fmt.Sprintf("host %q not in allowlist", req.TargetHost), RequiredTier: b.MinTier}
		}
	}
	for _, wp := range req.WritePaths {
		if !pathWithinAny(wp, b.Permissions.WritePaths) {
			return OpenShellDecision{Allowed: false, Reason: fmt.Sprintf("write path %q outside boundary", wp), RequiredTier: b.MinTier}
		}
	}
	return OpenShellDecision{Allowed: true, RequiredTier: b.MinTier}
}

func hostAllowed(host string, allowlist []string) bool {
	h := strings.ToLower(strings.TrimSpace(host))
	if h == "" {
		return false
	}
	for _, a := range allowlist {
		if strings.ToLower(strings.TrimSpace(a)) == h {
			return true
		}
	}
	return false
}

func pathWithinAny(target string, roots []string) bool {
	t := filepath.Clean(strings.TrimSpace(target))
	if t == "." || t == "" {
		return false
	}
	for _, r := range roots {
		root := filepath.Clean(strings.TrimSpace(r))
		if root == "" {
			continue
		}
		rel, err := filepath.Rel(root, t)
		if err != nil {
			continue
		}
		if rel == "." || (!strings.HasPrefix(rel, ".."+string(filepath.Separator)) && rel != "..") {
			return true
		}
	}
	return false
}
