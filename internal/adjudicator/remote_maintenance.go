package adjudicator

import (
	"fmt"
	"net"
	"net/url"
	"path"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

const MaxRemoteMaintenancePolicyResources = 64

// PositiveCLIReadOnly reports whether command is one invocation accepted by the
// existing gh/git read grammar. A caller read_only hint is never sufficient.
func PositiveCLIReadOnly(command string) bool {
	_, _, err := attenuateCLIGrammar(command)
	return err == nil
}

type RemoteMaintenancePolicyFacts struct {
	ProfileID       string                   `json:"profile_id"`
	OperationDigest string                   `json:"operation_digest"`
	Host            string                   `json:"host"`
	RemoteTarget    string                   `json:"remote_target"`
	Operation       string                   `json:"remote_operation"`
	Paths           []string                 `json:"remote_paths"`
	PathPrefixes    []remotemaint.PathPrefix `json:"remote_path_prefixes"`
	Units           []string                 `json:"remote_units"`
	ResourceIndex   int                      `json:"remote_resource_index"`
	ResourceKind    string                   `json:"remote_resource_kind"`
	Path            string                   `json:"path,omitempty"`
}

// ProjectRemoteMaintenancePolicyFacts maps every resolved resource through the
// existing path and egress hard floors. Only the closed, unit-bound systemd
// file/drop-in namespace is withheld from the local /etc scalar-path meaning;
// every other exact path or typed prefix is projected unchanged.
func ProjectRemoteMaintenancePolicyFacts(grant remotemaint.Grant) ([]RemoteMaintenancePolicyFacts, error) {
	count := len(grant.Paths) + len(grant.PathPrefixes)
	if count == 0 || count > MaxRemoteMaintenancePolicyResources {
		return nil, fmt.Errorf("remote maintenance policy resource count %d outside 1..%d", count, MaxRemoteMaintenancePolicyResources)
	}
	host, err := remoteMaintenancePolicyHost(grant.Target)
	if err != nil {
		return nil, err
	}
	base := RemoteMaintenancePolicyFacts{
		ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest, Host: host, RemoteTarget: grant.Target,
		Operation: grant.Operation, Paths: grant.Paths, PathPrefixes: grant.PathPrefixes, Units: grant.Units,
	}
	out := make([]RemoteMaintenancePolicyFacts, 0, count)
	for _, candidate := range grant.Paths {
		facts := base
		facts.ResourceIndex = len(out)
		facts.ResourceKind = "exact_path"
		facts.Path = remoteMaintenancePolicyPath(candidate, grant.Units)
		out = append(out, facts)
	}
	for _, prefix := range grant.PathPrefixes {
		facts := base
		facts.ResourceIndex = len(out)
		facts.ResourceKind = "filename_prefix"
		candidate := path.Join(prefix.Directory, prefix.FilenamePrefix)
		facts.Path = remoteMaintenancePolicyPath(candidate, grant.Units)
		out = append(out, facts)
	}
	return out, nil
}

func remoteMaintenancePolicyHost(target string) (string, error) {
	host := target
	if strings.Contains(target, "://") {
		u, err := url.Parse(target)
		if err != nil || u.Hostname() == "" {
			return "", fmt.Errorf("remote maintenance target is not representable as a policy host")
		}
		host = u.Hostname()
	} else {
		if at := strings.LastIndex(host, "@"); at >= 0 {
			host = host[at+1:]
		}
		if parsed, _, err := net.SplitHostPort(host); err == nil {
			host = parsed
		} else if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
			host = strings.Trim(host, "[]")
		}
	}
	if host == "" || strings.ContainsAny(host, "/\\@?# \t\r\n") {
		return "", fmt.Errorf("remote maintenance target is not representable as a policy host")
	}
	if strings.Contains(host, ":") && net.ParseIP(strings.Trim(host, "[]")) == nil {
		return "", fmt.Errorf("remote maintenance target is not representable as a policy host")
	}
	return host, nil
}

func remoteMaintenancePolicyPath(candidate string, units []string) string {
	if remoteMaintenanceSystemdResource(candidate, units) {
		// Keep basename-sensitive credential/self-modify globs effective without
		// giving a remote /etc path the local-host meaning of policy "/etc/".
		return "remote-systemd:" + strings.TrimPrefix(candidate, "/etc/")
	}
	return candidate
}

func remoteMaintenanceSystemdResource(candidate string, units []string) bool {
	const root = "/etc/systemd/system/"
	if !strings.HasPrefix(candidate, root) {
		return false
	}
	rel := strings.TrimPrefix(candidate, root)
	for _, unit := range units {
		if rel == unit {
			return true
		}
		if rel == unit+".d" {
			return true
		}
		dropIn := unit + ".d/"
		if strings.HasPrefix(rel, dropIn) {
			name := strings.TrimPrefix(rel, dropIn)
			return name != "" && !strings.Contains(name, "/") && (strings.HasSuffix(name, ".conf") || strings.HasSuffix(name, ".conf.fak-recovery-tmp"))
		}
	}
	return false
}
