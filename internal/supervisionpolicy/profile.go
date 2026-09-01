package supervisionpolicy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// ProcessClass is an immutable lifecycle and privilege class assigned at admission.
type ProcessClass string

const (
	ProcessClassRootService     ProcessClass = "root-service"
	ProcessClassRegularAgent    ProcessClass = "regular-agent"
	ProcessClassSubagent        ProcessClass = "subagent"
	ProcessClassServeController ProcessClass = "serve-controller"
	ProcessClassServeProxy      ProcessClass = "serve-proxy"
	ProcessClassServeReplica    ProcessClass = "serve-replica"
)

// RequiredProcessClasses is the closed process-class vocabulary.
var RequiredProcessClasses = []ProcessClass{
	ProcessClassRootService,
	ProcessClassRegularAgent,
	ProcessClassSubagent,
	ProcessClassServeController,
	ProcessClassServeProxy,
	ProcessClassServeReplica,
}

var (
	ErrPrivilegeWidening = errors.New("profile widens privileges")
	ErrResourceWidening  = errors.New("profile widens resources")
	ErrIdentityMutation  = errors.New("admitted identity is immutable")
	ErrInvalidTopology   = errors.New("invalid supervision topology")
	ErrRestartWidening   = errors.New("profile widens restart budget")
)

// MemberIdentity separates stable logical membership from replaceable process generations.
type MemberIdentity struct {
	Member       MemberID     `json:"member"`
	Generation   uint64       `json:"generation"`
	Class        ProcessClass `json:"class"`
	ParentDomain DomainID     `json:"parent_domain"`
}

// ValidateIdentityTransition permits generation replacement but not logical reclassification
// or movement between supervision domains.
func ValidateIdentityTransition(admitted, next MemberIdentity) error {
	if admitted.Member == "" || next.Member == "" || admitted.Member != next.Member ||
		admitted.Class != next.Class || admitted.ParentDomain != next.ParentDomain {
		return ErrIdentityMutation
	}
	if next.Generation < admitted.Generation {
		return fmt.Errorf("%w: generation regressed", ErrIdentityMutation)
	}
	return nil
}

// SecretHandle names an opaque secret reference. Secret material and bulk environment
// inheritance are deliberately outside the profile schema.
type SecretHandle struct {
	Name   string `json:"name"`
	Handle string `json:"handle"`
}

// ResourceCeiling is bounded by every profile ancestor. Zero means no allocation.
type ResourceCeiling struct {
	CPUUnits    uint64 `json:"cpu_units"`
	MemoryBytes uint64 `json:"memory_bytes"`
	Processes   uint32 `json:"processes"`
}

// RestartBudget bounds restarts inside a supervision domain.
type RestartBudget struct {
	MaxRestarts uint32        `json:"max_restarts"`
	Window      time.Duration `json:"window"`
}

// OperationalScalars use nearest-explicit-value inheritance.
type OperationalScalars struct {
	LogLevel      string        `json:"log_level"`
	ShutdownGrace time.Duration `json:"shutdown_grace"`
}

// Profile is a fully resolved, deterministic admission profile.
type Profile struct {
	Operational  OperationalScalars `json:"operational"`
	Capabilities []string           `json:"capabilities"`
	Resources    ResourceCeiling    `json:"resources"`
	Secrets      []SecretHandle     `json:"secrets"`
	Restart      RestartBudget      `json:"restart"`
}

// OperationalPatch carries explicit scalar values; nil means inherit.
type OperationalPatch struct {
	LogLevel      *string
	ShutdownGrace *time.Duration
}

// ResourcePatch carries explicit resource ceilings; nil means inherit.
type ResourcePatch struct {
	CPUUnits    *uint64
	MemoryBytes *uint64
	Processes   *uint32
}

// RestartPatch carries an explicit domain restart budget; nil means inherit.
type RestartPatch struct {
	MaxRestarts *uint32
	Window      *time.Duration
}

// ProfilePatch is one inheritance layer. Capabilities, when non-nil, are the
// complete desired set. A capability removed by an ancestor can only be restored
// when that ancestor explicitly listed it in GrantCapabilities.
type ProfilePatch struct {
	Operational       OperationalPatch
	Capabilities      *[]string
	GrantCapabilities []string
	Resources         ResourcePatch
	Secrets           *[]SecretHandle
	Restart           RestartPatch
}

// ProfileLayers fixes the only supported resolution order.
type ProfileLayers struct {
	CompiledDefaults ProfilePatch
	Installation     ProfilePatch
	ProcessClass     ProfilePatch
	ParentDomain     ProfilePatch
	Instance         ProfilePatch
}

// ResolveProfile resolves defaults -> installation -> process class -> parent
// domain -> instance and returns a digest of the canonical resolved profile.
func ResolveProfile(layers ProfileLayers) (Profile, string, error) {
	ordered := []ProfilePatch{layers.CompiledDefaults, layers.Installation, layers.ProcessClass, layers.ParentDomain, layers.Instance}
	var out Profile
	active := map[string]struct{}{}
	grants := map[string]struct{}{}
	initialized := false
	for i, patch := range ordered {
		if err := applyPatch(&out, active, grants, patch, initialized); err != nil {
			return Profile{}, "", fmt.Errorf("profile layer %d: %w", i, err)
		}
		initialized = true
	}
	canonicalizeProfile(&out)
	encoded, err := json.Marshal(out)
	if err != nil {
		return Profile{}, "", err
	}
	sum := sha256.Sum256(encoded)
	return out, hex.EncodeToString(sum[:]), nil
}

func applyPatch(out *Profile, active, grants map[string]struct{}, patch ProfilePatch, initialized bool) error {
	if patch.Operational.LogLevel != nil {
		out.Operational.LogLevel = *patch.Operational.LogLevel
	}
	if patch.Operational.ShutdownGrace != nil {
		if *patch.Operational.ShutdownGrace < 0 {
			return fmt.Errorf("negative shutdown grace")
		}
		out.Operational.ShutdownGrace = *patch.Operational.ShutdownGrace
	}

	if patch.Capabilities != nil {
		desired, err := stringSet(*patch.Capabilities)
		if err != nil {
			return err
		}
		if initialized {
			for capability := range desired {
				if _, held := active[capability]; held {
					continue
				}
				if _, granted := grants[capability]; !granted {
					return fmt.Errorf("%w: capability %q", ErrPrivilegeWidening, capability)
				}
			}
		}
		clear(active)
		for capability := range desired {
			active[capability] = struct{}{}
		}
	}
	for _, capability := range patch.GrantCapabilities {
		capability = strings.TrimSpace(capability)
		if capability == "" {
			return fmt.Errorf("empty capability grant")
		}
		if _, held := active[capability]; !held {
			return fmt.Errorf("%w: cannot grant unheld capability %q", ErrPrivilegeWidening, capability)
		}
		grants[capability] = struct{}{}
	}

	if err := applyResources(&out.Resources, patch.Resources, initialized); err != nil {
		return err
	}
	if patch.Secrets != nil {
		secrets, err := normalizeSecrets(*patch.Secrets)
		if err != nil {
			return err
		}
		out.Secrets = secrets
	}
	if err := applyRestart(&out.Restart, patch.Restart, initialized); err != nil {
		return err
	}
	out.Capabilities = setStrings(active)
	return nil
}

func applyResources(current *ResourceCeiling, patch ResourcePatch, bounded bool) error {
	if patch.CPUUnits != nil {
		if bounded && *patch.CPUUnits > current.CPUUnits {
			return fmt.Errorf("%w: cpu", ErrResourceWidening)
		}
		current.CPUUnits = *patch.CPUUnits
	}
	if patch.MemoryBytes != nil {
		if bounded && *patch.MemoryBytes > current.MemoryBytes {
			return fmt.Errorf("%w: memory", ErrResourceWidening)
		}
		current.MemoryBytes = *patch.MemoryBytes
	}
	if patch.Processes != nil {
		if bounded && *patch.Processes > current.Processes {
			return fmt.Errorf("%w: processes", ErrResourceWidening)
		}
		current.Processes = *patch.Processes
	}
	return nil
}

func applyRestart(current *RestartBudget, patch RestartPatch, bounded bool) error {
	if patch.MaxRestarts != nil {
		if bounded && *patch.MaxRestarts > current.MaxRestarts {
			return fmt.Errorf("%w: max restarts", ErrRestartWidening)
		}
		current.MaxRestarts = *patch.MaxRestarts
	}
	if patch.Window != nil {
		if *patch.Window <= 0 {
			return fmt.Errorf("restart window must be positive")
		}
		if bounded && current.Window > 0 && *patch.Window < current.Window {
			return fmt.Errorf("%w: shorter window", ErrRestartWidening)
		}
		current.Window = *patch.Window
	}
	return nil
}

// DomainSpec declares a supervision domain and its enclosing restart budget.
type DomainSpec struct {
	ID            DomainID
	Parent        DomainID
	RestartBudget RestartBudget
}

// MemberSpec declares stable topology membership; Generation identifies the current process.
type MemberSpec struct {
	Identity MemberIdentity
	Parent   MemberID
}

// Topology is a complete declared supervision graph.
type Topology struct {
	Domains []DomainSpec
	Members []MemberSpec
}

// ValidateTopology rejects unknown classes, undeclared domains, invalid parent
// relationships, cycles, duplicate identities, and restart budgets wider than parents.
func ValidateTopology(topology Topology) error {
	domains := make(map[DomainID]DomainSpec, len(topology.Domains))
	for _, domain := range topology.Domains {
		if domain.ID == "" || domain.RestartBudget.Window <= 0 {
			return fmt.Errorf("%w: malformed domain", ErrInvalidTopology)
		}
		if _, exists := domains[domain.ID]; exists {
			return fmt.Errorf("%w: duplicate domain %q", ErrInvalidTopology, domain.ID)
		}
		domains[domain.ID] = domain
	}
	for id, domain := range domains {
		seen := map[DomainID]struct{}{id: {}}
		for domain.Parent != "" {
			parent, ok := domains[domain.Parent]
			if !ok {
				return fmt.Errorf("%w: undeclared parent domain %q", ErrInvalidTopology, domain.Parent)
			}
			if _, cycle := seen[parent.ID]; cycle {
				return fmt.Errorf("%w: domain cycle", ErrInvalidTopology)
			}
			seen[parent.ID] = struct{}{}
			if domain.RestartBudget.MaxRestarts > parent.RestartBudget.MaxRestarts || domain.RestartBudget.Window < parent.RestartBudget.Window {
				return fmt.Errorf("%w: domain %q exceeds parent budget", ErrInvalidTopology, domain.ID)
			}
			domain = parent
		}
	}

	members := make(map[MemberID]MemberSpec, len(topology.Members))
	rootCount := 0
	for _, member := range topology.Members {
		id := member.Identity
		if id.Member == "" || id.Generation == 0 || !knownProcessClass(id.Class) {
			return fmt.Errorf("%w: malformed member", ErrInvalidTopology)
		}
		if _, ok := domains[id.ParentDomain]; !ok {
			return fmt.Errorf("%w: undeclared domain %q", ErrInvalidTopology, id.ParentDomain)
		}
		if _, exists := members[id.Member]; exists {
			return fmt.Errorf("%w: duplicate member %q", ErrInvalidTopology, id.Member)
		}
		members[id.Member] = member
		if id.Class == ProcessClassRootService {
			rootCount++
		}
	}
	if rootCount != 1 {
		return fmt.Errorf("%w: require exactly one root-service", ErrInvalidTopology)
	}
	for _, member := range members {
		if member.Identity.Class == ProcessClassRootService {
			if member.Parent != "" {
				return fmt.Errorf("%w: root-service has parent", ErrInvalidTopology)
			}
			continue
		}
		parent, ok := members[member.Parent]
		if !ok {
			return fmt.Errorf("%w: undeclared parent member %q", ErrInvalidTopology, member.Parent)
		}
		if !allowedParent(member.Identity.Class, parent.Identity.Class) {
			return fmt.Errorf("%w: %s cannot be parented by %s", ErrInvalidTopology, member.Identity.Class, parent.Identity.Class)
		}
		seen := map[MemberID]struct{}{member.Identity.Member: {}}
		cursor := member
		for cursor.Parent != "" {
			if _, cycle := seen[cursor.Parent]; cycle {
				return fmt.Errorf("%w: member cycle", ErrInvalidTopology)
			}
			seen[cursor.Parent] = struct{}{}
			cursor = members[cursor.Parent]
		}
	}
	return nil
}

func knownProcessClass(class ProcessClass) bool {
	for _, required := range RequiredProcessClasses {
		if class == required {
			return true
		}
	}
	return false
}

func allowedParent(child, parent ProcessClass) bool {
	switch child {
	case ProcessClassRegularAgent, ProcessClassServeController:
		return parent == ProcessClassRootService
	case ProcessClassSubagent:
		return parent == ProcessClassRegularAgent
	case ProcessClassServeProxy, ProcessClassServeReplica:
		return parent == ProcessClassServeController
	default:
		return false
	}
}

func stringSet(values []string) (map[string]struct{}, error) {
	set := make(map[string]struct{}, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			return nil, fmt.Errorf("empty capability")
		}
		set[value] = struct{}{}
	}
	return set, nil
}

func setStrings(set map[string]struct{}) []string {
	values := make([]string, 0, len(set))
	for value := range set {
		values = append(values, value)
	}
	sort.Strings(values)
	return values
}

func normalizeSecrets(values []SecretHandle) ([]SecretHandle, error) {
	seen := make(map[string]struct{}, len(values))
	out := append([]SecretHandle(nil), values...)
	for _, secret := range out {
		if strings.TrimSpace(secret.Name) == "" || strings.TrimSpace(secret.Handle) == "" {
			return nil, fmt.Errorf("invalid secret handle")
		}
		if _, duplicate := seen[secret.Name]; duplicate {
			return nil, fmt.Errorf("duplicate secret name %q", secret.Name)
		}
		seen[secret.Name] = struct{}{}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out, nil
}

func canonicalizeProfile(profile *Profile) {
	sort.Strings(profile.Capabilities)
	sort.Slice(profile.Secrets, func(i, j int) bool { return profile.Secrets[i].Name < profile.Secrets[j].Name })
}
