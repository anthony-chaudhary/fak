package harnessresolve

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnesscompose"
	"github.com/anthony-chaudhary/fak/internal/maputil"
	"github.com/anthony-chaudhary/fak/internal/stackresolve"
	"github.com/anthony-chaudhary/fak/pkg/harnesskit/lockv2"
)

// Invariant: resolution produces deterministic, permutation-invariant locks whose SHA-256 digests uniquely seal content.

// Schema specifies the canonical JSON schema URI for harness product specification manifests.
const Schema = "fak.harness-product/v1alpha1"

// LockSchemaV2 specifies the version 2 JSON schema URI for compiled multi-platform harness product locks.
const LockSchemaV2 = "fak.harness-product-lock/v2"

// ProductLockSchemaV2 is an alias for LockSchemaV2.
const ProductLockSchemaV2 = LockSchemaV2

// LockSchema specifies the version 1alpha2 JSON schema URI for compiled harness product locks.
const LockSchema = "fak.harness-product-lock/v1alpha2"

// LegacyLockSchema specifies the backward-compatible version 1alpha1 schema URI for launchable product locks.
const LegacyLockSchema = "fak.harness-product-lock/v1alpha1"

// SecretPlaintextLeakError is the typed refusal token when a secret asset contains an inline plaintext value.
const SecretPlaintextLeakError = lockv2.SecretPlaintextLeakError

// ErrSecretPlaintextLeak is an alias for SecretPlaintextLeakError.
const ErrSecretPlaintextLeak = SecretPlaintextLeakError

var secretRefPattern = regexp.MustCompile(`^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$`)

// PlatformRequirement defines an execution platform target (OS, architecture, contract).
type PlatformRequirement = lockv2.PlatformRequirement

// Manifest defines root components, environment constraints, and asset layers for a harness product.
type Manifest struct {
	Schema        string                  `json:"schema"`
	Roots         []string                `json:"roots"`
	Platforms     []PlatformRequirement   `json:"platforms,omitempty"`
	Components    []Component             `json:"components"`
	Compatibility Compatibility           `json:"compatibility,omitempty"`
	Budget        Budget                  `json:"budget,omitempty"`
	Assets        harnesscompose.Manifest `json:"assets"`
}

// Component describes a discrete capability provider or dependency within a harness manifest.
type Component struct {
	ID            string                `json:"id"`
	Version       string                `json:"version"`
	Digest        string                `json:"digest"`
	Source        string                `json:"source"`
	Provides      []string              `json:"provides,omitempty"`
	Requires      []Requirement         `json:"requires,omitempty"`
	Conflicts     []string              `json:"conflicts,omitempty"`
	Compatibility Compatibility         `json:"compatibility,omitempty"`
	Cost          Budget                `json:"cost,omitempty"`
	Adapters      []string              `json:"adapters,omitempty"`
	Evidence      stackresolve.Evidence `json:"evidence"`
}

// Requirement defines a required or optional dependency capability along with an optional version range.
type Requirement struct {
	Capability string `json:"capability"`
	Range      string `json:"range,omitempty"`
	Optional   bool   `json:"optional,omitempty"`
}

// Compatibility restricts component admission by operating system, CPU architecture, and runtime contract.
type Compatibility struct {
	OS       []string `json:"os,omitempty"`
	Arch     []string `json:"arch,omitempty"`
	Contract string   `json:"contract,omitempty"`
}

// Budget specifies resource upper bounds on context token consumption, resident memory, and concurrent workers.
type Budget struct {
	ContextTokens int `json:"context_tokens,omitempty"`
	MemoryMiB     int `json:"memory_mib,omitempty"`
	Workers       int `json:"workers,omitempty"`
}

// Environment defines execution platform attributes including target operating system, architecture, and contract.
type Environment struct {
	OS       string `json:"os"`
	Arch     string `json:"arch"`
	Contract string `json:"contract"`
}

// LockedComponent represents an admitted component with frozen version, provenance reason, and budget costs.
type LockedComponent struct {
	ID            string        `json:"id"`
	Version       string        `json:"version"`
	Digest        string        `json:"digest"`
	Source        string        `json:"source"`
	Reason        string        `json:"reason"`
	Provider      string        `json:"provider"`
	Provides      []string      `json:"provides,omitempty"`
	Requires      []Requirement `json:"requires,omitempty"`
	Conflicts     []string      `json:"conflicts,omitempty"`
	Compatibility Compatibility `json:"compatibility,omitempty"`
	Cost          Budget        `json:"cost,omitempty"`
	Adapters      []string      `json:"adapters,omitempty"`
}

// Lock provides a sealed, reproducible description of resolved components, effective assets, and decision traces.
type Lock struct {
	Schema      string                          `json:"schema"`
	ID          string                          `json:"id"`
	Platforms   []PlatformRequirement           `json:"platforms,omitempty"`
	Environment Environment                     `json:"environment,omitempty"`
	Budget      Budget                          `json:"budget"`
	Components  []LockedComponent               `json:"components"`
	Assets      []harnesscompose.EffectiveAsset `json:"assets"`
	AssetTrace  []harnesscompose.Trace          `json:"asset_trace"`
	Decisions   []stackresolve.Decision         `json:"decisions"`
}

// matchesPlatform reports whether the lock targets the given operating system and CPU architecture.
func (l Lock) matchesPlatform(targetOS, targetArch string) bool {
	if len(l.Platforms) == 0 {
		if l.Environment.OS != "" || l.Environment.Arch != "" {
			return (l.Environment.OS == "" || l.Environment.OS == targetOS) &&
				(l.Environment.Arch == "" || l.Environment.Arch == targetArch)
		}
		return true
	}
	for _, p := range l.Platforms {
		if (p.OS == "" || p.OS == targetOS) && (p.Arch == "" || p.Arch == targetArch) {
			return true
		}
	}
	return false
}

// validatePlatforms verifies that every component is compatible with declared platforms.
func (l Lock) validatePlatforms() error {
	platforms := l.Platforms
	if len(platforms) == 0 && (l.Environment.OS != "" || l.Environment.Arch != "") {
		platforms = []PlatformRequirement{{OS: l.Environment.OS, Arch: l.Environment.Arch, Contract: l.Environment.Contract}}
	}
	for _, p := range platforms {
		for _, comp := range l.Components {
			if len(comp.Compatibility.OS) > 0 {
				matched := false
				for _, os := range comp.Compatibility.OS {
					if os == p.OS {
						matched = true
						break
					}
				}
				if !matched {
					return fmt.Errorf("component %q incompatible with platform OS %q", comp.ID, p.OS)
				}
			}
			if len(comp.Compatibility.Arch) > 0 {
				matched := false
				for _, arch := range comp.Compatibility.Arch {
					if arch == p.Arch {
						matched = true
						break
					}
				}
				if !matched {
					return fmt.Errorf("component %q incompatible with platform arch %q", comp.ID, p.Arch)
				}
			}
			if comp.Compatibility.Contract != "" && p.Contract != "" && comp.Compatibility.Contract != p.Contract {
				return fmt.Errorf("component %q requires contract %q, platform has %q", comp.ID, comp.Compatibility.Contract, p.Contract)
			}
		}
	}
	return nil
}

// Explanation accounts for why a specific dependency capability was bound to a selected provider component.
type Explanation struct {
	Capability string `json:"capability"`
	Range      string `json:"range,omitempty"`
	From       string `json:"from"`
	Chosen     string `json:"chosen"`
	Reason     string `json:"reason"`
}

// Result bundles the compiled product lock alongside human-readable dependency resolution explanations.
type Result struct {
	Lock    Lock          `json:"lock"`
	Explain []Explanation `json:"explain"`
}

// Parse deserializes and validates raw JSON configuration bytes into a structured Manifest model.
// Precondition: raw input must contain well-formed JSON bytes conforming to the product schema with non-empty roots.
// Postcondition: returns a validated Manifest with verified semantic versioning and cryptographic digest formats, or a descriptive error.
func Parse(raw []byte) (Manifest, error) {
	var manifest Manifest
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return Manifest{}, fmt.Errorf("parse product: %w", err)
	}
	if manifest.Schema != Schema {
		return Manifest{}, fmt.Errorf("schema must be %q", Schema)
	}
	if len(manifest.Roots) == 0 {
		return Manifest{}, fmt.Errorf("at least one root is required")
	}
	seen := map[string]bool{}
	for _, component := range manifest.Components {
		if component.ID == "" || component.Version == "" || component.Digest == "" || component.Source == "" {
			return Manifest{}, fmt.Errorf("component id/version/digest/source are required")
		}
		key := component.ID + "@" + component.Version
		if seen[key] {
			return Manifest{}, fmt.Errorf("duplicate component %q", key)
		}
		seen[key] = true
		if _, err := parseVersion(component.Version); err != nil {
			return Manifest{}, fmt.Errorf("component %q: %w", component.ID, err)
		}
		if !strings.HasPrefix(component.Digest, "sha256:") || len(strings.TrimPrefix(component.Digest, "sha256:")) == 0 {
			return Manifest{}, fmt.Errorf("component %q digest must be sha256:<value>", component.ID)
		}
		for _, req := range component.Requires {
			if req.Capability == "" {
				return Manifest{}, fmt.Errorf("component %q has empty requirement", component.ID)
			}
			if _, err := parseRange(req.Range); err != nil {
				return Manifest{}, fmt.Errorf("component %q requirement %q: %w", component.ID, req.Capability, err)
			}
		}
	}
	return manifest, nil
}

// CheckMultiPlatformCompatibility verifies that all components are compatible with all target platforms.
func CheckMultiPlatformCompatibility(components []Component, platforms []PlatformRequirement) error {
	for _, p := range platforms {
		env := Environment{OS: p.OS, Arch: p.Arch, Contract: p.Contract}
		for _, c := range components {
			if err := checkCompatibility(c.Compatibility, env, "component "+c.ID); err != nil {
				return err
			}
		}
	}
	return nil
}

// Resolve compiles asset layers and component dependencies into an immutable product lock against target environment rules.
// Precondition: manifest must contain valid roots and component declarations matching the provided target environment.
// Postcondition: returns an immutable product lock with resolved dependencies and budget guarantees, or an error if constraints fail.
func Resolve(ctx context.Context, manifest Manifest, selectedLayers []string, env Environment) (Result, error) {
	assets, err := harnesscompose.Compose(manifest.Assets, selectedLayers)
	if err != nil {
		return Result{}, fmt.Errorf("compose assets: %w", err)
	}
	for _, a := range assets.Assets {
		if a.Kind == "secret" {
			if a.Value != "" {
				return Result{}, fmt.Errorf("%s: locked secret %q cannot contain plaintext value", SecretPlaintextLeakError, a.ID)
			}
			if a.Ref == "" || !secretRefPattern.MatchString(a.Ref) {
				return Result{}, fmt.Errorf("locked secret %q invalid or missing opaque reference %q (must match ^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$)", a.ID, a.Ref)
			}
		}
	}
	if len(manifest.Platforms) > 0 {
		if err := CheckMultiPlatformCompatibility(manifest.Components, manifest.Platforms); err != nil {
			return Result{}, err
		}
	} else {
		if err := checkCompatibility(manifest.Compatibility, env, "product"); err != nil {
			return Result{}, err
		}
	}
	components, err := bindRequirements(manifest.Components)
	if err != nil {
		return Result{}, err
	}
	if err := checkCycles(components); err != nil {
		return Result{}, err
	}
	stack := make([]stackresolve.Component, 0, len(components))
	byID := map[string]Component{}
	for _, component := range components {
		if len(manifest.Platforms) == 0 {
			if err := checkCompatibility(component.Compatibility, env, "component "+component.ID); err != nil {
				return Result{}, err
			}
		}
		byID[component.ID] = component
		relations := make([]stackresolve.Relation, 0, len(component.Requires)+len(component.Conflicts))
		for _, req := range component.Requires {
			relations = append(relations, stackresolve.Relation{Kind: relationKind(req.Optional), Target: req.Capability, Evidence: component.Evidence})
		}
		for _, conflict := range component.Conflicts {
			relations = append(relations, stackresolve.Relation{Kind: stackresolve.Conflicts, Target: conflict, Evidence: component.Evidence})
		}
		stack = append(stack, stackresolve.Component{ID: component.ID, Kind: "harness-component", Version: component.Version, Provides: component.Provides, Relations: relations, Evidence: component.Evidence})
	}
	receipt, err := stackresolve.Resolve(ctx, "harness-product", manifest.Roots, stackresolve.ManifestProvider{Manifest: stackresolve.Manifest{Schema: "fak.stack.manifest/v1", Components: stack}})
	if err != nil {
		return Result{}, err
	}
	if receipt.Status != "allow" {
		return Result{}, fmt.Errorf("dependency resolution refused: %s %s", conflictCode(receipt), conflictWanted(receipt))
	}
	used := Budget{}
	locked := make([]LockedComponent, 0, len(receipt.Selected))
	for _, selected := range receipt.Selected {
		component, ok := byID[selected.ID]
		if !ok {
			return Result{}, fmt.Errorf("selected component %q missing metadata", selected.ID)
		}
		used = addBudget(used, component.Cost)
		locked = append(locked, LockedComponent{ID: component.ID, Version: component.Version, Digest: component.Digest, Source: component.Source, Reason: decisionReason(receipt.Decisions, selected.ID), Provider: providerFor(receipt.Decisions, selected.ID), Provides: sortedStrings(component.Provides), Requires: sortedRequirements(component.Requires), Conflicts: sortedStrings(component.Conflicts), Compatibility: normalizedCompatibility(component.Compatibility), Cost: component.Cost, Adapters: sortedStrings(component.Adapters)})
	}
	if err := withinBudget(used, manifest.Budget); err != nil {
		return Result{}, err
	}
	sort.Slice(locked, func(i, j int) bool { return locked[i].ID < locked[j].ID })
	lockSchema := LockSchema
	if len(manifest.Platforms) > 0 {
		lockSchema = LockSchemaV2
	}
	lock := Lock{Schema: lockSchema, Platforms: manifest.Platforms, Environment: env, Budget: used, Components: locked, Assets: assets.Assets, AssetTrace: assets.Trace, Decisions: receipt.Decisions}
	lock.ID, err = lockID(lock)
	if err != nil {
		return Result{}, err
	}
	return Result{Lock: lock, Explain: explanations(components, receipt.Selected)}, nil
}

// Precondition: input component list must contain valid versions and non-empty requirement capability targets.
// Postcondition: maps capability requirements to matching provider IDs and returns an error on missing or ambiguous matches.
func bindRequirements(input []Component) ([]Component, error) {
	components := append([]Component(nil), input...)
	sort.Slice(components, func(i, j int) bool {
		if components[i].ID != components[j].ID {
			return components[i].ID < components[j].ID
		}
		return components[i].Version < components[j].Version
	})
	providers := map[string][]Component{}
	for _, component := range components {
		providers[component.ID] = appendProvider(providers[component.ID], component)
		for _, capability := range component.Provides {
			providers[capability] = appendProvider(providers[capability], component)
		}
	}
	for i := range components {
		for j := range components[i].Requires {
			req := &components[i].Requires[j]
			matches := make([]Component, 0)
			for _, candidate := range providers[req.Capability] {
				ok, err := matchRange(candidate.Version, req.Range)
				if err != nil {
					return nil, err
				}
				if ok {
					matches = append(matches, candidate)
				}
			}
			if len(matches) == 0 {
				if req.Optional {
					continue
				}
				return nil, fmt.Errorf("component %q missing dependency %s %s", components[i].ID, req.Capability, req.Range)
			}
			if len(matches) > 1 {
				return nil, fmt.Errorf("component %q dependency %s %s has ambiguous providers %s", components[i].ID, req.Capability, req.Range, providerNames(matches))
			}
			req.Capability = matches[0].ID
		}
	}
	return components, nil
}

// Precondition: components must have bound capability IDs with populated dependency edge requirements.
// Postcondition: returns a descriptive error if any directed dependency cycle exists among components.
func checkCycles(components []Component) error {
	edges := map[string][]string{}
	ids := map[string]bool{}
	for _, c := range components {
		ids[c.ID] = true
		for _, r := range c.Requires {
			if !r.Optional || ids[r.Capability] {
				edges[c.ID] = append(edges[c.ID], r.Capability)
			}
		}
	}
	state := map[string]int{}
	path := []string{}
	var visit func(string) error
	visit = func(id string) error {
		if state[id] == 1 {
			return fmt.Errorf("dependency cycle: %s -> %s", strings.Join(path, " -> "), id)
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		path = append(path, id)
		for _, next := range edges[id] {
			if err := visit(next); err != nil {
				return err
			}
		}
		path = path[:len(path)-1]
		state[id] = 2
		return nil
	}
	keys := maputil.SortedKeys(edges)
	for _, id := range keys {
		if err := visit(id); err != nil {
			return err
		}
	}
	return nil
}

// Precondition: target environment must declare non-empty OS, architecture, and runtime contract strings.
// Postcondition: returns an error if component or product compatibility rules reject the target environment.
func checkCompatibility(c Compatibility, env Environment, subject string) error {
	if len(c.OS) > 0 && !contains(c.OS, env.OS) {
		return fmt.Errorf("%s incompatible OS %q", subject, env.OS)
	}
	if len(c.Arch) > 0 && !contains(c.Arch, env.Arch) {
		return fmt.Errorf("%s incompatible arch %q", subject, env.Arch)
	}
	if c.Contract != "" && c.Contract != env.Contract {
		return fmt.Errorf("%s requires contract %q, got %q", subject, c.Contract, env.Contract)
	}
	return nil
}

// Precondition: both budget allocations must specify non-negative token, memory, and worker values.
// Postcondition: returns the summed resource demands across context tokens, memory, and worker pools.
func addBudget(a, b Budget) Budget {
	return Budget{ContextTokens: a.ContextTokens + b.ContextTokens, MemoryMiB: a.MemoryMiB + b.MemoryMiB, Workers: a.Workers + b.Workers}
}

// Precondition: used budget reflects actual aggregated consumption and limit specifies boundary caps.
// Postcondition: returns a descriptive error if any individual resource dimension exceeds its limit.
func withinBudget(used, limit Budget) error {
	if limit.ContextTokens > 0 && used.ContextTokens > limit.ContextTokens {
		return fmt.Errorf("context budget exceeded: %d > %d", used.ContextTokens, limit.ContextTokens)
	}
	if limit.MemoryMiB > 0 && used.MemoryMiB > limit.MemoryMiB {
		return fmt.Errorf("memory budget exceeded: %d > %d", used.MemoryMiB, limit.MemoryMiB)
	}
	if limit.Workers > 0 && used.Workers > limit.Workers {
		return fmt.Errorf("worker budget exceeded: %d > %d", used.Workers, limit.Workers)
	}
	return nil
}
func lockID(lock Lock) (string, error) {
	if lock.Schema == LockSchemaV2 {
		copy := lock
		copy.ID = ""
		raw, err := json.Marshal(copy)
		if err != nil {
			return "", err
		}
		canonical, err := lockv2.CanonicalizeJSON(raw)
		if err != nil {
			return "", err
		}
		sum := sha256.Sum256(canonical)
		return "sha256:" + hex.EncodeToString(sum[:]), nil
	}
	copy := lock
	copy.ID = ""
	raw, err := json.Marshal(copy)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
func providerNames(cs []Component) string {
	names := make([]string, len(cs))
	for i, c := range cs {
		names[i] = c.ID + "@" + c.Version
	}
	sort.Strings(names)
	return strings.Join(names, ",")
}
func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}

type version [3]int

func parseVersion(raw string) (version, error) {
	raw = strings.TrimPrefix(strings.TrimSpace(raw), "v")
	parts := strings.Split(raw, ".")
	if len(parts) != 3 {
		return version{}, fmt.Errorf("version %q must be MAJOR.MINOR.PATCH", raw)
	}
	var v version
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return version{}, fmt.Errorf("invalid version %q", raw)
		}
		v[i] = n
	}
	return v, nil
}

type versionRange struct {
	op string
	v  version
}

func parseRange(raw string) (versionRange, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return versionRange{op: ">=", v: version{}}, nil
	}
	for _, op := range []string{">=", "<=", "==", ">", "<"} {
		if strings.HasPrefix(raw, op) {
			v, err := parseVersion(strings.TrimSpace(strings.TrimPrefix(raw, op)))
			return versionRange{op: op, v: v}, err
		}
	}
	v, err := parseVersion(raw)
	return versionRange{op: "==", v: v}, err
}
func matchRange(raw, constraint string) (bool, error) {
	v, err := parseVersion(raw)
	if err != nil {
		return false, err
	}
	r, err := parseRange(constraint)
	if err != nil {
		return false, err
	}
	cmp := 0
	for i := 0; i < 3; i++ {
		if v[i] < r.v[i] {
			cmp = -1
			break
		}
		if v[i] > r.v[i] {
			cmp = 1
			break
		}
	}
	switch r.op {
	case ">=":
		return cmp >= 0, nil
	case "<=":
		return cmp <= 0, nil
	case ">":
		return cmp > 0, nil
	case "<":
		return cmp < 0, nil
	default:
		return cmp == 0, nil
	}
}

func relationKind(optional bool) stackresolve.RelationKind {
	if optional {
		return stackresolve.Optional
	}
	return stackresolve.Requires
}
func decisionReason(decisions []stackresolve.Decision, id string) string {
	for _, decision := range decisions {
		if decision.Chosen == id {
			return string(decision.Relation) + " " + decision.Wanted + " from " + decision.From
		}
	}
	return "selected root"
}
func providerFor(decisions []stackresolve.Decision, id string) string {
	for _, decision := range decisions {
		if decision.Chosen == id {
			return decision.Evidence.Source
		}
	}
	return "manifest"
}

func conflictCode(receipt stackresolve.Receipt) string {
	if receipt.Conflict == nil {
		return "UNKNOWN"
	}
	return receipt.Conflict.Code
}
func conflictWanted(receipt stackresolve.Receipt) string {
	if receipt.Conflict == nil {
		return "unknown"
	}
	return receipt.Conflict.Wanted
}

func explanations(components []Component, selected []stackresolve.Component) []Explanation {
	selectedSet := map[string]bool{}
	for _, c := range selected {
		selectedSet[c.ID] = true
	}
	out := []Explanation{}
	for _, c := range components {
		if !selectedSet[c.ID] {
			continue
		}
		if len(c.Requires) == 0 {
			out = append(out, Explanation{From: c.ID, Chosen: c.ID, Reason: "selected root or provider"})
		}
		for _, r := range c.Requires {
			if selectedSet[r.Capability] {
				out = append(out, Explanation{Capability: r.Capability, Range: r.Range, From: c.ID, Chosen: r.Capability, Reason: "required dependency"})
			}
		}
		for _, capability := range c.Provides {
			out = append(out, Explanation{Capability: capability, From: c.ID, Chosen: c.ID, Reason: "declared capability provider"})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Capability != out[j].Capability {
			return out[i].Capability < out[j].Capability
		}
		if out[i].From != out[j].From {
			return out[i].From < out[j].From
		}
		return out[i].Chosen < out[j].Chosen
	})
	return out
}
func sortedStrings(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

func sortedRequirements(in []Requirement) []Requirement {
	out := append([]Requirement(nil), in...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].Capability != out[j].Capability {
			return out[i].Capability < out[j].Capability
		}
		if out[i].Range != out[j].Range {
			return out[i].Range < out[j].Range
		}
		return !out[i].Optional && out[j].Optional
	})
	return out
}

func normalizedCompatibility(in Compatibility) Compatibility {
	in.OS = sortedStrings(in.OS)
	in.Arch = sortedStrings(in.Arch)
	return in
}

func appendProvider(in []Component, component Component) []Component {
	for _, existing := range in {
		if existing.ID == component.ID && existing.Version == component.Version {
			return in
		}
	}
	return append(in, component)
}

// VerifyLock checks that a lock has the current schema and that its ID matches
// the canonical contents. Callers must verify before treating an unchanged ID
// as an admission decision.
// Precondition: lock must declare a recognized schema identifier and non-empty content digest ID.
// Postcondition: returns nil if and only if the recomputed SHA-256 payload digest matches the claimed lock ID.
func VerifyLock(lock Lock) error {
	if lock.Schema != LockSchemaV2 && lock.Schema != LockSchema && lock.Schema != LegacyLockSchema {
		return fmt.Errorf("lock schema must be %q, %q, or legacy %q", LockSchemaV2, LockSchema, LegacyLockSchema)
	}
	if lock.ID == "" {
		return fmt.Errorf("lock id is required")
	}
	for _, a := range lock.Assets {
		if a.Kind == "secret" {
			if a.Value != "" {
				return fmt.Errorf("%s: locked secret %q cannot contain plaintext value", SecretPlaintextLeakError, a.ID)
			}
			if a.Ref == "" || !secretRefPattern.MatchString(a.Ref) {
				return fmt.Errorf("locked secret %q invalid or missing opaque reference %q (must match ^(env|file|vault|keyring):[a-zA-Z0-9_./#-]+$)", a.ID, a.Ref)
			}
		}
	}
	want := lock.ID
	got, err := lockID(lock)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("lock digest mismatch: got %s want %s", want, got)
	}
	return nil
}

// Mixable validates the evidence floor needed to combine a resolved lock with
// independently produced locks. Legacy locks remain valid launch inputs but
// cannot establish facts their schema never retained.
// Precondition: lock must pass canonical digest verification and conform strictly to the latest lock schema version.
// Postcondition: returns nil if all locked components preserve required reason provenance, contracts, and adapter evidence.
func Mixable(lock Lock) error {
	if err := VerifyLock(lock); err != nil {
		return err
	}
	if lock.Schema != LockSchema && lock.Schema != LockSchemaV2 {
		return fmt.Errorf("legacy product lock %q is launchable but not mixable; rebuild it from source as %q", lock.Schema, LockSchema)
	}
	for _, component := range lock.Components {
		if component.Reason == "" || component.Provider == "" {
			return fmt.Errorf("component %q has incomplete selection provenance", component.ID)
		}
		if component.Compatibility.Contract == "" {
			return fmt.Errorf("component %q has no compatibility contract", component.ID)
		}
		if len(component.Adapters) == 0 {
			return fmt.Errorf("component %q has no runtime adapter conformance evidence", component.ID)
		}
	}
	return nil
}

// ReidentifyLock recomputes the canonical identity after a trusted resolver or
// derivation has changed represented lock contents. It does not validate those
// changes; callers must enforce their operation-specific rules first.
// Precondition: pointer to lock structure must be non-nil prior to recomputing canonical digest identity.
// Postcondition: lock ID field is updated in-place with the fresh SHA-256 digest computed over all other fields.
func ReidentifyLock(lock *Lock) error {
	if lock == nil {
		return fmt.Errorf("lock is required")
	}
	lock.ID = ""
	id, err := lockID(*lock)
	if err != nil {
		return err
	}
	lock.ID = id
	return nil
}
