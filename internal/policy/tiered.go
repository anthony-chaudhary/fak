package policy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// ReasonConfigDrift is the closed refusal reason code emitted when a policy layer
// input changes mid-session or is modified, violating the policy epoch (#2408).
const ReasonConfigDrift abi.ReasonCode = 1110

// ReasonConfigDriftName is the stable name registered for ReasonConfigDrift.
const ReasonConfigDriftName = "CONFIG_DRIFT"

func init() {
	abi.RegisterReason(ReasonConfigDrift, ReasonConfigDriftName)
}

// Tier represents the trust and authority rank of a settings layer.
// Higher-trust tiers define the outer capability boundary; lower-trust
// tiers can only narrow the floor (meet-only merge).
type Tier string

const (
	// TierManaged is enterprise / organizational / MDM policy (highest authority).
	TierManaged Tier = "managed"
	// TierUser is user-global configuration (~/.fak/ or ~/.config/).
	TierUser Tier = "user"
	// TierProject is repository-level configuration (.fak/ checked into git).
	TierProject Tier = "project"
	// TierLocal is untracked local developer configuration (.fak/local).
	TierLocal Tier = "local"
)

// String returns the string representation of the tier.
func (t Tier) String() string {
	return string(t)
}

// Rank returns the numerical authority rank of the tier.
// Lower rank indicates higher authority: Managed (0) > User (1) > Project (2) > Local (3).
func (t Tier) Rank() int {
	switch strings.ToLower(strings.TrimSpace(string(t))) {
	case string(TierManaged):
		return 0
	case string(TierUser):
		return 1
	case string(TierProject):
		return 2
	case string(TierLocal):
		return 3
	default:
		return 99
	}
}

// IsValid reports whether t is a recognized settings tier.
func (t Tier) IsValid() bool {
	switch strings.ToLower(strings.TrimSpace(string(t))) {
	case string(TierManaged), string(TierUser), string(TierProject), string(TierLocal):
		return true
	default:
		return false
	}
}

// ParseTier parses a string into a Tier, case-insensitively.
func ParseTier(s string) (Tier, error) {
	norm := strings.ToLower(strings.TrimSpace(s))
	switch norm {
	case string(TierManaged):
		return TierManaged, nil
	case string(TierUser):
		return TierUser, nil
	case string(TierProject):
		return TierProject, nil
	case string(TierLocal):
		return TierLocal, nil
	default:
		return Tier(norm), fmt.Errorf("unknown policy tier: %q (want managed, user, project, local)", s)
	}
}

// TierLayer represents one tiered settings input.
type TierLayer struct {
	Tier     Tier     `json:"tier"`
	Path     string   `json:"path,omitempty"`
	Raw      []byte   `json:"raw,omitempty"`
	Manifest Manifest `json:"manifest"`
}

// NewTierLayer creates a TierLayer with an in-memory Manifest.
func NewTierLayer(tier Tier, m Manifest) TierLayer {
	return TierLayer{
		Tier:     tier,
		Manifest: m,
	}
}

// NewTierLayerFromBytes creates a TierLayer by parsing raw JSON manifest bytes fail-loudly.
func NewTierLayerFromBytes(tier Tier, raw []byte) (TierLayer, error) {
	m, err := ParseManifest(raw)
	if err != nil {
		return TierLayer{}, fmt.Errorf("tier %s parse manifest: %w", tier, err)
	}
	if _, err := m.ToRuntime(); err != nil {
		return TierLayer{}, fmt.Errorf("tier %s invalid manifest: %w", tier, err)
	}
	return TierLayer{
		Tier:     tier,
		Raw:      cloneBytes(raw),
		Manifest: m,
	}, nil
}

// NewTierLayerFromFile creates a TierLayer from a file path on disk, parsing fail-loudly.
func NewTierLayerFromFile(tier Tier, path string) (TierLayer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return TierLayer{}, fmt.Errorf("tier %s read file %s: %w", tier, path, err)
	}
	m, err := ParseManifest(data)
	if err != nil {
		return TierLayer{}, fmt.Errorf("tier %s parse file %s: %w", tier, path, err)
	}
	if _, err := m.ToRuntime(); err != nil {
		return TierLayer{}, fmt.Errorf("tier %s invalid file %s: %w", tier, path, err)
	}
	return TierLayer{
		Tier:     tier,
		Path:     path,
		Raw:      data,
		Manifest: m,
	}, nil
}

// RuleProvenance tracks which tier contributed or clamped a specific policy element.
type RuleProvenance struct {
	Rule   string `json:"rule"`
	Tier   Tier   `json:"tier"`
	Status string `json:"status"` // "admitted", "clamped", "dropped"
}

// CompiledTieredPolicy contains the result of compiling multiple settings tiers.
type CompiledTieredPolicy struct {
	Effective   Manifest          `json:"effective"`
	Epoch       string            `json:"epoch"`
	Provenance  map[string]Tier   `json:"provenance"`
	RuleActions []RuleProvenance  `json:"rule_actions"`
	Layers      []TierLayer       `json:"layers"`
	Hermetic    bool              `json:"hermetic"`
}

// TieredResolverOptions configures the compilation and resolution behavior.
type TieredResolverOptions struct {
	// Hermetic ignores all ambient tiers (User, Project, Local), retaining only
	// TierManaged. Hermetic mode is provable from the resulting policy epoch inputs.
	Hermetic bool `json:"hermetic"`
}

// CompileTieredManifest compiles user, project, local, and managed settings tiers into
// ONE effective Manifest using meet-only merge semantics. Lower-trust tiers can only
// narrow the policy: deny/ask rules merge, but affirmative allow rules from lower tiers
// exceeding the higher-trust floor are dropped or clamped.
func CompileTieredManifest(layers ...TierLayer) (Manifest, error) {
	compiled, err := CompileTieredPolicy(layers...)
	if err != nil {
		return Manifest{}, err
	}
	return compiled.Effective, nil
}

// CompileTieredPolicy compiles layers into a CompiledTieredPolicy including
// provenance and the deterministic policy epoch.
func CompileTieredPolicy(layers ...TierLayer) (*CompiledTieredPolicy, error) {
	return CompileTieredPolicyWithOptions(TieredResolverOptions{}, layers...)
}

// CompileTieredPolicyWithOptions compiles layers with explicit resolver options.
func CompileTieredPolicyWithOptions(opts TieredResolverOptions, layers ...TierLayer) (*CompiledTieredPolicy, error) {
	var activeLayers []TierLayer
	for _, l := range layers {
		if opts.Hermetic && l.Tier != TierManaged {
			continue
		}
		// Validate and parse layer if raw or file was supplied
		if len(l.Raw) > 0 {
			m, err := ParseManifest(l.Raw)
			if err != nil {
				return nil, fmt.Errorf("tier %s parse manifest: %w", l.Tier, err)
			}
			l.Manifest = m
		} else if l.Path != "" && len(l.Raw) == 0 {
			data, err := os.ReadFile(l.Path)
			if err != nil {
				return nil, fmt.Errorf("tier %s read file %s: %w", l.Tier, l.Path, err)
			}
			m, err := ParseManifest(data)
			if err != nil {
				return nil, fmt.Errorf("tier %s parse file %s: %w", l.Tier, l.Path, err)
			}
			l.Raw = data
			l.Manifest = m
		} else {
			// Validate Manifest struct by roundtripping to ensure fail-loud consistency
			b, err := json.Marshal(l.Manifest)
			if err != nil {
				return nil, fmt.Errorf("tier %s marshal manifest: %w", l.Tier, err)
			}
			if _, err := ParseManifest(b); err != nil {
				return nil, fmt.Errorf("tier %s validate manifest: %w", l.Tier, err)
			}
		}
		if _, err := l.Manifest.ToRuntime(); err != nil {
			return nil, fmt.Errorf("tier %s invalid manifest: %w", l.Tier, err)
		}
		activeLayers = append(activeLayers, l)
	}

	if len(activeLayers) == 0 {
		eff := Manifest{
			Version: Version,
			Posture: PostureFailClosed,
		}
		epoch := ComputePolicyEpoch(eff)
		return &CompiledTieredPolicy{
			Effective:  eff,
			Epoch:      epoch,
			Provenance: make(map[string]Tier),
			Layers:     nil,
			Hermetic:   opts.Hermetic,
		}, nil
	}

	// Sort layers stably by Tier authority rank (Managed first, Local last)
	sort.SliceStable(activeLayers, func(i, j int) bool {
		ri, rj := activeLayers[i].Tier.Rank(), activeLayers[j].Tier.Rank()
		if ri != rj {
			return ri < rj
		}
		return activeLayers[i].Path < activeLayers[j].Path
	})

	prov := make(map[string]Tier)
	var ruleActs []RuleProvenance

	// Accumulator initialized from the highest-trust tier
	acc := cloneManifest(activeLayers[0].Manifest)
	accTier := activeLayers[0].Tier

	// Tag initial rules with highest tier
	recordInitialProvenance(&acc, accTier, prov, &ruleActs)

	// Fold subsequent lower-trust tiers with meet-only merge
	for i := 1; i < len(activeLayers); i++ {
		layer := activeLayers[i]
		foldMeetOnly(&acc, layer.Manifest, accTier, layer.Tier, prov, &ruleActs)
	}

	// Canonicalize effective manifest
	canonicalizeManifest(&acc)

	epoch := ComputePolicyEpoch(acc, activeLayers...)

	return &CompiledTieredPolicy{
		Effective:   acc,
		Epoch:       epoch,
		Provenance:  prov,
		RuleActions: ruleActs,
		Layers:      activeLayers,
		Hermetic:    opts.Hermetic,
	}, nil
}

func recordInitialProvenance(m *Manifest, tier Tier, prov map[string]Tier, ruleActs *[]RuleProvenance) {
	for _, a := range m.Allow {
		prov["allow:"+a] = tier
		*ruleActs = append(*ruleActs, RuleProvenance{Rule: "allow:" + a, Tier: tier, Status: "admitted"})
	}
	for _, p := range m.AllowPrefix {
		prov["allow_prefix:"+p] = tier
		*ruleActs = append(*ruleActs, RuleProvenance{Rule: "allow_prefix:" + p, Tier: tier, Status: "admitted"})
	}
	for tool := range m.Deny {
		prov["deny:"+tool] = tier
		*ruleActs = append(*ruleActs, RuleProvenance{Rule: "deny:" + tool, Tier: tier, Status: "admitted"})
	}
	for _, c := range m.Complain {
		prov["complain:"+c] = tier
	}
	for _, g := range m.BlockedPathGlobs {
		prov["blocked_path:"+g] = tier
	}
	for _, r := range m.ArgRules {
		prov["arg_rule:"+r.Tool+":"+r.Arg] = tier
	}
}

// foldMeetOnly combines child (lower-trust tier) into acc (higher-trust accumulator).
// Lower tiers can only narrow; affirmative allow additions exceeding the higher floor are dropped.
func foldMeetOnly(acc *Manifest, child Manifest, accTier, childTier Tier, prov map[string]Tier, ruleActs *[]RuleProvenance) {
	// 1. Allow and AllowPrefix: lower tiers can only narrow.
	foldAllows(acc, child, childTier, prov, ruleActs)

	// 2. Deny: union (deny rules merge; if either denies, it is denied).
	if acc.Deny == nil {
		acc.Deny = make(map[string]string)
	}
	for tool, reason := range child.Deny {
		if _, exists := acc.Deny[tool]; !exists {
			acc.Deny[tool] = reason
			prov["deny:"+tool] = childTier
			*ruleActs = append(*ruleActs, RuleProvenance{
				Rule:   "deny:" + tool,
				Tier:   childTier,
				Status: "admitted",
			})
		}
	}

	// 3. Complain: union.
	acc.Complain = dedupeAndSort(append(acc.Complain, child.Complain...))
	for _, c := range child.Complain {
		if _, ok := prov["complain:"+c]; !ok {
			prov["complain:"+c] = childTier
		}
	}

	// 4. Blocked paths, self-modify globs, credential globs: union (more blocked paths = narrower).
	acc.BlockedPathGlobs = dedupeAndSort(append(acc.BlockedPathGlobs, child.BlockedPathGlobs...))
	for _, g := range child.BlockedPathGlobs {
		if _, ok := prov["blocked_path:"+g]; !ok {
			prov["blocked_path:"+g] = childTier
		}
	}
	acc.SelfModifyGlobs = dedupeAndSort(append(acc.SelfModifyGlobs, child.SelfModifyGlobs...))
	acc.CredentialPathGlobs = dedupeAndSort(append(acc.CredentialPathGlobs, child.CredentialPathGlobs...))

	// 5. Redact fields: union.
	acc.RedactFields = dedupeAndSort(append(acc.RedactFields, child.RedactFields...))

	// 6. ArgRules: each ArgRule restricts calls. Concatenate and deduplicate.
	for _, cr := range child.ArgRules {
		exists := false
		for _, ar := range acc.ArgRules {
			if argRulesEqual(ar, cr) {
				exists = true
				break
			}
		}
		if !exists {
			acc.ArgRules = append(acc.ArgRules, cr)
			key := "arg_rule:" + cr.Tool + ":" + cr.Arg
			if _, ok := prov[key]; !ok {
				prov[key] = childTier
			}
		}
	}

	// 7. Posture: most restrictive wins (fail_closed > admit_and_log > default_open).
	acc.Posture = mostRestrictivePosture(acc.Posture, child.Posture)

	// 8. RateLimit: non-zero caps take the minimum (tighten only).
	if acc.RateLimit == nil {
		acc.RateLimit = child.RateLimit
	} else if child.RateLimit != nil {
		rl := *acc.RateLimit
		if child.RateLimit.MaxCalls > 0 {
			if rl.MaxCalls <= 0 || child.RateLimit.MaxCalls < rl.MaxCalls {
				rl.MaxCalls = child.RateLimit.MaxCalls
			}
		}
		acc.RateLimit = &rl
	}

	// 9. SubagentDepth: take minimum depth.
	if acc.SubagentDepth == nil {
		acc.SubagentDepth = child.SubagentDepth
	} else if child.SubagentDepth != nil {
		sd := *acc.SubagentDepth
		if child.SubagentDepth.MaxDepth > 0 {
			if sd.MaxDepth <= 0 || child.SubagentDepth.MaxDepth < sd.MaxDepth {
				sd.MaxDepth = child.SubagentDepth.MaxDepth
			}
		}
		acc.SubagentDepth = &sd
	}

	// 10. LintWrites: if either enables it, it stays enabled.
	acc.LintWrites = acc.LintWrites || child.LintWrites

	// 11. SecretPosture: most restrictive wins.
	acc.SecretPosture = mostRestrictiveSecretPosture(acc.SecretPosture, child.SecretPosture)
	acc.SecretPatterns = dedupeAndSort(append(acc.SecretPatterns, child.SecretPatterns...))

	// 12. Egress:
	if acc.Egress == nil {
		acc.Egress = child.Egress
	} else if child.Egress != nil {
		eg := *acc.Egress
		eg.DenyHosts = dedupeAndSort(append(eg.DenyHosts, child.Egress.DenyHosts...))
		eg.BlockHosts = dedupeAndSort(append(eg.BlockHosts, child.Egress.BlockHosts...))
		eg.BlockLists = dedupeAndSort(append(eg.BlockLists, child.Egress.BlockLists...))
		eg.Restrict = eg.Restrict || child.Egress.Restrict

		if eg.AllowHosts != nil && child.Egress.AllowHosts != nil {
			eg.AllowHosts = intersectStrings(eg.AllowHosts, child.Egress.AllowHosts)
		} else if eg.AllowHosts == nil && child.Egress.AllowHosts != nil {
			eg.AllowHosts = dedupeAndSort(child.Egress.AllowHosts)
		}

		if eg.ResearchAllowHosts != nil && child.Egress.ResearchAllowHosts != nil {
			eg.ResearchAllowHosts = intersectStrings(eg.ResearchAllowHosts, child.Egress.ResearchAllowHosts)
		} else if eg.ResearchAllowHosts == nil && child.Egress.ResearchAllowHosts != nil {
			eg.ResearchAllowHosts = dedupeAndSort(child.Egress.ResearchAllowHosts)
		}
		acc.Egress = &eg
	}

	// 13. MountView: lower tier can only mount within higher tier views, read-only cannot become read-write.
	if len(acc.MountView) == 0 {
		acc.MountView = child.MountView
	} else if len(child.MountView) > 0 {
		var newMounts []MountRule
		for _, cm := range child.MountView {
			for _, am := range acc.MountView {
				if strings.HasPrefix(cm.Path, am.Path) || cm.Path == am.Path {
					rule := cm
					if am.readOnly() {
						rule.Mode = "ro"
					}
					newMounts = append(newMounts, rule)
					break
				}
			}
		}
		acc.MountView = newMounts
	}
}

// foldAllows enforces that lower-trust tier allow/allow_prefix rules can only
// narrow: tool allowances exceeding the accumulator's floor are dropped.
func foldAllows(acc *Manifest, child Manifest, childTier Tier, prov map[string]Tier, ruleActs *[]RuleProvenance) {
	// If child specifies no allow rules at all, it does not constrain allows.
	if child.Allow == nil && child.AllowPrefix == nil {
		return
	}

	accUnconstrained := (acc.Allow == nil && acc.AllowPrefix == nil)

	isAllowedByAcc := func(tool string) bool {
		if accUnconstrained {
			return true
		}
		for _, a := range acc.Allow {
			if a == tool {
				return true
			}
		}
		for _, p := range acc.AllowPrefix {
			if strings.HasPrefix(tool, p) {
				return true
			}
		}
		return false
	}

	var newAllows []string
	if child.Allow != nil {
		for _, tool := range child.Allow {
			if isAllowedByAcc(tool) {
				newAllows = append(newAllows, tool)
				prov["allow:"+tool] = childTier
				*ruleActs = append(*ruleActs, RuleProvenance{
					Rule:   "allow:" + tool,
					Tier:   childTier,
					Status: "admitted",
				})
			} else {
				// Lower tier allow exceeds higher floor -> DROPPED!
				*ruleActs = append(*ruleActs, RuleProvenance{
					Rule:   "allow:" + tool,
					Tier:   childTier,
					Status: "dropped_exceeds_floor",
				})
			}
		}
	} else if acc.Allow != nil {
		newAllows = append(newAllows, acc.Allow...)
	}

	var newPrefixes []string
	if child.AllowPrefix != nil {
		for _, pfx := range child.AllowPrefix {
			if accUnconstrained {
				newPrefixes = append(newPrefixes, pfx)
				prov["allow_prefix:"+pfx] = childTier
			} else {
				valid := false
				for _, apfx := range acc.AllowPrefix {
					if strings.HasPrefix(pfx, apfx) {
						valid = true
						break
					}
				}
				if valid {
					newPrefixes = append(newPrefixes, pfx)
					prov["allow_prefix:"+pfx] = childTier
					*ruleActs = append(*ruleActs, RuleProvenance{
						Rule:   "allow_prefix:" + pfx,
						Tier:   childTier,
						Status: "admitted",
					})
				} else {
					*ruleActs = append(*ruleActs, RuleProvenance{
						Rule:   "allow_prefix:" + pfx,
						Tier:   childTier,
						Status: "dropped_exceeds_floor",
					})
				}
			}
		}
	} else if acc.AllowPrefix != nil {
		newPrefixes = append(newPrefixes, acc.AllowPrefix...)
	}

	acc.Allow = dedupeAndSort(newAllows)
	acc.AllowPrefix = dedupeAndSort(newPrefixes)
}

func canonicalizeManifest(m *Manifest) {
	if m.Version == "" {
		m.Version = Version
	}
	m.Allow = dedupeAndSort(m.Allow)
	m.AllowPrefix = dedupeAndSort(m.AllowPrefix)
	m.Complain = dedupeAndSort(m.Complain)
	m.SelfModifyGlobs = dedupeAndSort(m.SelfModifyGlobs)
	m.BlockedPathGlobs = dedupeAndSort(m.BlockedPathGlobs)
	m.CredentialPathGlobs = dedupeAndSort(m.CredentialPathGlobs)
	m.RedactFields = dedupeAndSort(m.RedactFields)
	m.SecretPatterns = dedupeAndSort(m.SecretPatterns)

	// Denied tools should not be in affirmative Allow
	if len(m.Deny) > 0 && len(m.Allow) > 0 {
		var filtered []string
		for _, a := range m.Allow {
			if _, denied := m.Deny[a]; !denied {
				filtered = append(filtered, a)
			}
		}
		m.Allow = filtered
	}
}

func mostRestrictivePosture(p1, p2 string) string {
	rank := func(p string) int {
		switch strings.ToLower(p) {
		case PostureFailClosed, "":
			return 0
		case PostureAdmitAndLog:
			return 1
		case PostureDefaultOpen:
			return 2
		default:
			return 0
		}
	}
	if rank(p1) <= rank(p2) {
		if p1 == "" {
			return PostureFailClosed
		}
		return p1
	}
	return p2
}

func mostRestrictiveSecretPosture(s1, s2 string) string {
	rank := func(s string) int {
		switch strings.ToLower(s) {
		case "fail_closed":
			return 0
		case "quarantine", "":
			return 1
		case "admit_and_log":
			return 2
		default:
			return 1
		}
	}
	if rank(s1) <= rank(s2) {
		if s1 == "" {
			return "quarantine"
		}
		return s1
	}
	return s2
}

func argRulesEqual(a, b ArgRule) bool {
	if a.Tool != b.Tool || a.Arg != b.Arg || a.AllowGlob != b.AllowGlob ||
		a.DenyRegex != b.DenyRegex || a.MaxBytes != b.MaxBytes ||
		a.CLIReadOnly != b.CLIReadOnly || a.Reason != b.Reason || a.Advisory != b.Advisory {
		return false
	}
	if (a.AllowExact == nil) != (b.AllowExact == nil) {
		return false
	}
	if a.AllowExact != nil && *a.AllowExact != *b.AllowExact {
		return false
	}
	return true
}

func cloneManifest(m Manifest) Manifest {
	b, _ := json.Marshal(m)
	var out Manifest
	_ = json.Unmarshal(b, &out)
	return out
}

// ComputePolicyEpoch computes a deterministic SHA-256 hex string over the effective
// manifest and all contributing layer inputs.
func ComputePolicyEpoch(effective Manifest, layers ...TierLayer) string {
	h := sha256.New()

	// 1. Hash the effective manifest ruleset digest
	effDigest := ComputeRulesetDigest(effective)
	_, _ = fmt.Fprintf(h, "effective:%s\n", effDigest)

	// 2. Sort layers in canonical order (by Tier authority rank, then Path)
	sortedLayers := make([]TierLayer, len(layers))
	copy(sortedLayers, layers)
	sort.SliceStable(sortedLayers, func(i, j int) bool {
		ri, rj := sortedLayers[i].Tier.Rank(), sortedLayers[j].Tier.Rank()
		if ri != rj {
			return ri < rj
		}
		return sortedLayers[i].Path < sortedLayers[j].Path
	})

	// 3. Hash each layer's metadata and content digest
	for _, l := range sortedLayers {
		var contentDigest string
		if len(l.Raw) > 0 {
			contentDigest = ComputeContentDigest(l.Raw)
		} else {
			contentDigest = ComputeRulesetDigest(l.Manifest)
		}
		_, _ = fmt.Fprintf(h, "layer|tier:%s|path:%s|digest:%s\n", l.Tier, l.Path, contentDigest)
	}

	return hex.EncodeToString(h.Sum(nil))
}

type layerSnapshot struct {
	tier    Tier
	path    string
	hasPath bool
	rawHash string
	modTime time.Time
}

// ConfigDriftRefusal represents a structured refusal when a settings input
// drifts mid-session.
type ConfigDriftRefusal struct {
	Reason        string         `json:"reason"`
	ReasonCode    abi.ReasonCode `json:"reason_code"`
	Tier          Tier           `json:"tier,omitempty"`
	Path          string         `json:"path,omitempty"`
	ExpectedEpoch string         `json:"expected_epoch"`
	ActualEpoch   string         `json:"actual_epoch,omitempty"`
	Detail        string         `json:"detail,omitempty"`
	Verdict       abi.Verdict    `json:"verdict"`
}

func (r *ConfigDriftRefusal) Error() string {
	if r.Detail != "" {
		return fmt.Sprintf("policy config drift (%s): %s [tier=%s path=%s]", r.Reason, r.Detail, r.Tier, r.Path)
	}
	return fmt.Sprintf("policy config drift (%s): tier=%s path=%s", r.Reason, r.Tier, r.Path)
}

// TieredPolicyResolver provides runtime evaluation against a tiered settings floor,
// verifying configuration stability and stamping the policy epoch on all verdicts.
type TieredPolicyResolver struct {
	mu          sync.RWMutex
	layers      []TierLayer
	snapshots   []layerSnapshot
	effective   Manifest
	epoch       string
	provenance  map[string]Tier
	ruleActions []RuleProvenance
	runtime     Runtime
	adjudicator *adjudicator.Adjudicator
	hermetic    bool
}

// NewTieredPolicyResolver constructs a TieredPolicyResolver from the given layers.
func NewTieredPolicyResolver(layers ...TierLayer) (*TieredPolicyResolver, error) {
	return NewTieredPolicyResolverWithOptions(TieredResolverOptions{}, layers...)
}

// NewTieredPolicyResolverWithOptions constructs a TieredPolicyResolver with explicit options.
func NewTieredPolicyResolverWithOptions(opts TieredResolverOptions, layers ...TierLayer) (*TieredPolicyResolver, error) {
	compiled, err := CompileTieredPolicyWithOptions(opts, layers...)
	if err != nil {
		return nil, err
	}

	rt, err := compiled.Effective.ToRuntime()
	if err != nil {
		return nil, fmt.Errorf("compile runtime for effective manifest: %w", err)
	}

	adj := adjudicator.New(rt.Adjudicator)

	// Record snapshots for drift detection
	var snaps []layerSnapshot
	for _, l := range compiled.Layers {
		snap := layerSnapshot{
			tier: l.Tier,
			path: l.Path,
		}
		if l.Path != "" {
			snap.hasPath = true
			if fi, err := os.Stat(l.Path); err == nil {
				snap.modTime = fi.ModTime()
			}
			if len(l.Raw) > 0 {
				snap.rawHash = hex.EncodeToString(sha256Bytes(l.Raw))
			} else if data, err := os.ReadFile(l.Path); err == nil {
				snap.rawHash = hex.EncodeToString(sha256Bytes(data))
			}
		} else if len(l.Raw) > 0 {
			snap.rawHash = hex.EncodeToString(sha256Bytes(l.Raw))
		} else {
			snap.rawHash = ComputeRulesetDigest(l.Manifest)
		}
		snaps = append(snaps, snap)
	}

	return &TieredPolicyResolver{
		layers:      compiled.Layers,
		snapshots:   snaps,
		effective:   compiled.Effective,
		epoch:       compiled.Epoch,
		provenance:  compiled.Provenance,
		ruleActions: compiled.RuleActions,
		runtime:     rt,
		adjudicator: adj,
		hermetic:    opts.Hermetic,
	}, nil
}

// EffectiveManifest returns the resolved Manifest.
func (r *TieredPolicyResolver) EffectiveManifest() Manifest {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return cloneManifest(r.effective)
}

// PolicyEpoch returns the deterministic policy epoch hex string.
func (r *TieredPolicyResolver) PolicyEpoch() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.epoch
}

// Epoch is an alias for PolicyEpoch.
func (r *TieredPolicyResolver) Epoch() string {
	return r.PolicyEpoch()
}

// Layers returns the contributing layers.
func (r *TieredPolicyResolver) Layers() []TierLayer {
	r.mu.RLock()
	defer r.mu.RUnlock()
	res := make([]TierLayer, len(r.layers))
	copy(res, r.layers)
	return res
}

// Provenance returns the per-rule tier provenance map.
func (r *TieredPolicyResolver) Provenance() map[string]Tier {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cp := make(map[string]Tier, len(r.provenance))
	for k, v := range r.provenance {
		cp[k] = v
	}
	return cp
}

// RuleProvenance returns the ordered list of rule provenance actions.
func (r *TieredPolicyResolver) RuleProvenance() []RuleProvenance {
	r.mu.RLock()
	defer r.mu.RUnlock()
	res := make([]RuleProvenance, len(r.ruleActions))
	copy(res, r.ruleActions)
	return res
}

// Adjudicator returns the underlying *adjudicator.Adjudicator.
func (r *TieredPolicyResolver) Adjudicator() *adjudicator.Adjudicator {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.adjudicator
}

// Hermetic reports whether the resolver was compiled in hermetic mode.
func (r *TieredPolicyResolver) Hermetic() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.hermetic
}

// MutateLayer modifies an in-memory layer input (primarily used for test simulation).
func (r *TieredPolicyResolver) MutateLayer(tier Tier, newRaw []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := range r.layers {
		if r.layers[i].Tier == tier {
			r.layers[i].Raw = cloneBytes(newRaw)
			if m, err := ParseManifest(newRaw); err == nil {
				r.layers[i].Manifest = m
			}
		}
	}
}

// CheckConfigDrift checks whether any layer file or input has changed since resolution.
// If drift is detected, it returns a typed ConfigDriftRefusal.
func (r *TieredPolicyResolver) CheckConfigDrift() (*ConfigDriftRefusal, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.checkDriftLocked()
}

func (r *TieredPolicyResolver) checkDriftLocked() (*ConfigDriftRefusal, error) {
	for i, snap := range r.snapshots {
		if snap.hasPath {
			fi, err := os.Stat(snap.path)
			if err != nil {
				refusal := &ConfigDriftRefusal{
					Reason:        ReasonConfigDriftName,
					ReasonCode:    ReasonConfigDrift,
					Tier:          snap.tier,
					Path:          snap.path,
					ExpectedEpoch: r.epoch,
					Detail:        fmt.Sprintf("layer file inaccessible (%v)", err),
				}
				refusal.Verdict = r.driftVerdict(refusal)
				return refusal, refusal
			}
			curData, err := os.ReadFile(snap.path)
			if err != nil {
				refusal := &ConfigDriftRefusal{
					Reason:        ReasonConfigDriftName,
					ReasonCode:    ReasonConfigDrift,
					Tier:          snap.tier,
					Path:          snap.path,
					ExpectedEpoch: r.epoch,
					Detail:        fmt.Sprintf("layer file read failure (%v)", err),
				}
				refusal.Verdict = r.driftVerdict(refusal)
				return refusal, refusal
			}
			curHash := hex.EncodeToString(sha256Bytes(curData))
			if curHash != snap.rawHash {
				refusal := &ConfigDriftRefusal{
					Reason:        ReasonConfigDriftName,
					ReasonCode:    ReasonConfigDrift,
					Tier:          snap.tier,
					Path:          snap.path,
					ExpectedEpoch: r.epoch,
					Detail:        fmt.Sprintf("layer file %s mutated mid-session (tier: %s)", snap.path, snap.tier),
				}
				refusal.Verdict = r.driftVerdict(refusal)
				return refusal, refusal
			}
			_ = fi
		} else if i < len(r.layers) {
			l := r.layers[i]
			var curHash string
			if len(l.Raw) > 0 {
				curHash = hex.EncodeToString(sha256Bytes(l.Raw))
			} else {
				curHash = ComputeRulesetDigest(l.Manifest)
			}
			if curHash != snap.rawHash {
				refusal := &ConfigDriftRefusal{
					Reason:        ReasonConfigDriftName,
					ReasonCode:    ReasonConfigDrift,
					Tier:          snap.tier,
					ExpectedEpoch: r.epoch,
					Detail:        fmt.Sprintf("layer %s input mutated mid-session", snap.tier),
				}
				refusal.Verdict = r.driftVerdict(refusal)
				return refusal, refusal
			}
		}
	}
	return nil, nil
}

func (r *TieredPolicyResolver) driftVerdict(refusal *ConfigDriftRefusal) abi.Verdict {
	return abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: ReasonConfigDrift,
		Meta: map[string]string{
			"reason":       ReasonConfigDriftName,
			"policy_epoch": r.epoch,
			"drift_tier":   string(refusal.Tier),
			"drift_path":   refusal.Path,
			"drift_detail": refusal.Detail,
		},
	}
}

// CheckConfigDrift is a convenience function that checks drift on a resolver.
func CheckConfigDrift(r *TieredPolicyResolver) (*ConfigDriftRefusal, error) {
	if r == nil {
		return nil, nil
	}
	return r.CheckConfigDrift()
}

// Adjudicate adjudicates a tool call against the tiered capability floor.
// If configuration drift is detected, it immediately refuses with reason CONFIG_DRIFT.
// Otherwise, every resulting verdict carries the policy epoch in its Meta.
func (r *TieredPolicyResolver) Adjudicate(ctx context.Context, call *abi.ToolCall) abi.Verdict {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// 1. Configuration drift check
	if refusal, _ := r.checkDriftLocked(); refusal != nil {
		return refusal.Verdict
	}

	// 2. Adjudicate against compiled floor
	v := r.adjudicator.Adjudicate(ctx, call)

	// 3. Stamp policy epoch onto verdict Meta
	if v.Meta == nil {
		v.Meta = make(map[string]string)
	}
	v.Meta["policy_epoch"] = r.epoch

	return v
}

// FormatResolve formats the effective manifest with per-rule source-tier provenance.
func (r *TieredPolicyResolver) FormatResolve() string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var buf bytes.Buffer
	buf.WriteString(fmt.Sprintf("Effective Manifest (Epoch: %s):\n", r.epoch))
	buf.WriteString(fmt.Sprintf("  Posture: %s\n", r.effective.Posture))

	if len(r.effective.Allow) > 0 {
		buf.WriteString("  Allow:\n")
		for _, a := range r.effective.Allow {
			tier := r.provenance["allow:"+a]
			buf.WriteString(fmt.Sprintf("    - %s [%s]\n", a, tier))
		}
	}
	if len(r.effective.AllowPrefix) > 0 {
		buf.WriteString("  AllowPrefix:\n")
		for _, p := range r.effective.AllowPrefix {
			tier := r.provenance["allow_prefix:"+p]
			buf.WriteString(fmt.Sprintf("    - %s [%s]\n", p, tier))
		}
	}
	if len(r.effective.Deny) > 0 {
		buf.WriteString("  Deny:\n")
		var keys []string
		for k := range r.effective.Deny {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			tier := r.provenance["deny:"+k]
			buf.WriteString(fmt.Sprintf("    - %s: %s [%s]\n", k, r.effective.Deny[k], tier))
		}
	}
	return buf.String()
}

func sha256Bytes(b []byte) []byte {
	sum := sha256.Sum256(b)
	return sum[:]
}
