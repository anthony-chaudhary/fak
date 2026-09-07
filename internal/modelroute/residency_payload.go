package modelroute

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
)

// PayloadClassificationVersion is the schema tag for payload-level classification manifests.
const PayloadClassificationVersion = "fak-residency-payload/v1"

// EgressWitnessVersion is the schema tag for egress witness audit artifacts.
const EgressWitnessVersion = "fak-egress-witness/v1"

// Named reasons for residency enforcement and egress checks.
const (
	// ReasonResidencyEgressViolation is the named reason for residency egress refusal.
	ReasonResidencyEgressViolation = "RESIDENCY_EGRESS_VIOLATION"

	// ReasonPayloadClassifiedRemoteDenied is the fail-closed named refusal reason when a classified
	// payload is bound for a remote target and no local target is available.
	ReasonPayloadClassifiedRemoteDenied = "PAYLOAD_CLASSIFIED_REMOTE_DENIED"

	// ReasonClassificationInvalid is the fail-closed reason when classification rules cannot be loaded or validated.
	ReasonClassificationInvalid = "RESIDENCY_CLASSIFICATION_INVALID"

	// ReasonPayloadClassifiedReroutedLocal indicates a classified payload was deterministically rerouted to a local target.
	ReasonPayloadClassifiedReroutedLocal = "PAYLOAD_CLASSIFIED_REROUTED_LOCAL"

	// Canonical sensitivity and residency classes.
	ClassLocal        = "LOCAL"
	ClassClassified   = "CLASSIFIED"
	ClassRestricted   = "RESTRICTED"
	ClassConfidential = "CONFIDENTIAL"
	ClassSecret       = "SECRET"
	ClassTenant       = "TENANT"
	ClassPII          = "PII"
	ClassPublic       = "PUBLIC"
	ClassUnclassified = "UNCLASSIFIED"
)

// IsLocalOnlyClass reports whether class denotes a sensitivity or residency tier
// that requires on-box execution (cannot egress to a remote target).
func IsLocalOnlyClass(class string) bool {
	c := strings.ToUpper(strings.TrimSpace(class))
	if c == "" || c == ClassPublic || c == ClassUnclassified {
		return false
	}
	return true
}

// normalizePath converts path separators to slashes, strips leading prefixes, and cleans the path.
func normalizePath(p string) string {
	p = filepath.ToSlash(strings.TrimSpace(p))
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/")
	return path.Clean(p)
}

// matchPattern matches a glob pattern against a target path.
// Supports exact match, recursive "**", directory prefix, extension match, and path.Match.
func matchPattern(pattern, targetPath string) bool {
	normPat := filepath.ToSlash(strings.TrimSpace(pattern))
	normPat = strings.TrimPrefix(normPat, "./")
	normPat = strings.TrimPrefix(normPat, "/")
	normTarget := normalizePath(targetPath)

	if normPat == "" || normTarget == "" {
		return false
	}

	if normPat == normTarget {
		return true
	}

	if normPat == "**" || normPat == "*" {
		return true
	}

	// Suffix "/**" matches the directory and all descendants
	if strings.HasSuffix(normPat, "/**") {
		dirPrefix := strings.TrimSuffix(normPat, "/**")
		if normTarget == dirPrefix || strings.HasPrefix(normTarget, dirPrefix+"/") {
			return true
		}
	}

	// Prefix "**/" matches any parent directory prefix
	if strings.HasPrefix(normPat, "**/") {
		subPat := strings.TrimPrefix(normPat, "**/")
		if matchPattern(subPat, normTarget) {
			return true
		}
		parts := strings.Split(normTarget, "/")
		for i := 1; i < len(parts); i++ {
			subPath := strings.Join(parts[i:], "/")
			if matchPattern(subPat, subPath) {
				return true
			}
		}
	}

	// Pattern without slash matches file base name as well as whole path
	if !strings.Contains(normPat, "/") {
		if ok, _ := path.Match(normPat, normTarget); ok {
			return true
		}
		if ok, _ := path.Match(normPat, path.Base(normTarget)); ok {
			return true
		}
	}

	// Standard path.Match
	if ok, _ := path.Match(normPat, normTarget); ok {
		return true
	}

	// Directory prefix "dir/" matches "dir/anything"
	if strings.HasSuffix(normPat, "/") {
		if strings.HasPrefix(normTarget, normPat) {
			return true
		}
	}

	return false
}

// ClassificationRule binds a file path or glob pattern to a residency/sensitivity class.
type ClassificationRule struct {
	Pattern     string `json:"pattern"`
	Class       string `json:"class"`
	Description string `json:"description,omitempty"`
}

// Validate checks that the rule has a valid pattern and non-empty class.
func (r ClassificationRule) Validate() error {
	if strings.TrimSpace(r.Pattern) == "" {
		return fmt.Errorf("modelroute: classification rule has empty pattern")
	}
	if strings.TrimSpace(r.Class) == "" {
		return fmt.Errorf("modelroute: classification rule for %q has empty class", r.Pattern)
	}
	return nil
}

// RequiresLocal reports whether this rule demands local residency.
func (r ClassificationRule) RequiresLocal() bool {
	return IsLocalOnlyClass(r.Class)
}

// PayloadClassificationConfig declares the payload-level residency rules for a repository
// or deployment, along with an optional local target for deterministic rerouting.
type PayloadClassificationConfig struct {
	Version      string               `json:"version,omitempty"`
	Rules        []ClassificationRule `json:"rules"`
	LocalTarget  *Target              `json:"local_target,omitempty"`
	LocalAccount string               `json:"local_account,omitempty"`
}

// Validate verifies that all rules are well-formed and any declared LocalTarget is on-box.
func (c PayloadClassificationConfig) Validate() error {
	if c.Version != "" && !strings.HasPrefix(c.Version, PayloadClassificationVersion) {
		return fmt.Errorf("modelroute: payload classification version %q is not %s.x", c.Version, PayloadClassificationVersion)
	}
	for i, r := range c.Rules {
		if err := r.Validate(); err != nil {
			return fmt.Errorf("modelroute: rule %d: %w", i, err)
		}
	}
	if c.LocalTarget != nil {
		if !c.LocalTarget.Local() {
			return fmt.Errorf("modelroute: local_target %q must be local (kind %q is remote)", c.LocalTarget.EngineRoute(), c.LocalTarget.Kind)
		}
	}
	return nil
}

// JSON renders the configuration as indented JSON.
func (c PayloadClassificationConfig) JSON() []byte {
	out := c
	if out.Version == "" {
		out.Version = PayloadClassificationVersion
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return append(b, '\n')
}

// ParsePayloadClassification parses and validates a JSON payload classification manifest.
func ParsePayloadClassification(b []byte) (PayloadClassificationConfig, error) {
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	var cfg PayloadClassificationConfig
	if err := dec.Decode(&cfg); err != nil {
		return PayloadClassificationConfig{}, fmt.Errorf("modelroute: parse payload classification: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return PayloadClassificationConfig{}, err
	}
	return cfg, nil
}

// LoadPayloadClassification reads and validates a payload classification manifest from disk.
func LoadPayloadClassification(path string) (PayloadClassificationConfig, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return PayloadClassificationConfig{}, fmt.Errorf("modelroute: read payload classification %s: %w", path, err)
	}
	return ParsePayloadClassification(b)
}

// Override merges this configuration with a central configuration, giving central rules
// and settings precedence. Central rules are prepended so they evaluate first.
func (c PayloadClassificationConfig) Override(central PayloadClassificationConfig) PayloadClassificationConfig {
	out := c
	if central.Version != "" {
		out.Version = central.Version
	}
	if central.LocalTarget != nil {
		out.LocalTarget = central.LocalTarget
	}
	if central.LocalAccount != "" {
		out.LocalAccount = central.LocalAccount
	}
	if len(central.Rules) > 0 {
		merged := make([]ClassificationRule, 0, len(central.Rules)+len(c.Rules))
		merged = append(merged, central.Rules...)
		merged = append(merged, c.Rules...)
		out.Rules = merged
	}
	return out
}

// MergePayloadClassification merges a repository-level configuration with a central override.
func MergePayloadClassification(repo, central PayloadClassificationConfig) PayloadClassificationConfig {
	return repo.Override(central)
}

// LoadPayloadClassificationWithOverride loads both repo and central manifests, merging them fail-closed.
func LoadPayloadClassificationWithOverride(repoPath, centralPath string) (PayloadClassificationConfig, error) {
	var repoCfg, centralCfg PayloadClassificationConfig
	var err error

	if repoPath != "" {
		repoCfg, err = LoadPayloadClassification(repoPath)
		if err != nil {
			return PayloadClassificationConfig{}, fmt.Errorf("repo classification: %w", err)
		}
	}
	if centralPath != "" {
		centralCfg, err = LoadPayloadClassification(centralPath)
		if err != nil {
			return PayloadClassificationConfig{}, fmt.Errorf("central classification: %w", err)
		}
	}

	if repoPath == "" && centralPath == "" {
		return PayloadClassificationConfig{}, fmt.Errorf("modelroute: both repoPath and centralPath are empty")
	}
	if repoPath == "" {
		return centralCfg, nil
	}
	if centralPath == "" {
		return repoCfg, nil
	}
	return repoCfg.Override(centralCfg), nil
}

// PayloadItem represents one item (file path or data buffer) in a dispatch payload.
type PayloadItem struct {
	Path      string `json:"path"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
	Data      []byte `json:"-"`
}

// Payload represents the data context submitted for model dispatch.
type Payload struct {
	Paths     []string      `json:"paths,omitempty"`
	Items     []PayloadItem `json:"items,omitempty"`
	SizeBytes int64         `json:"size_bytes,omitempty"`
}

// NewPayload constructs a Payload from path strings.
func NewPayload(paths ...string) Payload {
	return Payload{Paths: paths}
}

// AllItems resolves all paths and items into a unified list of PayloadItems with non-zero sizes.
func (p Payload) AllItems() []PayloadItem {
	var items []PayloadItem
	for _, item := range p.Items {
		it := item
		if it.SizeBytes <= 0 {
			if len(it.Data) > 0 {
				it.SizeBytes = int64(len(it.Data))
			} else if len(it.Path) > 0 {
				it.SizeBytes = int64(len(it.Path))
			} else {
				it.SizeBytes = 1
			}
		}
		items = append(items, it)
	}
	for _, pathStr := range p.Paths {
		pathStr = strings.TrimSpace(pathStr)
		if pathStr == "" {
			continue
		}
		already := false
		for _, it := range items {
			if it.Path == pathStr {
				already = true
				break
			}
		}
		if !already {
			size := int64(len(pathStr))
			if size <= 0 {
				size = 1
			}
			items = append(items, PayloadItem{
				Path:      pathStr,
				SizeBytes: size,
			})
		}
	}
	return items
}

// TotalBytes returns the total payload byte size.
func (p Payload) TotalBytes() int64 {
	if p.SizeBytes > 0 {
		return p.SizeBytes
	}
	var total int64
	for _, item := range p.AllItems() {
		total += item.SizeBytes
	}
	return total
}

// ClassificationMatch records a rule that classified a payload item.
type ClassificationMatch struct {
	Path        string `json:"path"`
	Pattern     string `json:"pattern"`
	Class       string `json:"class"`
	Description string `json:"description,omitempty"`
	SizeBytes   int64  `json:"size_bytes"`
}

// ClassificationResult records the outcome of classifying a payload.
type ClassificationResult struct {
	Classified      bool                  `json:"classified"`
	ClassifiedBytes int64                 `json:"classified_bytes"`
	TotalBytes      int64                 `json:"total_bytes"`
	Matches         []ClassificationMatch `json:"matches,omitempty"`
	RulesEvaluated  int                   `json:"rules_evaluated"`
}

// ClassifyPayload evaluates a payload against the declared classification rules.
func ClassifyPayload(cfg PayloadClassificationConfig, payload Payload) ClassificationResult {
	items := payload.AllItems()
	totalBytes := payload.TotalBytes()
	var matches []ClassificationMatch
	var classifiedBytes int64
	rulesEvaluated := 0

	for _, item := range items {
		itemClassified := false
		for _, rule := range cfg.Rules {
			rulesEvaluated++
			if matchPattern(rule.Pattern, item.Path) {
				if rule.RequiresLocal() {
					matches = append(matches, ClassificationMatch{
						Path:        item.Path,
						Pattern:     rule.Pattern,
						Class:       rule.Class,
						Description: rule.Description,
						SizeBytes:   item.SizeBytes,
					})
					itemClassified = true
					break
				}
			}
		}
		if itemClassified {
			classifiedBytes += item.SizeBytes
		}
	}

	return ClassificationResult{
		Classified:      len(matches) > 0,
		ClassifiedBytes: classifiedBytes,
		TotalBytes:      totalBytes,
		Matches:         matches,
		RulesEvaluated:  rulesEvaluated,
	}
}

// DispatchAction indicates the resolution of a dispatch proposal.
type DispatchAction string

const (
	ActionAllow   DispatchAction = "ALLOW"
	ActionReroute DispatchAction = "REROUTE"
	ActionDeny    DispatchAction = "DENY"
)

// DispatchDecision records the adjudicated outcome of a pre-dispatch residency check.
type DispatchDecision struct {
	Action          DispatchAction        `json:"action"`
	Target          Target                `json:"target"`
	OriginalTarget  Target                `json:"original_target"`
	Reason          string                `json:"reason,omitempty"`
	Detail          string                `json:"detail,omitempty"`
	Classified      bool                  `json:"classified"`
	ClassifiedBytes int64                 `json:"classified_bytes"`
	TotalBytes      int64                 `json:"total_bytes"`
	Matches         []ClassificationMatch `json:"matches,omitempty"`
	RulesEvaluated  int                   `json:"rules_evaluated"`
}

// EgressViolationError represents an explicit refusal when a classified payload cannot egress off-box.
type EgressViolationError struct {
	Reason  string                `json:"reason"`
	Message string                `json:"message"`
	Target  Target                `json:"target"`
	Matches []ClassificationMatch `json:"matches,omitempty"`
}

func (e *EgressViolationError) Error() string {
	return fmt.Sprintf("modelroute: %s: target %q: %s", e.Reason, e.Target.EngineRoute(), e.Message)
}

// EgressDenial records a denied dispatch attempt in the witness artifact.
type EgressDenial struct {
	Reason          string                `json:"reason"`
	Target          Target                `json:"target"`
	PayloadPaths    []string              `json:"payload_paths,omitempty"`
	Matches         []ClassificationMatch `json:"matches,omitempty"`
	ClassifiedBytes int64                 `json:"classified_bytes"`
	Detail          string                `json:"detail,omitempty"`
}

// EgressReroute records a deterministic reroute to local in the witness artifact.
type EgressReroute struct {
	FromTarget      Target                `json:"from_target"`
	ToTarget        Target                `json:"to_target"`
	Reason          string                `json:"reason"`
	PayloadPaths    []string              `json:"payload_paths,omitempty"`
	Matches         []ClassificationMatch `json:"matches,omitempty"`
	ClassifiedBytes int64                 `json:"classified_bytes"`
}

// EgressWitness is the verifiable audit artifact proving that zero classified
// bytes were transmitted to remote model targets during an execution run.
type EgressWitness struct {
	Version                 string          `json:"version"`
	RunID                   string          `json:"run_id,omitempty"`
	TimestampUTC            string          `json:"timestamp_utc,omitempty"`
	RulesEvaluated          int             `json:"rules_evaluated"`
	ClassifiedBytesRemote   int64           `json:"classified_bytes_remote"` // Invariant: MUST be zero
	ClassifiedBytesLocal    int64           `json:"classified_bytes_local"`
	UnclassifiedBytesRemote int64           `json:"unclassified_bytes_remote"`
	UnclassifiedBytesLocal  int64           `json:"unclassified_bytes_local"`
	Denials                 []EgressDenial  `json:"denials,omitempty"`
	Reroutes                []EgressReroute `json:"reroutes,omitempty"`
}

// NewEgressWitness initializes an empty witness record for an execution run.
func NewEgressWitness(runID string) *EgressWitness {
	return &EgressWitness{
		Version:      EgressWitnessVersion,
		RunID:        runID,
		TimestampUTC: time.Now().UTC().Format(time.RFC3339),
	}
}

// Validate verifies the core invariant: ClassifiedBytesRemote MUST be zero.
func (w *EgressWitness) Validate() error {
	if w.ClassifiedBytesRemote != 0 {
		return fmt.Errorf("modelroute: egress witness invariant violated: %d classified bytes sent remotely (must be zero)", w.ClassifiedBytesRemote)
	}
	return nil
}

// JSON renders the witness artifact as indented JSON.
func (w EgressWitness) JSON() []byte {
	out := w
	if out.Version == "" {
		out.Version = EgressWitnessVersion
	}
	b, _ := json.MarshalIndent(out, "", "  ")
	return append(b, '\n')
}

// RecordDecision records a dispatch decision into the witness ledger.
func (w *EgressWitness) RecordDecision(d DispatchDecision, payloadPaths []string) error {
	w.RulesEvaluated += d.RulesEvaluated

	switch d.Action {
	case ActionDeny:
		w.Denials = append(w.Denials, EgressDenial{
			Reason:          d.Reason,
			Target:          d.Target,
			PayloadPaths:    payloadPaths,
			Matches:         d.Matches,
			ClassifiedBytes: d.ClassifiedBytes,
			Detail:          d.Detail,
		})

	case ActionReroute:
		w.Reroutes = append(w.Reroutes, EgressReroute{
			FromTarget:      d.OriginalTarget,
			ToTarget:        d.Target,
			Reason:          d.Reason,
			PayloadPaths:    payloadPaths,
			Matches:         d.Matches,
			ClassifiedBytes: d.ClassifiedBytes,
		})
		w.ClassifiedBytesLocal += d.ClassifiedBytes
		unclassified := d.TotalBytes - d.ClassifiedBytes
		if unclassified > 0 {
			w.UnclassifiedBytesLocal += unclassified
		}

	case ActionAllow:
		if d.Target.Local() {
			w.ClassifiedBytesLocal += d.ClassifiedBytes
			unclassified := d.TotalBytes - d.ClassifiedBytes
			if unclassified > 0 {
				w.UnclassifiedBytesLocal += unclassified
			}
		} else {
			if d.Classified {
				w.ClassifiedBytesRemote += d.ClassifiedBytes
				return fmt.Errorf("modelroute: illegal state: classified payload allowed to remote target")
			}
			w.UnclassifiedBytesRemote += d.TotalBytes
		}
	}

	return nil
}

// PayloadResidencyEnforcer evaluates payload residency policies before model dispatch.
type PayloadResidencyEnforcer struct {
	Config  PayloadClassificationConfig
	Roster  *Roster
	Witness *EgressWitness
	initErr error
}

// NewPayloadResidencyEnforcer creates an enforcer with the provided classification config.
func NewPayloadResidencyEnforcer(cfg PayloadClassificationConfig) *PayloadResidencyEnforcer {
	enforcer := &PayloadResidencyEnforcer{
		Config: cfg,
	}
	if err := cfg.Validate(); err != nil {
		enforcer.initErr = err
	}
	return enforcer
}

// WithRoster associates an account roster for resolving local account fallbacks.
func (e *PayloadResidencyEnforcer) WithRoster(r *Roster) *PayloadResidencyEnforcer {
	e.Roster = r
	return e
}

// WithWitness attaches an egress witness artifact to record decisions.
func (e *PayloadResidencyEnforcer) WithWitness(w *EgressWitness) *PayloadResidencyEnforcer {
	e.Witness = w
	return e
}

// Check evaluates whether a payload may be dispatched to target.
// Composes account-level locality (target.Local() / target.Remote()) with payload classification.
func (e *PayloadResidencyEnforcer) Check(target Target, payload Payload) (DispatchDecision, error) {
	paths := payload.Paths
	if len(paths) == 0 {
		for _, it := range payload.AllItems() {
			if it.Path != "" {
				paths = append(paths, it.Path)
			}
		}
	}

	// 1. Fail closed on invalid classification config.
	if e.initErr != nil {
		dec := DispatchDecision{
			Action:         ActionDeny,
			Target:         target,
			OriginalTarget: target,
			Reason:         ReasonClassificationInvalid,
			Detail:         fmt.Sprintf("classification configuration is invalid: %v", e.initErr),
		}
		if e.Witness != nil {
			_ = e.Witness.RecordDecision(dec, paths)
		}
		return dec, &EgressViolationError{
			Reason:  ReasonClassificationInvalid,
			Message: dec.Detail,
			Target:  target,
		}
	}

	// 2. Classify payload.
	classification := ClassifyPayload(e.Config, payload)

	// 3. Unclassified payload: pass-through allowed.
	if !classification.Classified {
		dec := DispatchDecision{
			Action:         ActionAllow,
			Target:         target,
			OriginalTarget: target,
			Classified:     false,
			TotalBytes:     classification.TotalBytes,
			RulesEvaluated: classification.RulesEvaluated,
		}
		if e.Witness != nil {
			_ = e.Witness.RecordDecision(dec, paths)
		}
		return dec, nil
	}

	// 4. Classified payload + Local target: allowed because bytes remain on-box.
	if target.Local() {
		dec := DispatchDecision{
			Action:          ActionAllow,
			Target:          target,
			OriginalTarget:  target,
			Classified:      true,
			ClassifiedBytes: classification.ClassifiedBytes,
			TotalBytes:      classification.TotalBytes,
			Matches:         classification.Matches,
			RulesEvaluated:  classification.RulesEvaluated,
		}
		if e.Witness != nil {
			_ = e.Witness.RecordDecision(dec, paths)
		}
		return dec, nil
	}

	// 5. Classified payload + Remote target: deny egress or reroute.
	var localTarget *Target
	if e.Config.LocalTarget != nil && e.Config.LocalTarget.Local() {
		localTarget = e.Config.LocalTarget
	} else if e.Config.LocalAccount != "" && e.Roster != nil {
		// First check if LocalAccount matches an Account.ID in the Roster.
		for _, a := range e.Roster.Accounts {
			if a.ID == e.Config.LocalAccount && a.Kind == KindLocal {
				baseURL := a.BaseURL
				if baseURL == "" {
					baseURL = KindBaseURL(a.Kind)
				}
				lt := Target{
					Model:             a.ID,
					Account:           a.ID,
					Kind:              a.Kind,
					BaseURL:           baseURL,
					CredEnv:           a.CredEnv,
					UpstreamModel:     a.ID,
					ContextTokens:     a.ContextTokens,
					MaxOutputTokens:   a.MaxOutputTokens,
					RequestsPerMinute: a.RequestsPerMinute,
					RequestsPerDay:    a.RequestsPerDay,
					TokensPerMinute:   a.TokensPerMinute,
					TokensPerDay:      a.TokensPerDay,
					Principals:        a.Principals,
					ManualOnly:        a.ManualOnly,
					AuthScheme:        a.AuthScheme,
				}
				localTarget = &lt
				break
			}
		}
		// If not matched as an account ID, check if it resolves as a model ID to a local target.
		if localTarget == nil {
			if lt, err := e.Roster.Resolve(e.Config.LocalAccount); err == nil && lt.Local() {
				localTarget = &lt
			}
		}
	}

	// 5a. Deterministic reroute to declared local target if configured.
	if localTarget != nil {
		dec := DispatchDecision{
			Action:          ActionReroute,
			Target:          *localTarget,
			OriginalTarget:  target,
			Reason:          ReasonPayloadClassifiedReroutedLocal,
			Detail:          fmt.Sprintf("rerouted classified payload from remote %s to local %s", target.EngineRoute(), localTarget.EngineRoute()),
			Classified:      true,
			ClassifiedBytes: classification.ClassifiedBytes,
			TotalBytes:      classification.TotalBytes,
			Matches:         classification.Matches,
			RulesEvaluated:  classification.RulesEvaluated,
		}
		if e.Witness != nil {
			_ = e.Witness.RecordDecision(dec, paths)
		}
		return dec, nil
	}

	// 5b. Explicit refusal when no local target exists.
	dec := DispatchDecision{
		Action:          ActionDeny,
		Target:          target,
		OriginalTarget:  target,
		Reason:          ReasonPayloadClassifiedRemoteDenied,
		Detail:          fmt.Sprintf("classified payload (%d bytes, %d matches) denied dispatch to remote target %s", classification.ClassifiedBytes, len(classification.Matches), target.EngineRoute()),
		Classified:      true,
		ClassifiedBytes: classification.ClassifiedBytes,
		TotalBytes:      classification.TotalBytes,
		Matches:         classification.Matches,
		RulesEvaluated:  classification.RulesEvaluated,
	}
	if e.Witness != nil {
		_ = e.Witness.RecordDecision(dec, paths)
	}
	return dec, &EgressViolationError{
		Reason:  ReasonPayloadClassifiedRemoteDenied,
		Message: dec.Detail,
		Target:  target,
		Matches: classification.Matches,
	}
}
