package trajhook

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/trajctlhook"
)

// Re-export closed intervention actions from trajctlhook
type Action = trajctlhook.Action

const (
	ActionSteer    = trajctlhook.ActionSteer
	ActionPause    = trajctlhook.ActionPause
	ActionRollback = trajctlhook.ActionRollback
	ActionRetry    = trajctlhook.ActionRetry
)

// StepClassification is the structured verdict emitted by a step classifier.
type StepClassification = trajctlhook.StepClassification

// StepClassifier defines the contract for user-defined and skill-based step classifiers.
// Deliverable 1 (#11410): StepClassifier interface with ClassifyPreCall and ClassifyPostResult.
type StepClassifier interface {
	ClassifyPreCall(ctx context.Context, tool string, args string) (StepClassification, error)
	ClassifyPostResult(ctx context.Context, tool string, args string, result string) (StepClassification, error)
}

// StepClassifierFunc is a function adapter implementing StepClassifier.
// It is invoked with an empty result on ClassifyPreCall, and with the result on ClassifyPostResult.
type StepClassifierFunc func(ctx context.Context, tool string, args string, result string) (StepClassification, error)

// ClassifyPreCall calls the underlying function with empty result string.
func (f StepClassifierFunc) ClassifyPreCall(ctx context.Context, tool string, args string) (StepClassification, error) {
	return f(ctx, tool, args, "")
}

// ClassifyPostResult calls the underlying function with the tool result.
func (f StepClassifierFunc) ClassifyPostResult(ctx context.Context, tool string, args string, result string) (StepClassification, error) {
	return f(ctx, tool, args, result)
}

// ClassifyStep calls the underlying function directly.
func (f StepClassifierFunc) ClassifyStep(ctx context.Context, tool string, args string, result string) (StepClassification, error) {
	return f(ctx, tool, args, result)
}

// PreCallClassifierFunc adapts a function to StepClassifier, running only pre-call.
type PreCallClassifierFunc func(ctx context.Context, tool string, args string) (StepClassification, error)

// ClassifyPreCall executes the pre-call check function.
func (f PreCallClassifierFunc) ClassifyPreCall(ctx context.Context, tool string, args string) (StepClassification, error) {
	return f(ctx, tool, args)
}

// ClassifyPostResult returns empty classification for pre-call only classifier.
func (f PreCallClassifierFunc) ClassifyPostResult(ctx context.Context, tool string, args string, result string) (StepClassification, error) {
	return StepClassification{}, nil
}

// PostResultClassifierFunc adapts a function to StepClassifier, running only post-result.
type PostResultClassifierFunc func(ctx context.Context, tool string, args string, result string) (StepClassification, error)

// ClassifyPreCall returns empty classification for post-result only classifier.
func (f PostResultClassifierFunc) ClassifyPreCall(ctx context.Context, tool string, args string) (StepClassification, error) {
	return StepClassification{}, nil
}

// ClassifyPostResult executes the post-result check function.
func (f PostResultClassifierFunc) ClassifyPostResult(ctx context.Context, tool string, args string, result string) (StepClassification, error) {
	return f(ctx, tool, args, result)
}

// ClassifierRegistry manages named step classifiers in deterministic registration order.
type ClassifierRegistry struct {
	mu    sync.RWMutex
	items map[string]StepClassifier
	order []string
}

// NewClassifierRegistry creates a new, empty ClassifierRegistry.
func NewClassifierRegistry() *ClassifierRegistry {
	return &ClassifierRegistry{
		items: make(map[string]StepClassifier),
	}
}

// Register adds or updates a named step classifier, preserving registration order.
func (r *ClassifierRegistry) Register(name string, sc StepClassifier) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.items[name]; !exists {
		r.order = append(r.order, name)
	}
	r.items[name] = sc
}

// GetClassifiers returns all registered step classifiers in registration order.
func (r *ClassifierRegistry) GetClassifiers() []StepClassifier {
	r.mu.RLock()
	defer r.mu.RUnlock()
	res := make([]StepClassifier, 0, len(r.order))
	for _, name := range r.order {
		if sc, ok := r.items[name]; ok {
			res = append(res, sc)
		}
	}
	return res
}

// Get returns the named step classifier if registered.
func (r *ClassifierRegistry) Get(name string) (StepClassifier, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	sc, ok := r.items[name]
	return sc, ok
}

// GetNames returns the names of all registered classifiers in registration order.
func (r *ClassifierRegistry) GetNames() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, len(r.order))
	copy(out, r.order)
	return out
}

// Len returns the count of registered classifiers.
func (r *ClassifierRegistry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.order)
}

// Has reports whether a classifier with the given name is registered.
func (r *ClassifierRegistry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.items[name]
	return ok
}

// Reset clears all registered classifiers (primarily for testing).
func (r *ClassifierRegistry) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.items = make(map[string]StepClassifier)
	r.order = nil
}

// EvaluatePreCall runs registered classifiers' ClassifyPreCall in registration order, short-circuiting on the first tripped predicate.
func (r *ClassifierRegistry) EvaluatePreCall(ctx context.Context, tool, args string) (StepClassification, bool) {
	classifiers := r.GetClassifiers()
	for _, sc := range classifiers {
		c, err := sc.ClassifyPreCall(ctx, tool, args)
		if err != nil {
			continue
		}
		if c.Action.IsValid() {
			return c, true
		}
	}
	return StepClassification{}, false
}

// EvaluatePostResult runs registered classifiers' ClassifyPostResult in registration order, short-circuiting on the first tripped predicate.
func (r *ClassifierRegistry) EvaluatePostResult(ctx context.Context, tool, args, result string) (StepClassification, bool) {
	classifiers := r.GetClassifiers()
	for _, sc := range classifiers {
		c, err := sc.ClassifyPostResult(ctx, tool, args, result)
		if err != nil {
			continue
		}
		if c.Action.IsValid() {
			return c, true
		}
	}
	return StepClassification{}, false
}

// EvaluateTurn runs the registered classifiers in order: first evaluating pre-call predicates,
// then evaluating post-result predicates. It short-circuits on the first tripped predicate.
func (r *ClassifierRegistry) EvaluateTurn(ctx context.Context, tool, args, result string) (StepClassification, bool) {
	if c, ok := r.EvaluatePreCall(ctx, tool, args); ok {
		return c, true
	}
	return r.EvaluatePostResult(ctx, tool, args, result)
}

var defaultClassifierRegistry = NewClassifierRegistry()

// RegisterStepClassifier registers a step classifier in the default registry.
func RegisterStepClassifier(name string, sc StepClassifier) {
	defaultClassifierRegistry.Register(name, sc)
}

// GetStepClassifiers returns all registered step classifiers from the default registry.
func GetStepClassifiers() []StepClassifier {
	return defaultClassifierRegistry.GetClassifiers()
}

// ResetStepClassifiersForTest resets the default classifier registry for test isolation.
func ResetStepClassifiersForTest() {
	defaultClassifierRegistry.Reset()
}

// EvaluatePreCall evaluates the registered classifier chain before tool execution.
func EvaluatePreCall(ctx context.Context, tool, args string) (StepClassification, bool) {
	return defaultClassifierRegistry.EvaluatePreCall(ctx, tool, args)
}

// EvaluatePostResult evaluates the registered classifier chain after tool execution.
func EvaluatePostResult(ctx context.Context, tool, args, result string) (StepClassification, bool) {
	return defaultClassifierRegistry.EvaluatePostResult(ctx, tool, args, result)
}

// EvaluateTurn evaluates the registered step classifier chain, short-circuiting on the first tripped predicate.
func EvaluateTurn(ctx context.Context, tool, args, result string) (StepClassification, bool) {
	return defaultClassifierRegistry.EvaluateTurn(ctx, tool, args, result)
}

// StepClassifierSemanticScreen implements abi.SemanticScreen to integrate seamlessly
// with the kernel's registered screen chain.
type StepClassifierSemanticScreen struct {
	Registry *ClassifierRegistry
}

var (
	_ abi.SemanticScreen = StepClassifierSemanticScreen{}
	_ abi.SemanticScreen = (*StepClassifierSemanticScreen)(nil)
)

// NewStepClassifierSemanticScreen creates a new screen instance backed by the default or custom registry.
func NewStepClassifierSemanticScreen(reg ...*ClassifierRegistry) *StepClassifierSemanticScreen {
	var r *ClassifierRegistry
	if len(reg) > 0 && reg[0] != nil {
		r = reg[0]
	}
	return &StepClassifierSemanticScreen{Registry: r}
}

// Register registers this screen with the kernel's ABI semantic screen registry.
func (s *StepClassifierSemanticScreen) Register() {
	abi.RegisterSemanticScreen(s)
}

// ScreenToolCall inspects a tool call before execution against registered pre-call classifiers.
func (s StepClassifierSemanticScreen) ScreenToolCall(ctx context.Context, c *abi.ToolCall) abi.ScreenAdvice {
	if c == nil {
		return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
	}
	tool := c.Tool
	args := ""
	if c.Meta != nil {
		if a, ok := c.Meta["args"]; ok {
			args = a
		} else if a, ok := c.Meta["arguments"]; ok {
			args = a
		}
	}
	if args == "" && len(c.Args.Inline) > 0 {
		args = string(c.Args.Inline)
	}

	reg := s.Registry
	if reg == nil {
		reg = defaultClassifierRegistry
	}

	classification, tripped := reg.EvaluatePreCall(ctx, tool, args)
	if !tripped {
		return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
	}

	switch classification.Action {
	case ActionPause, ActionRollback, ActionRetry:
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonTrustViolation,
			Digest:      classification.Reason,
			By:          "step_classifier:pre_call:" + string(classification.Action),
		}
	case ActionSteer:
		digest := classification.Guidance
		if digest == "" {
			digest = classification.Reason
		}
		return abi.ScreenAdvice{
			Disposition: abi.ScreenDigest,
			Digest:      digest,
			By:          "step_classifier:pre_call:steer",
		}
	default:
		return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
	}
}

// ScreenResult inspects the result body and tool call using the registered classifier chain.
func (s StepClassifierSemanticScreen) ScreenResult(ctx context.Context, c *abi.ToolCall, body []byte) abi.ScreenAdvice {
	tool := ""
	args := ""
	if c != nil {
		tool = c.Tool
		if c.Meta != nil {
			if a, ok := c.Meta["args"]; ok {
				args = a
			} else if a, ok := c.Meta["arguments"]; ok {
				args = a
			}
		}
		if args == "" && len(c.Args.Inline) > 0 {
			args = string(c.Args.Inline)
		}
	}
	result := string(body)

	reg := s.Registry
	if reg == nil {
		reg = defaultClassifierRegistry
	}

	classification, tripped := reg.EvaluateTurn(ctx, tool, args, result)
	if !tripped {
		return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
	}

	switch classification.Action {
	case ActionPause, ActionRollback, ActionRetry:
		return abi.ScreenAdvice{
			Disposition: abi.ScreenQuarantine,
			Reason:      abi.ReasonTrustViolation,
			Digest:      classification.Reason,
			By:          "step_classifier:" + string(classification.Action),
		}
	case ActionSteer:
		digest := classification.Guidance
		if digest == "" {
			digest = classification.Reason
		}
		return abi.ScreenAdvice{
			Disposition: abi.ScreenDigest,
			Digest:      digest,
			By:          "step_classifier:steer",
		}
	default:
		return abi.ScreenAdvice{Disposition: abi.ScreenAllow}
	}
}

// --- Skill Integration Hook (Deliverable 4: #11410) ---

// SkillClassifierDecl specifies a skill-declared step classifier for trajectory intervention.
// It allows skills (in .agents/skills/* or registered programmatically) to declare custom
// predicates and closed control actions (STEER, PAUSE, ROLLBACK, RETRY) to mount during session startup.
type SkillClassifierDecl struct {
	// Skill identifies the declaring skill (e.g., "goal", "dos-dispatch", "git-safety").
	Skill string `json:"skill"`

	// Name is an optional identifier for the classifier. If omitted, defaults to Skill.
	Name string `json:"name,omitempty"`

	// Description explains the purpose or scope of this classifier.
	Description string `json:"description,omitempty"`

	// Tools optionally restricts this classifier to specific tool names (e.g. ["bash", "write"]).
	// An empty slice matches any tool.
	Tools []string `json:"tools,omitempty"`

	// MatchArgs specifies substrings that arguments must contain to trigger the classifier.
	// If multiple substrings are provided, ALL must be present.
	MatchArgs []string `json:"match_args,omitempty"`

	// MatchAnyArgs specifies substrings where at least ONE must be present in args.
	MatchAnyArgs []string `json:"match_any_args,omitempty"`

	// MatchResult specifies substrings that the tool result must contain.
	// If multiple substrings are provided, ALL must be present.
	MatchResult []string `json:"match_result,omitempty"`

	// MatchAnyResult specifies substrings where at least ONE must be present in result.
	MatchAnyResult []string `json:"match_any_result,omitempty"`

	// Action specifies the closed trajectory control action: STEER, PAUSE, ROLLBACK, RETRY.
	Action Action `json:"action"`

	// Confidence indicates the classification confidence (0.0 - 1.0). Defaults to 1.0 if not specified.
	Confidence float64 `json:"confidence,omitempty"`

	// Reason is the human/audit explanation for the classification.
	Reason string `json:"reason"`

	// Guidance provides targeted steer or retry instructions for the agent.
	Guidance string `json:"guidance,omitempty"`

	// NegativeConstraints specifies negative constraints (DO NOT rules) for RETRY actions.
	NegativeConstraints []string `json:"negative_constraints,omitempty"`

	// Classifier is an optional programmatic StepClassifier implementation.
	// When provided, it takes precedence for classification logic while still
	// respecting the Tools filter if configured.
	Classifier StepClassifier `json:"-"`

	// MatchPredicate is an optional programmatic condition predicate.
	// When provided, it must evaluate to true for the classifier to trigger.
	MatchPredicate func(ctx context.Context, tool, args, result string) bool `json:"-"`
}

// FullName returns the canonical registered identifier for this declaration.
// If Name is provided without a "skill:" prefix, it is formatted as "skill:<skill>:<name>".
// If Name is omitted, it is formatted as "skill:<skill>".
func (d SkillClassifierDecl) FullName() string {
	skill := strings.TrimSpace(d.Skill)
	name := strings.TrimSpace(d.Name)
	if name != "" {
		if skill != "" && !strings.HasPrefix(name, "skill:") {
			return "skill:" + skill + ":" + name
		}
		return name
	}
	if skill != "" {
		if strings.HasPrefix(skill, "skill:") {
			return skill
		}
		return "skill:" + skill
	}
	return ""
}

// ToClassifier compiles the declaration into an executable StepClassifier.
func (d SkillClassifierDecl) ToClassifier() (StepClassifier, error) {
	name := d.FullName()
	if name == "" {
		return nil, fmt.Errorf("trajhook: skill classifier declaration must have a skill or name")
	}

	if d.Classifier != nil {
		if len(d.Tools) == 0 {
			return d.Classifier, nil
		}
		allowed := make(map[string]bool, len(d.Tools))
		for _, t := range d.Tools {
			allowed[strings.ToLower(strings.TrimSpace(t))] = true
		}
		return &toolFilteredClassifier{
			inner:        d.Classifier,
			allowedTools: allowed,
		}, nil
	}

	action := d.Action
	if !action.IsValid() {
		parsed, err := trajctlhook.ParseAction(string(d.Action))
		if err == nil && parsed.IsValid() {
			action = parsed
		}
	}
	if !action.IsValid() {
		return nil, fmt.Errorf("trajhook: invalid or missing action %q in skill classifier %q", d.Action, name)
	}

	confidence := d.Confidence
	if confidence <= 0 {
		confidence = 1.0
	}

	var allowedTools map[string]bool
	if len(d.Tools) > 0 {
		allowedTools = make(map[string]bool, len(d.Tools))
		for _, t := range d.Tools {
			allowedTools[strings.ToLower(strings.TrimSpace(t))] = true
		}
	}

	return &skillCompiledClassifier{
		decl:         d,
		action:       action,
		confidence:   confidence,
		allowedTools: allowedTools,
	}, nil
}

type toolFilteredClassifier struct {
	inner        StepClassifier
	allowedTools map[string]bool
}

func (f *toolFilteredClassifier) ClassifyPreCall(ctx context.Context, tool, args string) (StepClassification, error) {
	if f.allowedTools != nil && !f.allowedTools[strings.ToLower(strings.TrimSpace(tool))] {
		return StepClassification{}, nil
	}
	return f.inner.ClassifyPreCall(ctx, tool, args)
}

func (f *toolFilteredClassifier) ClassifyPostResult(ctx context.Context, tool, args, result string) (StepClassification, error) {
	if f.allowedTools != nil && !f.allowedTools[strings.ToLower(strings.TrimSpace(tool))] {
		return StepClassification{}, nil
	}
	return f.inner.ClassifyPostResult(ctx, tool, args, result)
}

func (f *toolFilteredClassifier) ClassifyStep(ctx context.Context, tool, args, result string) (StepClassification, error) {
	if res, err := f.ClassifyPreCall(ctx, tool, args); err == nil && res.Action.IsValid() {
		return res, nil
	}
	return f.ClassifyPostResult(ctx, tool, args, result)
}

type skillCompiledClassifier struct {
	decl         SkillClassifierDecl
	action       Action
	confidence   float64
	allowedTools map[string]bool
}

func (c *skillCompiledClassifier) ClassifyPreCall(ctx context.Context, tool, args string) (StepClassification, error) {
	// If this declaration requires matching result substrings, it cannot trip pre-call
	if len(c.decl.MatchResult) > 0 || len(c.decl.MatchAnyResult) > 0 {
		return StepClassification{}, nil
	}

	// 1. Tool filter
	if c.allowedTools != nil && !c.allowedTools[strings.ToLower(strings.TrimSpace(tool))] {
		return StepClassification{}, nil
	}

	// 2. MatchArgs: ALL must match
	for _, sub := range c.decl.MatchArgs {
		if sub != "" && !strings.Contains(args, sub) {
			return StepClassification{}, nil
		}
	}

	// 3. MatchAnyArgs: at least ONE must match
	if len(c.decl.MatchAnyArgs) > 0 {
		matched := false
		for _, sub := range c.decl.MatchAnyArgs {
			if sub != "" && strings.Contains(args, sub) {
				matched = true
				break
			}
		}
		if !matched {
			return StepClassification{}, nil
		}
	}

	// 4. MatchPredicate
	if c.decl.MatchPredicate != nil && !c.decl.MatchPredicate(ctx, tool, args, "") {
		return StepClassification{}, nil
	}

	// Tripped pre-call
	return StepClassification{
		Action:              c.action,
		Confidence:          c.confidence,
		Reason:              c.decl.Reason,
		Guidance:            c.decl.Guidance,
		NegativeConstraints: append([]string(nil), c.decl.NegativeConstraints...),
	}, nil
}

func (c *skillCompiledClassifier) ClassifyPostResult(ctx context.Context, tool, args, result string) (StepClassification, error) {
	// 1. Tool filter
	if c.allowedTools != nil && !c.allowedTools[strings.ToLower(strings.TrimSpace(tool))] {
		return StepClassification{}, nil
	}

	// 2. MatchArgs: ALL must match
	for _, sub := range c.decl.MatchArgs {
		if sub != "" && !strings.Contains(args, sub) {
			return StepClassification{}, nil
		}
	}

	// 3. MatchAnyArgs: at least ONE must match
	if len(c.decl.MatchAnyArgs) > 0 {
		matched := false
		for _, sub := range c.decl.MatchAnyArgs {
			if sub != "" && strings.Contains(args, sub) {
				matched = true
				break
			}
		}
		if !matched {
			return StepClassification{}, nil
		}
	}

	// 4. MatchResult: ALL must match
	for _, sub := range c.decl.MatchResult {
		if sub != "" && !strings.Contains(result, sub) {
			return StepClassification{}, nil
		}
	}

	// 5. MatchAnyResult: at least ONE must match
	if len(c.decl.MatchAnyResult) > 0 {
		matched := false
		for _, sub := range c.decl.MatchAnyResult {
			if sub != "" && strings.Contains(result, sub) {
				matched = true
				break
			}
		}
		if !matched {
			return StepClassification{}, nil
		}
	}

	// 6. MatchPredicate
	if c.decl.MatchPredicate != nil && !c.decl.MatchPredicate(ctx, tool, args, result) {
		return StepClassification{}, nil
	}

	// Tripped post-result
	return StepClassification{
		Action:              c.action,
		Confidence:          c.confidence,
		Reason:              c.decl.Reason,
		Guidance:            c.decl.Guidance,
		NegativeConstraints: append([]string(nil), c.decl.NegativeConstraints...),
	}, nil
}

func (c *skillCompiledClassifier) ClassifyStep(ctx context.Context, tool, args, result string) (StepClassification, error) {
	if res, err := c.ClassifyPreCall(ctx, tool, args); err == nil && res.Action.IsValid() {
		return res, nil
	}
	return c.ClassifyPostResult(ctx, tool, args, result)
}

// MountSkillClassifier mounts a skill classifier declaration into the specified registry.
// If reg is nil, the default registry is used.
func MountSkillClassifier(reg *ClassifierRegistry, decl SkillClassifierDecl) error {
	if reg == nil {
		reg = defaultClassifierRegistry
	}
	fullName := decl.FullName()
	if fullName == "" {
		return fmt.Errorf("trajhook: skill classifier declaration must have a skill or name")
	}
	sc, err := decl.ToClassifier()
	if err != nil {
		return err
	}
	reg.Register(fullName, sc)
	return nil
}

// DeclareSkillClassifier declares and mounts a skill classifier into the default registry.
func DeclareSkillClassifier(decl SkillClassifierDecl) error {
	return MountSkillClassifier(defaultClassifierRegistry, decl)
}

// MountSkillClassifiers mounts multiple skill classifier declarations into the specified registry.
func MountSkillClassifiers(reg *ClassifierRegistry, decls ...SkillClassifierDecl) error {
	for _, decl := range decls {
		if err := MountSkillClassifier(reg, decl); err != nil {
			return err
		}
	}
	return nil
}

// DeclareSkillClassifiers declares and mounts multiple skill classifiers into the default registry.
func DeclareSkillClassifiers(decls ...SkillClassifierDecl) error {
	return MountSkillClassifiers(defaultClassifierRegistry, decls...)
}

// LoadSkillClassifierDir inspects a skill directory (e.g. .agents/skills/<name>)
// and loads any declared step classifiers from classifiers.json, step_classifiers.json,
// trajhook.json, or SKILL.md frontmatter.
func LoadSkillClassifierDir(skillDir string) ([]SkillClassifierDecl, error) {
	skillDir = filepath.Clean(skillDir)
	fi, err := os.Stat(skillDir)
	if err != nil || !fi.IsDir() {
		return nil, nil
	}

	skillName := filepath.Base(skillDir)
	var decls []SkillClassifierDecl

	// 1. Dedicated JSON declaration files
	jsonCandidates := []string{"classifiers.json", "step_classifiers.json", "trajhook.json"}
	for _, candidate := range jsonCandidates {
		path := filepath.Join(skillDir, candidate)
		data, err := os.ReadFile(path)
		if err == nil && len(data) > 0 {
			parsed, err := parseClassifierJSON(skillName, data)
			if err == nil && len(parsed) > 0 {
				decls = append(decls, parsed...)
				break
			}
		}
	}

	// 2. SKILL.md or skill.md frontmatter declarations
	mdCandidates := []string{"SKILL.md", "skill.md"}
	for _, candidate := range mdCandidates {
		path := filepath.Join(skillDir, candidate)
		data, err := os.ReadFile(path)
		if err == nil && len(data) > 0 {
			parsed, err := parseSkillMDClassifiers(skillName, data)
			if err == nil && len(parsed) > 0 {
				decls = append(decls, parsed...)
				break
			}
		}
	}

	return decls, nil
}

func parseClassifierJSON(defaultSkill string, data []byte) ([]SkillClassifierDecl, error) {
	var list []SkillClassifierDecl
	if err := json.Unmarshal(data, &list); err == nil {
		for i := range list {
			if list[i].Skill == "" {
				list[i].Skill = defaultSkill
			}
		}
		return list, nil
	}

	var single SkillClassifierDecl
	if err := json.Unmarshal(data, &single); err == nil {
		if single.Skill == "" {
			single.Skill = defaultSkill
		}
		return []SkillClassifierDecl{single}, nil
	}

	return nil, fmt.Errorf("trajhook: failed to unmarshal classifier JSON")
}

func parseSkillMDClassifiers(defaultSkill string, data []byte) ([]SkillClassifierDecl, error) {
	content := string(data)
	lines := strings.Split(content, "\n")
	if len(lines) < 3 {
		return nil, nil
	}
	firstLine := strings.TrimSpace(strings.TrimRight(lines[0], "\r"))
	if firstLine != "---" {
		return nil, nil
	}
	closingIdx := -1
	for i := 1; i < len(lines); i++ {
		l := strings.TrimSpace(strings.TrimRight(lines[i], "\r"))
		if l == "---" {
			closingIdx = i
			break
		}
	}
	if closingIdx == -1 {
		return nil, nil
	}

	fmLines := lines[1:closingIdx]
	metaAction := ""
	metaTool := ""
	metaMatch := ""
	metaReason := ""
	metaGuidance := ""
	name := defaultSkill

	for _, line := range fmLines {
		trimmed := strings.TrimSpace(strings.TrimRight(line, "\r"))
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		k, v, ok := strings.Cut(trimmed, ":")
		if !ok {
			continue
		}
		k = strings.TrimSpace(strings.ToLower(k))
		v = strings.Trim(strings.TrimSpace(v), `"'`)
		switch k {
		case "name":
			if v != "" {
				name = v
			}
		case "trajectory_action", "trajectory-action":
			metaAction = v
		case "trajectory_tool", "trajectory-tool":
			metaTool = v
		case "trajectory_match", "trajectory-match":
			metaMatch = v
		case "trajectory_reason", "trajectory-reason":
			metaReason = v
		case "trajectory_guidance", "trajectory-guidance":
			metaGuidance = v
		}
	}

	if metaAction != "" {
		act, err := trajctlhook.ParseAction(metaAction)
		if err == nil && act.IsValid() {
			decl := SkillClassifierDecl{
				Skill:    name,
				Action:   act,
				Reason:   metaReason,
				Guidance: metaGuidance,
			}
			if metaTool != "" {
				decl.Tools = []string{metaTool}
			}
			if metaMatch != "" {
				decl.MatchArgs = []string{metaMatch}
			}
			return []SkillClassifierDecl{decl}, nil
		}
	}

	return nil, nil
}

// DiscoverSkillClassifiers scans the workspace (.agents/skills and .claude/skills, plus extraDirs)
// for skill directories declaring step classifiers.
func DiscoverSkillClassifiers(workspaceRoot string, extraDirs ...string) ([]SkillClassifierDecl, error) {
	if workspaceRoot == "" {
		if cwd, err := os.Getwd(); err == nil {
			workspaceRoot = cwd
		}
	}
	workspaceRoot = filepath.Clean(workspaceRoot)

	searchRoots := []string{
		filepath.Join(workspaceRoot, ".agents", "skills"),
		filepath.Join(workspaceRoot, ".claude", "skills"),
	}
	for _, d := range extraDirs {
		d = strings.TrimSpace(d)
		if d == "" {
			continue
		}
		if !filepath.IsAbs(d) {
			d = filepath.Join(workspaceRoot, d)
		}
		searchRoots = append(searchRoots, filepath.Clean(d))
	}

	var allDecls []SkillClassifierDecl
	seenDirs := make(map[string]bool)

	for _, root := range searchRoots {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			dirPath := filepath.Join(root, entry.Name())
			canonical := filepath.Clean(dirPath)
			if seenDirs[canonical] {
				continue
			}
			seenDirs[canonical] = true

			decls, err := LoadSkillClassifierDir(dirPath)
			if err == nil && len(decls) > 0 {
				allDecls = append(allDecls, decls...)
			}
		}
	}

	return allDecls, nil
}

// MountDiscoveredSkillClassifiers discovers all skill classifiers in workspaceRoot
// and mounts them into the specified registry. If reg is nil, defaultClassifierRegistry is used.
// Returns the count of mounted classifiers.
func MountDiscoveredSkillClassifiers(reg *ClassifierRegistry, workspaceRoot string, extraDirs ...string) (int, error) {
	if reg == nil {
		reg = defaultClassifierRegistry
	}
	decls, err := DiscoverSkillClassifiers(workspaceRoot, extraDirs...)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, decl := range decls {
		if err := MountSkillClassifier(reg, decl); err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}
