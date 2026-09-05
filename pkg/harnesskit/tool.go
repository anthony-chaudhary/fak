package harnesskit

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
)

// ToolContractVersion is the compatibility version for the declarative tool integration contract.
const ToolContractVersion = "fak.harness.tools/v1"

// Mutability describes the side-effect profile of a tool.
type Mutability string

const (
	// MutabilityReadOnly indicates side-effect-free, idempotent tools (e.g. reading files, querying status).
	MutabilityReadOnly Mutability = "read_only"
	// MutabilityMutating indicates state-changing but bounded tools (e.g. writing files, creating draft PRs).
	MutabilityMutating Mutability = "mutating"
	// MutabilityDestructive indicates irreversible, high-impact tools (e.g. pushing to remotes, deleting branches/records).
	MutabilityDestructive Mutability = "destructive"
)

// AuthType classifies how credentials are paged into a tool invocation.
type AuthType string

const (
	// AuthTypeNone indicates no credentials required.
	AuthTypeNone AuthType = "none"
	// AuthTypeFleetSecret indicates an API key or token paged from the fleet secret manager just-in-time.
	AuthTypeFleetSecret AuthType = "fleet_secret"
	// AuthTypeOAuth2 indicates an OAuth 2.0 access token resolved and refreshed by the harness runtime.
	AuthTypeOAuth2 AuthType = "oauth2"
)

// ConditionKind specifies the mechanism used to evaluate a dynamic tool condition.
type ConditionKind string

const (
	// ConditionKindPrerequisite requires that a specific tool has already succeeded in the current session.
	ConditionKindPrerequisite ConditionKind = "prerequisite"
	// ConditionKindPhase requires that the current harness workflow is in a specific phase (e.g. "plan", "execute", "verify").
	ConditionKindPhase ConditionKind = "phase"
	// ConditionKindRole requires that the calling agent or user holds a specific capability role.
	ConditionKindRole ConditionKind = "role"
	// ConditionKindCustom evaluates an arbitrary predicate against the execution context.
	ConditionKindCustom ConditionKind = "custom"
)

// RateLimit specifies invocation ceilings for a tool.
type RateLimit struct {
	MaxPerTurn    int           `json:"max_per_turn,omitempty"`
	MaxPerSession int           `json:"max_per_session,omitempty"`
	Timeout       time.Duration `json:"timeout,omitempty"`
}

// ToolScope defines the capability boundaries, mutability, and constraints of a tool.
type ToolScope struct {
	WorkspacePaths       []string     `json:"workspace_paths,omitempty"`
	ReadOnly             bool         `json:"read_only,omitempty"`
	NetworkAllowed       bool         `json:"network_allowed,omitempty"`
	MaxTurns             int          `json:"max_turns,omitempty"`
	Mutability           Mutability   `json:"mutability,omitempty"`
	PathScopes           []string     `json:"path_scopes,omitempty"`
	NetworkScopes        []string     `json:"network_scopes,omitempty"`
	RateLimit            RateLimit    `json:"rate_limit,omitempty"`
	RequiredCapabilities []Capability `json:"required_capabilities,omitempty"`
}

// ToolCondition specifies conditional scoping, session allow/block list constraints, and dynamic preconditions.
type ToolCondition struct {
	AllowList    []string                                         `json:"allow_list,omitempty"`
	BlockList    []string                                         `json:"block_list,omitempty"`
	Precondition func(ctx context.Context, sessionID string) bool `json:"-"`
}

// AuthBinding declares credential dependencies resolved just-in-time at the execution boundary and scrubbing policy.
type AuthBinding struct {
	Type                    AuthType `json:"type,omitempty"`
	SecretKey               string   `json:"secret_key,omitempty"`
	SecretRefs              []string `json:"secret_refs,omitempty"`
	JITAuthPaging           bool     `json:"jit_auth_paging,omitempty"`
	OAuthProvider           string   `json:"oauth_provider,omitempty"`
	ScrubSecretsFromResults bool     `json:"scrub_secrets_from_results,omitempty"`
}

// Condition represents a dynamic prerequisite or activation gate for a tool.
type Condition struct {
	Kind        ConditionKind                   `json:"kind"`
	Target      string                          `json:"target,omitempty"`
	Description string                          `json:"description,omitempty"`
	Predicate   func(ctx ExecutionContext) bool `json:"-"`
}

// AuthRequirement declares credential dependencies resolved just-in-time at the execution boundary.
// Invariant: The model context and serialized tool parameters NEVER contain these secrets.
type AuthRequirement struct {
	Type          AuthType `json:"type"`
	SecretKey     string   `json:"secret_key,omitempty"`
	OAuthProvider string   `json:"oauth_provider,omitempty"`
	OAuthScopes   []string `json:"oauth_scopes,omitempty"`
}

// OAuthToken represents a validated, non-expired OAuth 2.0 credential.
type OAuthToken struct {
	AccessToken string    `json:"access_token"`
	TokenType   string    `json:"token_type"`
	ExpiresAt   time.Time `json:"expires_at"`
	Scopes      []string  `json:"scopes"`
}

// Valid reports whether the token is currently valid and not expired.
func (t *OAuthToken) Valid() bool {
	if t == nil || t.AccessToken == "" {
		return false
	}
	if !t.ExpiresAt.IsZero() && time.Now().After(t.ExpiresAt) {
		return false
	}
	return true
}

// SecretStore resolves named credentials from a secure fleet secret vault.
type SecretStore interface {
	GetSecret(ctx context.Context, key string) (string, error)
}

// OAuthTokenProvider resolves and refreshes OAuth tokens.
type OAuthTokenProvider interface {
	GetToken(ctx context.Context, provider string, scopes []string) (*OAuthToken, error)
	RefreshToken(ctx context.Context, provider string) (*OAuthToken, error)
}

// ExecutionContext provides call-scoped state, identity, and safe credential retrieval to a tool handler.
type ExecutionContext interface {
	Context() context.Context
	SessionID() string
	TurnID() string
	Phase() string
	Role() string
	PriorSuccessfulTools() []string
	HasSucceeded(toolName string) bool
	GetSecret(key string) (string, error)
	GetOAuthToken(provider string) (*OAuthToken, error)
}

// ToolHandler is the execution function of a tool.
type ToolHandler func(ctx ExecutionContext, args json.RawMessage) (Result, error)

// ToolDefinition is the declarative integration specification for a tool.
type ToolDefinition struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Schema      map[string]any    `json:"schema,omitempty"`
	Scope       ToolScope         `json:"scope"`
	Condition   ToolCondition     `json:"condition,omitempty"`
	Auth        *AuthBinding      `json:"auth,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`

	// Integration & compatibility fields:
	Integration string          `json:"integration,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Conditions  []Condition     `json:"conditions,omitempty"`
	LegacyAuth  AuthRequirement `json:"legacy_auth,omitempty"`
	Handler     ToolHandler     `json:"-"`
}

// NewTool starts a new fluent tool definition.
func NewTool(name string) *ToolDefinition {
	return &ToolDefinition{
		Name: name,
		Scope: ToolScope{
			ReadOnly:   true,
			Mutability: MutabilityReadOnly,
		},
		Auth: &AuthBinding{
			Type:                    AuthTypeNone,
			ScrubSecretsFromResults: true,
		},
		LegacyAuth: AuthRequirement{Type: AuthTypeNone},
	}
}

// Validate checks whether the tool definition meets structural invariants.
func (t *ToolDefinition) Validate() error {
	if strings.TrimSpace(t.Name) == "" {
		return &Error{Code: CodeInvalid, Op: "tool_validate", Err: errors.New("tool name is required")}
	}
	if t.Schema != nil {
		if len(t.Schema) == 0 {
			return &Error{Code: CodeInvalid, Op: "tool_validate", Err: errors.New("tool schema cannot be empty")}
		}
		if rawType, ok := t.Schema["type"]; ok {
			typeStr, isStr := rawType.(string)
			if !isStr || strings.TrimSpace(typeStr) == "" {
				return &Error{Code: CodeInvalid, Op: "tool_validate", Err: errors.New("tool schema type must be a non-empty string")}
			}
		}
		if _, err := json.Marshal(t.Schema); err != nil {
			return &Error{Code: CodeInvalid, Op: "tool_validate", Err: fmt.Errorf("tool schema is not valid JSON: %w", err)}
		}
	}
	return nil
}

// CheckPermission evaluates whether the invocation is permitted under the tool's conditions and scope.
func (t *ToolDefinition) CheckPermission(sessionID string, requestedPath string, isWrite bool) (bool, string) {
	// 1. Evaluate Condition BlockList
	if sessionID != "" {
		for _, blocked := range t.Condition.BlockList {
			if MatchToolPattern(blocked, sessionID) {
				return false, fmt.Sprintf("session %q is blocked by block list", sessionID)
			}
		}
	}

	// 2. Evaluate Condition AllowList
	if len(t.Condition.AllowList) > 0 {
		allowed := false
		for _, allow := range t.Condition.AllowList {
			if MatchToolPattern(allow, sessionID) {
				allowed = true
				break
			}
		}
		if !allowed {
			return false, fmt.Sprintf("session %q is not in allow list", sessionID)
		}
	}

	// 3. Evaluate Condition Precondition
	if t.Condition.Precondition != nil {
		if !t.Condition.Precondition(context.Background(), sessionID) {
			return false, "precondition failed"
		}
	}

	// 4. Evaluate Scope ReadOnly vs isWrite
	if isWrite && (t.Scope.ReadOnly || t.Scope.Mutability == MutabilityReadOnly) {
		return false, "tool is read-only"
	}

	// 5. Evaluate Scope WorkspacePaths / PathScopes
	paths := append([]string(nil), t.Scope.WorkspacePaths...)
	paths = append(paths, t.Scope.PathScopes...)
	if requestedPath != "" && len(paths) > 0 {
		if !PathWithinScope(requestedPath, paths) {
			return false, fmt.Sprintf("requested path %q outside workspace scope", requestedPath)
		}
	}

	return true, ""
}

// ScrubResult redacts secret references, authorization tokens, and credentials from tool output.
func (t *ToolDefinition) ScrubResult(content []byte) []byte {
	if len(content) == 0 {
		return content
	}
	scrubbed := scrubSensitiveBytes(content)
	if t.Auth != nil && (t.Auth.ScrubSecretsFromResults || t.Auth.JITAuthPaging) {
		s := string(scrubbed)
		for _, ref := range t.Auth.SecretRefs {
			refTrim := strings.TrimSpace(ref)
			if refTrim != "" && strings.Contains(s, refTrim) {
				s = strings.ReplaceAll(s, refTrim, "[REDACTED]")
			}
		}
		if t.Auth.SecretKey != "" && strings.Contains(s, t.Auth.SecretKey) {
			s = strings.ReplaceAll(s, t.Auth.SecretKey, "[REDACTED]")
		}
		scrubbed = []byte(s)
	}
	return scrubbed
}

// WithSchema sets the declarative map schema for tool arguments.
func (t *ToolDefinition) WithSchema(schema map[string]any) *ToolDefinition {
	t.Schema = schema
	return t
}

// WithToolCondition sets the conditional scoping and dynamic preconditions.
func (t *ToolDefinition) WithToolCondition(cond ToolCondition) *ToolDefinition {
	t.Condition = cond
	return t
}

// WithAuthBinding configures declarative auth binding.
func (t *ToolDefinition) WithAuthBinding(auth *AuthBinding) *ToolDefinition {
	t.Auth = auth
	return t
}

// WithMetadata attaches arbitrary key-value metadata to the tool definition.
func (t *ToolDefinition) WithMetadata(meta map[string]string) *ToolDefinition {
	t.Metadata = meta
	return t
}

// WithIntegration assigns the integration namespace for this tool.
func (t *ToolDefinition) WithIntegration(integration string) *ToolDefinition {
	t.Integration = integration
	return t
}

// WithDescription sets the semantic description exposed to the model.
func (t *ToolDefinition) WithDescription(desc string) *ToolDefinition {
	t.Description = desc
	return t
}

// WithParameters sets the JSON schema for tool arguments.
func (t *ToolDefinition) WithParameters(schema json.RawMessage) *ToolDefinition {
	t.Parameters = schema
	return t
}

// WithScope configures mutability, paths, network, and rate limits.
func (t *ToolDefinition) WithScope(scope ToolScope) *ToolDefinition {
	if scope.ReadOnly && scope.Mutability == "" {
		scope.Mutability = MutabilityReadOnly
	} else if scope.Mutability == MutabilityReadOnly {
		scope.ReadOnly = true
	}
	if len(scope.WorkspacePaths) > 0 && len(scope.PathScopes) == 0 {
		scope.PathScopes = append([]string(nil), scope.WorkspacePaths...)
	} else if len(scope.PathScopes) > 0 && len(scope.WorkspacePaths) == 0 {
		scope.WorkspacePaths = append([]string(nil), scope.PathScopes...)
	}
	t.Scope = scope
	return t
}

// WithCondition appends an activation precondition.
func (t *ToolDefinition) WithCondition(cond Condition) *ToolDefinition {
	t.Conditions = append(t.Conditions, cond)
	return t
}

// WithAuth configures just-in-time credential requirements.
func (t *ToolDefinition) WithAuth(auth AuthRequirement) *ToolDefinition {
	t.LegacyAuth = auth
	if t.Auth == nil {
		t.Auth = &AuthBinding{}
	}
	t.Auth.Type = auth.Type
	t.Auth.SecretKey = auth.SecretKey
	if auth.SecretKey != "" {
		t.Auth.SecretRefs = []string{auth.SecretKey}
	}
	t.Auth.OAuthProvider = auth.OAuthProvider
	if auth.Type == AuthTypeFleetSecret || auth.Type == AuthTypeOAuth2 {
		t.Auth.JITAuthPaging = true
		t.Auth.ScrubSecretsFromResults = true
	}
	return t
}

// WithHandler binds the execution handler.
func (t *ToolDefinition) WithHandler(h ToolHandler) *ToolDefinition {
	t.Handler = h
	return t
}

// RequirePriorSuccess creates a condition requiring that priorTool succeeded in the current session.
func RequirePriorSuccess(priorTool string) Condition {
	return Condition{
		Kind:        ConditionKindPrerequisite,
		Target:      priorTool,
		Description: fmt.Sprintf("requires prior successful execution of %q", priorTool),
		Predicate: func(ctx ExecutionContext) bool {
			return ctx.HasSucceeded(priorTool)
		},
	}
}

// RequirePhase creates a condition requiring that the current session is in the named phase.
func RequirePhase(phase string) Condition {
	return Condition{
		Kind:        ConditionKindPhase,
		Target:      phase,
		Description: fmt.Sprintf("only active during %q phase", phase),
		Predicate: func(ctx ExecutionContext) bool {
			return strings.EqualFold(ctx.Phase(), phase)
		},
	}
}

// RequireRole creates a condition requiring that the caller possesses the specified role.
func RequireRole(role string) Condition {
	return Condition{
		Kind:        ConditionKindRole,
		Target:      role,
		Description: fmt.Sprintf("requires caller role %q", role),
		Predicate: func(ctx ExecutionContext) bool {
			return strings.EqualFold(ctx.Role(), role)
		},
	}
}

// IntegrationDefinition is a cohesive set of tools representing an integration to an external system.
type IntegrationDefinition struct {
	Name        string           `json:"name"`
	Description string           `json:"description"`
	DefaultAuth AuthRequirement  `json:"default_auth"`
	Tools       []ToolDefinition `json:"tools"`
}

// NewIntegration initializes a cohesive integration set.
func NewIntegration(name, description string) *IntegrationDefinition {
	return &IntegrationDefinition{
		Name:        name,
		Description: description,
		DefaultAuth: AuthRequirement{Type: AuthTypeNone},
	}
}

// WithDefaultAuth sets the default credential requirement for all tools in this integration.
func (i *IntegrationDefinition) WithDefaultAuth(auth AuthRequirement) *IntegrationDefinition {
	i.DefaultAuth = auth
	return i
}

// WithTool adds a tool to the integration, inheriting integration name and default auth if unset.
func (i *IntegrationDefinition) WithTool(tool *ToolDefinition) *IntegrationDefinition {
	if tool == nil {
		return i
	}
	cp := *tool
	if cp.Integration == "" {
		cp.Integration = i.Name
	}
	if cp.Auth == nil || cp.Auth.Type == "" || cp.Auth.Type == AuthTypeNone {
		cp.Auth = &AuthBinding{
			Type:                    i.DefaultAuth.Type,
			SecretKey:               i.DefaultAuth.SecretKey,
			OAuthProvider:           i.DefaultAuth.OAuthProvider,
			JITAuthPaging:           i.DefaultAuth.Type != "" && i.DefaultAuth.Type != AuthTypeNone,
			ScrubSecretsFromResults: true,
		}
		if i.DefaultAuth.SecretKey != "" {
			cp.Auth.SecretRefs = []string{i.DefaultAuth.SecretKey}
		}
	}
	i.Tools = append(i.Tools, cp)
	return i
}

// ToolFilterPolicy specifies session-level allow, block, and masking rules.
type ToolFilterPolicy struct {
	AllowList      []string `json:"allow_list,omitempty"`
	BlockList      []string `json:"block_list,omitempty"`
	MaskBlocked    bool     `json:"mask_blocked"`
	MaxActiveTools int      `json:"max_active_tools,omitempty"`
}

// MatchToolPattern reports whether a tool name matches a filter pattern (supporting '*' wildcard prefix or suffix).
func MatchToolPattern(pattern, toolName string) bool {
	pattern = strings.TrimSpace(pattern)
	if pattern == "" || pattern == "*" {
		return true
	}
	if strings.HasPrefix(pattern, "*") {
		suffix := strings.TrimPrefix(pattern, "*")
		return strings.HasSuffix(toolName, suffix)
	}
	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(toolName, prefix)
	}
	return pattern == toolName
}

// IsAllowed evaluates a tool against an allow/block filter policy.
func (p ToolFilterPolicy) IsAllowed(toolName string) bool {
	for _, block := range p.BlockList {
		if MatchToolPattern(block, toolName) {
			return false
		}
	}
	if len(p.AllowList) == 0 {
		return true
	}
	for _, allow := range p.AllowList {
		if MatchToolPattern(allow, toolName) {
			return true
		}
	}
	return false
}

// ToolTelemetry captures structured execution metrics and audit facts for Grafana dashboarding.
type ToolTelemetry struct {
	SessionID   string        `json:"session_id"`
	TurnID      string        `json:"turn_id"`
	Tool        string        `json:"tool"`
	Integration string        `json:"integration,omitempty"`
	Mutability  Mutability    `json:"mutability"`
	Verdict     string        `json:"verdict"`
	Reason      string        `json:"reason,omitempty"`
	AuthPaged   bool          `json:"auth_paged"`
	AuthType    AuthType      `json:"auth_type"`
	Duration    time.Duration `json:"duration"`
	InputBytes  int           `json:"input_bytes"`
	OutputBytes int           `json:"output_bytes"`
	Error       string        `json:"error,omitempty"`
}

// ToolRegistry manages registered tools, evaluates dynamic conditions, performs JIT auth, and records telemetry.
type ToolRegistry struct {
	mu           sync.RWMutex
	tools        map[string]ToolDefinition
	callCounts   map[string]map[string]int // sessionID -> tool -> count
	turnCounts   map[string]map[string]int // sessionID:turnID -> tool -> count
	secretStore  SecretStore
	oauthStore   OAuthTokenProvider
	telemetrySub func(ToolTelemetry)
}

// NewToolRegistry initializes a thread-safe tool registry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools:      make(map[string]ToolDefinition),
		callCounts: make(map[string]map[string]int),
		turnCounts: make(map[string]map[string]int),
	}
}

// WithSecretStore attaches a fleet secret store for JIT credential paging.
func (r *ToolRegistry) WithSecretStore(s SecretStore) *ToolRegistry {
	r.secretStore = s
	return r
}

// WithOAuthTokenProvider attaches an OAuth token provider.
func (r *ToolRegistry) WithOAuthTokenProvider(p OAuthTokenProvider) *ToolRegistry {
	r.oauthStore = p
	return r
}

// OnTelemetry attaches a telemetry subscriber (e.g. for Prometheus or Grafana log sink).
func (r *ToolRegistry) OnTelemetry(fn func(ToolTelemetry)) *ToolRegistry {
	r.telemetrySub = fn
	return r
}

// Register adds a tool definition to the registry.
func (r *ToolRegistry) Register(tool ToolDefinition) error {
	if err := tool.Validate(); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[tool.Name]; exists {
		return &Error{Code: CodeConflict, Op: "tool_register", Err: fmt.Errorf("tool %q already registered", tool.Name)}
	}
	r.tools[tool.Name] = tool
	return nil
}

// RegisterIntegration registers all tools from an integration definition.
func (r *ToolRegistry) RegisterIntegration(integration IntegrationDefinition) error {
	for _, tool := range integration.Tools {
		if err := r.Register(tool); err != nil {
			return err
		}
	}
	return nil
}

// Get retrieves a tool by name.
func (r *ToolRegistry) Get(name string) (ToolDefinition, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	return tool, ok
}

// List returns all registered tools sorted by name.
func (r *ToolRegistry) List() []ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]ToolDefinition, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t)
	}
	slices.SortFunc(out, func(a, b ToolDefinition) int {
		return strings.Compare(a.Name, b.Name)
	})
	return out
}

// FilterForTurn returns the active tools exposed to the model for a given turn.
// Blocked tools and tools failing preconditions are filtered out if policy.MaskBlocked is true.
func (r *ToolRegistry) FilterForTurn(ctx ExecutionContext, policy ToolFilterPolicy) []ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var result []ToolDefinition
	for _, tool := range r.tools {
		if !policy.IsAllowed(tool.Name) {
			if policy.MaskBlocked {
				continue
			}
		}
		// Evaluate conditions
		conditionsMet := true
		for _, cond := range tool.Conditions {
			if cond.Predicate != nil && !cond.Predicate(ctx) {
				conditionsMet = false
				break
			}
		}
		if !conditionsMet && policy.MaskBlocked {
			continue
		}
		result = append(result, tool)
		if policy.MaxActiveTools > 0 && len(result) >= policy.MaxActiveTools {
			break
		}
	}
	slices.SortFunc(result, func(a, b ToolDefinition) int {
		return strings.Compare(a.Name, b.Name)
	})
	return result
}

// Dispatch executes an adjudicated tool call, enforcing scoping, conditions, JIT auth paging, and telemetry.
func (r *ToolRegistry) Dispatch(ctx ExecutionContext, inv Invocation) (Result, ToolTelemetry, error) {
	start := time.Now()
	toolName := inv.Tool
	inputBytes := len(inv.Arguments)

	telem := ToolTelemetry{
		SessionID:  ctx.SessionID(),
		TurnID:     ctx.TurnID(),
		Tool:       toolName,
		InputBytes: inputBytes,
		Verdict:    "ALLOW",
	}

	r.mu.RLock()
	tool, ok := r.tools[toolName]
	r.mu.RUnlock()

	emit := func(res Result, err error) (Result, ToolTelemetry, error) {
		telem.Duration = time.Since(start)
		if err != nil {
			telem.Error = err.Error()
		}
		telem.OutputBytes = len(res.Content)
		if r.telemetrySub != nil {
			r.telemetrySub(telem)
		}
		return res, telem, err
	}

	if !ok {
		telem.Verdict = "DENY"
		telem.Reason = "TOOL_NOT_FOUND"
		return emit(Result{}, &Error{Code: CodeInvalid, Op: "dispatch", Err: fmt.Errorf("tool %q not registered", toolName)})
	}

	telem.Integration = tool.Integration
	telem.Mutability = tool.Scope.Mutability
	if tool.Auth != nil {
		telem.AuthType = tool.Auth.Type
	}

	// 1. Condition Verification
	for _, cond := range tool.Conditions {
		if cond.Predicate != nil && !cond.Predicate(ctx) {
			telem.Verdict = "DENY"
			telem.Reason = "PRECONDITION_FAILED"
			desc := cond.Description
			if desc == "" {
				desc = string(cond.Kind)
			}
			return emit(Result{}, &Error{Code: CodeDenied, Op: "dispatch", Err: fmt.Errorf("precondition failed for %q: %s", toolName, desc)})
		}
	}

	// 2. Rate Limits
	r.mu.Lock()
	sessionKey := ctx.SessionID()
	turnKey := fmt.Sprintf("%s:%s", ctx.SessionID(), ctx.TurnID())

	if _, exists := r.callCounts[sessionKey]; !exists {
		r.callCounts[sessionKey] = make(map[string]int)
	}
	if _, exists := r.turnCounts[turnKey]; !exists {
		r.turnCounts[turnKey] = make(map[string]int)
	}

	sessionCalls := r.callCounts[sessionKey][toolName]
	turnCalls := r.turnCounts[turnKey][toolName]

	if tool.Scope.RateLimit.MaxPerTurn > 0 && turnCalls >= tool.Scope.RateLimit.MaxPerTurn {
		r.mu.Unlock()
		telem.Verdict = "DENY"
		telem.Reason = "TURN_RATE_LIMIT_EXCEEDED"
		return emit(Result{}, &Error{Code: CodeBackpressure, Op: "dispatch", Err: fmt.Errorf("rate limit exceeded: max %d calls per turn for %q", tool.Scope.RateLimit.MaxPerTurn, toolName)})
	}
	if tool.Scope.RateLimit.MaxPerSession > 0 && sessionCalls >= tool.Scope.RateLimit.MaxPerSession {
		r.mu.Unlock()
		telem.Verdict = "DENY"
		telem.Reason = "SESSION_RATE_LIMIT_EXCEEDED"
		return emit(Result{}, &Error{Code: CodeBackpressure, Op: "dispatch", Err: fmt.Errorf("quota exceeded: max %d calls per session for %q", tool.Scope.RateLimit.MaxPerSession, toolName)})
	}

	r.callCounts[sessionKey][toolName]++
	r.turnCounts[turnKey][toolName]++
	r.mu.Unlock()

	// 3. JIT Credential Paging Verification
	if tool.Auth != nil {
		switch tool.Auth.Type {
		case AuthTypeFleetSecret:
			key := tool.Auth.SecretKey
			if key == "" && len(tool.Auth.SecretRefs) > 0 {
				key = tool.Auth.SecretRefs[0]
			}
			secret, err := ctx.GetSecret(key)
			if err != nil || secret == "" {
				telem.Verdict = "DENY"
				telem.Reason = "AUTH_SECRET_MISSING"
				return emit(Result{}, &Error{Code: CodeDenied, Op: "auth_paging", Err: fmt.Errorf("required fleet secret %q could not be resolved", key)})
			}
			telem.AuthPaged = true
		case AuthTypeOAuth2:
			token, err := ctx.GetOAuthToken(tool.Auth.OAuthProvider)
			if err != nil || token == nil || !token.Valid() {
				telem.Verdict = "DENY"
				telem.Reason = "OAUTH_TOKEN_EXPIRED"
				return emit(Result{}, &Error{Code: CodeDenied, Op: "oauth_paging", Err: fmt.Errorf("valid OAuth token for provider %q unavailable", tool.Auth.OAuthProvider)})
			}
			telem.AuthPaged = true
		default:
			if tool.Auth.JITAuthPaging {
				for _, ref := range tool.Auth.SecretRefs {
					secret, err := ctx.GetSecret(ref)
					if err != nil || secret == "" {
						telem.Verdict = "DENY"
						telem.Reason = "AUTH_SECRET_MISSING"
						return emit(Result{}, &Error{Code: CodeDenied, Op: "auth_paging", Err: fmt.Errorf("required fleet secret %q could not be resolved", ref)})
					}
				}
				if tool.Auth.OAuthProvider != "" {
					token, err := ctx.GetOAuthToken(tool.Auth.OAuthProvider)
					if err != nil || token == nil || !token.Valid() {
						telem.Verdict = "DENY"
						telem.Reason = "OAUTH_TOKEN_EXPIRED"
						return emit(Result{}, &Error{Code: CodeDenied, Op: "oauth_paging", Err: fmt.Errorf("valid OAuth token for provider %q unavailable", tool.Auth.OAuthProvider)})
					}
				}
				telem.AuthPaged = true
			}
		}
	}

	if tool.Handler == nil {
		telem.Verdict = "DENY"
		telem.Reason = "HANDLER_UNIMPLEMENTED"
		return emit(Result{}, &Error{Code: CodeUnsupported, Op: "dispatch", Err: fmt.Errorf("no handler registered for %q", toolName)})
	}

	// 4. Execution with Timeout
	runCtx := ctx.Context()
	if tool.Scope.RateLimit.Timeout > 0 {
		var cancel context.CancelFunc
		runCtx, cancel = context.WithTimeout(runCtx, tool.Scope.RateLimit.Timeout)
		defer cancel()
	}

	wrappedCtx := &childExecutionContext{
		parent: ctx,
		ctx:    runCtx,
	}

	rawResult, err := tool.Handler(wrappedCtx, inv.Arguments)
	if err != nil {
		telem.Verdict = "DENY"
		telem.Reason = "EXECUTION_ERROR"
		return emit(Result{}, err)
	}

	// 5. Output Sanitization: scrub secret references, Bearer tokens, and sensitive headers
	sanitizedContent := tool.ScrubResult(rawResult.Content)
	return emit(Result{Content: sanitizedContent}, nil)
}

// PathWithinScope reports whether targetPath is contained within any of the allowed pathScopes.
func PathWithinScope(targetPath string, pathScopes []string) bool {
	if len(pathScopes) == 0 {
		return true
	}
	targetClean := filepath.Clean(targetPath)
	for _, scope := range pathScopes {
		scopeClean := filepath.Clean(scope)
		if targetClean == scopeClean {
			return true
		}
		rel, err := filepath.Rel(scopeClean, targetClean)
		if err == nil && !strings.HasPrefix(rel, "..") && rel != "." {
			return true
		}
		scopePrefix := scopeClean
		if !strings.HasSuffix(scopePrefix, string(filepath.Separator)) {
			scopePrefix += string(filepath.Separator)
		}
		if strings.HasPrefix(targetClean, scopePrefix) {
			return true
		}
	}
	return false
}

// scrubSensitiveBytes strips recognizable API keys and Authorization headers from tool returns.
func scrubSensitiveBytes(b []byte) []byte {
	if len(b) == 0 {
		return b
	}
	s := string(b)
	for _, prefix := range []string{"Bearer ", "bearer ", "token "} {
		if idx := strings.Index(s, prefix); idx != -1 {
			end := strings.IndexAny(s[idx+len(prefix):], " \r\n\",}")
			if end == -1 {
				s = s[:idx+len(prefix)] + "[REDACTED]"
			} else {
				s = s[:idx+len(prefix)] + "[REDACTED]" + s[idx+len(prefix)+end:]
			}
		}
	}
	return []byte(s)
}

// childExecutionContext embeds an ExecutionContext while overriding the context.Context.
type childExecutionContext struct {
	parent ExecutionContext
	ctx    context.Context
}

func (c *childExecutionContext) Context() context.Context { return c.ctx }
func (c *childExecutionContext) SessionID() string        { return c.parent.SessionID() }
func (c *childExecutionContext) TurnID() string           { return c.parent.TurnID() }
func (c *childExecutionContext) Phase() string            { return c.parent.Phase() }
func (c *childExecutionContext) Role() string             { return c.parent.Role() }
func (c *childExecutionContext) PriorSuccessfulTools() []string {
	return c.parent.PriorSuccessfulTools()
}
func (c *childExecutionContext) HasSucceeded(tool string) bool        { return c.parent.HasSucceeded(tool) }
func (c *childExecutionContext) GetSecret(key string) (string, error) { return c.parent.GetSecret(key) }
func (c *childExecutionContext) GetOAuthToken(p string) (*OAuthToken, error) {
	return c.parent.GetOAuthToken(p)
}

// ToolContract describes the machine-readable tool capability and governance contract.
type ToolContract struct {
	Version       string   `json:"version"`
	Mutabilities  []string `json:"mutabilities"`
	AuthTypes     []string `json:"auth_types"`
	Conditions    []string `json:"conditions"`
	Security      string   `json:"security_invariant"`
	Observability string   `json:"observability_standard"`
}

// PublicToolContract returns the normative tool contract metadata.
func PublicToolContract() ToolContract {
	return ToolContract{
		Version:       ToolContractVersion,
		Mutabilities:  []string{string(MutabilityReadOnly), string(MutabilityMutating), string(MutabilityDestructive)},
		AuthTypes:     []string{string(AuthTypeNone), string(AuthTypeFleetSecret), string(AuthTypeOAuth2)},
		Conditions:    []string{string(ConditionKindPrerequisite), string(ConditionKindPhase), string(ConditionKindRole), string(ConditionKindCustom)},
		Security:      "secrets are paged just-in-time and never enter model context or serialized tool parameters",
		Observability: "every invocation emits structured latency, mutability, auth-paging, and verdict telemetry",
	}
}
