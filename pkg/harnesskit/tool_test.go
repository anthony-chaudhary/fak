package harnesskit

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
	"time"
)

// mockExecutionContext implements ExecutionContext for testing.
type mockExecutionContext struct {
	ctx         context.Context
	sessionID   string
	turnID      string
	phase       string
	role        string
	priorTools  []string
	secrets     map[string]string
	oauthTokens map[string]*OAuthToken
}

func (m *mockExecutionContext) Context() context.Context       { return m.ctx }
func (m *mockExecutionContext) SessionID() string              { return m.sessionID }
func (m *mockExecutionContext) TurnID() string                 { return m.turnID }
func (m *mockExecutionContext) Phase() string                  { return m.phase }
func (m *mockExecutionContext) Role() string                   { return m.role }
func (m *mockExecutionContext) PriorSuccessfulTools() []string { return m.priorTools }
func (m *mockExecutionContext) HasSucceeded(tool string) bool {
	return slices.Contains(m.priorTools, tool)
}
func (m *mockExecutionContext) GetSecret(key string) (string, error) {
	if val, ok := m.secrets[key]; ok {
		return val, nil
	}
	return "", errors.New("secret not found")
}
func (m *mockExecutionContext) GetOAuthToken(provider string) (*OAuthToken, error) {
	if tok, ok := m.oauthTokens[provider]; ok {
		return tok, nil
	}
	return nil, errors.New("oauth token not found")
}

func TestToolDefinitionAndRegistration(t *testing.T) {
	registry := NewToolRegistry()

	schema := json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`)
	tool := NewTool("fs.read_file").
		WithIntegration("filesystem").
		WithDescription("Reads a file safely").
		WithParameters(schema).
		WithScope(ToolScope{
			Mutability: MutabilityReadOnly,
			PathScopes: []string{"/workspace/project"},
		}).
		WithHandler(func(ctx ExecutionContext, args json.RawMessage) (Result, error) {
			return Result{Content: json.RawMessage(`{"content":"hello world"}`)}, nil
		})

	if err := registry.Register(*tool); err != nil {
		t.Fatalf("Register failed: %v", err)
	}

	// Conflict registration
	if err := registry.Register(*tool); err == nil {
		t.Fatalf("Expected conflict error for duplicate registration, got nil")
	}

	// Retrieval
	def, ok := registry.Get("fs.read_file")
	if !ok {
		t.Fatalf("Get returned false for registered tool")
	}
	if def.Integration != "filesystem" || def.Scope.Mutability != MutabilityReadOnly {
		t.Errorf("Unexpected tool def: %+v", def)
	}

	// List
	list := registry.List()
	if len(list) != 1 || list[0].Name != "fs.read_file" {
		t.Errorf("Unexpected list result: %+v", list)
	}
}

func TestIntegrationDefinitionBundling(t *testing.T) {
	registry := NewToolRegistry()

	integration := NewIntegration("github", "GitHub integration suite").
		WithDefaultAuth(AuthRequirement{
			Type:          AuthTypeOAuth2,
			OAuthProvider: "github",
			OAuthScopes:   []string{"repo"},
		}).
		WithTool(NewTool("issues.list").WithDescription("Lists issues")).
		WithTool(NewTool("pulls.create").WithDescription("Creates a PR").WithScope(ToolScope{Mutability: MutabilityMutating}))

	if err := registry.RegisterIntegration(*integration); err != nil {
		t.Fatalf("RegisterIntegration failed: %v", err)
	}

	issueTool, ok := registry.Get("issues.list")
	if !ok {
		t.Fatalf("issues.list not registered")
	}
	if issueTool.Integration != "github" {
		t.Errorf("Integration not inherited: %s", issueTool.Integration)
	}
	if issueTool.Auth.Type != AuthTypeOAuth2 || issueTool.Auth.OAuthProvider != "github" {
		t.Errorf("Auth requirement not inherited: %+v", issueTool.Auth)
	}

	prTool, ok := registry.Get("pulls.create")
	if !ok {
		t.Fatalf("pulls.create not registered")
	}
	if prTool.Scope.Mutability != MutabilityMutating {
		t.Errorf("Scope not preserved: %+v", prTool.Scope)
	}
}

func TestToolFilteringAllowBlockAndMasking(t *testing.T) {
	registry := NewToolRegistry()

	_ = registry.Register(*NewTool("github.read_issue").WithDescription("read"))
	_ = registry.Register(*NewTool("github.create_pr").WithDescription("write"))
	_ = registry.Register(*NewTool("shell.exec").WithDescription("exec"))
	_ = registry.Register(*NewTool("db.delete").WithDescription("delete"))

	mockCtx := &mockExecutionContext{ctx: context.Background(), sessionID: "s1", turnID: "t1"}

	// 1. Blocklist with masking
	policy := ToolFilterPolicy{
		BlockList:   []string{"shell.*", "*.delete"},
		MaskBlocked: true,
	}
	tools := registry.FilterForTurn(mockCtx, policy)
	if len(tools) != 2 {
		t.Fatalf("Expected 2 tools, got %d", len(tools))
	}
	for _, tool := range tools {
		if strings.HasPrefix(tool.Name, "shell.") || strings.HasSuffix(tool.Name, ".delete") {
			t.Errorf("Blocked tool leaked into active turn: %s", tool.Name)
		}
	}

	// 2. Allowlist
	policyAllow := ToolFilterPolicy{
		AllowList:   []string{"github.*"},
		MaskBlocked: true,
	}
	toolsAllow := registry.FilterForTurn(mockCtx, policyAllow)
	if len(toolsAllow) != 2 {
		t.Fatalf("Expected 2 tools under allowlist, got %d", len(toolsAllow))
	}

	// 3. Max active tools cap
	policyCapped := ToolFilterPolicy{
		MaxActiveTools: 1,
		MaskBlocked:    true,
	}
	toolsCapped := registry.FilterForTurn(mockCtx, policyCapped)
	if len(toolsCapped) != 1 {
		t.Fatalf("Expected 1 tool under cap, got %d", len(toolsCapped))
	}
}

func TestToolConditionsAndPreconditions(t *testing.T) {
	registry := NewToolRegistry()

	deployTool := NewTool("cloud.deploy").
		WithDescription("Deploy service to production").
		WithCondition(RequirePhase("deploy")).
		WithCondition(RequirePriorSuccess("test.run")).
		WithCondition(RequireRole("admin")).
		WithHandler(func(ctx ExecutionContext, args json.RawMessage) (Result, error) {
			return Result{Content: json.RawMessage(`{"status":"deployed"}`)}, nil
		})

	_ = registry.Register(*deployTool)

	// Context with unmet conditions (phase is "plan", test.run not run, role is "guest")
	unmetCtx := &mockExecutionContext{
		ctx:        context.Background(),
		sessionID:  "s1",
		turnID:     "t1",
		phase:      "plan",
		role:       "guest",
		priorTools: []string{},
	}

	// Masking: deploy tool should be omitted from turn
	filtered := registry.FilterForTurn(unmetCtx, ToolFilterPolicy{MaskBlocked: true})
	if len(filtered) != 0 {
		t.Fatalf("Expected deploy tool to be masked when conditions unmet, got %d", len(filtered))
	}

	// Direct dispatch without meeting conditions: must return DENY with PRECONDITION_FAILED
	_, telem, err := registry.Dispatch(unmetCtx, Invocation{Tool: "cloud.deploy", Arguments: json.RawMessage(`{}`)})
	if err == nil {
		t.Fatalf("Expected error dispatching tool with unmet preconditions")
	}
	if telem.Verdict != "DENY" || telem.Reason != "PRECONDITION_FAILED" {
		t.Errorf("Unexpected telemetry: %+v", telem)
	}

	// Satisfied context
	metCtx := &mockExecutionContext{
		ctx:        context.Background(),
		sessionID:  "s1",
		turnID:     "t2",
		phase:      "deploy",
		role:       "admin",
		priorTools: []string{"test.run"},
	}

	filteredMet := registry.FilterForTurn(metCtx, ToolFilterPolicy{MaskBlocked: true})
	if len(filteredMet) != 1 {
		t.Fatalf("Expected deploy tool to be visible when conditions met, got %d", len(filteredMet))
	}

	res, telemMet, err := registry.Dispatch(metCtx, Invocation{Tool: "cloud.deploy", Arguments: json.RawMessage(`{}`)})
	if err != nil {
		t.Fatalf("Dispatch failed on satisfied context: %v", err)
	}
	if telemMet.Verdict != "ALLOW" {
		t.Errorf("Expected ALLOW verdict, got: %+v", telemMet)
	}
	if !strings.Contains(string(res.Content), "deployed") {
		t.Errorf("Unexpected result: %s", string(res.Content))
	}
}

func TestToolRateLimitsPerTurnAndSession(t *testing.T) {
	registry := NewToolRegistry()

	limitedTool := NewTool("expensive.ai_call").
		WithScope(ToolScope{
			RateLimit: RateLimit{
				MaxPerTurn:    2,
				MaxPerSession: 3,
			},
		}).
		WithHandler(func(ctx ExecutionContext, args json.RawMessage) (Result, error) {
			return Result{Content: json.RawMessage(`{"cost":"$$"}`)}, nil
		})

	_ = registry.Register(*limitedTool)

	ctxTurn1 := &mockExecutionContext{ctx: context.Background(), sessionID: "sess-1", turnID: "turn-1"}

	// Call 1 in turn 1: OK
	if _, _, err := registry.Dispatch(ctxTurn1, Invocation{Tool: "expensive.ai_call"}); err != nil {
		t.Fatalf("Call 1 failed: %v", err)
	}
	// Call 2 in turn 1: OK
	if _, _, err := registry.Dispatch(ctxTurn1, Invocation{Tool: "expensive.ai_call"}); err != nil {
		t.Fatalf("Call 2 failed: %v", err)
	}
	// Call 3 in turn 1: Should fail turn rate limit
	_, telemTurnLimit, err := registry.Dispatch(ctxTurn1, Invocation{Tool: "expensive.ai_call"})
	if err == nil || telemTurnLimit.Reason != "TURN_RATE_LIMIT_EXCEEDED" {
		t.Fatalf("Expected turn rate limit exceeded, got telem=%+v, err=%v", telemTurnLimit, err)
	}

	// Call in turn 2 of same session:
	ctxTurn2 := &mockExecutionContext{ctx: context.Background(), sessionID: "sess-1", turnID: "turn-2"}
	// Call 3 total (session call 3): OK
	if _, _, err := registry.Dispatch(ctxTurn2, Invocation{Tool: "expensive.ai_call"}); err != nil {
		t.Fatalf("Call 3 failed: %v", err)
	}
	// Call 4 total (session call 4): Should fail session limit (max 3)
	_, telemSessLimit, err := registry.Dispatch(ctxTurn2, Invocation{Tool: "expensive.ai_call"})
	if err == nil || telemSessLimit.Reason != "SESSION_RATE_LIMIT_EXCEEDED" {
		t.Fatalf("Expected session rate limit exceeded, got telem=%+v, err=%v", telemSessLimit, err)
	}
}

func TestJITFleetSecretPagingAndOutputScrubbing(t *testing.T) {
	registry := NewToolRegistry()

	secretTool := NewTool("stripe.create_charge").
		WithAuth(AuthRequirement{
			Type:      AuthTypeFleetSecret,
			SecretKey: "STRIPE_SECRET_KEY",
		}).
		WithHandler(func(ctx ExecutionContext, args json.RawMessage) (Result, error) {
			// Retrieve secret safely in handler
			sec, err := ctx.GetSecret("STRIPE_SECRET_KEY")
			if err != nil {
				return Result{}, err
			}
			if sec != "sk_live_supersecret_12345" {
				return Result{}, errors.New("incorrect secret resolved")
			}
			// Simulate a response that accidentally echoes back the Bearer authorization header
			return Result{Content: json.RawMessage(`{"status":"charged","debug_auth":"Bearer sk_live_supersecret_12345"}`)}, nil
		})

	_ = registry.Register(*secretTool)

	// 1. Missing secret in execution context
	missingSecretCtx := &mockExecutionContext{
		ctx:       context.Background(),
		sessionID: "s1",
		turnID:    "t1",
		secrets:   map[string]string{},
	}
	_, telemMissing, err := registry.Dispatch(missingSecretCtx, Invocation{Tool: "stripe.create_charge"})
	if err == nil || telemMissing.Reason != "AUTH_SECRET_MISSING" {
		t.Fatalf("Expected AUTH_SECRET_MISSING, got telem=%+v err=%v", telemMissing, err)
	}

	// 2. Present secret: executes, pages secret, and scrubs echo in result
	presentSecretCtx := &mockExecutionContext{
		ctx:       context.Background(),
		sessionID: "s1",
		turnID:    "t1",
		secrets: map[string]string{
			"STRIPE_SECRET_KEY": "sk_live_supersecret_12345",
		},
	}
	res, telemSuccess, err := registry.Dispatch(presentSecretCtx, Invocation{Tool: "stripe.create_charge"})
	if err != nil {
		t.Fatalf("Dispatch failed with present secret: %v", err)
	}
	if !telemSuccess.AuthPaged || telemSuccess.AuthType != AuthTypeFleetSecret {
		t.Errorf("Telemetry did not record JIT auth paging: %+v", telemSuccess)
	}

	// Invariant: Output must have scrubbed the secret!
	outputStr := string(res.Content)
	if strings.Contains(outputStr, "sk_live_supersecret_12345") {
		t.Fatalf("CRITICAL LEAK: Raw secret found in returned tool result: %s", outputStr)
	}
	if !strings.Contains(outputStr, "[REDACTED]") {
		t.Fatalf("Expected [REDACTED] in scrubbed tool output, got: %s", outputStr)
	}
}

func TestManagedOAuthTokenResolutionAndExpiration(t *testing.T) {
	registry := NewToolRegistry()

	oauthTool := NewTool("slack.send_message").
		WithAuth(AuthRequirement{
			Type:          AuthTypeOAuth2,
			OAuthProvider: "slack",
			OAuthScopes:   []string{"chat:write"},
		}).
		WithHandler(func(ctx ExecutionContext, args json.RawMessage) (Result, error) {
			token, err := ctx.GetOAuthToken("slack")
			if err != nil {
				return Result{}, err
			}
			return Result{Content: json.RawMessage(`{"posted":true,"token_type":"` + token.TokenType + `"}`)}, nil
		})

	_ = registry.Register(*oauthTool)

	// 1. Expired OAuth token
	expiredCtx := &mockExecutionContext{
		ctx:       context.Background(),
		sessionID: "s1",
		turnID:    "t1",
		oauthTokens: map[string]*OAuthToken{
			"slack": {
				AccessToken: "xoxp-expired-token",
				TokenType:   "Bearer",
				ExpiresAt:   time.Now().Add(-1 * time.Hour),
			},
		},
	}
	_, telemExpired, err := registry.Dispatch(expiredCtx, Invocation{Tool: "slack.send_message"})
	if err == nil || telemExpired.Reason != "OAUTH_TOKEN_EXPIRED" {
		t.Fatalf("Expected OAUTH_TOKEN_EXPIRED, got telem=%+v err=%v", telemExpired, err)
	}

	// 2. Valid OAuth token
	validCtx := &mockExecutionContext{
		ctx:       context.Background(),
		sessionID: "s1",
		turnID:    "t1",
		oauthTokens: map[string]*OAuthToken{
			"slack": {
				AccessToken: "xoxp-valid-live-token",
				TokenType:   "Bearer",
				ExpiresAt:   time.Now().Add(1 * time.Hour),
			},
		},
	}
	res, telemValid, err := registry.Dispatch(validCtx, Invocation{Tool: "slack.send_message"})
	if err != nil {
		t.Fatalf("Dispatch failed with valid OAuth token: %v", err)
	}
	if !telemValid.AuthPaged || telemValid.AuthType != AuthTypeOAuth2 {
		t.Errorf("Expected AuthPaged=true, got: %+v", telemValid)
	}
	if !strings.Contains(string(res.Content), "posted") {
		t.Errorf("Unexpected result: %s", string(res.Content))
	}
}

func TestToolTelemetrySubscriber(t *testing.T) {
	registry := NewToolRegistry()

	var recorded []ToolTelemetry
	registry.OnTelemetry(func(item ToolTelemetry) {
		recorded = append(recorded, item)
	})

	tool := NewTool("calc.add").
		WithIntegration("calculator").
		WithHandler(func(ctx ExecutionContext, args json.RawMessage) (Result, error) {
			return Result{Content: json.RawMessage(`{"result":42}`)}, nil
		})
	_ = registry.Register(*tool)

	ctx := &mockExecutionContext{ctx: context.Background(), sessionID: "sess-tel", turnID: "turn-tel"}
	_, _, err := registry.Dispatch(ctx, Invocation{Tool: "calc.add", Arguments: json.RawMessage(`{"a":20,"b":22}`)})
	if err != nil {
		t.Fatal(err)
	}

	if len(recorded) != 1 {
		t.Fatalf("Expected 1 telemetry record, got %d", len(recorded))
	}
	rec := recorded[0]
	if rec.SessionID != "sess-tel" || rec.TurnID != "turn-tel" || rec.Tool != "calc.add" || rec.Integration != "calculator" {
		t.Errorf("Telemetry fields mismatched: %+v", rec)
	}
	if rec.Verdict != "ALLOW" || rec.InputBytes == 0 || rec.OutputBytes == 0 {
		t.Errorf("Telemetry metrics incomplete: %+v", rec)
	}
}

func TestPathWithinScopeEnforcement(t *testing.T) {
	scopes := []string{"/workspace/repo", "/tmp/fak-cache"}

	// Within scope
	if !PathWithinScope("/workspace/repo/pkg/main.go", scopes) {
		t.Errorf("Expected /workspace/repo/pkg/main.go to be within scope")
	}
	if !PathWithinScope("/tmp/fak-cache/obj1", scopes) {
		t.Errorf("Expected /tmp/fak-cache/obj1 to be within scope")
	}

	// Escape attempts
	if PathWithinScope("/workspace/other_repo/file.txt", scopes) {
		t.Errorf("Expected /workspace/other_repo/file.txt to be rejected")
	}
	if PathWithinScope("/etc/passwd", scopes) {
		t.Errorf("Expected /etc/passwd to be rejected")
	}
	if PathWithinScope("/workspace/repo/../../etc/passwd", scopes) {
		t.Errorf("Expected traversal path to be rejected")
	}
}

func TestToolDefinitionValidation(t *testing.T) {
	// 1. Valid tool definition
	validTool := ToolDefinition{
		Name:        "fs.read_file",
		Description: "Reads a file safely within workspace bounds",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"path": map[string]any{"type": "string"},
			},
			"required": []any{"path"},
		},
		Scope: ToolScope{
			WorkspacePaths: []string{"/workspace/project"},
			ReadOnly:       true,
			NetworkAllowed: false,
			MaxTurns:       10,
		},
		Condition: ToolCondition{
			AllowList: []string{"sess-1"},
		},
		Auth: &AuthBinding{
			SecretRefs:              []string{"FLEET_TOKEN_1"},
			JITAuthPaging:           true,
			OAuthProvider:           "github",
			ScrubSecretsFromResults: true,
		},
		Metadata: map[string]string{
			"category": "filesystem",
		},
	}
	if err := validTool.Validate(); err != nil {
		t.Fatalf("Expected valid tool to pass Validate(), got err: %v", err)
	}

	// 2. Empty name rejection
	emptyNameTool := ToolDefinition{
		Name: "",
		Schema: map[string]any{
			"type": "object",
		},
	}
	if err := emptyNameTool.Validate(); err == nil {
		t.Fatalf("Expected empty name to fail Validate(), got nil")
	}

	// 3. Whitespace name rejection
	whitespaceTool := ToolDefinition{
		Name: "   ",
		Schema: map[string]any{
			"type": "object",
		},
	}
	if err := whitespaceTool.Validate(); err == nil {
		t.Fatalf("Expected whitespace name to fail Validate(), got nil")
	}

	// 4. Empty schema rejection
	emptySchemaTool := ToolDefinition{
		Name:   "fs.write",
		Schema: map[string]any{},
	}
	if err := emptySchemaTool.Validate(); err == nil {
		t.Fatalf("Expected empty schema map to fail Validate(), got nil")
	}

	// 5. Invalid schema type rejection
	invalidTypeTool := ToolDefinition{
		Name: "fs.write",
		Schema: map[string]any{
			"type": 12345,
		},
	}
	if err := invalidTypeTool.Validate(); err == nil {
		t.Fatalf("Expected non-string schema type to fail Validate(), got nil")
	}
}

func TestToolPermissionScopeChecks(t *testing.T) {
	// Read-only scope enforcement
	roTool := ToolDefinition{
		Name: "fs.read",
		Scope: ToolScope{
			WorkspacePaths: []string{"/workspace/project"},
			ReadOnly:       true,
		},
	}

	// Read operation on read-only tool is permitted
	allowed, reason := roTool.CheckPermission("sess-1", "/workspace/project/README.md", false)
	if !allowed {
		t.Fatalf("Expected read on read-only tool to be allowed, got reason: %s", reason)
	}

	// Write operation on read-only tool is refused
	allowed, reason = roTool.CheckPermission("sess-1", "/workspace/project/README.md", true)
	if allowed || !strings.Contains(reason, "read-only") {
		t.Fatalf("Expected write on read-only tool to be rejected, got allowed=%v reason=%s", allowed, reason)
	}

	// Mutating tool permits write
	mutTool := ToolDefinition{
		Name: "fs.write",
		Scope: ToolScope{
			WorkspacePaths: []string{"/workspace/project"},
			ReadOnly:       false,
			Mutability:     MutabilityMutating,
		},
	}
	allowed, reason = mutTool.CheckPermission("sess-1", "/workspace/project/main.go", true)
	if !allowed {
		t.Fatalf("Expected write on mutating tool to be allowed, got reason: %s", reason)
	}

	// Path prefix and containment matching
	scopedTool := ToolDefinition{
		Name: "fs.access",
		Scope: ToolScope{
			WorkspacePaths: []string{"/workspace/project", "/tmp/scratch"},
			ReadOnly:       true,
		},
	}

	// Paths within scope
	if ok, r := scopedTool.CheckPermission("sess-1", "/workspace/project/sub/file.go", false); !ok {
		t.Errorf("Expected subfile to be allowed, got %s", r)
	}
	if ok, r := scopedTool.CheckPermission("sess-1", "/tmp/scratch/temp.txt", false); !ok {
		t.Errorf("Expected /tmp/scratch path to be allowed, got %s", r)
	}

	// Paths outside scope
	if ok, r := scopedTool.CheckPermission("sess-1", "/etc/passwd", false); ok || !strings.Contains(r, "outside workspace scope") {
		t.Errorf("Expected /etc/passwd to be rejected, got ok=%v reason=%s", ok, r)
	}
	if ok, _ := scopedTool.CheckPermission("sess-1", "/workspace/other_project/main.go", false); ok {
		t.Errorf("Expected /workspace/other_project to be rejected")
	}

	// Directory traversal path escape
	if ok, _ := scopedTool.CheckPermission("sess-1", "/workspace/project/../../etc/shadow", false); ok {
		t.Errorf("Expected traversal path to be rejected")
	}
}

func TestToolConditionAllowBlockEvaluation(t *testing.T) {
	// AllowList evaluation
	allowTool := ToolDefinition{
		Name: "admin.run",
		Condition: ToolCondition{
			AllowList: []string{"session-prod-*", "admin-session"},
		},
	}
	if ok, _ := allowTool.CheckPermission("admin-session", "", false); !ok {
		t.Errorf("Expected admin-session in allow list to be permitted")
	}
	if ok, _ := allowTool.CheckPermission("session-prod-99", "", false); !ok {
		t.Errorf("Expected session-prod-* match to be permitted")
	}
	if ok, reason := allowTool.CheckPermission("session-dev-1", "", false); ok || !strings.Contains(reason, "not in allow list") {
		t.Errorf("Expected session-dev-1 to be rejected by allow list, got ok=%v reason=%s", ok, reason)
	}
	if ok, _ := allowTool.CheckPermission("", "", false); ok {
		t.Errorf("Expected empty session to be rejected when allow list is non-empty")
	}

	// BlockList evaluation
	blockTool := ToolDefinition{
		Name: "shell.run",
		Condition: ToolCondition{
			BlockList: []string{"untrusted-*", "blocked-session"},
		},
	}
	if ok, reason := blockTool.CheckPermission("untrusted-worker-1", "", false); ok || !strings.Contains(reason, "blocked") {
		t.Errorf("Expected untrusted worker to be blocked, got ok=%v reason=%s", ok, reason)
	}
	if ok, reason := blockTool.CheckPermission("blocked-session", "", false); ok || !strings.Contains(reason, "blocked") {
		t.Errorf("Expected blocked-session to be blocked, got ok=%v reason=%s", ok, reason)
	}
	if ok, _ := blockTool.CheckPermission("clean-session", "", false); !ok {
		t.Errorf("Expected clean-session not on block list to be permitted")
	}

	// Precedence: BlockList takes priority over AllowList
	priorityTool := ToolDefinition{
		Name: "data.export",
		Condition: ToolCondition{
			AllowList: []string{"worker-*"},
			BlockList: []string{"worker-quarantined"},
		},
	}
	if ok, _ := priorityTool.CheckPermission("worker-clean", "", false); !ok {
		t.Errorf("Expected worker-clean to be permitted")
	}
	if ok, _ := priorityTool.CheckPermission("worker-quarantined", "", false); ok {
		t.Errorf("Expected worker-quarantined to be blocked despite matching allow pattern")
	}
	if ok, _ := priorityTool.CheckPermission("external-agent", "", false); ok {
		t.Errorf("Expected external-agent to be rejected by allow list")
	}

	// Precondition evaluation
	precondTool := ToolDefinition{
		Name: "deploy.service",
		Condition: ToolCondition{
			Precondition: func(ctx context.Context, sessionID string) bool {
				return strings.HasPrefix(sessionID, "verified-")
			},
		},
	}
	if ok, _ := precondTool.CheckPermission("verified-session-123", "", false); !ok {
		t.Errorf("Expected verified session to pass precondition")
	}
	if ok, reason := precondTool.CheckPermission("unverified-session-123", "", false); ok || !strings.Contains(reason, "precondition failed") {
		t.Errorf("Expected unverified session to fail precondition, got ok=%v reason=%s", ok, reason)
	}
}

func TestToolSecretScrubbingWithJITAuth(t *testing.T) {
	tool := &ToolDefinition{
		Name: "payment.charge",
		Auth: &AuthBinding{
			SecretRefs:              []string{"sk_live_998877665544", "DB_PASSWORD_123"},
			JITAuthPaging:           true,
			OAuthProvider:           "stripe",
			ScrubSecretsFromResults: true,
		},
	}

	raw := []byte(`{"status":"ok","token":"sk_live_998877665544","db_pass":"DB_PASSWORD_123","auth":"Bearer sec_live_bearer_9999"}`)
	scrubbed := tool.ScrubResult(raw)
	scrubbedStr := string(scrubbed)

	if strings.Contains(scrubbedStr, "sk_live_998877665544") {
		t.Fatalf("Secret sk_live_998877665544 leaked in scrubbed output: %s", scrubbedStr)
	}
	if strings.Contains(scrubbedStr, "DB_PASSWORD_123") {
		t.Fatalf("Secret DB_PASSWORD_123 leaked in scrubbed output: %s", scrubbedStr)
	}
	if strings.Contains(scrubbedStr, "sec_live_bearer_9999") {
		t.Fatalf("Bearer token sec_live_bearer_9999 leaked in scrubbed output: %s", scrubbedStr)
	}
	if !strings.Contains(scrubbedStr, "[REDACTED]") {
		t.Fatalf("Expected [REDACTED] in scrubbed output: %s", scrubbedStr)
	}

	// When ScrubSecretsFromResults is false, SecretRefs are not redacted
	noScrubTool := &ToolDefinition{
		Name: "debug.echo",
		Auth: &AuthBinding{
			SecretRefs:              []string{"debug_token_xyz"},
			ScrubSecretsFromResults: false,
		},
	}
	rawNoScrub := []byte(`{"debug":"debug_token_xyz"}`)
	scrubbedNoScrub := noScrubTool.ScrubResult(rawNoScrub)
	if !strings.Contains(string(scrubbedNoScrub), "debug_token_xyz") {
		t.Fatalf("Expected debug_token_xyz to be preserved when ScrubSecretsFromResults=false")
	}
}
