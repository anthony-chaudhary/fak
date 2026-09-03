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
