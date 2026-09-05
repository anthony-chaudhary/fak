package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

const (
	dispatchWorkerPreflightReady              = "READY"
	dispatchWorkerPreflightAuthInvalid        = "AUTH_INVALID"
	dispatchWorkerPreflightAuthMissing        = "AUTH_MISSING"
	dispatchWorkerPreflightAuthExpired        = "AUTH_EXPIRED"
	dispatchWorkerPreflightAuthMismatched     = "AUTH_MISMATCHED"
	dispatchWorkerPreflightGatewayRejected    = "GATEWAY_REJECTED"
	dispatchWorkerPreflightModelUnsupported   = "MODEL_UNSUPPORTED"
	dispatchWorkerPreflightQuotaExhausted     = "QUOTA_EXHAUSTED"
	dispatchWorkerPreflightTransientUpstream  = "TRANSIENT_UPSTREAM"
	dispatchWorkerPreflightRouteMisconfigured = "ROUTE_MISCONFIGURED"

	dispatchWorkerPreflightEvidenceTTL = 30 * time.Second
	dispatchWorkerPreflightBackoff     = 30 * time.Second
	dispatchWorkerPreflightTimeout     = 12 * time.Second
)

type dispatchWorkerPreflightRequest struct {
	Backend         string
	Account         dispatchtick.Account
	Model           string
	Workspace       string
	WorkKind        string
	DeadlineSeconds int
	Guarded         bool
	RouteDigest     string
	LaunchCommand   []string
}

type dispatchCodexPreflightObservation struct {
	Authenticated  bool
	AccountType    string
	Models         []string
	AuthError      string
	ModelError     string
	QuotaError     string
	RouteError     string
	QuotaExhausted bool
	RetryAt        time.Time
	GatewayVerdict string
}

type dispatchWorkerPreflightResult struct {
	Ready         bool
	Verdict       string
	Reason        string
	Model         string
	AccountType   string
	SeatToken     string
	Evidence      string
	CheckedAt     time.Time
	ExpiresAt     time.Time
	CooldownUntil time.Time

	backend         string
	accountTag      string
	accountDir      string
	workspace       string
	workKind        string
	deadlineSeconds int
	guarded         bool
	routeDigest     string
}

// AllowsStartup is admission, not a readiness claim. The caller must hold a lane
// lease and observe progress after the single guarded, deadline-bound launch.
func (r dispatchWorkerPreflightResult) AllowsStartup() bool {
	return r.Ready || (r.Verdict == dispatchWorkerPreflightTransientUpstream && r.guarded && r.deadlineSeconds > 0)
}

func (r dispatchWorkerPreflightResult) Map() map[string]any {
	out := map[string]any{
		"id":              "worker_identity",
		"evaluated":       true,
		"ok":              r.Ready,
		"startup_allowed": r.AllowsStartup(),
		"verdict":         r.Verdict,
		"reason":          r.Reason,
		"seat_token":      r.SeatToken,
		"model":           r.Model,
		"evidence":        r.Evidence,
		"checked_at":      r.CheckedAt.UTC().Format(time.RFC3339Nano),
		"expires_at":      r.ExpiresAt.UTC().Format(time.RFC3339Nano),
		"evidence_ttl":    int(dispatchWorkerPreflightEvidenceTTL / time.Second),
		"guarded":         r.guarded,
		"route_digest":    r.routeDigest,
		"work_kind":       r.workKind,
		"deadline_s":      r.deadlineSeconds,
		"workspace":       dispatchWorkerPreflightDigest("workspace", r.workspace),
	}
	if r.AccountType != "" {
		out["credential_class"] = r.AccountType
	}
	if !r.CooldownUntil.IsZero() {
		out["cooldown_until"] = r.CooldownUntil.UTC().Format(time.RFC3339Nano)
	}
	return out
}

func (r dispatchWorkerPreflightResult) Binds(req dispatchWorkerPreflightRequest, now time.Time) bool {
	return r.AllowsStartup() &&
		!now.After(r.ExpiresAt) &&
		r.backend == strings.TrimSpace(req.Backend) &&
		r.accountTag == strings.TrimSpace(req.Account.Tag) &&
		r.accountDir == dispatchPreflightCleanPath(req.Account.Dir) &&
		r.Model == strings.TrimSpace(req.Model) &&
		r.workspace == dispatchPreflightCleanPath(req.Workspace) &&
		r.workKind == strings.TrimSpace(req.WorkKind) &&
		r.deadlineSeconds == req.DeadlineSeconds &&
		r.guarded == req.Guarded &&
		r.routeDigest == strings.TrimSpace(req.RouteDigest)
}

func newDispatchWorkerPreflightRequest(root string, opts dispatchTickOptions, account dispatchtick.Account, launch dispatchtick.WorkerLaunch, launchCommand []string, guarded bool) dispatchWorkerPreflightRequest {
	return dispatchWorkerPreflightRequest{
		Backend:         strings.TrimSpace(opts.Backend),
		Account:         account,
		Model:           strings.TrimSpace(launch.Model),
		Workspace:       root,
		WorkKind:        strings.TrimSpace(opts.WorkKind),
		DeadlineSeconds: opts.WorkerTimeoutS,
		Guarded:         guarded,
		RouteDigest:     dispatchWorkerPreflightDigest(append([]string{"route"}, launchCommand...)...),
		LaunchCommand:   append([]string(nil), launchCommand...),
	}
}

var dispatchCodexWorkerPreflightProbe = runDispatchCodexAppServerPreflight

func dispatchWorkerPreflight(ctx context.Context, req dispatchWorkerPreflightRequest, now time.Time) dispatchWorkerPreflightResult {
	now = now.UTC()
	result := dispatchWorkerPreflightResult{
		Model:           strings.TrimSpace(req.Model),
		SeatToken:       dispatchWorkerPreflightDigest("seat", strings.TrimSpace(req.Account.Tag), dispatchPreflightCleanPath(req.Account.Dir)),
		CheckedAt:       now,
		ExpiresAt:       now.Add(dispatchWorkerPreflightEvidenceTTL),
		backend:         strings.TrimSpace(req.Backend),
		accountTag:      strings.TrimSpace(req.Account.Tag),
		accountDir:      dispatchPreflightCleanPath(req.Account.Dir),
		workspace:       dispatchPreflightCleanPath(req.Workspace),
		workKind:        strings.TrimSpace(req.WorkKind),
		deadlineSeconds: req.DeadlineSeconds,
		guarded:         req.Guarded,
		routeDigest:     strings.TrimSpace(req.RouteDigest),
	}
	if !req.Guarded {
		result.Verdict = dispatchWorkerPreflightRouteMisconfigured
		result.Reason = "Codex worker launch is not guard-fronted; enable FLEET_DOGFOOD_GUARD (subscription: `fak guard -- codex`; API/local route: configure FLEET_DOGFOOD_GUARD_BASEURL or use `fak guard --base-url <url> -- codex`)"
		return result.finishEvidence(nil)
	}
	if result.Model == "" {
		result.Verdict = dispatchWorkerPreflightRouteMisconfigured
		result.Reason = "Codex launch has no resolved model to preflight"
		return result.finishEvidence(nil)
	}
	if result.accountDir == "" {
		result.Verdict = dispatchWorkerPreflightAuthMissing
		result.Reason = "Codex launch has no account home to refresh"
		return result.finishEvidence(nil)
	}
	if req.Guarded {
		if baseURL := dispatchProviderURL(req.LaunchCommand); baseURL != "" {
			if _, err := dispatchDeepHealthURL(baseURL); err != nil {
				result.Verdict = dispatchWorkerPreflightRouteMisconfigured
				result.Reason = "Codex guard route is malformed"
				return result.finishEvidence(nil)
			}
		}
	}

	if req.Workspace != "" && (strings.EqualFold(req.Backend, "codex") || strings.EqualFold(req.Backend, "opencode")) {
		_, _ = projectassets.Ensure(req.Workspace, true)
	}

	obs, err := dispatchCodexWorkerPreflightProbe(ctx, req)
	result.AccountType = strings.TrimSpace(obs.AccountType)
	switch {
	case strings.TrimSpace(obs.RouteError) != "":
		result.Verdict = dispatchWorkerPreflightErrorVerdict(obs.RouteError)
		if result.Verdict == dispatchWorkerPreflightTransientUpstream {
			result.Verdict = dispatchWorkerPreflightRouteMisconfigured
		}
	case strings.TrimSpace(obs.AuthError) != "":
		result.Verdict = dispatchWorkerPreflightErrorVerdict(obs.AuthError)
		if result.Verdict != dispatchWorkerPreflightRouteMisconfigured &&
			result.Verdict != dispatchWorkerPreflightTransientUpstream &&
			result.Verdict != dispatchWorkerPreflightAuthMissing &&
			result.Verdict != dispatchWorkerPreflightAuthExpired &&
			result.Verdict != dispatchWorkerPreflightAuthMismatched &&
			result.Verdict != dispatchWorkerPreflightGatewayRejected {
			result.Verdict = dispatchWorkerPreflightAuthInvalid
		}
	case strings.TrimSpace(obs.GatewayVerdict) != "":
		result.Verdict = obs.GatewayVerdict
	case strings.TrimSpace(obs.ModelError) != "":
		result.Verdict = dispatchWorkerPreflightErrorVerdict(obs.ModelError)
	case strings.TrimSpace(obs.QuotaError) != "":
		result.Verdict = dispatchWorkerPreflightErrorVerdict(obs.QuotaError)
	case obs.QuotaExhausted:
		result.Verdict = dispatchWorkerPreflightQuotaExhausted
	case err != nil:
		result.Verdict = dispatchWorkerPreflightErrorVerdict(err.Error())
	case !obs.Authenticated:
		result.Verdict = dispatchWorkerPreflightAuthMissing
	case !dispatchPreflightModelAvailable(result.Model, obs.Models):
		result.Verdict = dispatchWorkerPreflightModelUnsupported
	default:
		result.Ready = true
		result.Verdict = dispatchWorkerPreflightReady
	}

	// A partial response can contain both a transport failure and an explicit
	// denial. No inconclusive rung may erase a denial from another RPC reply.
	if result.Verdict == dispatchWorkerPreflightTransientUpstream {
		if obs.QuotaExhausted {
			result.Verdict = dispatchWorkerPreflightQuotaExhausted
		}
		for _, text := range []string{obs.AuthError, obs.ModelError, obs.QuotaError} {
			if verdict := dispatchWorkerPreflightErrorVerdict(text); strings.TrimSpace(text) != "" && verdict != dispatchWorkerPreflightTransientUpstream {
				result.Verdict = verdict
				break
			}
		}
		if obs.GatewayVerdict != "" && obs.GatewayVerdict != dispatchWorkerPreflightTransientUpstream {
			result.Verdict = obs.GatewayVerdict
		}
	}

	switch result.Verdict {
	case dispatchWorkerPreflightReady:
		result.Reason = fmt.Sprintf("Codex seat is ready for model %q", result.Model)
	case dispatchWorkerPreflightAuthInvalid:
		result.Reason = "Codex credential refresh failed for the selected seat"
	case dispatchWorkerPreflightAuthMissing:
		seat := result.accountTag
		if seat == "" {
			seat = "default"
		}
		cleanHome := redactLocalPath(result.accountDir)
		result.Reason = fmt.Sprintf("Codex credential is missing for selected seat %q (source: %s); run `fak accounts enroll-current --harness codex` or `CODEX_HOME=%s fak m -- codex login`", seat, cleanHome, cleanHome)
	case dispatchWorkerPreflightAuthExpired:
		result.Reason = "Codex credential is expired for the selected seat"
	case dispatchWorkerPreflightAuthMismatched:
		result.Reason = "Codex credential does not match the selected seat"
	case dispatchWorkerPreflightGatewayRejected:
		result.Reason = "Codex gateway rejected the selected seat credential"
	case dispatchWorkerPreflightModelUnsupported:
		result.Reason = fmt.Sprintf("model %q is unavailable to credential class %q", result.Model, firstString(result.AccountType, "unknown"))
	case dispatchWorkerPreflightQuotaExhausted:
		result.Reason = "Codex account quota is exhausted"
		result.CooldownUntil = obs.RetryAt.UTC()
		if result.CooldownUntil.IsZero() {
			result.CooldownUntil = now.Add(5 * time.Minute)
		}
	case dispatchWorkerPreflightRouteMisconfigured:
		result.Reason = "Codex launch route is misconfigured"
	case dispatchWorkerPreflightTransientUpstream:
		result.Reason = "Codex preflight did not receive a stable upstream response"
		result.CooldownUntil = now.Add(dispatchWorkerPreflightBackoff)
	default:
		result.Reason = "Codex preflight explicitly refused startup"
	}
	return result.finishEvidence(&obs)
}

func (r dispatchWorkerPreflightResult) finishEvidence(obs *dispatchCodexPreflightObservation) dispatchWorkerPreflightResult {
	models := []string(nil)
	if obs != nil {
		models = append(models, obs.Models...)
		sort.Strings(models)
	}
	r.Evidence = dispatchWorkerPreflightDigest(
		"worker-preflight-v1",
		r.backend,
		r.SeatToken,
		r.Model,
		r.AccountType,
		r.workspace,
		r.workKind,
		strconv.Itoa(r.deadlineSeconds),
		strconv.FormatBool(r.guarded),
		r.routeDigest,
		r.Verdict,
		r.CheckedAt.Format(time.RFC3339Nano),
		r.CooldownUntil.Format(time.RFC3339Nano),
		strings.Join(models, "\x1f"),
	)
	return r
}

func dispatchWorkerPreflightErrorVerdict(text string) string {
	lower := strings.ToLower(strings.TrimSpace(text))
	switch {
	case strings.Contains(lower, "route misconfigured"):
		return dispatchWorkerPreflightRouteMisconfigured
	case strings.Contains(lower, "policy_block"), strings.Contains(lower, "policy denied"), strings.Contains(lower, "host_denied"), strings.Contains(lower, "403 forbidden"):
		return dispatchWorkerPreflightGatewayRejected
	case strings.Contains(lower, "credential missing"),
		strings.Contains(lower, "no codex login"),
		strings.Contains(lower, "not logged in"),
		strings.Contains(lower, "login required"):
		return dispatchWorkerPreflightAuthMissing
	case strings.Contains(lower, "credential expired"),
		strings.Contains(lower, "expired token"),
		strings.Contains(lower, "token expired"):
		return dispatchWorkerPreflightAuthExpired
	case strings.Contains(lower, "credential mismatched"),
		strings.Contains(lower, "different account home"),
		strings.Contains(lower, "credential provenance mismatch"):
		return dispatchWorkerPreflightAuthMismatched
	case strings.Contains(lower, "gateway rejected"),
		strings.Contains(lower, "invalid_refresh_token"),
		strings.Contains(lower, "invalid refresh token"),
		strings.Contains(lower, "invalid token"),
		strings.Contains(lower, "401 unauthorized"),
		strings.Contains(lower, "unauthorized"):
		return dispatchWorkerPreflightGatewayRejected
	case strings.Contains(lower, "model_provider"),
		strings.Contains(lower, "env_key"),
		strings.Contains(lower, "provider configuration"),
		strings.Contains(lower, "invalid configuration"),
		strings.Contains(lower, "config.toml"),
		strings.Contains(lower, "executable file not found"):
		return dispatchWorkerPreflightRouteMisconfigured
	case strings.Contains(lower, "unsupported model"),
		strings.Contains(lower, "model unsupported"),
		strings.Contains(lower, "model is not supported"),
		strings.Contains(lower, "model not found"):
		return dispatchWorkerPreflightModelUnsupported
	case strings.Contains(lower, "usage limit"),
		strings.Contains(lower, "rate limit reached"),
		strings.Contains(lower, "credits depleted"),
		strings.Contains(lower, "quota exhausted"),
		strings.Contains(lower, "quota exceeded"),
		strings.Contains(lower, "insufficient quota"),
		strings.Contains(lower, "spend control reached"):
		return dispatchWorkerPreflightQuotaExhausted
	case strings.Contains(lower, "refresh token"),
		strings.Contains(lower, "authentication failed"):
		return dispatchWorkerPreflightAuthInvalid
	default:
		return dispatchWorkerPreflightTransientUpstream
	}
}

func dispatchPreflightModelAvailable(model string, available []string) bool {
	model = strings.ToLower(strings.TrimSpace(model))
	for _, candidate := range available {
		if strings.ToLower(strings.TrimSpace(candidate)) == model {
			return true
		}
	}
	return false
}

func dispatchPreflightCleanPath(path string) string {
	if strings.TrimSpace(path) == "" {
		return ""
	}
	return filepath.Clean(path)
}

func dispatchWorkerPreflightDigest(parts ...string) string {
	h := sha256.New()
	for _, part := range parts {
		_, _ = io.WriteString(h, part)
		_, _ = h.Write([]byte{0})
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

type dispatchCodexRPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data"`
}

type dispatchCodexRPCMessage struct {
	ID     json.RawMessage        `json:"id"`
	Result json.RawMessage        `json:"result"`
	Error  *dispatchCodexRPCError `json:"error"`
}

func runDispatchCodexAppServerPreflight(ctx context.Context, req dispatchWorkerPreflightRequest) (dispatchCodexPreflightObservation, error) {
	exe := resolveDispatchWorkerExecutable("codex", "codex")
	cmd := exec.CommandContext(ctx, exe, "app-server", "--stdio")
	env := envMap(os.Environ())
	env["CODEX_HOME"] = req.Account.Dir
	delete(env, "CODEX_THREAD_ID")
	cmd.Env = envSliceFromMap(env)
	configureDispatchHelperCommand(cmd)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return dispatchCodexPreflightObservation{}, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return dispatchCodexPreflightObservation{}, err
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return dispatchCodexPreflightObservation{}, err
	}
	defer func() {
		if cmd.ProcessState == nil {
			_ = stdin.Close()
			if cmd.Process != nil {
				_ = cmd.Process.Kill()
			}
			_ = cmd.Wait()
		}
	}()
	enc := json.NewEncoder(stdin)
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 64*1024), 8*1024*1024)
	if err := enc.Encode(map[string]any{
		"id":     0,
		"method": "initialize",
		"params": map[string]any{
			"clientInfo":   map[string]string{"name": "fak-dispatch-preflight", "version": "1"},
			"capabilities": map[string]any{},
		},
	}); err != nil {
		return dispatchCodexPreflightObservation{}, err
	}
	initMsg, err := dispatchReadCodexRPCMessage(ctx, scanner, "0")
	if err != nil {
		return dispatchCodexPreflightObservation{}, err
	}
	if initMsg.Error != nil {
		return dispatchCodexPreflightObservation{RouteError: initMsg.Error.Message}, nil
	}
	var initialized struct {
		CodexHome string `json:"codexHome"`
	}
	if err := json.Unmarshal(initMsg.Result, &initialized); err != nil {
		return dispatchCodexPreflightObservation{RouteError: "invalid initialize response"}, nil
	}
	if dispatchPreflightCleanPath(initialized.CodexHome) != dispatchPreflightCleanPath(req.Account.Dir) {
		return dispatchCodexPreflightObservation{RouteError: "Codex app-server used a different account home"}, nil
	}
	if err := enc.Encode(map[string]any{"method": "initialized", "params": map[string]any{}}); err != nil {
		return dispatchCodexPreflightObservation{}, err
	}
	requests := []map[string]any{
		{"id": 1, "method": "account/read", "params": map[string]any{"refreshToken": true}},
		{"id": 2, "method": "model/list", "params": map[string]any{"limit": 1000, "includeHidden": true}},
		{"id": 3, "method": "account/rateLimits/read", "params": map[string]any{}},
	}
	for _, rpc := range requests {
		if err := enc.Encode(rpc); err != nil {
			return dispatchCodexPreflightObservation{}, err
		}
	}
	messages, err := dispatchReadCodexRPCMessages(ctx, scanner, []string{"1", "2", "3"})
	obs := dispatchCodexObservationFromRPC(messages)
	if err != nil {
		return obs, err
	}
	if err := stdin.Close(); err != nil {
		return obs, err
	}
	if err := cmd.Wait(); err != nil {
		if ctx.Err() != nil {
			return obs, ctx.Err()
		}
		return obs, fmt.Errorf("Codex app-server preflight exited: %w", err)
	}
	if obs.Authenticated && obs.AuthError == "" {
		obs.GatewayVerdict = dispatchCodexGatewayCredentialPreflight(ctx, req, time.Now().UTC())
	}
	return obs, nil
}

var dispatchCodexGatewayHTTPClient = &http.Client{Timeout: dispatchProviderProbeTimeout}

// dispatchCodexGatewayCredentialPreflight checks the same responses route and matched
// credential pair guard will proxy for the child. A syntactically invalid request is
// deliberate: any non-auth HTTP response proves the credential crossed the route without
// spending model quota.
func dispatchCodexGatewayCredentialPreflight(ctx context.Context, req dispatchWorkerPreflightRequest, now time.Time) string {
	cred, err := readCodexSubscriptionCredential(filepath.Join(req.Account.Dir, codexAuthFileName))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) || strings.Contains(strings.ToLower(err.Error()), "no codex login") || strings.Contains(strings.ToLower(err.Error()), "no codex chatgpt-subscription token") {
			return dispatchWorkerPreflightAuthMissing
		}
		return dispatchWorkerPreflightAuthInvalid
	}
	if exp, ok := dispatchJWTExpiry(cred.AccessToken); ok && !exp.After(now) {
		return dispatchWorkerPreflightAuthExpired
	}
	endpoint := dispatchProviderURL(req.LaunchCommand)
	if endpoint == "" {
		endpoint = guardCodexChatGPTBackendBaseURL
	}
	u, err := url.Parse(endpoint)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return dispatchWorkerPreflightRouteMisconfigured
	}
	u.Path = strings.TrimSuffix(u.Path, "/") + "/responses"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, u.String(), strings.NewReader(`{}`))
	if err != nil {
		return dispatchWorkerPreflightRouteMisconfigured
	}
	httpReq.Header.Set("Authorization", "Bearer "+cred.AccessToken)
	for key, value := range guardCodexSubscriptionHeaders(cred) {
		httpReq.Header.Set(key, value)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := dispatchCodexGatewayHTTPClient.Do(httpReq)
	if err != nil {
		return dispatchWorkerPreflightTransientUpstream
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusUnauthorized:
		return dispatchWorkerPreflightGatewayRejected
	case http.StatusForbidden:
		return dispatchWorkerPreflightAuthMismatched
	case http.StatusTooManyRequests:
		return dispatchWorkerPreflightQuotaExhausted
	case http.StatusNotFound:
		return dispatchWorkerPreflightRouteMisconfigured
	default:
		if resp.StatusCode >= 500 {
			return dispatchWorkerPreflightTransientUpstream
		}
		return ""
	}
}

func dispatchJWTExpiry(raw string) (time.Time, bool) {
	parts := strings.Split(raw, ".")
	if len(parts) != 3 {
		return time.Time{}, false
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return time.Time{}, false
	}
	var claims struct {
		Exp int64 `json:"exp"`
	}
	if json.Unmarshal(payload, &claims) != nil || claims.Exp <= 0 {
		return time.Time{}, false
	}
	return time.Unix(claims.Exp, 0).UTC(), true
}

func dispatchReadCodexRPCMessage(ctx context.Context, scanner *bufio.Scanner, wantID string) (dispatchCodexRPCMessage, error) {
	messages, err := dispatchReadCodexRPCMessages(ctx, scanner, []string{wantID})
	return messages[wantID], err
}

func dispatchReadCodexRPCMessages(ctx context.Context, scanner *bufio.Scanner, wantIDs []string) (map[string]dispatchCodexRPCMessage, error) {
	wanted := make(map[string]bool, len(wantIDs))
	for _, id := range wantIDs {
		wanted[id] = true
	}
	found := make(map[string]dispatchCodexRPCMessage, len(wantIDs))
	for len(found) < len(wanted) && scanner.Scan() {
		var msg dispatchCodexRPCMessage
		if err := json.Unmarshal(scanner.Bytes(), &msg); err != nil {
			continue
		}
		id := strings.Trim(string(msg.ID), `"`)
		if wanted[id] {
			found[id] = msg
		}
	}
	if len(found) == len(wanted) {
		return found, nil
	}
	if err := scanner.Err(); err != nil {
		return found, err
	}
	if err := ctx.Err(); err != nil {
		return found, err
	}
	return found, errors.New("Codex app-server ended before preflight responses arrived")
}

func dispatchCodexObservationFromRPC(messages map[string]dispatchCodexRPCMessage) dispatchCodexPreflightObservation {
	var obs dispatchCodexPreflightObservation
	accountMsg := messages["1"]
	if accountMsg.Error != nil {
		obs.AuthError = accountMsg.Error.Message
	} else if len(accountMsg.Result) > 0 {
		var result struct {
			Account *struct {
				Type string `json:"type"`
			} `json:"account"`
			RequiresOpenAIAuth bool `json:"requiresOpenaiAuth"`
		}
		if err := json.Unmarshal(accountMsg.Result, &result); err != nil {
			obs.AuthError = "invalid account/read response"
		} else {
			obs.Authenticated = result.Account != nil
			if result.Account != nil {
				obs.AccountType = result.Account.Type
			}
			if result.RequiresOpenAIAuth && result.Account == nil {
				obs.AuthError = "not logged in"
			}
		}
	}

	modelMsg := messages["2"]
	if modelMsg.Error != nil {
		obs.ModelError = modelMsg.Error.Message
	} else if len(modelMsg.Result) > 0 {
		var result struct {
			Data []struct {
				ID    string `json:"id"`
				Model string `json:"model"`
			} `json:"data"`
		}
		if err := json.Unmarshal(modelMsg.Result, &result); err != nil {
			obs.ModelError = "invalid model/list response"
		} else {
			for _, model := range result.Data {
				obs.Models = append(obs.Models, model.ID, model.Model)
			}
		}
	}

	quotaMsg := messages["3"]
	if quotaMsg.Error != nil {
		obs.QuotaError = quotaMsg.Error.Message
	} else if len(quotaMsg.Result) > 0 {
		obs.QuotaExhausted, obs.RetryAt, obs.QuotaError = dispatchCodexQuotaFromResult(quotaMsg.Result)
	}
	return obs
}

type dispatchCodexRateLimitWindow struct {
	UsedPercent int    `json:"usedPercent"`
	ResetsAt    *int64 `json:"resetsAt"`
}

type dispatchCodexSpendLimit struct {
	RemainingPercent int   `json:"remainingPercent"`
	ResetsAt         int64 `json:"resetsAt"`
}

type dispatchCodexRateLimitSnapshot struct {
	Primary              *dispatchCodexRateLimitWindow `json:"primary"`
	Secondary            *dispatchCodexRateLimitWindow `json:"secondary"`
	IndividualLimit      *dispatchCodexSpendLimit      `json:"individualLimit"`
	RateLimitReachedType *string                       `json:"rateLimitReachedType"`
	SpendControlReached  *bool                         `json:"spendControlReached"`
}

func dispatchCodexQuotaFromResult(raw json.RawMessage) (bool, time.Time, string) {
	var result struct {
		RateLimits          dispatchCodexRateLimitSnapshot            `json:"rateLimits"`
		RateLimitsByLimitID map[string]dispatchCodexRateLimitSnapshot `json:"rateLimitsByLimitId"`
	}
	if err := json.Unmarshal(raw, &result); err != nil {
		return false, time.Time{}, "invalid account/rateLimits/read response"
	}
	snapshots := []dispatchCodexRateLimitSnapshot{result.RateLimits}
	for _, snapshot := range result.RateLimitsByLimitID {
		snapshots = append(snapshots, snapshot)
	}
	exhausted := false
	var retryAt time.Time
	considerReset := func(unix *int64) {
		if unix == nil || *unix <= 0 {
			return
		}
		at := time.Unix(*unix, 0).UTC()
		if retryAt.IsZero() || at.Before(retryAt) {
			retryAt = at
		}
	}
	for _, snapshot := range snapshots {
		reached := snapshot.RateLimitReachedType != nil && strings.TrimSpace(*snapshot.RateLimitReachedType) != ""
		spendReached := snapshot.SpendControlReached != nil && *snapshot.SpendControlReached
		primaryReached := snapshot.Primary != nil && snapshot.Primary.UsedPercent >= 100
		secondaryReached := snapshot.Secondary != nil && snapshot.Secondary.UsedPercent >= 100
		individualReached := snapshot.IndividualLimit != nil && snapshot.IndividualLimit.RemainingPercent <= 0
		if !(reached || spendReached || primaryReached || secondaryReached || individualReached) {
			continue
		}
		exhausted = true
		if snapshot.Primary != nil {
			considerReset(snapshot.Primary.ResetsAt)
		}
		if snapshot.Secondary != nil {
			considerReset(snapshot.Secondary.ResetsAt)
		}
		if snapshot.IndividualLimit != nil {
			considerReset(&snapshot.IndividualLimit.ResetsAt)
		}
	}
	return exhausted, retryAt, ""
}

// Only successful command execution is progress; banners, a thread ID, and a
// surviving PID do not establish that the worker can actually perform work.
var dispatchObserveStartup = dispatchObserveCodexStartup

func dispatchObserveCodexStartup(logPath string, budget time.Duration) bool {
	deadline := time.Now().Add(budget)
	for {
		if dispatchCodexLogHasProgress(logPath) {
			return true
		}
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return false
		}
		time.Sleep(min(100*time.Millisecond, remaining))
	}
}

func dispatchCodexLogHasProgress(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	scanner := bufio.NewScanner(io.LimitReader(f, 4<<20))
	scanner.Buffer(make([]byte, 4096), 1<<20)
	for scanner.Scan() {
		var event struct {
			Type string `json:"type"`
			Item struct {
				Type     string `json:"type"`
				Status   string `json:"status"`
				ExitCode *int   `json:"exit_code"`
			} `json:"item"`
		}
		if json.Unmarshal(scanner.Bytes(), &event) == nil && event.Type == "item.completed" && event.Item.Type == "command_execution" && event.Item.Status == "completed" && event.Item.ExitCode != nil && *event.Item.ExitCode == 0 {
			return true
		}
	}
	return false
}

func dispatchTransientProviderCheck(check map[string]any) bool {
	reason := dispatchMapString(check, "reason")
	if dispatchWorkerPreflightErrorVerdict(reason) != dispatchWorkerPreflightTransientUpstream {
		return false
	}
	for _, key := range []string{"status", "upstream_status"} {
		if status, ok := check[key].(int); ok && status >= 400 && status < 500 {
			return false
		}
	}
	// Do not reinterpret arbitrary explicit health denials as uncertainty.
	lower := strings.ToLower(reason)
	return strings.Contains(lower, "timeout") || strings.Contains(lower, "connection") || strings.Contains(lower, "http 50") || strings.Contains(lower, "malformed json") || strings.Contains(lower, "did not evaluate") || strings.Contains(lower, "temporar")
}
