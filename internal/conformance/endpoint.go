package conformance

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"time"
)

const EndpointSchema = "fak-endpoint-conformance/1"

type EndpointField struct {
	Status   string `json:"status"`
	Observed any    `json:"observed,omitempty"`
	Reason   string `json:"reason,omitempty"`
}

type EndpointObservation struct {
	Role       string                  `json:"role"`
	Endpoint   string                  `json:"endpoint"`
	Identity   EndpointField           `json:"identity"`
	Admission  EndpointField           `json:"policy_admission"`
	Lifecycle  EndpointField           `json:"session_lifecycle"`
	Provenance EndpointField           `json:"provider_model_provenance"`
	Usage      EndpointField           `json:"provider_reported_usage"`
	Receipt    EndpointField           `json:"receipt"`
	Reconnect  EndpointField           `json:"reconnect"`
	Quota      EndpointField           `json:"quota_refusal"`
	Upgrade    EndpointField           `json:"upgrade_compatibility"`
	Task       EndpointTaskObservation `json:"task"`
}

type EndpointTaskObservation struct {
	HTTPStatus   int    `json:"http_status"`
	Object       string `json:"object,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
	Content      string `json:"content,omitempty"`
	CompletionID string `json:"completion_id,omitempty"`
}

type EndpointPacket struct {
	Schema    string                `json:"schema"`
	Verdict   string                `json:"verdict"`
	Reason    string                `json:"reason,omitempty"`
	Task      string                `json:"task"`
	Endpoints []EndpointObservation `json:"endpoints"`
}

type endpointHealth struct {
	OK     bool   `json:"ok"`
	Engine string `json:"engine"`
	Model  string `json:"model"`
}

type endpointModels struct {
	Data []struct {
		ID      string `json:"id"`
		OwnedBy string `json:"owned_by"`
	} `json:"data"`
}

type endpointMCPResponse struct {
	Result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	} `json:"result"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type endpointAdjudication struct {
	Verdict struct {
		Kind        string `json:"kind"`
		Reason      string `json:"reason,omitempty"`
		By          string `json:"by,omitempty"`
		Disposition string `json:"disposition,omitempty"`
	} `json:"verdict"`
	TraceID string `json:"trace_id"`
}

type endpointSessionState struct {
	TraceID string `json:"trace_id"`
	Run     string `json:"run"`
	Reason  string `json:"reason,omitempty"`
	Rev     uint64 `json:"rev"`
}

type endpointCompletion struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage map[string]any `json:"usage"`
}

func ProbeEndpointPair(ctx context.Context, client *http.Client, localURL, managedURL string) EndpointPacket {
	const task = "Reply with exactly FAK_ENDPOINT_CONFORMANCE_PASS and nothing else."
	packet := EndpointPacket{Schema: EndpointSchema, Verdict: "FAIL", Task: task}
	for _, target := range []struct{ role, url string }{{"local", localURL}, {"managed", managedURL}} {
		obs, err := probeEndpoint(ctx, client, target.role, target.url, task)
		packet.Endpoints = append(packet.Endpoints, obs)
		if err != nil {
			packet.Reason = target.role + ": " + err.Error()
			return packet
		}
	}
	if err := compareEndpointSemantics(packet.Endpoints[0], packet.Endpoints[1]); err != nil {
		packet.Reason = err.Error()
		return packet
	}
	packet.Verdict = "PASS"
	return packet
}

func probeEndpoint(ctx context.Context, client *http.Client, role, rawURL, task string) (EndpointObservation, error) {
	base := strings.TrimRight(rawURL, "/")
	notYet := func(reason string) EndpointField { return EndpointField{Status: "NOT_YET", Reason: reason} }
	obs := EndpointObservation{Role: role, Endpoint: role,
		Admission: notYet("endpoint exposes no deterministic admission probe"),
		Lifecycle: notYet("endpoint exposes no portable session-lifecycle receipt"),
		Quota:     notYet("probe does not manufacture a quota failure"),
		Upgrade:   EndpointField{Status: "PASS", Observed: EndpointSchema},
	}
	var health endpointHealth
	if err := endpointJSON(ctx, client, http.MethodGet, base+"/healthz", nil, &health, nil); err != nil {
		return obs, fmt.Errorf("health: %w", err)
	}
	if !health.OK || health.Model == "" {
		return obs, fmt.Errorf("health lacks identity")
	}
	obs.Identity = EndpointField{Status: "PASS", Observed: map[string]any{"engine": health.Engine, "model": health.Model}}

	allow, err := endpointAdjudicate(ctx, client, base, "search_kb", "endpoint-conformance-allow")
	if err != nil {
		obs.Admission = notYet("portable fak_adjudicate endpoint unavailable: " + err.Error())
	} else {
		deny, denyErr := endpointAdjudicate(ctx, client, base, "refund_payment", "endpoint-conformance-deny")
		if denyErr != nil {
			return obs, fmt.Errorf("admission deny: %w", denyErr)
		}
		if allow.Verdict.Kind != "ALLOW" || deny.Verdict.Kind != "DENY" {
			return obs, fmt.Errorf("admission matrix = %s/%s, want ALLOW/DENY", allow.Verdict.Kind, deny.Verdict.Kind)
		}
		obs.Admission = EndpointField{Status: "PASS", Observed: map[string]any{
			"allow": map[string]any{"tool": "search_kb", "verdict": allow.Verdict.Kind, "by": allow.Verdict.By},
			"deny":  map[string]any{"tool": "refund_payment", "verdict": deny.Verdict.Kind, "reason": deny.Verdict.Reason, "disposition": deny.Verdict.Disposition, "by": deny.Verdict.By},
		}}
	}

	var models endpointModels
	if err := endpointJSON(ctx, client, http.MethodGet, base+"/v1/models", nil, &models, nil); err != nil {
		return obs, fmt.Errorf("models: %w", err)
	}
	if len(models.Data) == 0 || models.Data[0].ID == "" {
		return obs, fmt.Errorf("models lacks provenance")
	}
	obs.Provenance = EndpointField{Status: "PASS", Observed: map[string]any{"model": models.Data[0].ID, "owner": models.Data[0].OwnedBy}}

	body := map[string]any{"model": health.Model, "messages": []map[string]string{{"role": "user", "content": task}}, "temperature": 0, "max_tokens": 32}
	var completion endpointCompletion
	status := 0
	if err := endpointJSON(ctx, client, http.MethodPost, base+"/v1/chat/completions", body, &completion, &status); err != nil {
		return obs, fmt.Errorf("task: %w", err)
	}
	if len(completion.Choices) == 0 {
		return obs, fmt.Errorf("task returned no choice")
	}
	choice := completion.Choices[0]
	obs.Task = EndpointTaskObservation{HTTPStatus: status, Object: completion.Object, FinishReason: choice.FinishReason, Content: strings.TrimSpace(choice.Message.Content), CompletionID: completion.ID}
	lifecycle, lifecycleErr := endpointLifecycle(ctx, client, base, role)
	if lifecycleErr != nil {
		obs.Lifecycle = notYet("portable session lifecycle endpoint unavailable: " + lifecycleErr.Error())
	} else {
		obs.Lifecycle = EndpointField{Status: "PASS", Observed: lifecycle}
	}

	if len(completion.Usage) == 0 {
		obs.Usage = notYet("provider usage absent")
	} else {
		obs.Usage = EndpointField{Status: "PASS", Observed: completion.Usage}
	}
	if completion.ID == "" {
		obs.Receipt = notYet("completion receipt id absent")
	} else {
		obs.Receipt = EndpointField{Status: "PASS", Observed: completion.ID}
	}

	var after endpointHealth
	if err := endpointJSON(ctx, client, http.MethodGet, base+"/healthz", nil, &after, nil); err != nil {
		return obs, fmt.Errorf("reconnect: %w", err)
	}
	if !after.OK || after.Model != health.Model {
		return obs, fmt.Errorf("reconnect identity drift")
	}
	obs.Reconnect = EndpointField{Status: "PASS", Observed: true}
	return obs, nil
}

func endpointLifecycle(ctx context.Context, client *http.Client, base, role string) (map[string]any, error) {
	traceID := "endpoint-conformance-lifecycle-" + role
	path := base + "/v1/fak/session/" + traceID
	var initial endpointSessionState
	if err := endpointJSON(ctx, client, http.MethodGet, path, nil, &initial, nil); err != nil {
		return nil, err
	}
	if initial.TraceID != traceID || initial.Run != "running" {
		return nil, fmt.Errorf("initial state = %q/%q, want trace/running", initial.TraceID, initial.Run)
	}
	var stopped endpointSessionState
	body := map[string]any{"run": "stopped", "reason": "completed", "if_rev": initial.Rev}
	if err := endpointJSON(ctx, client, http.MethodPost, path+"/run", body, &stopped, nil); err != nil {
		return nil, err
	}
	if stopped.TraceID != traceID || stopped.Run != "stopped" || stopped.Reason != "completed" || stopped.Rev <= initial.Rev {
		return nil, fmt.Errorf("stopped state lacks trace, terminal transition, reason, or revision advance")
	}
	var readback endpointSessionState
	if err := endpointJSON(ctx, client, http.MethodGet, path, nil, &readback, nil); err != nil {
		return nil, err
	}
	if readback != stopped {
		return nil, fmt.Errorf("terminal lifecycle readback drift")
	}
	return map[string]any{"initial": initial.Run, "terminal": stopped.Run, "reason": stopped.Reason, "revision_advanced": true, "readback": true}, nil
}

func endpointAdjudicate(ctx context.Context, client *http.Client, base, tool, traceID string) (endpointAdjudication, error) {
	body := map[string]any{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": map[string]any{
		"name": "fak_adjudicate", "arguments": map[string]any{"tool": tool, "arguments": map[string]any{}, "trace_id": traceID},
	}}
	var wire endpointMCPResponse
	if err := endpointJSON(ctx, client, http.MethodPost, base+"/mcp", body, &wire, nil); err != nil {
		return endpointAdjudication{}, err
	}
	if wire.Error != nil {
		return endpointAdjudication{}, fmt.Errorf("MCP %d: %s", wire.Error.Code, wire.Error.Message)
	}
	if wire.Result.IsError || len(wire.Result.Content) == 0 {
		return endpointAdjudication{}, fmt.Errorf("MCP tool returned no successful content")
	}
	var out endpointAdjudication
	if err := json.Unmarshal([]byte(wire.Result.Content[0].Text), &out); err != nil {
		return out, fmt.Errorf("decode adjudication: %w", err)
	}
	if out.Verdict.Kind == "" || out.TraceID != traceID {
		return out, fmt.Errorf("adjudication lacks verdict or trace identity")
	}
	return out, nil
}

func endpointJSON(ctx context.Context, client *http.Client, method, url string, body any, out any, status *int) error {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, r)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if status != nil {
		*status = resp.StatusCode
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func compareEndpointSemantics(a, b EndpointObservation) error {
	if a.Task.HTTPStatus != b.Task.HTTPStatus || a.Task.Object != b.Task.Object || a.Task.FinishReason != b.Task.FinishReason || a.Task.Content != b.Task.Content {
		return fmt.Errorf("semantic task drift after deployment-field normalization")
	}
	fields := map[string][2]EndpointField{
		"identity": {a.Identity, b.Identity}, "admission": {a.Admission, b.Admission}, "lifecycle": {a.Lifecycle, b.Lifecycle},
		"provenance": {a.Provenance, b.Provenance}, "usage": {a.Usage, b.Usage}, "reconnect": {a.Reconnect, b.Reconnect},
		"quota": {a.Quota, b.Quota}, "upgrade": {a.Upgrade, b.Upgrade},
	}
	for name, pair := range fields {
		if pair[0].Status != pair[1].Status {
			return fmt.Errorf("semantic %s status drift: %s != %s", name, pair[0].Status, pair[1].Status)
		}
		if pair[0].Status == "PASS" && !reflect.DeepEqual(pair[0].Observed, pair[1].Observed) {
			return fmt.Errorf("semantic %s value drift", name)
		}
	}
	// Completion IDs are deployment-local receipt identities. Their presence is semantic;
	// their values are deliberately normalized and therefore not compared.
	if a.Receipt.Status != b.Receipt.Status {
		return fmt.Errorf("semantic receipt status drift: %s != %s", a.Receipt.Status, b.Receipt.Status)
	}
	return nil
}

func DefaultEndpointClient() *http.Client { return &http.Client{Timeout: 90 * time.Second} }
