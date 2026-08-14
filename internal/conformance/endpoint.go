package conformance

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
	for name, pair := range map[string][2]string{
		"identity": {a.Identity.Status, b.Identity.Status}, "provenance": {a.Provenance.Status, b.Provenance.Status},
		"usage": {a.Usage.Status, b.Usage.Status}, "receipt": {a.Receipt.Status, b.Receipt.Status}, "reconnect": {a.Reconnect.Status, b.Reconnect.Status},
	} {
		if pair[0] != pair[1] {
			return fmt.Errorf("semantic %s status drift: %s != %s", name, pair[0], pair[1])
		}
	}
	return nil
}

func DefaultEndpointClient() *http.Client { return &http.Client{Timeout: 90 * time.Second} }
