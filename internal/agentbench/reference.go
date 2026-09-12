package agentbench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
)

const (
	referenceMinimumContext = 32768
	referenceResponseLimit  = 1 << 20
)

var referenceDigestPattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

type referenceRequest struct {
	Messages     []chatMessage     `json:"messages"`
	Tools        []json.RawMessage `json:"tools,omitempty"`
	OutputTokens int               `json:"max_tokens"`
}

type referenceEncoding struct {
	ModelID              string `json:"model_id"`
	RendererID           string `json:"renderer_id"`
	TokenizerID          string `json:"tokenizer_id"`
	TokenIDs             []int  `json:"token_ids"`
	PromptTokens         int    `json:"prompt_tokens"`
	ContextWindowTokens  int    `json:"context_window_tokens"`
	ReservedOutputTokens int    `json:"reserved_output_tokens"`
	RenderedSHA256       string `json:"rendered_sha256"`
}

type nativeReference struct {
	ModelID             string
	ContextWindowTokens int
	TokenizeURL         string

	client      *http.Client
	mu          sync.Mutex
	rendererID  string
	tokenizerID string
}

func discoverNativeReference(ctx context.Context, client *http.Client, endpoint, model string) (*nativeReference, error) {
	base, err := referenceBaseURL(endpoint)
	if err != nil {
		return nil, err
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return nil, errors.New("reference tokenizer requires an explicit model")
	}
	safeClient := referenceHTTPClient(client)
	modelsURL := base.ResolveReference(&url.URL{Path: "/v1/models"})
	var response struct {
		Data []struct {
			ID              string `json:"id"`
			ContextLength   int    `json:"context_length"`
			ContextWindow   int    `json:"context_window"`
			FakCapabilities struct {
				PromptTokenization struct {
					Endpoint string `json:"endpoint"`
				} `json:"prompt_tokenization"`
			} `json:"fak_capabilities"`
		} `json:"data"`
	}
	if err := referenceJSON(ctx, safeClient, http.MethodGet, modelsURL.String(), nil, &response); err != nil {
		return nil, fmt.Errorf("discover native reference: %w", err)
	}
	for _, row := range response.Data {
		if row.ID != model {
			continue
		}
		contextTokens := row.ContextLength
		if contextTokens == 0 {
			contextTokens = row.ContextWindow
		}
		if row.ContextLength > 0 && row.ContextWindow > 0 && row.ContextLength != row.ContextWindow {
			return nil, errors.New("native reference context metadata disagrees")
		}
		if contextTokens < referenceMinimumContext {
			return nil, fmt.Errorf("native reference model context %d is below %d", contextTokens, referenceMinimumContext)
		}
		tokenizeURL, err := sameOriginReferenceURL(base, row.FakCapabilities.PromptTokenization.Endpoint)
		if err != nil {
			return nil, err
		}
		return &nativeReference{ModelID: model, ContextWindowTokens: contextTokens, TokenizeURL: tokenizeURL, client: safeClient}, nil
	}
	return nil, fmt.Errorf("native reference model %q was not advertised", model)
}

func (r *nativeReference) Encode(ctx context.Context, req referenceRequest) (referenceEncoding, error) {
	if r == nil || r.client == nil || r.ModelID == "" || r.TokenizeURL == "" {
		return referenceEncoding{}, errors.New("native reference is unavailable")
	}
	if req.OutputTokens < 0 {
		return referenceEncoding{}, errors.New("reference output reservation cannot be negative")
	}
	wire := struct {
		Model     string            `json:"model"`
		Messages  []chatMessage     `json:"messages"`
		Tools     []json.RawMessage `json:"tools,omitempty"`
		MaxTokens int               `json:"max_tokens"`
		Stream    bool              `json:"stream"`
	}{r.ModelID, req.Messages, req.Tools, req.OutputTokens, false}
	body, err := json.Marshal(wire)
	if err != nil {
		return referenceEncoding{}, err
	}
	var encoded referenceEncoding
	if err := referenceJSON(ctx, r.client, http.MethodPost, r.TokenizeURL, body, &encoded); err != nil {
		return referenceEncoding{}, fmt.Errorf("encode native reference: %w", err)
	}
	if err := r.validateEncoding(encoded, req.OutputTokens); err != nil {
		return referenceEncoding{}, err
	}
	encoded.TokenIDs = append([]int(nil), encoded.TokenIDs...)
	return encoded, nil
}

func (r *nativeReference) validateEncoding(encoded referenceEncoding, outputTokens int) error {
	if encoded.ModelID != r.ModelID || encoded.ContextWindowTokens != r.ContextWindowTokens {
		return errors.New("native reference encoding metadata does not match discovery")
	}
	if encoded.RendererID == "" || encoded.TokenizerID == "" || !referenceDigestPattern.MatchString(encoded.RenderedSHA256) {
		return errors.New("native reference encoding identity is incomplete")
	}
	if encoded.PromptTokens <= 0 || encoded.PromptTokens != len(encoded.TokenIDs) || (outputTokens > 0 && encoded.ReservedOutputTokens != outputTokens) || encoded.ReservedOutputTokens < 0 {
		return errors.New("native reference token accounting is inconsistent")
	}
	for _, id := range encoded.TokenIDs {
		if id < 0 {
			return errors.New("native reference returned a negative token ID")
		}
	}
	reservation := outputTokens
	if reservation == 0 {
		reservation = encoded.ReservedOutputTokens
	}
	if encoded.PromptTokens+reservation > referenceMinimumContext {
		return errors.New("native reference prompt and output reservation exceed 32768 tokens")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.rendererID == "" {
		r.rendererID, r.tokenizerID = encoded.RendererID, encoded.TokenizerID
	} else if r.rendererID != encoded.RendererID || r.tokenizerID != encoded.TokenizerID {
		return errors.New("native reference renderer or tokenizer identity changed")
	}
	return nil
}

func referenceBaseURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" || parsed.User != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, errors.New("reference endpoint must be an http(s) origin")
	}
	parsed.Path, parsed.RawPath, parsed.RawQuery, parsed.Fragment = "", "", "", ""
	return parsed, nil
}

func sameOriginReferenceURL(base *url.URL, advertised string) (string, error) {
	capability, err := url.Parse(strings.TrimSpace(advertised))
	if err != nil || advertised == "" {
		return "", errors.New("native prompt tokenization capability is absent")
	}
	resolved := base.ResolveReference(capability)
	if resolved.Scheme != base.Scheme || resolved.Host != base.Host || resolved.User != nil || resolved.Path != "/v1/fak/tokenize" || resolved.RawQuery != "" || resolved.Fragment != "" {
		return "", errors.New("native prompt tokenization capability is not the bounded same-origin endpoint")
	}
	return resolved.String(), nil
}

func referenceHTTPClient(client *http.Client) *http.Client {
	if client == nil {
		client = http.DefaultClient
	}
	clone := *client
	clone.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	return &clone
}

func referenceJSON(ctx context.Context, client *http.Client, method, endpoint string, body []byte, out any) error {
	request, err := http.NewRequestWithContext(ctx, method, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Accept", "application/json")
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	limited := io.LimitReader(response.Body, referenceResponseLimit+1)
	payload, err := io.ReadAll(limited)
	if err != nil {
		return err
	}
	if len(payload) > referenceResponseLimit {
		return errors.New("native reference response exceeds limit")
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("native reference HTTP status %d", response.StatusCode)
	}
	if err := json.Unmarshal(payload, out); err != nil {
		return fmt.Errorf("decode native reference response: %w", err)
	}
	return nil
}
