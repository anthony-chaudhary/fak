package polymodel

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Provider identifies a supported external request vocabulary.
type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderGemini    Provider = "gemini"
)

// CanonicalRequest is the provider-neutral request boundary used by polymodel.
type CanonicalRequest struct {
	Model       string
	System      string
	Messages    []CanonicalMessage
	Temperature *float64
	MaxTokens   int
}

type CanonicalMessage struct {
	Role    string
	Content string
}

// NormalizeRequest translates a provider request into the canonical form. It
// fails closed on unsupported providers, roles, or non-text content rather than
// silently changing request semantics.
func NormalizeRequest(provider Provider, raw []byte) (CanonicalRequest, error) {
	switch provider {
	case ProviderOpenAI:
		return normalizeOpenAI(raw)
	case ProviderAnthropic:
		return normalizeAnthropic(raw)
	case ProviderGemini:
		return normalizeGemini(raw)
	default:
		return CanonicalRequest{}, fmt.Errorf("polymodel: unsupported provider %q", provider)
	}
}

func normalizeOpenAI(raw []byte) (CanonicalRequest, error) {
	var in struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
		Temperature *float64 `json:"temperature"`
		MaxTokens   int      `json:"max_tokens"`
	}
	if err := strictJSON(raw, &in); err != nil {
		return CanonicalRequest{}, err
	}
	out := CanonicalRequest{Model: in.Model, Temperature: in.Temperature, MaxTokens: in.MaxTokens}
	for _, m := range in.Messages {
		text, err := textContent(m.Content)
		if err != nil {
			return CanonicalRequest{}, err
		}
		if m.Role == "system" {
			if out.System != "" {
				out.System += "\n"
			}
			out.System += text
			continue
		}
		if m.Role != "user" && m.Role != "assistant" {
			return CanonicalRequest{}, fmt.Errorf("polymodel: unsupported OpenAI role %q", m.Role)
		}
		out.Messages = append(out.Messages, CanonicalMessage{Role: m.Role, Content: text})
	}
	return validateCanonical(out)
}

func normalizeAnthropic(raw []byte) (CanonicalRequest, error) {
	var in struct {
		Model    string          `json:"model"`
		System   json.RawMessage `json:"system"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
		Temperature *float64 `json:"temperature"`
		MaxTokens   int      `json:"max_tokens"`
	}
	if err := strictJSON(raw, &in); err != nil {
		return CanonicalRequest{}, err
	}
	out := CanonicalRequest{Model: in.Model, Temperature: in.Temperature, MaxTokens: in.MaxTokens}
	var err error
	if len(in.System) != 0 {
		out.System, err = textContent(in.System)
		if err != nil {
			return CanonicalRequest{}, err
		}
	}
	for _, m := range in.Messages {
		if m.Role != "user" && m.Role != "assistant" {
			return CanonicalRequest{}, fmt.Errorf("polymodel: unsupported Anthropic role %q", m.Role)
		}
		text, err := textContent(m.Content)
		if err != nil {
			return CanonicalRequest{}, err
		}
		out.Messages = append(out.Messages, CanonicalMessage{Role: m.Role, Content: text})
	}
	return validateCanonical(out)
}

func normalizeGemini(raw []byte) (CanonicalRequest, error) {
	type content struct {
		Role  string `json:"role"`
		Parts []struct {
			Text *string `json:"text"`
		} `json:"parts"`
	}
	var in struct {
		Model      string    `json:"model"`
		System     *content  `json:"system_instruction"`
		Contents   []content `json:"contents"`
		Generation *struct {
			Temperature *float64 `json:"temperature"`
			MaxTokens   int      `json:"max_output_tokens"`
		} `json:"generation_config"`
	}
	if err := strictJSON(raw, &in); err != nil {
		return CanonicalRequest{}, err
	}
	out := CanonicalRequest{Model: in.Model}
	if in.Generation != nil {
		out.Temperature = in.Generation.Temperature
		out.MaxTokens = in.Generation.MaxTokens
	}
	join := func(parts []struct {
		Text *string `json:"text"`
	}) (string, error) {
		var s []string
		for _, p := range parts {
			if p.Text == nil {
				return "", errors.New("polymodel: Gemini non-text part unsupported")
			}
			s = append(s, *p.Text)
		}
		return strings.Join(s, ""), nil
	}
	if in.System != nil {
		var err error
		out.System, err = join(in.System.Parts)
		if err != nil {
			return CanonicalRequest{}, err
		}
	}
	for _, m := range in.Contents {
		role := m.Role
		if role == "model" {
			role = "assistant"
		}
		if role != "user" && role != "assistant" {
			return CanonicalRequest{}, fmt.Errorf("polymodel: unsupported Gemini role %q", m.Role)
		}
		text, err := join(m.Parts)
		if err != nil {
			return CanonicalRequest{}, err
		}
		out.Messages = append(out.Messages, CanonicalMessage{Role: role, Content: text})
	}
	return validateCanonical(out)
}

func strictJSON(raw []byte, out any) error {
	d := json.NewDecoder(strings.NewReader(string(raw)))
	d.DisallowUnknownFields()
	if err := d.Decode(out); err != nil {
		return fmt.Errorf("polymodel: invalid provider request: %w", err)
	}
	var trailing any
	if err := d.Decode(&trailing); err == nil {
		return errors.New("polymodel: trailing JSON")
	}
	return nil
}
func textContent(raw json.RawMessage) (string, error) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) != nil {
		return "", errors.New("polymodel: non-text content unsupported")
	}
	var out []string
	for _, b := range blocks {
		if b.Type != "text" {
			return "", fmt.Errorf("polymodel: content type %q unsupported", b.Type)
		}
		out = append(out, b.Text)
	}
	return strings.Join(out, ""), nil
}
func validateCanonical(r CanonicalRequest) (CanonicalRequest, error) {
	if strings.TrimSpace(r.Model) == "" {
		return CanonicalRequest{}, errors.New("polymodel: model is required")
	}
	if len(r.Messages) == 0 {
		return CanonicalRequest{}, errors.New("polymodel: messages are required")
	}
	if r.MaxTokens < 0 {
		return CanonicalRequest{}, errors.New("polymodel: max tokens cannot be negative")
	}
	return r, nil
}
