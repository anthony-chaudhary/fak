package main

import (
	"reflect"
	"testing"
)

// mkResults builds an ordered probe-result slice over the real guardLocalBackends() list,
// marking which backends are live and what models each reports, so the precedence tests
// exercise the same backend table production uses.
func mkResults(live map[string]bool, models map[string][]string) []localProbeResult {
	bs := guardLocalBackends()
	out := make([]localProbeResult, 0, len(bs))
	for _, b := range bs {
		out = append(out, localProbeResult{backend: b, live: live[b.name], models: models[b.name]})
	}
	return out
}

func TestGuardChooseLocalBackendPrecedence(t *testing.T) {
	ollamaV1 := guardOpenAIV1Base("http://127.0.0.1:11434")
	lmstudioV1 := guardOpenAIV1Base("http://127.0.0.1:1234")
	llamaV1 := guardOpenAIV1Base("http://127.0.0.1:8080")

	cases := []struct {
		name      string
		live      map[string]bool
		models    map[string][]string
		wantBase  string
		wantModel string
		wantLabel string
		wantFound bool
	}{
		{
			name:      "nothing live",
			live:      map[string]bool{},
			wantFound: false,
		},
		{
			name:      "only llama.cpp live",
			live:      map[string]bool{"llama.cpp": true},
			models:    map[string][]string{"llama.cpp": {"local-model"}},
			wantBase:  llamaV1,
			wantModel: "local-model",
			wantLabel: "llama.cpp",
			wantFound: true,
		},
		{
			name:      "ollama wins over lm studio when both live",
			live:      map[string]bool{"Ollama": true, "LM Studio": true},
			models:    map[string][]string{"Ollama": {"llama3"}, "LM Studio": {"phi"}},
			wantBase:  ollamaV1,
			wantModel: "llama3",
			wantLabel: "Ollama",
			wantFound: true,
		},
		{
			name:      "skips dead ollama, picks lm studio",
			live:      map[string]bool{"Ollama": false, "LM Studio": true},
			models:    map[string][]string{"LM Studio": {"qwen2.5-coder:7b"}},
			wantBase:  lmstudioV1,
			wantModel: "qwen2.5-coder:7b",
			wantLabel: "LM Studio",
			wantFound: true,
		},
		{
			name:      "live but zero models -> chosen with empty model",
			live:      map[string]bool{"Ollama": true},
			models:    map[string][]string{"Ollama": nil},
			wantBase:  ollamaV1,
			wantModel: "",
			wantLabel: "Ollama",
			wantFound: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			base, model, label, found := guardChooseLocalBackend(mkResults(tc.live, tc.models))
			if found != tc.wantFound || base != tc.wantBase || model != tc.wantModel || label != tc.wantLabel {
				t.Fatalf("guardChooseLocalBackend = (%q, %q, %q, %v); want (%q, %q, %q, %v)",
					base, model, label, found, tc.wantBase, tc.wantModel, tc.wantLabel, tc.wantFound)
			}
		})
	}
}

func TestGuardPickLocalModel(t *testing.T) {
	cases := []struct {
		name   string
		models []string
		want   string
	}{
		{"empty", nil, ""},
		{"single", []string{"llama3"}, "llama3"},
		{"prefers coder over alphabetical first", []string{"zephyr", "qwen2.5-coder:7b", "alpha"}, "qwen2.5-coder:7b"},
		{"prefers code substring", []string{"mistral", "starcoder2"}, "starcoder2"},
		{"no coder -> sorted first", []string{"mistral", "llama3", "phi"}, "llama3"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := guardPickLocalModel(tc.models); got != tc.want {
				t.Errorf("guardPickLocalModel(%v) = %q, want %q", tc.models, got, tc.want)
			}
		})
	}
}

func TestParseOllamaTags(t *testing.T) {
	body := []byte(`{"models":[{"name":"qwen2.5-coder:7b"},{"name":"llama3:8b"},{"name":""}]}`)
	got := parseOllamaTags(body)
	want := []string{"qwen2.5-coder:7b", "llama3:8b"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseOllamaTags = %v, want %v", got, want)
	}
	if parseOllamaTags([]byte("not json")) != nil {
		t.Error("parseOllamaTags(garbage) should return nil")
	}
}

func TestParseOpenAIModels(t *testing.T) {
	body := []byte(`{"object":"list","data":[{"id":"qwen2.5-coder-3b","object":"model"},{"id":"  "},{"id":"phi-3"}]}`)
	got := parseOpenAIModels(body)
	want := []string{"qwen2.5-coder-3b", "phi-3"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseOpenAIModels = %v, want %v", got, want)
	}
	if parseOpenAIModels([]byte("{bad")) != nil {
		t.Error("parseOpenAIModels(garbage) should return nil")
	}
}

func TestGuardOllamaHostBase(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", ""},
		{"  ", ""},
		{"127.0.0.1:11434", "http://127.0.0.1:11434"},
		{"http://box:11434", "http://box:11434"},
		{"https://box:11434/", "https://box:11434"},
		{"remote-host:9999/", "http://remote-host:9999"},
	}
	for _, tc := range cases {
		if got := guardOllamaHostBase(tc.in); got != tc.want {
			t.Errorf("guardOllamaHostBase(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
