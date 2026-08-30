package main

import (
	"flag"
	"reflect"
	"strings"
	"testing"
)

func TestGuardNativeControlsUseExplicitFlagsOverAmbientValues(t *testing.T) {
	for name, value := range map[string]string{
		"FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS": "4096",
		"FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE":     "0",
		"FAK_Q4K_GATEUP_SLAB":                        "0",
		"FAK_PREFIX_PROFILE":                         "ambient.jsonl",
		"FAK_VULKAN_Q4K_PROFILE":                     "0",
		"FAK_VULKAN_STAGE_Q4K":                       "0",
	} {
		t.Setenv(name, value)
	}
	fs := flag.NewFlagSet("guard-native", flag.ContinueOnError)
	flags := registerGuardNativeControlFlags(fs)
	if err := fs.Parse([]string{
		"--native-qwen-q4k-prefill-chunk-tokens=8192",
		"--native-qwen35-metal-gdn-sequence",
		"--native-q4k-gateup-slab",
		"--native-prefix-profile=guard.jsonl",
		"--vulkan-q4k-profile",
		"--vulkan-stage-q4k",
	}); err != nil {
		t.Fatal(err)
	}
	if err := validateNativeQwenQ4KPrefillChunk(*flags.prefillChunk); err != nil {
		t.Fatalf("guard rejected the established explicit 8192 contract: %v", err)
	}
	got := flags.config()
	if got.Planner.QwenQ4KPrefillChunkTokens != 8192 || !got.Planner.Qwen35MetalGDNSequence || !got.Planner.Q4KGateUpOutputSlab || got.PrefixProfile != "guard.jsonl" || !got.VulkanQ4KProfile || !got.VulkanStageQ4K {
		t.Fatalf("guard native config did not preserve explicit flags: %+v", got)
	}
}

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
	qwen36V1 := guardOpenAIV1Base("http://127.0.0.1:8131")
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
			name:      "qwen36 dogfood port is discoverable",
			live:      map[string]bool{"Qwen3.6 local": true},
			models:    map[string][]string{"Qwen3.6 local": {"lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"}},
			wantBase:  qwen36V1,
			wantModel: "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M",
			wantLabel: "Qwen3.6 local",
			wantFound: true,
		},
		{
			name:      "lm studio default port wins over qwen dogfood port",
			live:      map[string]bool{"LM Studio": true, "Qwen3.6 local": true},
			models:    map[string][]string{"LM Studio": {"local-default"}, "Qwen3.6 local": {"qwen3.6"}},
			wantBase:  lmstudioV1,
			wantModel: "local-default",
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

func TestGuardLocalNothingDetectedMessageMentionsQwen36Port(t *testing.T) {
	msg := guardLocalNothingDetectedMessage()
	for _, want := range []string{"Ollama 127.0.0.1:11434", "LM Studio 127.0.0.1:1234", "Qwen3.6 dogfood 127.0.0.1:8131", "llama.cpp 127.0.0.1:8080"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("message missing %q:\n%s", want, msg)
		}
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

func TestGuardLocalProviderExtraBodyForQwen36(t *testing.T) {
	for _, tc := range []struct {
		label string
		model string
	}{
		{label: "Qwen3.6 local", model: ""},
		{label: "LM Studio", model: "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"},
		{label: "llama.cpp", model: "qwen36-27b-q4_k_m"},
	} {
		if got := guardLocalProviderExtraBody(tc.label, tc.model); got != guardQwen36ProviderExtraBodyJSON {
			t.Fatalf("guardLocalProviderExtraBody(%q, %q) = %q, want Qwen3.6 body", tc.label, tc.model, got)
		}
	}
	if got := guardLocalProviderExtraBody("LM Studio", "qwen2.5-coder:7b"); got != "" {
		t.Fatalf("non-Qwen3.6 model got extra body: %q", got)
	}
}

func TestGuardApplyLocalProviderExtraBody(t *testing.T) {
	env := map[string]string{}
	applied, alreadySet, value, err := guardApplyLocalProviderExtraBody(
		"Qwen3.6 local",
		"lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M",
		func(k string) string { return env[k] },
		func(k, v string) error { env[k] = v; return nil },
	)
	if err != nil || !applied || alreadySet || value != guardQwen36ProviderExtraBodyJSON {
		t.Fatalf("apply = applied=%v alreadySet=%v value=%q err=%v", applied, alreadySet, value, err)
	}
	if env["FAK_PROVIDER_EXTRA_BODY_JSON"] != guardQwen36ProviderExtraBodyJSON {
		t.Fatalf("env body = %q, want Qwen3.6 body", env["FAK_PROVIDER_EXTRA_BODY_JSON"])
	}

	env["FAK_PROVIDER_EXTRA_BODY_JSON"] = `{"top_k":8}`
	applied, alreadySet, value, err = guardApplyLocalProviderExtraBody(
		"Qwen3.6 local",
		"lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M",
		func(k string) string { return env[k] },
		func(k, v string) error { env[k] = v; return nil },
	)
	if err != nil || applied || !alreadySet || value != guardQwen36ProviderExtraBodyJSON {
		t.Fatalf("override apply = applied=%v alreadySet=%v value=%q err=%v", applied, alreadySet, value, err)
	}
	if env["FAK_PROVIDER_EXTRA_BODY_JSON"] != `{"top_k":8}` {
		t.Fatalf("operator override was overwritten: %q", env["FAK_PROVIDER_EXTRA_BODY_JSON"])
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

func TestGuardLocalDetectedBannerBytes(t *testing.T) {
	got := guardLocalDetectedBanner("Ollama", "http://127.0.0.1:11434/v1", "qwen2.5-coder:7b") + "\n"
	want := "-> local backend: Ollama http://127.0.0.1:11434/v1 (model: qwen2.5-coder:7b)\n"
	if got != want {
		t.Fatalf("banner bytes = %q, want %q", got, want)
	}

	got = guardLocalDetectedBanner("LM Studio", "http://127.0.0.1:1234/v1", "") + "\n"
	want = "-> local backend: LM Studio http://127.0.0.1:1234/v1 (model: server default)\n"
	if got != want {
		t.Fatalf("default-model banner bytes = %q, want %q", got, want)
	}
}
