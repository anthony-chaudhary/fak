package agent

import "testing"

func TestProviderDefaultMaxTokensEnv(t *testing.T) {
	p, err := NewProviderHTTPPlanner(string(ProviderOpenAI), "http://127.0.0.1:1/v1", "m", "k")
	if err != nil {
		t.Fatal(err)
	}
	if p.MaxTokens != 1024 {
		t.Fatalf("unset default = %d, want 1024", p.MaxTokens)
	}
	t.Setenv("FAK_PROVIDER_DEFAULT_MAX_TOKENS", "8192")
	t.Setenv("FAK_PROVIDER_MAX_TOKENS", "")
	p, err = NewProviderHTTPPlanner(string(ProviderOpenAI), "http://127.0.0.1:1/v1", "m", "k")
	if err != nil {
		t.Fatal(err)
	}
	if p.MaxTokens != 8192 {
		t.Fatalf("override = %d, want 8192", p.MaxTokens)
	}
	t.Setenv("FAK_PROVIDER_DEFAULT_MAX_TOKENS", "junk")
	p, _ = NewProviderHTTPPlanner(string(ProviderOpenAI), "http://127.0.0.1:1/v1", "m", "k")
	if p.MaxTokens != 1024 {
		t.Fatalf("junk override = %d, want 1024", p.MaxTokens)
	}
}
