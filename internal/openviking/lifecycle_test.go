package openviking

import (
	"testing"
)

// Invariant: OpenViking HTTP client configuration must strictly validate endpoint base URLs and API credentials.
// Guard: NewClient rejects empty base URLs or missing API keys.

func TestOpenVikingLifecycle(t *testing.T) {
	t.Parallel()

	cfg := Config{
		BaseURL: "http://127.0.0.1:0",
		APIKey:  "test-api-key",
		Account: "test-account",
		User:    "test-user",
	}

	client, err := NewClient(cfg)
	if err != nil {
		t.Fatalf("failed creating OpenViking client: %v", err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}

	// Missing base URL
	badCfg := cfg
	badCfg.BaseURL = ""
	if _, err := NewClient(badCfg); err == nil {
		t.Fatal("expected error on empty base URL")
	}
}
