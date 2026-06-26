package main

import (
	"strings"
	"testing"
)

func TestDefaultGatewayBearerTokenUsesSecretLoader(t *testing.T) {
	t.Setenv(gatewayBearerEnv, "plain-gateway-token-value")

	if got := defaultGatewayBearerToken(); got != "plain-gateway-token-value" {
		t.Fatalf("defaultGatewayBearerToken = %q, want env token", got)
	}

	loader := gatewayBearerLoader(gatewayBearerEnv)
	tok, src, ok := loader.LookupSource(gatewayBearerEnv)
	if !ok || tok != "plain-gateway-token-value" || src != "os-env" {
		t.Fatalf("LookupSource = (%q,%q,%v), want os-env token", tok, src, ok)
	}
	if out := loader.Redact("gateway key " + tok); strings.Contains(out, tok) {
		t.Fatalf("resolved gateway bearer was not redacted: %q", out)
	}
}
