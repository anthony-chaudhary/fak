package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestGatewayUsageProvenanceIncludesExposeProfile(t *testing.T) {
	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless"})
	if err != nil {
		t.Fatal(err)
	}
	if got := gatewayUsageProvenance(srv).ExposeProfile; got != "headless" {
		t.Fatalf("expose_profile = %q, want headless", got)
	}

	interactive, err := gateway.New(gateway.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if got := gatewayUsageProvenance(interactive).ExposeProfile; got != "interactive" {
		t.Fatalf("default expose_profile = %q, want interactive", got)
	}
}
