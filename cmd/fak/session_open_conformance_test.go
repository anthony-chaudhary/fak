package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestRenderSessionOpenExposesActionParity(t *testing.T) {
	result := gateway.SessionClientAttachResponse{Descriptor: gateway.SessionClientDescriptor{SessionID: "parity", ExecutionEpoch: "epoch-1", Capabilities: []string{"observe"}, Actions: gateway.SessionCapabilityCorpus([]string{"observe"})}}
	var out bytes.Buffer
	renderSessionOpen(&out, result)
	got := out.String()
	for _, needle := range []string{"action=observe available", "action=pause unavailable=CAPABILITY_NOT_ADVERTISED", "handoff=Open the terminal client"} {
		if !strings.Contains(got, needle) {
			t.Fatalf("render missing %q:\n%s", needle, got)
		}
	}
}
