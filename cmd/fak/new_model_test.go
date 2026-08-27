package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

func TestNewModelManifestCLIEmitsPacketAndTypedRefusal(t *testing.T) {
	fixture := func(name string) string {
		return filepath.Join("..", "..", "internal", "newmodel", "testdata", name)
	}

	var stdout, stderr bytes.Buffer
	if code := runNewModel(&stdout, &stderr, []string{"--from-manifest", fixture("qwen38-release.json"), "--json"}); code != 0 {
		t.Fatalf("valid manifest exit=%d stderr=%s", code, stderr.String())
	}
	var packet newmodel.OnboardingPacket
	if err := json.Unmarshal(stdout.Bytes(), &packet); err != nil {
		t.Fatal(err)
	}
	if packet.Schema != newmodel.OnboardingPacketSchema || packet.Execution.Engine != "fak-native" || packet.Execution.ExternalFallback {
		t.Fatalf("packet execution = %+v", packet.Execution)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runNewModel(&stdout, &stderr, []string{"--from-manifest", fixture("qwen38-semantic-delta.json"), "--json"}); code != 3 {
		t.Fatalf("semantic-delta exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var refusal newmodel.Refusal
	if err := json.Unmarshal(stderr.Bytes(), &refusal); err != nil {
		t.Fatal(err)
	}
	if refusal.Code != newmodel.RefusalUnresolvedSemanticAxis || refusal.Phase != "pre-allocation" || refusal.Axis != "rotary_scaling_semantics" {
		t.Fatalf("refusal = %+v", refusal)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runNewModel(&stdout, &stderr, []string{"--family", "fixture", "--json"}); code != 0 {
		t.Fatalf("legacy scaffold exit=%d stderr=%s", code, stderr.String())
	}
}
