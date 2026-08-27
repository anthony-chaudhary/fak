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
	if code := runNewModel(&stdout, &stderr, []string{"--from-manifest", fixture("qwen38-valid.json"), "--json"}); code != 0 {
		t.Fatalf("valid manifest exit=%d stderr=%s", code, stderr.String())
	}
	var packet newmodel.Packet
	if err := json.Unmarshal(stdout.Bytes(), &packet); err != nil {
		t.Fatal(err)
	}
	if packet.Schema != newmodel.PacketSchema || packet.Engine != "fak-native" || packet.ExternalRuntimeFallback {
		t.Fatalf("packet execution = %+v", packet)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runNewModel(&stdout, &stderr, []string{"--from-manifest", fixture("qwen38-unknown-delta.json"), "--json"}); code != 3 {
		t.Fatalf("semantic-delta exit=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	var refusal newmodel.Refusal
	if err := json.Unmarshal(stderr.Bytes(), &refusal); err != nil {
		t.Fatal(err)
	}
	if refusal.Reason != newmodel.RefusalUnknownSemanticDelta || refusal.Phase != "pre-allocation" || refusal.Axis != "attention" {
		t.Fatalf("refusal = %+v", refusal)
	}

	stdout.Reset()
	stderr.Reset()
	if code := runNewModel(&stdout, &stderr, []string{"--family", "fixture", "--json"}); code != 0 {
		t.Fatalf("legacy scaffold exit=%d stderr=%s", code, stderr.String())
	}
}
