package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

func TestNewModelManifestReadOnlyJSONPath(t *testing.T) {
	path := filepath.Join("..", "..", "internal", "newmodel", "testdata", "qwen38-valid.json")
	var first, second, stderr bytes.Buffer
	if code := runNewModel(&first, &stderr, []string{"--from-manifest", path, "--json"}); code != 0 {
		t.Fatalf("first compile exit=%d stderr=%s", code, stderr.String())
	}
	stderr.Reset()
	if code := runNewModel(&second, &stderr, []string{"--from-manifest", path, "--json"}); code != 0 {
		t.Fatalf("second compile exit=%d stderr=%s", code, stderr.String())
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("CLI path emitted different bytes for the same pinned manifest")
	}
	var packet newmodel.Packet
	if err := json.Unmarshal(first.Bytes(), &packet); err != nil {
		t.Fatal(err)
	}
	if packet.Engine != "fak-native" || packet.ExternalRuntimeFallback || packet.RegistrationClosure.Closed {
		t.Fatalf("CLI emitted a promoted or fallback packet: %+v", packet)
	}

	stderr.Reset()
	var refused bytes.Buffer
	bad := filepath.Join("..", "..", "internal", "newmodel", "testdata", "qwen38-unknown-delta.json")
	if code := runNewModel(&refused, &stderr, []string{"--from-manifest", bad, "--json"}); code != 3 {
		t.Fatalf("unknown delta exit=%d stdout=%s stderr=%s", code, refused.String(), stderr.String())
	}
	if refused.Len() != 0 {
		t.Fatalf("refusal emitted an executable/scaffold packet: %s", refused.String())
	}
	var refusal newmodel.Refusal
	if err := json.Unmarshal(stderr.Bytes(), &refusal); err != nil {
		t.Fatal(err)
	}
	if refusal.Reason != newmodel.RefusalUnknownSemanticDelta || refusal.Phase != "pre-allocation" || refusal.Axis != "attention" {
		t.Fatalf("refusal = %+v", refusal)
	}
}
