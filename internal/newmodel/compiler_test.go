package newmodel

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modeldescriptor"
)

func TestManifestCompilerDeterministicWitness(t *testing.T) {
	valid := fixture(t, "qwen38-valid.json")
	first, err := CompileManifestJSON(valid)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileManifestJSON(valid)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("same pinned manifest produced different packet bytes")
	}

	var packet Packet
	if err := json.Unmarshal(first, &packet); err != nil {
		t.Fatal(err)
	}
	if packet.Schema != PacketSchema || packet.Engine != "fak-native" || packet.ExternalRuntimeFallback {
		t.Fatalf("unsafe packet identity: %+v", packet)
	}
	if packet.Descriptor.Engine != "fak-native" {
		t.Fatalf("descriptor engine = %q, want fak-native", packet.Descriptor.Engine)
	}
	if packet.Release.EvidenceClass != "synthetic-non-claiming" {
		t.Fatalf("fixture evidence class = %q, want synthetic-non-claiming", packet.Release.EvidenceClass)
	}
	if packet.ManifestDigest == "" || packet.Source.ManifestSHA256 == "" || packet.Artifact.SHA256 == "" || packet.Artifact.TokenizerSHA256 == "" || packet.Artifact.ChatTemplateSHA256 == "" || packet.Artifact.ContextSHA256 == "" {
		t.Fatalf("packet lost a pinned digest: %+v", packet)
	}
	if packet.Rollback == "" {
		t.Fatal("packet lost the explicit rollback action")
	}
	if err := modeldescriptor.Validate(packet.Descriptor.ModelDescriptor()); err != nil {
		t.Fatalf("emitted descriptor does not validate through modeldescriptor: %v", err)
	}
	if got, want := packet.Descriptor.Aliases, []string{"qwen38", "qwen38forcausallm"}; !equalStrings(got, want) {
		t.Fatalf("normalized aliases = %v, want %v", got, want)
	}
	if len(packet.SupportLadder) != 5 || packet.SupportLadder[0].Status != "complete" || packet.SupportLadder[2].Status != "pending" {
		t.Fatalf("support ladder does not separate intake from promotion: %+v", packet.SupportLadder)
	}
	if packet.RegistrationClosure.Closed || len(packet.RegistrationClosure.Open) != 6 {
		t.Fatalf("registration closure hid unresolved work: %+v", packet.RegistrationClosure)
	}
	if !packet.Coupling.WithinBudget || packet.Coupling.DescriptorDigest == "" {
		t.Fatalf("coupling report missing or outside declared budget: %+v", packet.Coupling)
	}

	for path, reason := range map[string]RefusalReason{
		"qwen38-unknown-delta.json":       RefusalUnknownSemanticDelta,
		"qwen38-contradictory-delta.json": RefusalContradictorySemantic,
	} {
		t.Run(path, func(t *testing.T) {
			_, err := CompileManifestJSON(fixture(t, path))
			var refusal *Refusal
			if !errors.As(err, &refusal) || refusal.Reason != reason || refusal.Phase != "pre-allocation" || refusal.Axis != "attention" {
				t.Fatalf("refusal = %#v, err = %v; want reason %s on attention", refusal, err, reason)
			}
		})
	}
}

func TestManifestCompilerNormalizesStateAndRejectsNegativeCoupling(t *testing.T) {
	var manifest ReleaseManifest
	if err := json.Unmarshal(fixture(t, "qwen38-valid.json"), &manifest); err != nil {
		t.Fatal(err)
	}
	withoutRollback := manifest
	withoutRollback.Rollback = ""
	raw, err := json.Marshal(withoutRollback)
	if err != nil {
		t.Fatal(err)
	}
	_, err = CompileManifest(raw)
	var refusal *Refusal
	if !errors.As(err, &refusal) || refusal.Reason != RefusalManifestInvalid || refusal.Axis != "rollback" {
		t.Fatalf("missing rollback refusal = %#v, err = %v", refusal, err)
	}
	manifest.Descriptor.State = []modeldescriptor.Geometry{
		{Kind: " KV_State ", Shape: []int{32, 128}, BytesPerElement: 2},
		{Kind: "GDN_State", Shape: []int{64}, BytesPerElement: 4},
	}
	raw, err = json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	packet, err := CompileManifest(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := []string{packet.Descriptor.State[0].Kind, packet.Descriptor.State[1].Kind}, []string{"gdn-state", "kv-state"}; !equalStrings(got, want) {
		t.Fatalf("normalized state kinds = %v, want %v", got, want)
	}

	manifest.Coupling.Counts.CoreSwitches = -1
	raw, err = json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	_, err = CompileManifest(raw)
	refusal = nil
	if !errors.As(err, &refusal) || refusal.Reason != RefusalManifestInvalid || refusal.Axis != "coupling.counts.core_switches" {
		t.Fatalf("negative coupling refusal = %#v, err = %v", refusal, err)
	}
}

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
