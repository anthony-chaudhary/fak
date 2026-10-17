package newmodel

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
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
	for _, name := range []string{"release-pinned", "descriptor-validated"} {
		if rung := requireSupportRung(t, packet.SupportLadder, name); rung.Status != "complete" {
			t.Fatalf("support rung %q status = %q, want complete", name, rung.Status)
		}
	}
	for _, name := range []string{"semantic-reference", "fak-native", "optimized"} {
		if rung := requireSupportRung(t, packet.SupportLadder, name); rung.Status != "pending" {
			t.Fatalf("support rung %q status = %q, want pending", name, rung.Status)
		}
	}
	obligationRung := map[string]string{
		"semantic":    "semantic-reference",
		"oracle":      "semantic-reference",
		"backend":     "fak-native",
		"test":        "fak-native",
		"docs":        "fak-native",
		"performance": "optimized",
	}
	for _, obligation := range packet.Obligations {
		rung := requireSupportRung(t, packet.SupportLadder, obligationRung[obligation.Kind])
		if !contains(rung.Obligations, obligation.ID) {
			t.Fatalf("obligation %s:%s missing from support rung %q: %+v", obligation.Kind, obligation.ID, rung.Name, rung)
		}
	}
	wantOpen := make([]string, 0, len(packet.Obligations))
	for _, obligation := range packet.Obligations {
		wantOpen = append(wantOpen, obligation.Kind+":"+obligation.ID)
	}
	sort.Strings(wantOpen)
	if packet.RegistrationClosure.Closed != (len(wantOpen) == 0) || !equalStrings(packet.RegistrationClosure.Open, wantOpen) {
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

func TestPinRefusalPrecedenceIsDeterministic(t *testing.T) {
	var manifest ReleaseManifest
	if err := json.Unmarshal(fixture(t, "qwen38-valid.json"), &manifest); err != nil {
		t.Fatal(err)
	}
	manifest.Source.URI = ""
	manifest.Artifact.URI = ""
	manifest.Source.ManifestSHA256 = "bad"
	manifest.Artifact.SHA256 = "bad"
	raw, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		_, err := CompileManifest(raw)
		var refusal *Refusal
		if !errors.As(err, &refusal) || refusal.Reason != RefusalPinInvalid || refusal.Axis != "source.uri" {
			t.Fatalf("iteration %d refusal = %#v, err = %v; want source.uri first", i, refusal, err)
		}
	}

	manifest.Source.URI = "fixture://source"
	manifest.Artifact.URI = "fixture://artifact"
	raw, err = json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		_, err := CompileManifest(raw)
		var refusal *Refusal
		if !errors.As(err, &refusal) || refusal.Axis != "source.manifest_sha256" {
			t.Fatalf("iteration %d refusal = %#v, err = %v; want source.manifest_sha256 first", i, refusal, err)
		}
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

func requireSupportRung(t *testing.T, ladder []SupportRung, name string) SupportRung {
	t.Helper()
	for _, rung := range ladder {
		if rung.Name == name {
			return rung
		}
	}
	t.Fatalf("support ladder missing rung %q: %+v", name, ladder)
	return SupportRung{}
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

// fak-test:runtime fast est=10ms lane=default
func TestManifestAlwaysInventoriesPretrainedDrafterAsP0(t *testing.T) {
	t.Parallel()
	var manifest ReleaseManifest
	if err := json.Unmarshal(fixture(t, "qwen38-valid.json"), &manifest); err != nil {
		t.Fatal(err)
	}
	// Self-reported readiness must not pre-close the draft checks. The compiler
	// must discover the same obligations for a future, previously unseen family.
	manifest.Descriptor.Readiness = []string{"implemented", "tested", "hardware-qualified"}
	for _, family := range []string{"qwen38", "deepseek-v41", "unseen-family"} {
		t.Run(family, func(t *testing.T) {
			manifest.Release.ID = "synthetic-" + family
			manifest.Release.Family = family
			raw, err := json.Marshal(manifest)
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := CompileManifestJSON(raw)
			if err != nil {
				t.Fatal(err)
			}
			// Decode the wire contract independently so this regression also builds
			// against the old compiler and fails on its missing inventory field.
			var got struct {
				PretrainedDrafter struct {
					Schema                string `json:"schema"`
					Priority              string `json:"priority"`
					TargetID              string `json:"target_id"`
					TargetRevision        string `json:"target_revision"`
					TargetArtifactSHA256  string `json:"target_artifact_sha256"`
					TargetTokenizerSHA256 string `json:"target_tokenizer_sha256"`
					ArtifactAvailability  string `json:"artifact_availability"`
					Implementation        string `json:"implementation"`
					SoftwareVerification  string `json:"software_verification"`
					HardwareQualification string `json:"hardware_qualification"`
					ServingActivation     string `json:"serving_activation"`
					ExecutionAdmitted     bool   `json:"execution_admitted"`
					DefaultEnabled        bool   `json:"default_enabled"`
					Checks                []struct {
						ObligationID     string   `json:"obligation_id"`
						Kind             string   `json:"kind"`
						Status           string   `json:"status"`
						Description      string   `json:"description"`
						EvidenceRequired []string `json:"evidence_required"`
					} `json:"checks"`
				} `json:"pretrained_drafter"`
			}
			if err := json.Unmarshal(encoded, &got); err != nil {
				t.Fatal(err)
			}
			inv := got.PretrainedDrafter
			if inv.Schema != "fak.new-model-pretrained-drafter/1" || inv.Priority != "P0" {
				t.Fatalf("missing mandatory P0 inventory: %+v", inv)
			}
			if inv.TargetID != manifest.Release.ID || inv.TargetRevision != manifest.Release.Revision || inv.TargetArtifactSHA256 != manifest.Artifact.SHA256 || inv.TargetTokenizerSHA256 != manifest.Artifact.TokenizerSHA256 {
				t.Fatalf("inventory lost exact target identity: %+v", inv)
			}
			if inv.ArtifactAvailability != "not-surveyed" || inv.Implementation != "unassessed" || inv.SoftwareVerification != "not-established" || inv.HardwareQualification != "not-established" || inv.ServingActivation != "not-admitted" || inv.ExecutionAdmitted || inv.DefaultEnabled {
				t.Fatalf("intake promoted unverified draft support: %+v", inv)
			}
			var packet Packet
			if err := json.Unmarshal(encoded, &packet); err != nil {
				t.Fatal(err)
			}
			want := map[string]string{
				"pretrained-drafter-artifact-availability-and-pins":          "weight_shard_sha256",
				"pretrained-drafter-model-tokenizer-tap-quant-compatibility": "hidden_state_tap_indices_and_convention",
				"pretrained-drafter-tensor-inventory-and-loader":             "real_header_witness",
				"pretrained-drafter-native-forward-and-state-cache":          "independent_forward_oracle",
				"pretrained-drafter-exact-verification-rollback-stop":        "eos_and_stop",
				"pretrained-drafter-serving-activation-and-defaults":         "executed_route_receipt",
				"pretrained-drafter-measured-end-to-end-benefit":             "accepted_tokens_per_round",
			}
			if len(inv.Checks) != len(want) {
				t.Fatalf("draft checklist has %d checks, want %d", len(inv.Checks), len(want))
			}
			rungs := map[string]string{"semantic": "semantic-reference", "backend": "fak-native", "test": "fak-native", "performance": "optimized"}
			for _, check := range inv.Checks {
				evidence, exists := want[check.ObligationID]
				if !exists || check.Status != "pending" || check.Description == "" || !contains(check.EvidenceRequired, evidence) {
					t.Fatalf("incomplete or duplicate draft check: %+v", check)
				}
				delete(want, check.ObligationID)
				if !contains(packet.RegistrationClosure.Open, check.Kind+":"+check.ObligationID) || packet.RegistrationClosure.Closed {
					t.Fatalf("closure hid P0 obligation %q", check.ObligationID)
				}
				rung := requireSupportRung(t, packet.SupportLadder, rungs[check.Kind])
				if rung.Status != "pending" || !contains(rung.Obligations, check.ObligationID) {
					t.Fatalf("support ladder hid P0 obligation %q", check.ObligationID)
				}
			}
		})
	}
	for _, id := range []string{"pretrained-drafter-artifact-availability-and-pins", "PRETRAINED_DRAFTER_ARTIFACT_AVAILABILITY_AND_PINS"} {
		spoofed := manifest
		spoofed.Obligations = append(append([]Obligation(nil), manifest.Obligations...), Obligation{
			ID: id, Kind: "semantic", Description: "already complete",
		})
		raw, err := json.Marshal(spoofed)
		if err != nil {
			t.Fatal(err)
		}
		_, err = CompileManifest(raw)
		var refusal *Refusal
		if !errors.As(err, &refusal) || refusal.Reason != RefusalObligationsIncomplete || refusal.Axis != "pretrained-drafter-artifact-availability-and-pins" {
			t.Fatalf("manifest replaced a compiler-owned P0 check via %q: refusal=%+v err=%v", id, refusal, err)
		}
	}
}
