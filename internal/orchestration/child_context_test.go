package orchestration

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestChildContextReceiptContract(t *testing.T) {
	digest := func(ch byte) string { return strings.Repeat(string(ch), 64) }
	fixture := ChildContextReceipt{
		Schema:           ChildContextReceiptSchema,
		ParentRunID:      "run-parent-42",
		ParentSessionID:  "session-parent-7",
		ChildNodeID:      "child-review",
		ParentPlanDigest: digest('1'),
		ChildDeclaration: ChildContextDeclaration{
			ChildNodeID:      "child-review",
			ParentPlanDigest: digest('1'),
			Digest:           digest('2'),
		},
		AccessDeclarationDigest:             digest('3'),
		CapabilityEnvelopeDigest:            digest('4'),
		ReservedBudget:                      ChildContextBudget{Workers: 1, Tokens: 4096},
		WorkspaceStateEpoch:                 "git:f2a802ef1",
		InputArtifactRefs:                   []string{"artifact:brief@sha256:" + digest('5'), "artifact:plan@sha256:" + digest('6')},
		ExpectedOutputWitnessContractDigest: digest('7'),
	}

	canonical, err := fixture.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeChildContextReceiptJSON(canonical)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, fixture) {
		t.Fatalf("round trip changed receipt:\n got: %#v\nwant: %#v", decoded, fixture)
	}
	first, err := fixture.Digest()
	if err != nil {
		t.Fatal(err)
	}
	second, err := decoded.Digest()
	if err != nil || second != first {
		t.Fatalf("digest after round trip = %q, %v; want %q", second, err, first)
	}

	mutations := map[string]func(*ChildContextReceipt){
		"parent": func(r *ChildContextReceipt) { r.ParentRunID = "run-parent-43" },
		"plan": func(r *ChildContextReceipt) {
			r.ParentPlanDigest = digest('8')
			r.ChildDeclaration.ParentPlanDigest = digest('8')
		},
		"envelope": func(r *ChildContextReceipt) { r.CapabilityEnvelopeDigest = digest('9') },
		"epoch":    func(r *ChildContextReceipt) { r.WorkspaceStateEpoch = "git:f2a802ef2" },
	}
	for name, mutate := range mutations {
		t.Run("identity changes with "+name, func(t *testing.T) {
			changed := fixture
			changed.InputArtifactRefs = append([]string(nil), fixture.InputArtifactRefs...)
			mutate(&changed)
			got, err := changed.Digest()
			if err != nil {
				t.Fatal(err)
			}
			if got == first {
				t.Fatalf("%s change retained digest %q", name, got)
			}
		})
	}

	for name, mutate := range map[string]func(*ChildContextReceipt){
		"missing parent run":     func(r *ChildContextReceipt) { r.ParentRunID = "" },
		"missing parent session": func(r *ChildContextReceipt) { r.ParentSessionID = "" },
		"unbound child":          func(r *ChildContextReceipt) { r.ChildDeclaration.ChildNodeID = "other-child" },
		"unbound plan":           func(r *ChildContextReceipt) { r.ChildDeclaration.ParentPlanDigest = digest('8') },
	} {
		t.Run("rejects "+name, func(t *testing.T) {
			invalid := fixture
			invalid.InputArtifactRefs = append([]string(nil), fixture.InputArtifactRefs...)
			mutate(&invalid)
			if err := invalid.Validate(); err == nil {
				t.Fatal("validation accepted invalid receipt")
			}
		})
	}

	var object map[string]any
	if err := json.Unmarshal(canonical, &object); err != nil {
		t.Fatal(err)
	}
	object["unknown"] = true
	unknown, err := json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	for name, raw := range map[string][]byte{
		"unknown field":           unknown,
		"trailing value":          append(append([]byte(nil), canonical...), []byte(` {}`)...),
		"noncanonical whitespace": append([]byte(" "), canonical...),
	} {
		t.Run("strict decode rejects "+name, func(t *testing.T) {
			if _, err := DecodeChildContextReceiptJSON(raw); err == nil {
				t.Fatal("strict decoder accepted noncanonical JSON")
			}
		})
	}
}
