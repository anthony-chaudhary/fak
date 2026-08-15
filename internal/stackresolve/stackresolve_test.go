package stackresolve

import (
	"context"
	"encoding/json"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestSpineFixturesResolveAndExplainTransitiveConflict(t *testing.T) {
	allow, refuse, err := Selfcheck(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if allow.Status != "allow" {
		t.Fatalf("allow status = %q", allow.Status)
	}
	if got := selectedIDs(allow); !containsAll(got, "harness:ponytail@r8", "model:coder-awq@sha256:111", "backend:awq@1.4.2", "kernel:awq-portable@0.9", "runtime:cuda@12.8", "infra:l4-cuda12@r3") {
		t.Fatalf("selected = %v", got)
	}
	if !hasSubstitute(allow, "kernel.awq.fast", "kernel:awq-portable@0.9") {
		t.Fatalf("receipt does not explain kernel substitute: %+v", allow.Decisions)
	}
	if !hasWarning(allow, "RECOMMENDATION_UNMET", "infra.gpu.sm90") || !hasWarning(allow, "OPTIONAL_UNAVAILABLE", "tool.browser") {
		t.Fatalf("recommendation/optional warnings missing: %+v", allow.Warnings)
	}

	wantChain := []string{"harness:ponytail@r8", "model:coder-awq@sha256:111", "backend:awq@1.4.2", "kernel:awq-fast@0.9", "device.cuda.sm80"}
	if refuse.Conflict == nil || !reflect.DeepEqual(refuse.Conflict.Chain, wantChain) {
		t.Fatalf("conflict = %+v, want chain %v", refuse.Conflict, wantChain)
	}
	if refuse.Conflict.Evidence.Authority != "kernel" || refuse.Conflict.Evidence.Tier != "device-decode" {
		t.Fatalf("conflict evidence = %+v", refuse.Conflict.Evidence)
	}
	text := Format(refuse)
	for _, want := range []string{"REFUSE", strings.Join(wantChain, " -> "), "collect or refresh evidence"} {
		if !strings.Contains(text, want) {
			t.Fatalf("formatted refusal missing %q:\n%s", want, text)
		}
	}
}

func TestCapturedSelfcheckWitnessMatches(t *testing.T) {
	allow, refuse, err := Selfcheck(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.MarshalIndent(struct {
		Schema string  `json:"schema"`
		Allow  Receipt `json:"allow"`
		Refuse Receipt `json:"refuse"`
	}{Schema: "fak-stack-selfcheck/1", Allow: allow, Refuse: refuse}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile("testdata/selfcheck-witness.json")
	if err != nil {
		t.Fatal(err)
	}
	var gotJSON, wantJSON any
	if err := json.Unmarshal(got, &gotJSON); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(want, &wantJSON); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotJSON, wantJSON) {
		t.Fatalf("captured witness drifted; regenerate with go run ./cmd/stackresolvedemo -selfcheck -json")
	}
}

func TestResolveIsDeterministicAcrossCatalogOrder(t *testing.T) {
	raw, err := os.ReadFile("testdata/coding-stack.json")
	if err != nil {
		t.Fatal(err)
	}
	manifest, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	first, err := Resolve(context.Background(), manifest.Workload, manifest.Roots, ManifestProvider{Manifest: manifest})
	if err != nil {
		t.Fatal(err)
	}
	for left, right := 0, len(manifest.Components)-1; left < right; left, right = left+1, right-1 {
		manifest.Components[left], manifest.Components[right] = manifest.Components[right], manifest.Components[left]
	}
	second, err := Resolve(context.Background(), manifest.Workload, manifest.Roots, ManifestProvider{Manifest: manifest})
	if err != nil {
		t.Fatal(err)
	}
	one, _ := json.Marshal(first)
	two, _ := json.Marshal(second)
	if string(one) != string(two) {
		t.Fatalf("receipts differ by catalog order\n%s\n%s", one, two)
	}
}

func TestRecommendationNeverBecomesHardGate(t *testing.T) {
	component := Component{
		ID: "harness:h@1", Kind: "harness", Version: "1",
		Relations: []Relation{{Kind: Recommends, Target: "infra.fast", Evidence: Evidence{Authority: "bench", Source: "b1"}}},
		Evidence:  Evidence{Authority: "kit", Source: "h1"},
	}
	receipt, err := Resolve(context.Background(), "work@1", []string{component.ID}, ManifestProvider{Manifest: Manifest{Components: []Component{component}}})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Status != "allow" || !hasWarning(receipt, "RECOMMENDATION_UNMET", "infra.fast") {
		t.Fatalf("receipt = %+v", receipt)
	}
}

func TestConflictEvaluatedAfterDependencyClosure(t *testing.T) {
	evidence := Evidence{Authority: "owner", Source: "manifest"}
	components := []Component{
		{ID: "a@1", Kind: "root", Version: "1", Relations: []Relation{{Kind: Conflicts, Target: "cap.b", Evidence: evidence}, {Kind: Requires, Target: "cap.b", Evidence: evidence}}, Evidence: evidence},
		{ID: "b@1", Kind: "dependency", Version: "1", Provides: []string{"cap.b"}, Evidence: evidence},
	}
	receipt, err := Resolve(context.Background(), "work@1", []string{"a@1"}, ManifestProvider{Manifest: Manifest{Components: components}})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Status != "refuse" || receipt.Conflict == nil || receipt.Conflict.Code != "COMPONENT_CONFLICT" {
		t.Fatalf("receipt = %+v", receipt)
	}
}

func TestPackageDoesNotImportDomainOwners(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range files {
		parsed, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatal(err)
		}
		for _, imported := range parsed.Imports {
			path := strings.Trim(imported.Path.Value, `"`)
			if strings.Contains(path, "github.com/anthony-chaudhary/fak/internal/") {
				t.Fatalf("generic resolver imports domain package %q", path)
			}
		}
	}
}

func selectedIDs(receipt Receipt) []string {
	out := make([]string, 0, len(receipt.Selected))
	for _, component := range receipt.Selected {
		out = append(out, component.ID)
	}
	return out
}

func containsAll(got []string, want ...string) bool {
	set := map[string]bool{}
	for _, item := range got {
		set[item] = true
	}
	for _, item := range want {
		if !set[item] {
			return false
		}
	}
	return true
}

func hasSubstitute(receipt Receipt, wanted, chosen string) bool {
	for _, decision := range receipt.Decisions {
		if decision.Wanted == wanted && decision.Chosen == chosen && decision.Substitute {
			return true
		}
	}
	return false
}

func hasWarning(receipt Receipt, code, wanted string) bool {
	for _, warning := range receipt.Warnings {
		if warning.Code == code && warning.Wanted == wanted {
			return true
		}
	}
	return false
}
