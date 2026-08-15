package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestHoldCompletedCategoryBaselineRedirectsPolish(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, ".fak", "category-baselines.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	registry := `{"schema":"fak-category-baselines/1","categories":[{"name":"serving","layers":["medium-model","l2-cache","l3-cache"],"completed_layer":"medium-model","next_layer":"l2-cache","witness":"fak serve --selfcheck"}]}`
	if err := os.WriteFile(path, []byte(registry), 0o600); err != nil {
		t.Fatal(err)
	}

	payload := dispatchtick.RouterPayload{
		Issues: []dispatchtick.IssueRoute{
			{Number: 10, Title: "feat(serving): tune medium model", Lane: "model", Category: "serving", Layer: "medium-model", ExpectedSteps: 1},
			{Number: 20, Title: "feat(cache): implement L2", Lane: "cache", Category: "serving", Layer: "l2-cache", ExpectedSteps: 1},
			{Number: 30, Title: "fix(serving): repair medium model regression", Lane: "model", Category: "serving", Layer: "medium-model", ExpectedSteps: 1},
			{Number: 40, Title: "legacy work", Lane: "docs", ExpectedSteps: 1},
		},
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"model": {Issues: []int{10, 30}, Count: 2, StepBudget: 2},
			"cache": {Issues: []int{20}, Count: 1, StepBudget: 1},
			"docs":  {Issues: []int{40}, Count: 1, StepBudget: 1},
		},
		Counts: dispatchtick.RouterCounts{Routed: 4, RoutedStepBudget: 4},
	}
	got := holdCompletedCategoryBaselines(root, payload)
	for _, route := range got.Issues {
		if route.Number == 10 {
			t.Fatal("completed-layer polish remained routable")
		}
	}
	for _, want := range []int{20, 30, 40} {
		found := false
		for _, route := range got.Issues {
			found = found || route.Number == want
		}
		if !found {
			t.Fatalf("routable issue #%d was held: %+v", want, got.Issues)
		}
	}
	var held dispatchtick.SkippedIssue
	for _, row := range got.SkippedHumanBlocked {
		if row.Number == 10 {
			held = row
		}
	}
	if held.Reason != reasonCategoryBaselineComplete || !strings.Contains(held.NextAction, "serving/medium-model") || !strings.Contains(held.NextAction, "serving/l2-cache") {
		t.Fatalf("held row = %+v", held)
	}
}

func TestHoldCompletedCategoryBaselinesFailsOpenWithoutRegistry(t *testing.T) {
	payload := dispatchtick.RouterPayload{Issues: []dispatchtick.IssueRoute{{Number: 10, Category: "serving", Layer: "medium-model"}}}
	got := holdCompletedCategoryBaselines(t.TempDir(), payload)
	if len(got.Issues) != 1 {
		t.Fatalf("missing registry changed payload: %+v", got)
	}
}

func TestRouteCarriesExplicitCategoryLayer(t *testing.T) {
	body := "## Category\n\nserving\n\n## Layer\n\nmedium-model\n\n## Path hints\n\n- `internal/model/model.go`\n\n## Work unit\n\nleaf\n\n## Expected steps\n\n2\n\n## Acceptance\n\n- [ ] works"
	route := dispatchtick.RouteIssue(dispatchtick.Issue{Number: 10, Title: "model leaf", Body: body}, dispatchtick.LaneTaxonomy{Trees: map[string][]string{"model": {"internal/model/**"}}}, dispatchtick.RouteOptions{})
	if route.Category != "serving" || route.Layer != "medium-model" {
		t.Fatalf("route metadata = %+v", route)
	}
}
