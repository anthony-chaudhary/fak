package devcmd

import (
	"strings"
	"testing"
)

func TestAuthorDecomposeChildrenAddsWeightedProductionScope(t *testing.T) {
	r := decomposeRow{Parent: 36, Children: []decomposeChildResult{{Args: []string{"issue", "create", "--title", "child", "--body", "## Parent context\nDecomposed from #36"}}}}
	err := authorDecomposeChildren(&r, 20, "production", "- concurrent users: 10 users", "- concurrent users: 10 users")
	if err != nil {
		t.Fatal(err)
	}
	body := decomposeArgValue(r.Children[0].Args, "--body")
	for _, want := range []string{"## Work estimate", "Estimate: 8 points", "Contribution: 8/20 points", "## Completion standard\nproduction"} {
		if !strings.Contains(body, want) {
			t.Fatalf("missing %q in:\n%s", want, body)
		}
	}
}
func TestAuthorDecomposeChildrenPreservesDemoMaturity(t *testing.T) {
	r := decomposeRow{Parent: 36, Children: []decomposeChildResult{{Args: []string{"--body", "x"}}}}
	if err := authorDecomposeChildren(&r, 20, "demo", "", ""); err != nil {
		t.Fatal(err)
	}
	body := decomposeArgValue(r.Children[0].Args, "--body")
	if !strings.Contains(body, "## Completion standard\ndemo") {
		t.Fatalf("demo hidden:\n%s", body)
	}
}
