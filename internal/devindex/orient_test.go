package devindex

import (
	"path/filepath"
	"strings"
	"testing"
)

func writeOrientRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := `[lanes.trees]
gateway = ["internal/gateway/**"]
cmd     = ["cmd/**"]
docs    = ["docs/**", "README.md"]
`
	mustWrite(t, root, "dos.toml", dosToml)
	mustMkdir(t, root, "internal", "gateway")
	mustWrite(t, filepath.Join(root, "internal", "gateway"), "gateway.go", "package gateway\n")
	mustMkdir(t, root, "internal", "architest")
	mustWrite(t, filepath.Join(root, "internal", "architest"), "architest_test.go", `package architest

var tier = map[string]int{
	"gateway": 4,
	"devindex": 1,
}
`)
	return root
}

func TestOrientPathConventions(t *testing.T) {
	c, err := Load(writeOrientRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	rows := c.Orient([]string{"internal/gateway/server.go"}, []OrientationLease{
		{ID: "resolve-gateway", Holder: "peer", Tree: []string{"internal/gateway/**"}, TTLSeconds: 60},
		{ID: "resolve-docs", Holder: "peer", Tree: []string{"docs/**"}, TTLSeconds: 60},
	})
	if len(rows) != 1 {
		t.Fatalf("Orient rows = %d, want 1", len(rows))
	}
	got := rows[0]
	if got.Lane != "gateway" || got.Stamp != "(fak gateway)" {
		t.Fatalf("lane/stamp = %q/%q, want gateway/(fak gateway)", got.Lane, got.Stamp)
	}
	if got.Tier == nil || *got.Tier != 4 || got.TierName != "integrator" {
		t.Fatalf("tier = %v %q, want 4 integrator", got.Tier, got.TierName)
	}
	if got.TestTarget != "go test ./internal/gateway" {
		t.Fatalf("owning test = %q, want go test ./internal/gateway", got.TestTarget)
	}
	if strings.Join(got.LaneTree, " ") != "internal/gateway/**" {
		t.Fatalf("lane tree = %v", got.LaneTree)
	}
	if len(got.LiveLeases) != 1 || got.LiveLeases[0].ID != "resolve-gateway" {
		t.Fatalf("live leases = %+v, want resolve-gateway only", got.LiveLeases)
	}
}

func TestOrientGlobOwningTest(t *testing.T) {
	c, err := Load(writeOrientRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	rows := c.Orient([]string{"cmd/fak/*.go"}, nil)
	if len(rows) != 1 {
		t.Fatalf("Orient rows = %d, want 1", len(rows))
	}
	if rows[0].Lane != "cmd" || rows[0].Stamp != "(fak cmd)" {
		t.Fatalf("cmd glob lane/stamp = %q/%q", rows[0].Lane, rows[0].Stamp)
	}
	if rows[0].TestTarget != "go test ./cmd/fak" {
		t.Fatalf("cmd glob owning test = %q, want go test ./cmd/fak", rows[0].TestTarget)
	}
	if rows[0].Tier != nil {
		t.Fatalf("cmd paths should not carry an architest tier, got %v", *rows[0].Tier)
	}
}
