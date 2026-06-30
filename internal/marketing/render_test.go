package marketing

import (
	"strings"
	"testing"
	"time"
)

func mustClaim(t *testing.T, text string, s Ship) Claim {
	t.Helper()
	c, err := NewClaim(text, s)
	if err != nil {
		t.Fatalf("NewClaim(%q): %v", text, err)
	}
	return c
}

func TestArtifactTextRendersShaOnEveryBullet(t *testing.T) {
	d1 := time.Date(2026, 6, 28, 10, 0, 0, 0, time.UTC)
	d2 := time.Date(2026, 6, 27, 10, 0, 0, 0, time.UTC)
	art := Artifact{
		Kind:  KindWeeklyDigest,
		Title: "fak — what shipped",
		Claims: []Claim{
			mustClaim(t, "New: add reclaim path", Ship{SHA: "aaaa1111", Leaf: "gateway", Kind: "trailer", Date: d1}),
			mustClaim(t, "Faster: Q4_K reducer", Ship{SHA: "bbbb2222", Leaf: "model", Kind: "direct", Date: d2}),
		},
		Activity: Activity{Commits: 5, Ships: 2},
		Source:   "ci",
	}
	txt := art.Text()
	for _, sha := range []string{"aaaa1111", "bbbb2222"} {
		if !strings.Contains(txt, sha) {
			t.Errorf("Text() missing sha %q:\n%s", sha, txt)
		}
	}
	if !strings.Contains(txt, "WITNESSED") {
		t.Errorf("Text() missing WITNESSED label:\n%s", txt)
	}
	// honest footer: 2 ships, 3 other commits
	if !strings.Contains(txt, "2 witnessed ships") || !strings.Contains(txt, "3 other commits") {
		t.Errorf("Text() footer wrong:\n%s", txt)
	}
	// newest-first: gateway (06-28) before model (06-27)
	if strings.Index(txt, "aaaa1111") > strings.Index(txt, "bbbb2222") {
		t.Errorf("claims not newest-first:\n%s", txt)
	}
}

func TestArtifactTextHonestWhenEmpty(t *testing.T) {
	art := Artifact{Kind: KindWeeklyDigest, Activity: Activity{Commits: 3, Ships: 0}}
	txt := art.Text()
	if !strings.Contains(txt, "No witnessed ships") {
		t.Errorf("empty artifact should say 'No witnessed ships', got:\n%s", txt)
	}
	if !strings.Contains(txt, "0 witnessed ships") {
		t.Errorf("empty footer should say 0 witnessed ships:\n%s", txt)
	}
}

func TestArtifactBlocksCarrySameShas(t *testing.T) {
	art := Artifact{
		Kind:     KindLaunchBlurb,
		Claims:   []Claim{mustClaim(t, "New: add x", Ship{SHA: "deadbeef", Leaf: "gateway", Kind: "trailer"})},
		Activity: Activity{Commits: 1, Ships: 1},
	}
	blocks := art.Blocks()
	if len(blocks) == 0 {
		t.Fatal("Blocks() empty")
	}
	// flatten to a string and check the sha survived into the block payload
	flat := flattenBlocks(blocks)
	if !strings.Contains(flat, "deadbeef") {
		t.Errorf("Blocks() missing sha deadbeef:\n%s", flat)
	}
}

func TestClaimTextStripsPrefixAndStamp(t *testing.T) {
	cases := []struct {
		subject string
		want    string
	}{
		{"feat(gateway): add the reclaim path (fak gateway)", "New: add the reclaim path"},
		{"fix(model): correct the Q4_K scale (fak model)", "Fixed: correct the Q4_K scale"},
		{"perf(kvmmu): batch the evict (fak kvmmu)", "Faster: batch the evict"},
		{"fak/model: implement reducer", "Shipped: implement reducer"}, // direct stamp, no type prefix
	}
	for _, tc := range cases {
		got := claimText(Ship{Subject: tc.subject})
		if got != tc.want {
			t.Errorf("claimText(%q) = %q, want %q", tc.subject, got, tc.want)
		}
	}
}

func TestClaimTextRelabelsBindingLayerSyncShips(t *testing.T) {
	cases := []struct {
		name string
		ship Ship
		want string
	}{
		{
			name: "vendor landing",
			ship: Ship{
				Subject: "docs(docs): sync shared documentation (fak docs)",
				Paths:   []string{"docs/vendor/README.md"},
			},
			want: "Documented: add binding-layer AEO vendor routing",
		},
		{
			name: "minimum backend example",
			ship: Ship{
				Subject: "chore(compute): sync shared worktree (fak compute)",
				Paths:   []string{"internal/compute/minimal_backend_example_test.go"},
			},
			want: "New: add the compiling minimum backend example for vendor backends",
		},
		{
			name: "unrelated sync stays generic",
			ship: Ship{
				Subject: "chore(windowgate): sync shared worktree (fak windowgate)",
				Paths:   []string{"internal/windowgate/windowgate.go"},
			},
			want: "Shipped: sync shared worktree",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := claimText(tc.ship); got != tc.want {
				t.Fatalf("claimText = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestWeeklyDigestBuildsWitnessedClaims(t *testing.T) {
	ships := []Ship{
		{SHA: "aaaa1111", Leaf: "gateway", Kind: "trailer", Subject: "feat(gateway): add x (fak gateway)"},
		{SHA: "bbbb2222", Leaf: "model", Kind: "direct", Subject: "fak/model: add y"},
	}
	art := WeeklyDigest(ships, Activity{Commits: 4, Ships: 2}, nil, time.Time{})
	if len(art.Claims) != 2 {
		t.Fatalf("digest claims = %d, want 2", len(art.Claims))
	}
	if art.DedupeKey == "" {
		t.Error("digest has no DedupeKey")
	}
	// the dedupe key is stable across a re-render of the same ships
	art2 := WeeklyDigest(ships, Activity{Commits: 4, Ships: 2}, nil, time.Time{})
	if art.DedupeKey != art2.DedupeKey {
		t.Errorf("DedupeKey not stable: %q vs %q", art.DedupeKey, art2.DedupeKey)
	}
}

func TestWeeklyDigestExcludedSurfacedInFooter(t *testing.T) {
	ships := []Ship{{SHA: "aaaa1111", Leaf: "gateway", Kind: "trailer", Subject: "feat(gateway): add x (fak gateway)"}}
	excluded := []ExcludedShip{{Ship: Ship{Leaf: "preflight"}, Reason: "stub"}}
	art := WeeklyDigest(ships, Activity{Commits: 2, Ships: 1}, excluded, time.Time{})
	txt := art.Text()
	if !strings.Contains(txt, "1 held") {
		t.Errorf("footer should surface the held ship:\n%s", txt)
	}
}

// flattenBlocks walks the Block Kit payload and concatenates every string value, so a test
// can assert a sha survived into the structured card without asserting on exact nesting.
func flattenBlocks(blocks []any) string {
	var b strings.Builder
	var walk func(v any)
	walk = func(v any) {
		switch t := v.(type) {
		case string:
			b.WriteString(t)
			b.WriteByte(' ')
		case map[string]any:
			for _, vv := range t {
				walk(vv)
			}
		case []any:
			for _, vv := range t {
				walk(vv)
			}
		}
	}
	for _, blk := range blocks {
		walk(blk)
	}
	return b.String()
}
