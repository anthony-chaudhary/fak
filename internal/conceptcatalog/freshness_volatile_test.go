package conceptcatalog

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// volatileFixture lays out a scratch "generated" dir holding the committed README and
// INDEX, plus a root whose tracked README the caller mutates. The committed README is
// the realistic input: it carries every volatile row shape the generator emits.
func volatileFixture(t *testing.T, mutate func(string) string) (root, generated string) {
	t.Helper()
	repo, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	root = t.TempDir()
	generated = t.TempDir()
	for _, art := range generatedArtifacts {
		b, err := os.ReadFile(filepath.Join(repo, filepath.FromSlash(art.Tracked)))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(generated, art.Name), b, 0644); err != nil {
			t.Fatal(err)
		}
		tracked := string(b)
		if art.Tracked == GeneratedReadme {
			tracked = mutate(tracked)
			if tracked == string(b) {
				t.Fatal("mutation changed nothing: fixture marker missing")
			}
		}
		dst := filepath.Join(root, filepath.FromSlash(art.Tracked))
		if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dst, []byte(tracked), 0644); err != nil {
			t.Fatal(err)
		}
	}
	return root, generated
}

func replaceLine(t *testing.T, doc, prefix, with string) string {
	t.Helper()
	lines := strings.Split(doc, "\n")
	for i, ln := range lines {
		if strings.HasPrefix(ln, prefix) {
			lines[i] = with
			return strings.Join(lines, "\n")
		}
	}
	t.Fatalf("no line with prefix %q", prefix)
	return doc
}

// bumpFamilyRuns bumps every per-family coverage count and swaps adjacent rows,
// the shape an ordinary commit that adds Go identifiers produces.
func bumpFamilyRuns(doc string) string {
	lines := strings.Split(doc, "\n")
	var idx []int
	for i, ln := range lines {
		if m := volatileTableFamily.FindStringSubmatch(ln); m != nil {
			lines[i] = "| " + m[1] + " | 999 | 1999 | 1000 |"
			idx = append(idx, i)
		} else if m := volatileChartFamily.FindStringSubmatch(ln); m != nil {
			lines[i] = "  " + m[1] + " ##.......................... 999/1999"
			idx = append(idx, i)
		}
	}
	// Swap the first two rows of each run so gap-ordering drift is exercised too.
	for k := 0; k+1 < len(idx); k++ {
		if idx[k+1] == idx[k]+1 {
			lines[idx[k]], lines[idx[k+1]] = lines[idx[k+1]], lines[idx[k]]
			k++
		}
	}
	return strings.Join(lines, "\n")
}

// TestFreshnessIgnoresVolatileWholeTreeCounts is the fak-private#3072 witness: an
// ordinary commit that only moves whole-tree counts (discovered/covered tokens, debt,
// legacy score, per-family coverage and its gap order) must not make the README stale.
func TestFreshnessIgnoresVolatileWholeTreeCounts(t *testing.T) {
	root, generated := volatileFixture(t, func(doc string) string {
		doc = replaceLine(t, doc, "| **Disambiguation-debt (drive to 0)** |", "| **Disambiguation-debt (drive to 0)** | **12345** (clarity 1 + coverage 12344) |")
		doc = replaceLine(t, doc, "| **Confusable tokens positioned (covered / discovered)** |", "| **Confusable tokens positioned (covered / discovered)** | **1 / 99999** (0.0% of the discovered confusable space) |")
		doc = replaceLine(t, doc, "| Legacy bounded score (saturates; not the driver) |", "| Legacy bounded score (saturates; not the driver) | 12.3/100 (grade F) |")
		doc = replaceLine(t, doc, "concept-disambiguation chart - ", "concept-disambiguation chart - 2902 concepts - score 12.3/100 (grade F) - disambiguation-debt 12345")
		doc = replaceLine(t, doc, "namespace coverage  [", "namespace coverage  [#...............................] 0.0%  (1/99999 confusable tokens positioned)")
		return bumpFamilyRuns(doc)
	})
	res, err := compareGeneratedFreshness(root, generated)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Fresh {
		t.Fatalf("volatile whole-tree count drift reported stale: %+v", res)
	}
}

// TestFreshnessStillFailsOnCatalogDrift keeps the hard edge: anything the concept catalog
// decides - a concept's verdict, the crystal counter, a per-KPI clarity row, the catalog
// concept count, or the SET of families - is still stale when it disagrees.
func TestFreshnessStillFailsOnCatalogDrift(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(t *testing.T, doc string) string
	}{
		{"verdict row", func(t *testing.T, doc string) string {
			return strings.Replace(doc, "| * | **managed cache** (`managed-cache`) | crystal |", "| o | **managed cache** (`managed-cache`) | defined |", 1)
		}},
		{"crystal counter", func(t *testing.T, doc string) string {
			return replaceLine(t, doc, "| **Crystal-clear concepts (and climbing)** |", "| **Crystal-clear concepts (and climbing)** | **1** crystal of 2902 positioned |")
		}},
		{"per-KPI clarity", func(t *testing.T, doc string) string {
			return replaceLine(t, doc, "| distinctness | `defined` |", "| distinctness | `defined` | 90 | 3 | 3 undefined concept(s) |")
		}},
		{"catalog concept count", func(t *testing.T, doc string) string {
			return replaceLine(t, doc, "concept-disambiguation chart - ", "concept-disambiguation chart - 2903 concepts - score 89.6/100 (grade B) - disambiguation-debt 541")
		}},
		{"family set", func(t *testing.T, doc string) string {
			return strings.Replace(doc, "| vfs | 0 | 0 | 0 |", "| vfs-renamed | 0 | 0 | 0 |", 1)
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root, generated := volatileFixture(t, func(doc string) string { return tc.mutate(t, doc) })
			res, err := compareGeneratedFreshness(root, generated)
			if err != nil {
				t.Fatal(err)
			}
			if res.Fresh || len(res.StalePaths) != 1 || res.StalePaths[0] != GeneratedReadme {
				t.Fatalf("catalog drift must stay stale, got %+v", res)
			}
		})
	}
}
