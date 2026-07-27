package modver

import (
	"encoding/json"
	"strings"
	"testing"
)

// coverageOutFixture is a real-shaped `go test -coverprofile=coverage.out`
// artifact. It exercises every mapping rule the fold has to get right:
// a leaf with two files, an UNCOVERED block, a SUBPACKAGE that folds into its
// parent leaf module, a cmd/ binary, a root-package file that belongs to no
// module, and a dependency row from outside this Go module.
const coverageOutFixture = `mode: set
github.com/anthony-chaudhary/fak/internal/modver/modver.go:74.42,76.2 1 1
github.com/anthony-chaudhary/fak/internal/modver/modver.go:90.78,97.29 3 1
github.com/anthony-chaudhary/fak/internal/modver/stamp.go:20.30,24.16 2 0
github.com/anthony-chaudhary/fak/internal/modver/sub/helper.go:10.20,12.3 1 1
github.com/anthony-chaudhary/fak/cmd/fak/version_modules.go:22.66,33.24 8 1
github.com/anthony-chaudhary/fak/cmd/fak/version.go:30.28,37.3 2 0
github.com/anthony-chaudhary/fak/main.go:5.13,7.2 1 1
example.com/dep/pkg/dep.go:1.1,3.2 4 1
`

const fixtureModulePath = "github.com/anthony-chaudhary/fak"

func TestCoverageScoresFoldsProfileByModule(t *testing.T) {
	got, err := CoverageScores([]byte(coverageOutFixture), fixtureModulePath)
	if err != nil {
		t.Fatal(err)
	}
	// internal/modver: 1+3 covered, 2 uncovered, +1 covered from the subpackage
	// = 5 of 7 statements. cmd/fak: 8 of 10. The root file and the dependency row
	// map to no module and must not appear.
	want := map[string]float64{
		"internal/modver": 71.4,
		"cmd/fak":         80,
	}
	if len(got) != len(want) {
		t.Fatalf("coverage scores = %#v, want %#v", got, want)
	}
	for module, pct := range want {
		if got[module] != pct {
			t.Errorf("%s coverage = %v, want %v", module, got[module], pct)
		}
	}

	// The fold must feed the existing --scores decoder unchanged: it is the same
	// flat {module: number} shape, so no schema move is involved.
	encoded, err := MarshalModuleScores(got)
	if err != nil {
		t.Fatal(err)
	}
	var flat map[string]float64
	if err := json.Unmarshal(encoded, &flat); err != nil {
		t.Fatalf("fold output is not flat module-number JSON: %v", err)
	}
	joined, err := LoadScores(encoded)
	if err != nil {
		t.Fatalf("LoadScores(fold output): %v\n%s", err, encoded)
	}
	if joined["internal/modver"].Score != 71.4 {
		t.Fatalf("joined modver score = %v, want 71.4", joined["internal/modver"].Score)
	}
}

func TestCoverageScoresIsStatementWeightedNotFileAveraged(t *testing.T) {
	// A 1-statement fully covered file next to a 9-statement uncovered one is
	// 10% weighted by statements, but 50% if the files were averaged. The
	// statement-weighted number is the one `go tool cover -func` reports.
	profile := `mode: set
example.com/repo/internal/leaf/tiny.go:1.1,2.2 1 1
example.com/repo/internal/leaf/big.go:1.1,20.2 9 0
`
	got, err := CoverageScores([]byte(profile), "example.com/repo")
	if err != nil {
		t.Fatal(err)
	}
	if got["internal/leaf"] != 10 {
		t.Fatalf("internal/leaf coverage = %v, want 10 (statement-weighted)", got["internal/leaf"])
	}
}

func TestCoverageScoresMergesRepeatedBlocks(t *testing.T) {
	// A merged profile can list the same block twice (two runs concatenated).
	// Its statements must be counted ONCE, and the block reads as covered when
	// any occurrence executed it — otherwise a merged profile inflates the
	// denominator and understates the module.
	profile := `mode: count
example.com/repo/internal/leaf/a.go:1.1,2.2 2 0
example.com/repo/internal/leaf/a.go:1.1,2.2 2 7
`
	got, err := CoverageScores([]byte(profile), "example.com/repo")
	if err != nil {
		t.Fatal(err)
	}
	if got["internal/leaf"] != 100 {
		t.Fatalf("internal/leaf coverage = %v, want 100 (repeated block folded once, covered)", got["internal/leaf"])
	}
}

func TestCoverageScoresRejectsMalformedProfile(t *testing.T) {
	cases := map[string]struct{ profile, want string }{
		"no mode header": {"internal/leaf/a.go:1.1,2.2 1 1\n", "mode:"},
		"empty":          {"\n\n", "empty"},
		"header only":    {"mode: set\n", "no block rows"},
		"short row":      {"mode: set\nexample.com/repo/internal/leaf/a.go:1.1,2.2 1\n", "<stmts> <count>"},
		"bad count":      {"mode: set\nexample.com/repo/internal/leaf/a.go:1.1,2.2 1 x\n", "bad execution count"},
		"no module":      {"mode: set\nexample.com/repo/main.go:1.1,2.2 1 1\n", "no versioned module"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := CoverageScores([]byte(tc.profile), "example.com/repo")
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want one containing %q", err, tc.want)
			}
		})
	}
}

func TestCoverageStampRowsCarryCoverageAsScore(t *testing.T) {
	// The done condition, at the library seam: fold -> join -> the delta rows a
	// stamp appends carry the coverage percent as the row's score, labeled
	// witnessed because it was measured off a real run's artifact.
	coverage, err := CoverageScores([]byte(coverageOutFixture), fixtureModulePath)
	if err != nil {
		t.Fatal(err)
	}
	rep := Report{
		Head:       "abcdef12",
		AppVersion: "0.1.0",
		Modules: []Module{
			{Name: "cmd/fak", Kind: "cmd", Rev: 900, LastCommit: "abcdef12", LastDate: "2026-07-26T00:00:00Z"},
			{Name: "internal/modver", Kind: "internal", Rev: 12, LastCommit: "abcdef12", LastDate: "2026-07-26T00:00:00Z"},
			{Name: "internal/gateway", Kind: "internal", Rev: 40, LastCommit: "abcdef12", LastDate: "2026-07-26T00:00:00Z"},
		},
	}
	if matched := rep.JoinScores(CoverageEntries(coverage)); matched != 2 {
		t.Fatalf("joined %d coverage scores, want 2", matched)
	}
	// A module the profile never measured keeps a nil score: an unmeasured module
	// must not read as zero coverage.
	if rep.Modules[2].Score != nil {
		t.Fatalf("internal/gateway score = %v, want nil (never measured)", *rep.Modules[2].Score)
	}

	rows := DeltaRows(rep, nil, "2026-07-26T00:00:00Z")
	byModule := map[string]LedgerRow{}
	for _, r := range rows {
		byModule[r.Module] = r
	}
	row, ok := byModule["internal/modver"]
	if !ok {
		t.Fatalf("no ledger row for internal/modver in %#v", rows)
	}
	if row.Score == nil || *row.Score != 71.4 {
		t.Fatalf("ledger row score = %v, want 71.4", row.Score)
	}
	if row.ScoreProvenance != ProvenanceWitnessed {
		t.Fatalf("ledger row provenance = %q, want %q", row.ScoreProvenance, ProvenanceWitnessed)
	}
	if row.Schema != Schema {
		t.Fatalf("ledger row schema = %q, want %q (additive join, no schema move)", row.Schema, Schema)
	}
	if row.Version != "r12+gabcdef12" {
		t.Fatalf("ledger row version = %q, want r12+gabcdef12", row.Version)
	}

	// Re-stamping the same coverage at the same rev appends nothing: the score
	// join must not manufacture ledger movement.
	lines, err := AppendLines(rows)
	if err != nil {
		t.Fatal(err)
	}
	if again := DeltaRows(rep, lines, "2026-07-26T00:01:00Z"); len(again) != 0 {
		t.Fatalf("re-stamp appended %d rows, want 0", len(again))
	}
}
