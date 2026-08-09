package modver

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// maturityScorecardFixture is a real-shaped `fak maturity --json` payload. It
// carries one capability at each interesting point of the ladder: the top rung,
// a mid-ladder leaf, a bare prototype, a never-started capability, and a
// LADDER-SKIP (fak runs it but it has no tests) — the one row whose grade is
// docked rather than merely low.
const maturityScorecardFixture = `{
  "schema": "fak-maturity-scorecard/1",
  "ok": false,
  "corpus": {
    "score": 20,
    "ladder": ["proposed", "prototyped", "tested", "dogfooded", "default"]
  },
  "capabilities": [
    {"lane": "modver",    "dir": "internal/modver",    "rung": 4, "top_evidence": 4, "skip": false},
    {"lane": "gateway",   "dir": "internal/gateway",   "rung": 2, "top_evidence": 2, "skip": false},
    {"lane": "ctxplan",   "dir": "internal/ctxplan",   "rung": 1, "top_evidence": 1, "skip": false},
    {"lane": "blob",      "dir": "internal/blob",      "rung": 0, "top_evidence": 0, "skip": false},
    {"lane": "laneadmit", "dir": "internal/laneadmit", "rung": 1, "top_evidence": 3, "skip": true}
  ]
}`

func TestMaturityScoresKeyLadderGradesByModule(t *testing.T) {
	got, err := MaturityScores([]byte(maturityScorecardFixture))
	if err != nil {
		t.Fatal(err)
	}
	// Ladder position as a percentage of the top rung (4), minus a whole
	// capability for the ladder-skip: internal/laneadmit sits at rung 1 (25) but
	// carries high-rung evidence over an unmet lower rung, so it is charged 100.
	want := map[string]float64{
		"internal/modver":    100,
		"internal/gateway":   50,
		"internal/ctxplan":   25,
		"internal/blob":      0,
		"internal/laneadmit": -75,
	}
	if len(got) != len(want) {
		t.Fatalf("maturity scores = %#v, want %#v", got, want)
	}
	for module, score := range want {
		if got[module] != score {
			t.Errorf("%s maturity score = %v, want %v", module, got[module], score)
		}
	}

	// The fold must feed the existing --scores decoder unchanged: same flat
	// {module: number} shape, so no schema move is involved.
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
	if joined["internal/modver"].Score != 100 {
		t.Fatalf("joined modver score = %v, want 100", joined["internal/modver"].Score)
	}
}

func TestMaturityScoresMeanReproducesFleetIndex(t *testing.T) {
	// The anti-drift invariant: this adapter RE-KEYS internal/maturity's grade, it
	// does not mint a second maturity scale. internal/maturity's published fleet
	// index is
	//
	//	100*sum(rung)/(n*max) - 100*skips/n
	//
	// which is exactly the arithmetic mean of the per-module scores this adapter
	// emits. If either side ever changes its formula, this test reds.
	got, err := MaturityScores([]byte(maturityScorecardFixture))
	if err != nil {
		t.Fatal(err)
	}
	var fixture MaturityScorecard
	if err := json.Unmarshal([]byte(maturityScorecardFixture), &fixture); err != nil {
		t.Fatal(err)
	}
	n, maxRung := len(fixture.Capabilities), len(fixture.Corpus.Ladder)-1
	rungSum, skips := 0, 0
	for _, row := range fixture.Capabilities {
		rungSum += row.Rung
		if row.Skip {
			skips++
		}
	}
	wantIndex := 100*float64(rungSum)/(float64(n)*float64(maxRung)) - 100*float64(skips)/float64(n)

	total := 0.0
	for _, score := range got {
		total += score
	}
	mean := total / float64(len(got))
	if math.Abs(mean-wantIndex) > 1e-9 {
		t.Fatalf("mean module maturity score = %v, want the fleet index %v", mean, wantIndex)
	}

	// And the fixture's own published headline agrees with that mean, rounded the
	// way internal/maturity rounds it — so the per-module column and the
	// scorecard's one-number grade cannot tell an operator different things.
	var published struct {
		Corpus struct {
			Score int `json:"score"`
		} `json:"corpus"`
	}
	if err := json.Unmarshal([]byte(maturityScorecardFixture), &published); err != nil {
		t.Fatal(err)
	}
	if rounded := int(mean + 0.5); rounded != published.Corpus.Score {
		t.Fatalf("rounded mean = %d, want the payload's published score %d", rounded, published.Corpus.Score)
	}
}

func TestMaturityScoresRescalesWithTheLadder(t *testing.T) {
	// The top rung is read from the payload, never hardcoded: a capability at the
	// top of a 3-rung ladder is 100, not 50. Hardcoding max=4 would silently
	// halve every module the day internal/maturity adds or drops a rung.
	short := `{"corpus":{"ladder":["proposed","prototyped","tested"]},
	           "capabilities":[{"dir":"internal/leaf","rung":2,"skip":false},
	                           {"dir":"internal/other","rung":1,"skip":false}]}`
	got, err := MaturityScores([]byte(short))
	if err != nil {
		t.Fatal(err)
	}
	if got["internal/leaf"] != 100 {
		t.Errorf("internal/leaf = %v, want 100 (top of a 3-rung ladder)", got["internal/leaf"])
	}
	if got["internal/other"] != 50 {
		t.Errorf("internal/other = %v, want 50 (mid of a 3-rung ladder)", got["internal/other"])
	}
}

func TestMaturityScoresRejectsUnjoinableRows(t *testing.T) {
	const ladder = `"corpus":{"ladder":["proposed","prototyped","tested","dogfooded","default"]}`
	cases := map[string]struct{ payload, want string }{
		"not json": {`{`, "decode maturity scorecard"},
		"no capabilities": {`{` + ladder + `,"capabilities":[]}`,
			"no capabilities"},
		"no ladder": {`{"capabilities":[{"dir":"internal/leaf","rung":1}]}`,
			"no rung ladder"},
		"one-rung ladder": {`{"corpus":{"ladder":["proposed"]},"capabilities":[{"dir":"internal/leaf","rung":0}]}`,
			"no rung ladder"},
		"empty dir": {`{` + ladder + `,"capabilities":[{"dir":"  ","rung":1}]}`,
			"empty dir"},
		"unkeyable dir": {`{` + ladder + `,"capabilities":[{"dir":"docs/nightrun/module-versions.jsonl","rung":1}]}`,
			"cannot map maturity capability dir"},
		// A subpackage dir would classify to its PARENT module; folding it there
		// would attribute a subpackage's grade to the leaf that contains it.
		"subpackage dir": {`{` + ladder + `,"capabilities":[{"dir":"internal/leaf/sub","rung":1}]}`,
			"cannot map maturity capability dir"},
		"duplicate module": {`{` + ladder + `,"capabilities":[{"dir":"internal/leaf","rung":1},{"dir":"internal/leaf/","rung":2}]}`,
			"duplicate maturity capability"},
		"rung above ladder": {`{` + ladder + `,"capabilities":[{"dir":"internal/leaf","rung":9}]}`,
			"outside the ladder"},
		"negative rung": {`{` + ladder + `,"capabilities":[{"dir":"internal/leaf","rung":-1}]}`,
			"outside the ladder"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := MaturityScores([]byte(tc.payload))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want one containing %q", err, tc.want)
			}
		})
	}
}

func TestMaturityStampRowsCarryLadderGradeAsScore(t *testing.T) {
	// The done condition, at the library seam: fold -> join -> the delta rows a
	// stamp appends carry each leaf's ladder grade as the row's score, labeled
	// witnessed because every rung is re-derived from evidence on disk.
	scores, err := MaturityScores([]byte(maturityScorecardFixture))
	if err != nil {
		t.Fatal(err)
	}
	rep := Report{
		Head:       "abcdef12",
		AppVersion: "0.1.0",
		Modules: []Module{
			{Name: "cmd/fak", Kind: "cmd", Rev: 900, LastCommit: "abcdef12", LastDate: "2026-07-27T00:00:00Z"},
			{Name: "internal/laneadmit", Kind: "internal", Rev: 7, LastCommit: "abcdef12", LastDate: "2026-07-27T00:00:00Z"},
			{Name: "internal/modver", Kind: "internal", Rev: 12, LastCommit: "abcdef12", LastDate: "2026-07-27T00:00:00Z"},
		},
	}
	// cmd/fak is not a graded capability (the roster is internal/<leaf> lanes), so
	// it keeps a nil score: ungraded must not read as "graded at zero".
	if matched := rep.JoinScores(MaturityEntries(scores)); matched != 2 {
		t.Fatalf("joined %d maturity scores, want 2", matched)
	}
	if rep.Modules[0].Score != nil {
		t.Fatalf("cmd/fak score = %v, want nil (never graded)", *rep.Modules[0].Score)
	}

	rows := DeltaRows(rep, nil, "2026-07-27T00:00:00Z")
	byModule := map[string]LedgerRow{}
	for _, r := range rows {
		byModule[r.Module] = r
	}
	row, ok := byModule["internal/modver"]
	if !ok {
		t.Fatalf("no ledger row for internal/modver in %#v", rows)
	}
	if row.Score == nil || *row.Score != 100 {
		t.Fatalf("ledger row score = %v, want 100", row.Score)
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
	// The ladder-skip's negative grade must survive into the ledger: clamping it
	// at zero would make an overclaiming module read as merely immature.
	skipped, ok := byModule["internal/laneadmit"]
	if !ok {
		t.Fatalf("no ledger row for internal/laneadmit in %#v", rows)
	}
	if skipped.Score == nil || *skipped.Score != -75 {
		t.Fatalf("ladder-skip row score = %v, want -75", skipped.Score)
	}

	// Re-stamping the same grades at the same rev appends nothing: the score join
	// must not manufacture ledger movement.
	lines, err := AppendLines(rows)
	if err != nil {
		t.Fatal(err)
	}
	if again := DeltaRows(rep, lines, "2026-07-27T00:01:00Z"); len(again) != 0 {
		t.Fatalf("re-stamp appended %d rows, want 0", len(again))
	}
}
