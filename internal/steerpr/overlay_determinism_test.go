// Overlay-fold determinism (#5035): the overlay is read by a human across
// ticks and by a scorecard that trends osp_residual. Both break if the fold is
// unstable, so these tests pin the design-note claim "the fold is
// deterministic over git history" for the overlay's inputs: band, ordering,
// partial (unstamped) state, and the carried curve.
//
// Determinism here means SAME INPUTS -> SAME OUTPUT. It does not mean frozen
// over time: a different range, or a witness landing, legitimately changes the
// answer. What must never change the answer is a re-run, map-iteration order,
// wall-clock, or ambient env.
package steerpr

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// detLogRecord builds one record in the `git log --no-merges --name-only
// --format=%x1e%H%x1f%s%x1f%b%x1f` wire format the real fold consumes. It is a
// local twin of logRecord so this file stays self-contained against sibling
// test files in the package.
func detLogRecord(sha, subject, body string, files ...string) string {
	return "\x1e" + sha + "\x1f" + subject + "\x1f" + body + "\x1f" + strings.Join(files, "\n")
}

// detPinnedRaw is the pinned range fixture: multiple leaves, deliberate TIES
// (two leaves with equal commit count and equal band), mixed verdicts, issue
// refs in subject and body, and an unstamped orphan so the partial state is in
// the payload too. Ties are the part map iteration would scramble.
func detPinnedRaw() string {
	return strings.Join([]string{
		// git log yields newest-first.
		detLogRecord("e1f", "fix(gateway): treat same-tick ready as positive (#42) (fak gateway)",
			"Body mentions #99.", "internal/gateway/a.go"),
		detLogRecord("d2e", "feat(steerpr): carry the curve (#5038) (fak steerpr)",
			"", "internal/steerpr/curve.go"),
		detLogRecord("c3d", "test(steerpr): prove the band fold (#5034) (fak steerpr)",
			"relates to #5015", "internal/steerpr/band_test.go"),
		// TIE PAIR: cache and dojo both have exactly one commit and identical
		// verdict coverage -> equal band, equal size; only the leaf tiebreak
		// keeps them from swapping between runs.
		detLogRecord("b4c", "docs(dojo): explain the ladder (fak dojo)", "", "docs/dojo.md"),
		detLogRecord("a5b", "docs(cache): explain the other ladder (fak cache)", "", "docs/cache.md"),
		detLogRecord("f6a", "chore: no stamp at all", "see #7", "misc.txt"),
	}, "")
}

// detVerdicts is the caller-supplied witness verdict map for the pinned range,
// keyed by SHA the way `dos commit-audit --json` rows are. Attaching from a
// map (randomized iteration in matchVerdict-style loops) must not reach the
// output because attachment is keyed per-commit, not ordered.
func detVerdicts() map[string]Verdict {
	return map[string]Verdict{
		"e1f": VerdictWitnessed,
		"d2e": VerdictUnwitnessed, // reds the steerpr unit
		"c3d": VerdictWitnessed,
		"b4c": VerdictAbstain,
		"a5b": VerdictAbstain,
		// f6a deliberately ungraded: VerdictUnknown -> UNVERIFIABLE, never CLEARED.
	}
}

// detCurveLookup is a deterministic bound-objective lookup: only the steerpr
// unit carries a curve, so the Curve field's serialization is under test too.
func detCurveLookup(u Unit) (Curve, bool) {
	if u.Leaf != "steerpr" {
		return Curve{}, false
	}
	return Curve{ObjectiveID: "obj-steerpr", Signal: CurveHealthy, Rung: RungW2, Latest: 0.75, Delta: 0.05}, true
}

// detFoldPayload runs the whole overlay pipeline exactly the way
// buildSteerPRsView composes it — parse, attach verdicts, fold, worst-first
// sort, attach curves, marshal — and returns the machine payload bytes. The
// range-identity fields are pinned constants because the range is pinned.
func detFoldPayload(t *testing.T) []byte {
	t.Helper()
	commits := ParseLog(detPinnedRaw())
	verdicts := detVerdicts()
	for i := range commits {
		if v, ok := verdicts[commits[i].SHA]; ok {
			commits[i].Verdict = v
		}
	}
	units, unstamped := FoldUnits(commits)
	SortWorstFirst(units)
	AttachCurves(units, detCurveLookup)
	payload, err := json.Marshal(map[string]any{
		"schema":          Schema,
		"base_sha":        "BASE0000",
		"head_sha":        "HEAD0000",
		"range":           "BASE0000..HEAD0000",
		"commit_count":    len(commits),
		"unit_count":      len(units),
		"unstamped_count": len(unstamped),
		"residual_count":  Residual(units),
		"units":           units,
		"unstamped":       unstamped,
	})
	if err != nil {
		t.Fatalf("marshal payload: %v", err)
	}
	return payload
}

// TestOverlayFoldDeterminismDoubleFoldByteIdentical is the acceptance gate:
// folding the SAME pinned range twice yields byte-identical payloads — same
// units, same order, same bands, same partial state. Run under -count=5
// -shuffle=on to shake out map-order luck; the 50 in-process repeats give
// Go's randomized map iteration room to misbehave within a single run too.
func TestOverlayFoldDeterminismDoubleFoldByteIdentical(t *testing.T) {
	first := detFoldPayload(t)
	for i := 1; i < 50; i++ {
		got := detFoldPayload(t)
		if string(got) != string(first) {
			t.Fatalf("double-fold diverged on re-fold %d:\n first=%s\n   got=%s", i, first, got)
		}
	}
}

// TestOverlayFoldDeterminismOrderingIsTotal pins the ordering as a TOTAL
// order: worst-band-first, then biggest-first, then BY LEAF as the final
// tiebreak — so two units with equal band and equal commit count never swap
// between runs. The fixture's cache/dojo tie is the case map iteration would
// scramble if the tiebreak regressed.
func TestOverlayFoldDeterminismOrderingIsTotal(t *testing.T) {
	wantOrder := []string{
		"steerpr", // RESIDUAL (unwitnessed member), 2 commits — worst first
		"cache",   // UNVERIFIABLE tie pair: leaf tiebreak, alphabetical
		"dojo",
		"gateway", // CLEARED last: attention buys nothing there
	}
	for i := 0; i < 100; i++ {
		commits := ParseLog(detPinnedRaw())
		verdicts := detVerdicts()
		for j := range commits {
			if v, ok := verdicts[commits[j].SHA]; ok {
				commits[j].Verdict = v
			}
		}
		units, _ := FoldUnits(commits)
		SortWorstFirst(units)
		if len(units) != len(wantOrder) {
			t.Fatalf("iteration %d: %d units, want %d", i, len(units), len(wantOrder))
		}
		for k, w := range wantOrder {
			if units[k].Leaf != w {
				t.Fatalf("iteration %d: units[%d].Leaf = %q, want %q (ordering must be a total order; equal band+size ties break by leaf)",
					i, k, units[k].Leaf, w)
			}
		}
		// The tie pair really is a tie: same band, same size. If a fixture edit
		// breaks the tie, this test stops proving the tiebreak — fail loudly.
		if units[1].Band != units[2].Band || len(units[1].Commits) != len(units[2].Commits) {
			t.Fatalf("fixture regression: cache/dojo must tie on band and size (got %s/%d vs %s/%d)",
				units[1].Band, len(units[1].Commits), units[2].Band, len(units[2].Commits))
		}
	}
}

// TestOverlayFoldDeterminismPureFunctionOfRange proves the fold is a pure
// function of the range: two folds separated by wall-clock time and by ambient
// env changes produce identical bytes, and no timestamp-shaped field exists in
// the payload schema for wall-clock to leak through. A generated_at field
// would break byte-identity — if the schema ever needs one it must be
// injectable and excluded here, decided deliberately, not discovered in CI.
func TestOverlayFoldDeterminismPureFunctionOfRange(t *testing.T) {
	before := detFoldPayload(t)

	// Perturb the ambient environment between folds. None of this may reach
	// the payload.
	t.Setenv("TZ", "Pacific/Kiritimati")
	t.Setenv("SOURCE_DATE_EPOCH", "0")
	t.Setenv("FAK_STEERPR_DETERMINISM_PROBE", "perturbed")

	after := detFoldPayload(t)
	if string(before) != string(after) {
		t.Fatalf("fold is not a pure function of the range: env/wall-clock leaked\n before=%s\n  after=%s", before, after)
	}

	// Schema pin: every serialized Unit and Commit key must come from the
	// closed sets below. A new time-bearing field (generated_at, seen_at, ...)
	// shows up here as a failing key, making the determinism decision explicit
	// at the field's birth instead of as trend noise later.
	// grouped_by / leaves (#5040) are pinned here deliberately: both are pure
	// functions of the range plus the caller's wave bindings — no clock, no env —
	// and grouped_by is unconditional precisely because a unit that does not say
	// which basis it was grouped on is the failure that issue fences out.
	unitKeys := map[string]bool{
		"leaf": true, "grouped_by": true, "leaves": true,
		"title": true, "commits": true, "types": true,
		"resolves": true, "mentions": true, "files": true, "band": true, "curve": true,
	}
	commitKeys := map[string]bool{
		"sha": true, "subject": true, "leaf": true, "type": true,
		"resolves": true, "mentions": true, "files": true, "verdict": true, "band": true,
	}
	curveKeys := map[string]bool{
		"objective_id": true, "signal": true, "rung": true,
		"latest": true, "delta": true, "detail": true,
	}
	var decoded struct {
		Units []map[string]json.RawMessage `json:"units"`
	}
	if err := json.Unmarshal(after, &decoded); err != nil {
		t.Fatalf("unmarshal payload: %v", err)
	}
	if len(decoded.Units) == 0 {
		t.Fatal("payload has no units; fixture must exercise the schema pin")
	}
	for _, u := range decoded.Units {
		for key := range u {
			if !unitKeys[key] {
				t.Errorf("Unit serializes unknown key %q: pin it into the total order or keep it out of the payload", key)
			}
		}
		if rawCommits, ok := u["commits"]; ok {
			var cs []map[string]json.RawMessage
			if err := json.Unmarshal(rawCommits, &cs); err != nil {
				t.Fatalf("unmarshal commits: %v", err)
			}
			for _, c := range cs {
				for key := range c {
					if !commitKeys[key] {
						t.Errorf("Commit serializes unknown key %q: pin it into the total order or keep it out of the payload", key)
					}
				}
			}
		}
		if rawCurve, ok := u["curve"]; ok {
			var cv map[string]json.RawMessage
			if err := json.Unmarshal(rawCurve, &cv); err != nil {
				t.Fatalf("unmarshal curve: %v", err)
			}
			for key := range cv {
				if !curveKeys[key] {
					t.Errorf("Curve serializes unknown key %q: pin it into the total order or keep it out of the payload", key)
				}
			}
		}
	}
}

// TestOverlayFoldDeterminismReTickIsQuiescent proves re-tick equivalence: the
// maintenance loop re-ticking over an UNCHANGED range produces no spurious
// state change. Tick 2 re-parses the same range but carries tick 1's folded
// bands as the cached Band on each commit (the supplied-band path through
// commitBand); the pessimistic reconcile of a band with the verdict that
// produced it must be a fixed point — identical units, identical bytes.
func TestOverlayFoldDeterminismReTickIsQuiescent(t *testing.T) {
	fold := func(banded map[string]Band) ([]Unit, []Commit) {
		commits := ParseLog(detPinnedRaw())
		verdicts := detVerdicts()
		for i := range commits {
			if v, ok := verdicts[commits[i].SHA]; ok {
				commits[i].Verdict = v
			}
			if banded != nil {
				commits[i].Band = banded[commits[i].SHA]
			}
		}
		units, unstamped := FoldUnits(commits)
		SortWorstFirst(units)
		return units, unstamped
	}

	// Tick 1: fresh fold.
	units1, unstamped1 := fold(nil)

	// Cache tick 1's per-commit bands, the way a persisted overlay would.
	cached := map[string]Band{}
	for _, u := range units1 {
		for _, c := range u.Commits {
			cached[c.SHA] = c.Band
		}
	}
	for _, c := range unstamped1 {
		cached[c.SHA] = c.Band
	}

	// Tick 2: same range, cached bands supplied.
	units2, unstamped2 := fold(cached)

	if !reflect.DeepEqual(units1, units2) {
		t.Errorf("re-tick changed the units over an unchanged range:\n tick1=%+v\n tick2=%+v", units1, units2)
	}
	if !reflect.DeepEqual(unstamped1, unstamped2) {
		t.Errorf("re-tick changed the unstamped set over an unchanged range:\n tick1=%+v\n tick2=%+v", unstamped1, unstamped2)
	}
	pay := func(units []Unit, unstamped []Commit) string {
		b, err := json.Marshal(map[string]any{"schema": Schema, "units": units, "unstamped": unstamped})
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		return string(b)
	}
	if p1, p2 := pay(units1, unstamped1), pay(units2, unstamped2); p1 != p2 {
		t.Errorf("re-tick payload diverged:\n tick1=%s\n tick2=%s", p1, p2)
	}
	// And the headline number the scorecard trends is stable too.
	if r1, r2 := Residual(units1), Residual(units2); r1 != r2 {
		t.Errorf("re-tick moved osp_residual over an unchanged range: %d -> %d", r1, r2)
	}
}
