package main

// steer_prs_cohort_test.go — issue #5040's operator surface: `fak steer prs
// --cohort PLAN.json` regroups the commits bound to a planned cohort wave into
// one unit per wave, leaves everything else on the leaf fallback, and states the
// basis on every unit. These tests drive the real CLI entry point over the
// existing fake-git seam, so what is witnessed is what an operator would see.

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// steerCohortLog is the MIXED case the acceptance gate names: two wave-bound
// commits in DIFFERENT leaves (#701 gateway, #702 model — one wave), and one
// commit with no wave binding at all (#900 bench), which must stay leaf-grouped.
const steerCohortLog = "\x1eaaa1111111111111111111111111111111111111\x1ffeat(gateway): arm the cold-tool lever (#701) (fak gateway)\x1f\x1f\ninternal/gateway/g.go\n" +
	"\x1ebbb2222222222222222222222222222222222222\x1ffix(model): unbreak the decode path (#702) (fak model)\x1f\x1f\ninternal/model/m.go\n" +
	"\x1eccc3333333333333333333333333333333333333\x1ffeat(bench): add the inventory row (#900) (fak bench)\x1f\x1f\ninternal/bench/b.go\n"

// steerCohortPlan writes a real issuecohort.Plan to disk — the same type `fak
// issue cohort --json` emits — so the overlay is proven to read the EXISTING
// plan rather than some test-only shape.
func steerCohortPlan(t *testing.T, waves ...[]int) string {
	t.Helper()
	plan := issuecohort.Plan{Schema: issuecohort.Schema}
	for i, members := range waves {
		w := issuecohort.Wave{Index: i, Size: len(members)}
		for _, n := range members {
			w.Members = append(w.Members, issuecohort.WaveMember{
				Key:         "fanout-x-" + string(rune('a'+n%26)),
				IssueNumber: n,
			})
		}
		plan.Waves = append(plan.Waves, w)
	}
	plan.NumWaves = len(plan.Waves)
	raw, err := json.Marshal(plan)
	if err != nil {
		t.Fatalf("marshal plan: %v", err)
	}
	path := filepath.Join(t.TempDir(), "cohort.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write plan: %v", err)
	}
	return path
}

// TestSteerPRsCohortFoldsWaveIntoOneUnit is the acceptance gate on the operator
// surface: the wave-bound commits from two leaves land in ONE wave unit, the
// unbound commit keeps its leaf unit, and both units say which basis they used.
func TestSteerPRsCohortFoldsWaveIntoOneUnit(t *testing.T) {
	withSteerFakes(t, steerCohortLog, steerpr.VerdictWitnessed)
	plan := steerCohortPlan(t, []int{701, 702})

	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--json", "--cohort", plan, "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	var payload struct {
		WaveUnitCount int `json:"wave_unit_count"`
		Units         []struct {
			Leaf      string   `json:"leaf"`
			GroupedBy string   `json:"grouped_by"`
			Leaves    []string `json:"leaves"`
			Commits   []struct {
				SHA string `json:"sha"`
			} `json:"commits"`
		} `json:"units"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("decode json: %v\n%s", err, stdout.String())
	}
	if payload.WaveUnitCount != 1 {
		t.Fatalf("wave_unit_count = %d, want 1", payload.WaveUnitCount)
	}
	if len(payload.Units) != 2 {
		t.Fatalf("units = %#v, want 2 (one wave unit + one leaf fallback)", payload.Units)
	}
	var wave, leaf int
	for _, u := range payload.Units {
		switch u.GroupedBy {
		case "wave":
			wave++
			if u.Leaf != steerpr.WaveKey(0) {
				t.Fatalf("wave unit keyed %q, want %q", u.Leaf, steerpr.WaveKey(0))
			}
			if len(u.Commits) != 2 {
				t.Fatalf("wave unit holds %d commit(s), want the 2 wave-bound ones", len(u.Commits))
			}
			if !strings.Contains(strings.Join(u.Leaves, ","), "gateway") || !strings.Contains(strings.Join(u.Leaves, ","), "model") {
				t.Fatalf("wave unit leaves = %v, want it to span gateway and model", u.Leaves)
			}
		case "leaf":
			leaf++
			if u.Leaf != "bench" {
				t.Fatalf("leaf fallback unit = %q, want bench (the unbound commit)", u.Leaf)
			}
		default:
			t.Fatalf("unit %q has grouped_by %q, want wave or leaf — the basis is mandatory", u.Leaf, u.GroupedBy)
		}
	}
	if wave != 1 || leaf != 1 {
		t.Fatalf("grouping split = %d wave / %d leaf, want 1 / 1", wave, leaf)
	}
}

// TestSteerPRsWithoutCohortStaysLeafGrouped: the fallback is the default. With
// no --cohort the very same range folds by leaf, and every unit still says so.
func TestSteerPRsWithoutCohortStaysLeafGrouped(t *testing.T) {
	withSteerFakes(t, steerCohortLog, steerpr.VerdictWitnessed)

	view, err := buildSteerPRsView(t.TempDir(), "baseref", "headref")
	if err != nil {
		t.Fatalf("buildSteerPRsView: %v", err)
	}
	if got := view["wave_unit_count"].(int); got != 0 {
		t.Fatalf("wave_unit_count = %d, want 0 without a cohort plan", got)
	}
	units := view["units"].([]steerpr.Unit)
	if len(units) != 3 {
		t.Fatalf("units = %d, want 3 leaf units (gateway, model, bench)", len(units))
	}
	for _, u := range units {
		if u.GroupedBy != steerpr.GroupedByLeaf {
			t.Fatalf("unit %q grouped_by = %q, want leaf", u.Leaf, u.GroupedBy)
		}
	}
}

// TestSteerPRsCohortRenderNamesTheBasis: the basis is visible in the human
// render on EVERY unit, plus a count up front. An operator who cannot tell why
// a unit holds what it holds is the failure this issue fences out, so the render
// is part of the gate, not a nicety.
func TestSteerPRsCohortRenderNamesTheBasis(t *testing.T) {
	withSteerFakes(t, steerCohortLog, steerpr.VerdictWitnessed)
	plan := steerCohortPlan(t, []int{701, 702})

	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--cohort", plan, "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"1 unit(s) grouped by cohort WAVE",
		"## [CLEARED] wave:0 — 2 commit(s) · grouped_by: wave",
		"Wave spans 2 lane(s): gateway, model.",
		// bench carries no verdict in the fixture, so it bands UNVERIFIABLE — and
		// it still states its basis. The basis is printed for every unit, not only
		// for the novel wave ones.
		"## [UNVERIFIABLE] bench — 1 commit(s) · grouped_by: leaf",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q:\n%s", want, out)
		}
	}
}

// TestSteerPRsCohortWaveBandFoldsWorstMember: a wave unit is as pessimistic as a
// leaf unit. One unwitnessed member reds the whole wave — regrouping must never
// be a way to clear a band.
func TestSteerPRsCohortWaveBandFoldsWorstMember(t *testing.T) {
	withSteerFakes(t, steerCohortLog, steerpr.VerdictUnwitnessed)
	plan := steerCohortPlan(t, []int{701, 702})

	view, err := buildSteerPRsViewWaves(t.TempDir(), "baseref", "headref", []steerpr.WaveBinding{{Index: 0, Issues: []string{"#701", "#702"}}})
	if err != nil {
		t.Fatalf("buildSteerPRsViewWaves: %v", err)
	}
	units := view["units"].([]steerpr.Unit)
	if units[0].Leaf != steerpr.WaveKey(0) || units[0].Band != steerpr.BandResidual {
		t.Fatalf("worst-first unit = %q band %q, want wave:0 RESIDUAL", units[0].Leaf, units[0].Band)
	}
	if view["residual_count"].(int) != 1 {
		t.Fatalf("residual_count = %v, want 1", view["residual_count"])
	}
	// And the same through the CLI, so the flag path is not a second code path.
	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--check", "--cohort", plan, "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("--check exit = %d, want 1 (the wave unit is RESIDUAL); stderr=%s", code, stderr.String())
	}
}

// TestSteerPRsCohortRefusesAnUnbindablePlan: the three ways an operator could
// ask for the wave view and silently be handed the leaf view instead are all
// errors, not quiet fallbacks. Being shown a leaf view you believe is a wave
// view is exactly the confusion this issue removes.
func TestSteerPRsCohortRefusesAnUnbindablePlan(t *testing.T) {
	withSteerFakes(t, steerCohortLog, steerpr.VerdictWitnessed)
	dir := t.TempDir()

	missing := filepath.Join(dir, "nope.json")
	unparseable := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(unparseable, []byte("not json"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	// A cohort planned over not-yet-filed candidates: real waves, no issue
	// numbers, so nothing a landed commit could bind to.
	numberless := steerCohortPlan(t, []int{0})

	for name, path := range map[string]string{
		"missing":     missing,
		"unparseable": unparseable,
		"numberless":  numberless,
	} {
		var stdout, stderr bytes.Buffer
		code := runSteerPRs(&stdout, &stderr, []string{"--cohort", path, "--base", "baseref", "--head", "headref"})
		if code != 2 {
			t.Fatalf("%s: exit = %d, want 2 (refuse rather than fall back to leaf); stdout=%s", name, code, stdout.String())
		}
		if !strings.Contains(stderr.String(), "cohort plan") {
			t.Fatalf("%s: refusal should name the cohort plan: %s", name, stderr.String())
		}
	}
}
