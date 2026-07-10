//go:build wip_sessionfleet

// GATED WIP — twin of info_fleet.go: depends on the not-yet-committed gateway.SessionFleet
// surface, fenced behind //go:build wip_sessionfleet so the default build/test/vet stays green.
// Remove this line once gateway.SessionFleet lands.
package main

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/claimcheck"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func fleetFixture() *gateway.SessionFleet {
	return &gateway.SessionFleet{
		Verdict:           "ACTION",
		Machines:          3,
		Stale:             1,
		Action:            1,
		Sessions:          9,
		AuthBlocked:       2,
		VersionMismatches: 1,
		Rows: []gateway.SessionFleetMachine{
			{ID: "alpha", State: "OK", AgeMin: 3, Sessions: 5, Version: "v9.9.9"},
			{ID: "beta", State: "STALE", AgeMin: 120},
			{ID: "gamma", State: "ACTION"},
		},
	}
}

func TestGuardInfoFleetPanelFull(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{Fleet: fleetFixture()}, width: 120}
	rows := guardInfoFleetPanelRows(ctx, guardPanelFull)
	joined := strings.Join(rows, "\n")
	// Head: verdict word + machine count + the action/stale breakdown.
	if !strings.Contains(joined, "action") || !strings.Contains(joined, "3 machines") {
		t.Fatalf("fleet head missing verdict/machine count:\n%s", joined)
	}
	if !strings.Contains(joined, "1 action") || !strings.Contains(joined, "1 stale") {
		t.Fatalf("fleet head missing action/stale breakdown:\n%s", joined)
	}
	// Totals row rolls up sessions / auth-blocked / version-skew.
	if !strings.Contains(joined, "9 sessions") || !strings.Contains(joined, "2 auth-blocked") || !strings.Contains(joined, "1 version-skew") {
		t.Fatalf("fleet totals row missing:\n%s", joined)
	}
	// Machine rows: id, state, age, sessions, version.
	if !strings.Contains(joined, "alpha (OK)") || !strings.Contains(joined, "5 sess") || !strings.Contains(joined, "v9.9.9") {
		t.Fatalf("alpha machine row missing detail:\n%s", joined)
	}
	// 120 minutes folds to "2h".
	if !strings.Contains(joined, "beta (STALE)") || !strings.Contains(joined, "2h") {
		t.Fatalf("beta machine row / age fold missing:\n%s", joined)
	}
}

func TestGuardInfoFleetPanelMini(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{Fleet: fleetFixture()}, width: 80}
	rows := guardInfoFleetPanelRows(ctx, guardPanelMini)
	if len(rows) != 1 {
		t.Fatalf("mini fleet = %d rows, want exactly 1", len(rows))
	}
	line := rows[0]
	if !strings.Contains(line, "3 machines") || !strings.Contains(line, "2 need attention") {
		t.Fatalf("mini fleet line = %q, want machine count + attention fold", line)
	}
}

func TestGuardInfoFleetPanelSilentWhenAbsent(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{}, width: 80}
	if rows := guardInfoFleetPanelRows(ctx, guardPanelFull); rows != nil {
		t.Fatalf("fleet panel with no block = %v, want nil (silent)", rows)
	}
	// A present-but-zero-machine block is also silent.
	ctx = guardInfoPanelCtx{v: guardInfoVars{Fleet: &gateway.SessionFleet{}}, width: 80}
	if rows := guardInfoFleetPanelRows(ctx, guardPanelFull); rows != nil {
		t.Fatalf("fleet panel with 0 machines = %v, want nil (silent)", rows)
	}
}

func TestGuardFleetAgeText(t *testing.T) {
	cases := map[float64]string{0.4: "<1m", 5: "5m", 59: "59m", 60: "1h", 90: "1h 30m", 120: "2h"}
	for in, want := range cases {
		if got := guardFleetAgeText(in); got != want {
			t.Fatalf("guardFleetAgeText(%v) = %q, want %q", in, got, want)
		}
	}
}

// TestFleetSourceProvenanceRule pins the rule the pane encodes (issue #3605): a poll-sourced
// figure (relayed from a peer /debug/vars snapshot) is OBSERVED, a git-sourced figure (read
// from local git by fak) is WITNESSED.
func TestFleetSourceProvenanceRule(t *testing.T) {
	if got := fleetSourceProvenance(fleetSourcePoll); got != claimcheck.Observed {
		t.Fatalf("poll-sourced provenance = %q, want %q", got, claimcheck.Observed)
	}
	if got := fleetSourceProvenance(fleetSourceGit); got != claimcheck.Witnessed {
		t.Fatalf("git-sourced provenance = %q, want %q", got, claimcheck.Witnessed)
	}
}

// TestFleetMetricPollFieldsObserved asserts every declared poll-sourced fleet figure resolves
// to the OBSERVED label (none is a fak-authored WITNESSED number today).
func TestFleetMetricPollFieldsObserved(t *testing.T) {
	for field, src := range fleetMetricProvenance {
		if src != fleetSourcePoll {
			continue
		}
		p, ok := fleetFieldProvenance(field)
		if !ok || p != claimcheck.Observed {
			t.Fatalf("fleetFieldProvenance(%q) = %q,%v; want %q,true", field, p, ok, claimcheck.Observed)
		}
	}
}

// TestFleetProvenanceRendersDistinctly asserts an OBSERVED figure and a WITNESSED figure
// render with different, non-empty glyphs, an unlabeled provenance renders nothing, and the
// live pane figure for a poll-sourced field carries the OBSERVED glyph.
func TestFleetProvenanceRendersDistinctly(t *testing.T) {
	obs := fleetProvenanceGlyph(claimcheck.Observed)
	wit := fleetProvenanceGlyph(claimcheck.Witnessed)
	if obs == "" || wit == "" {
		t.Fatalf("provenance glyphs must be non-empty: observed=%q witnessed=%q", obs, wit)
	}
	if obs == wit {
		t.Fatalf("observed and witnessed glyphs must render distinctly, both = %q", obs)
	}
	if none := fleetProvenanceGlyph(claimcheck.ProvNone); none != "" {
		t.Fatalf("unlabeled provenance glyph = %q, want empty", none)
	}
	// The SAME number renders differently under the two trust classes.
	if a, b := fmt.Sprintf("3 units%s", obs), fmt.Sprintf("3 units%s", wit); a == b {
		t.Fatalf("observed vs witnessed figures render identically: %q", a)
	}
	// A live poll-sourced figure carries the observed glyph as its suffix.
	if got := fleetFigure(3, "machines", "Machines"); !strings.HasSuffix(got, obs) {
		t.Fatalf("fleetFigure(Machines) = %q, want observed-glyph %q suffix", got, obs)
	}
}

// TestFleetMetricProvenanceGateEveryFigureDeclared is the detection→gate: it reflects over
// gateway.SessionFleet and FAILS if any integer figure field is missing a provenance
// declaration (or a stale declaration names a field the struct no longer has), so a new
// fleet metric can never ship unlabeled.
func TestFleetMetricProvenanceGateEveryFigureDeclared(t *testing.T) {
	ft := reflect.TypeOf(gateway.SessionFleet{})
	for i := 0; i < ft.NumField(); i++ {
		f := ft.Field(i)
		if f.Type.Kind() != reflect.Int {
			continue // only integer count figures carry a provenance glyph
		}
		if _, ok := fleetMetricProvenance[f.Name]; !ok {
			t.Fatalf("SessionFleet figure field %q has no provenance in fleetMetricProvenance "+
				"(add fleetSourcePoll for /debug/vars-polled, fleetSourceGit for git-witnessed)", f.Name)
		}
	}
	for name := range fleetMetricProvenance {
		if _, ok := ft.FieldByName(name); !ok {
			t.Fatalf("fleetMetricProvenance declares %q but gateway.SessionFleet has no such field", name)
		}
	}
}

// TestFleetPanelRendersObservedGlyphs proves the wiring end-to-end: the rendered full pane
// tags its poll-sourced figures with the OBSERVED glyph.
func TestFleetPanelRendersObservedGlyphs(t *testing.T) {
	ctx := guardInfoPanelCtx{v: guardInfoVars{Fleet: fleetFixture()}, width: 120}
	joined := strings.Join(guardInfoFleetPanelRows(ctx, guardPanelFull), "\n")
	obs := fleetProvenanceGlyph(claimcheck.Observed)
	for _, want := range []string{"3 machines" + obs, "9 sessions" + obs, "1 version-skew" + obs} {
		if !strings.Contains(joined, want) {
			t.Fatalf("fleet pane missing observed-tagged figure %q:\n%s", want, joined)
		}
	}
}
