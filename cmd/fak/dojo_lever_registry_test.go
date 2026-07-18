package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

// leverSeamProbe is a lever-seam probe registered through the additive
// RegisterLever seam in this separate _test file — the proof that a KPI-cell
// leaf can land its lever + catalog row WITHOUT editing the central
// allDojoLevers / dojoLeverCatalogBase literals in dojo.go (#5108, mirroring
// TestRegisterClaimResolvesViaRegistry in internal/dojo). It carries the env it
// was built with so the fold test can prove the run parameters thread through.
type leverSeamProbe struct{ env dojoLeverEnv }

func (leverSeamProbe) Name() string { return "lever-seam-probe" }

func (leverSeamProbe) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	// An honest empty fold: the probe proves registration plumbing, not scoring.
	return nil, nil
}

var leverSeamProbeInfo = RegisterLever(dojoLeverInfo{
	Name:    "lever-seam-probe",
	Summary: "a lever-seam probe registered additively from a _test file (#5108)",
	Metrics: []dojoMetricInfo{
		{Name: "additive_resolves", Theory: "a lever registered via RegisterLever folds into allDojoLevers and dojoLeverCatalog without editing dojo.go"},
	},
}, func(env dojoLeverEnv) dojo.Lever { return leverSeamProbe{env: env} })

// TestRegisterLeverFoldsIntoLeverSetAndCatalog is the additive-seam witness: a
// lever registered via RegisterLever (in this separate file) is listed by
// dojoLeverCatalog, built by allDojoLevers with the run env threaded through,
// part of the default `dojo run` fold, and selectable via --lever — with no
// edit to the central literals.
func TestRegisterLeverFoldsIntoLeverSetAndCatalog(t *testing.T) {
	if leverSeamProbeInfo.Name != "lever-seam-probe" {
		t.Fatalf("RegisterLever must return the registered info, got %+v", leverSeamProbeInfo)
	}

	var row *dojoLeverInfo
	for _, lv := range dojoLeverCatalog() {
		if lv.Name == "lever-seam-probe" {
			r := lv
			row = &r
			break
		}
	}
	if row == nil {
		t.Fatal("dojoLeverCatalog does not fold the additively registered lever row")
	}
	if len(row.Metrics) != 1 || row.Metrics[0].Name != "additive_resolves" {
		t.Fatalf("folded catalog row lost its metrics: %+v", row)
	}

	all := allDojoLevers("probe-root", resume.TTL5m, 7)
	var got dojo.Lever
	for _, lv := range all {
		if lv.Name() == "lever-seam-probe" {
			got = lv
			break
		}
	}
	if got == nil {
		t.Fatalf("allDojoLevers does not fold the additively registered lever: %v", dojoLeverNames(all))
	}
	probe, ok := got.(leverSeamProbe)
	if !ok {
		t.Fatalf("folded lever is not the registered probe: %T", got)
	}
	if probe.env.Root != "probe-root" || probe.env.TTL != resume.TTL5m || probe.env.MaxFiles != 7 {
		t.Fatalf("run env did not thread through the builder: %+v", probe.env)
	}

	defaults := dojoLeverNames(defaultDojoLevers(".", resume.TTL5m, 0))
	if !dojoHasString(defaults, "lever-seam-probe") {
		t.Fatalf("default dojo levers must fold the registered lever: %v", defaults)
	}

	sel := filterDojoLevers(all, []string{"lever-seam-probe"})
	if len(sel) != 1 || sel[0].Name() != "lever-seam-probe" {
		t.Fatalf("--lever selection should resolve the registered lever, got %v", dojoLeverNames(sel))
	}
}

// TestRegisterLeverRefusesDuplicates pins the loud-collision contract: a
// duplicate lever name — against the additive seam or against the central
// catalog literal — panics at init instead of silently shadowing a lever,
// mirroring RegisterClaim's duplicate refusal.
func TestRegisterLeverRefusesDuplicates(t *testing.T) {
	build := func(dojoLeverEnv) dojo.Lever { return leverSeamProbe{} }

	mustPanic := func(name string, info dojoLeverInfo) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Fatalf("RegisterLever did not panic on %s", name)
			}
		}()
		RegisterLever(info, build)
	}

	// A duplicate of an additively registered lever.
	mustPanic("a duplicate registered lever", dojoLeverInfo{Name: "lever-seam-probe"})
	// A duplicate of a lever in the central catalog literal.
	mustPanic("a lever already in the central catalog", dojoLeverInfo{Name: "resume-posture"})
}
