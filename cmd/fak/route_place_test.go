package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Witnesses for `fak route --place`, the placement oracle of epic #5416.
//
// The property under test is not "the ladder walks" — internal/modelroute owns that —
// but that the SURFACE an operator reads cannot overstate what fak knows. Two ways it
// could: by descending to a cheap rung on a capability nobody measured, and by printing
// the unmeasured default as though it were a grade. Both are witnessed below.

// routePlaceRoster is a three-rung roster in miniature: one on-box model, one the org
// runs, one a vendor runs — plus the two binding shapes that are legacy SPELLINGS of a
// model already in the pool rather than additional hardware.
func routePlaceRoster() modelroute.Roster {
	return modelroute.Roster{
		Version: "1",
		Accounts: []modelroute.Account{
			{ID: "box", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
			{ID: "corp", Kind: modelroute.KindFleet, BaseURL: "http://glm.infer.corp.internal:8000/v1", CredEnv: "FAK_CORP_TOKEN"},
			{ID: "lab", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_API_KEY"},
		},
		Bindings: []modelroute.Binding{
			{Model: "rung-device", Account: "box", UpstreamModel: "qwen/qwen3.6-4b"},
			{Model: "rung-fleet", Account: "corp", UpstreamModel: "glm-5.2"},
			{Model: "rung-vendor", Account: "lab", UpstreamModel: "gpt-frontier"},
			{Model: "rung-vendor-oldname", Account: "lab", CompatibilityOnly: true},
			{Model: "rung-vendor-legacy", Account: "lab", DeprecatedAliasFor: "rung-vendor"},
		},
	}
}

// routePlaceRun invokes the oracle and returns (exit, stdout, stderr).
func routePlaceRun(t *testing.T, roster *modelroute.Roster, labels map[string]string, capSpec string, asJSON bool) (int, string, string) {
	t.Helper()
	return routePlaceRunOpts(t, roster, labels, placeOptions{CapSpec: capSpec, JSON: asJSON})
}

// routePlaceRunOpts is the same, for the cases that exercise more than --capability.
func routePlaceRunOpts(t *testing.T, roster *modelroute.Roster, labels map[string]string, opts placeOptions) (int, string, string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	code := runRoutePlace(&out, &errBuf, roster, modelroute.Subject{Labels: labels}, opts)
	return code, out.String(), errBuf.String()
}

func TestPlaceWithoutARosterIsAUsageErrorNotAGuess(t *testing.T) {
	code, out, errOut := routePlaceRun(t, nil, map[string]string{"work_class": "routine"}, "", false)
	if code != 2 {
		t.Fatalf("--place with no roster: exit = %d, want 2 (usage)", code)
	}
	if out != "" {
		t.Fatalf("a usage error printed a placement on stdout: %q", out)
	}
	if !strings.Contains(errOut, "--accounts") {
		t.Fatalf("the error does not say what is missing: %q", errOut)
	}
}

func TestAnUnmeasuredCandidateCannotWinACheapRungAtTheCLI(t *testing.T) {
	r := routePlaceRoster()
	code, out, errOut := routePlaceRun(t, &r, map[string]string{"work_class": "routine"}, "", false)
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	// Routine work, an on-box model bound and available — and it still goes to the
	// vendor, because nothing has measured what the on-box model can do. That is the
	// honest answer, and the surface has to be willing to give it.
	if !strings.Contains(out, "zone=vendor") {
		t.Fatalf("unmeasured candidates descended the ladder:\n%s", out)
	}
	if !strings.Contains(out, "self-hosted  no") {
		t.Fatalf("placement is not reported as off self-hosted hardware:\n%s", out)
	}
	for _, want := range []string{
		modelroute.ReasonZoneUnmeasured,    // the device and fleet rungs said why they lost
		modelroute.ReasonTopRungUnmeasured, // and the vendor rung said what it was admitted on
		modelroute.ReasonEscalatedPast,     // an escalation, not an absence of options
		"0 with a MEASURED capability",     // the count, so the cause is countable
		"--capability model=t0|t1|t2",      // and the lever that changes it
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output is missing %q:\n%s", want, out)
		}
	}
}

func TestDeclaringAMeasuredCapabilityMovesTheWorkOntoTheDevice(t *testing.T) {
	r := routePlaceRoster()
	code, out, errOut := routePlaceRun(t, &r,
		map[string]string{"work_class": "routine"}, "rung-device=t2,rung-fleet=t1", false)
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	// The SAME subject as the previous test. Only the evidence changed.
	if !strings.Contains(out, "zone=device") || !strings.Contains(out, "model=rung-device") {
		t.Fatalf("a measured on-box model did not take routine work:\n%s", out)
	}
	if !strings.Contains(out, "self-hosted  yes") {
		t.Fatalf("the placement is not reported as self-hosted:\n%s", out)
	}
	if !strings.Contains(out, "upstream=qwen/qwen3.6-4b") {
		t.Fatalf("the wire model the operator will actually be billed for (or not) is absent:\n%s", out)
	}
	if strings.Contains(out, modelroute.ReasonEscalatedPast) {
		t.Errorf("the cheapest rung took the work, so nothing escalated:\n%s", out)
	}
	if strings.Contains(out, "UNMEASURED") {
		t.Errorf("a measured winner is reported as unmeasured:\n%s", out)
	}
}

func TestAnUndeclaredWorkClassIsReportedAsTheCauseRatherThanBlamedOnHardware(t *testing.T) {
	r := routePlaceRoster()
	// A measured on-box model, and STILL the vendor: an undeclared class takes the
	// strictest floor (T0), which a T2 model does not meet. The operator's fix is a
	// label, not a purchase, and the surface has to say which.
	code, out, errOut := routePlaceRun(t, &r, nil, "rung-device=t2,rung-fleet=t1", false)
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "zone=vendor") {
		t.Fatalf("undeclared work was placed on a cheap rung:\n%s", out)
	}
	if !strings.Contains(out, "UNDECLARED") {
		t.Fatalf("the classification is not reported as undeclared:\n%s", out)
	}
	if !strings.Contains(out, "--labels work_class=") {
		t.Fatalf("the surface does not name the lever that fixes this:\n%s", out)
	}
	if !strings.Contains(out, modelroute.ReasonZoneUnderTier) {
		t.Fatalf("the cheap rungs did not report WHY they were refused:\n%s", out)
	}
	if !strings.Contains(out, "not more hardware") {
		t.Errorf("the surface lets an operator read this as a capacity problem:\n%s", out)
	}
}

func TestSecurityWorkNeverFallsToASmallLocalModel(t *testing.T) {
	r := routePlaceRoster()
	// The device model is measured, declared, present, and cheap. The floor still
	// refuses it: the floor is fixed by the WORK, not by what is available.
	code, out, errOut := routePlaceRun(t, &r,
		map[string]string{"work_class": string(modelroute.ClassSecurityRelease)},
		"rung-device=t2,rung-fleet=t2", false)
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if strings.Contains(out, "zone=device") || strings.Contains(out, "zone=fleet") {
		t.Fatalf("security/release/destructive work landed on a cheap rung:\n%s", out)
	}
	if !strings.Contains(out, modelroute.ReasonZoneUnderTier) {
		t.Errorf("the refusal reason is not on the ladder:\n%s", out)
	}
}

func TestABadCapabilityDeclarationIsAUsageErrorNotAnIgnoredToken(t *testing.T) {
	for _, spec := range []string{"rung-device=t9", "rung-device", "=t2", "rung-device=t2,rung-fleet=nope"} {
		r := routePlaceRoster()
		code, out, errOut := routePlaceRun(t, &r, map[string]string{"work_class": "routine"}, spec, false)
		if code != 2 {
			t.Errorf("--capability %q: exit = %d, want 2 (usage); stdout = %q", spec, code, out)
		}
		if out != "" {
			t.Errorf("--capability %q: printed a placement anyway: %q", spec, out)
		}
		if !strings.Contains(errOut, "capability") {
			t.Errorf("--capability %q: error does not name the flag: %q", spec, errOut)
		}
	}
}

func TestTheJSONReportCarriesBothTheClassAndThePlacement(t *testing.T) {
	r := routePlaceRoster()
	code, out, errOut := routePlaceRun(t, &r,
		map[string]string{"work_class": "routine"}, "rung-device=t2", true)
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	var got placementReport
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("--json emitted something that is not a report: %v\n%s", err, out)
	}
	if !got.Classification.Declared || got.Classification.Class != modelroute.ClassRoutine {
		t.Errorf("classification half is wrong: %+v", got.Classification)
	}
	if got.Placement.Zone != modelroute.ZoneDevice || !got.Placement.SelfHosted() {
		t.Errorf("placement half is wrong: zone=%s self-hosted=%v", got.Placement.Zone, got.Placement.SelfHosted())
	}
	if !got.Placement.Measured {
		t.Errorf("the measured flag did not survive the JSON round trip: %+v", got.Placement)
	}
	if got.MeasuredCount != 1 {
		t.Errorf("measured_candidates = %d, want 1", got.MeasuredCount)
	}
	if len(got.Placement.Ladder) != len(modelroute.Zones()) {
		t.Errorf("the JSON ladder is not the whole ladder: %+v", got.Placement.Ladder)
	}
}

func TestLegacySpellingsAreNotCountedAsExtraHardware(t *testing.T) {
	r := routePlaceRoster()
	got := placementCandidates(r, nil)
	if len(got) != 3 {
		t.Fatalf("candidate pool = %d, want 3 (one per rung); got %+v", len(got), got)
	}
	for _, c := range got {
		if c.Model == "rung-vendor-oldname" || c.Model == "rung-vendor-legacy" {
			t.Errorf("a compatibility/deprecated spelling entered the pool as a candidate: %+v", got)
		}
		if c.Measured {
			t.Errorf("an undeclared candidate claims a measured capability: %+v", c)
		}
	}
	// Deterministic order, because a placement that depends on map iteration is not a
	// placement policy anyone can be held to.
	if got[0].Model != "rung-device" || got[1].Model != "rung-fleet" || got[2].Model != "rung-vendor" {
		t.Errorf("candidate order is not deterministic-by-id: %+v", got)
	}
}

func TestAnUnmeasuredWinnerIsNeverRenderedAsAGrade(t *testing.T) {
	// The trap this guards: an unmeasured candidate enters Admit at the ZERO-VALUE
	// tier, which is the most demanding one, so the raw TierChoice reads
	// "capability T0, over-tier" about a model nobody has ever graded.
	unmeasured := modelroute.Placement{
		Measured: false,
		Choice:   modelroute.TierChoice{Capability: modelroute.TierT0, OverTier: true},
	}
	cell := capabilityCell(unmeasured)
	if !strings.Contains(cell, "UNMEASURED") {
		t.Errorf("capabilityCell(unmeasured) = %q, want it to say so", cell)
	}
	if strings.Contains(cell, "OVER-TIER") {
		t.Errorf("capabilityCell asserted waste about an ungraded model: %q", cell)
	}
	if strings.Contains(cell, modelroute.TierT0.String()) {
		t.Errorf("capabilityCell printed the default as though it were a grade: %q", cell)
	}

	measured := modelroute.Placement{
		Measured: true,
		Choice:   modelroute.TierChoice{Capability: modelroute.TierT0, OverTier: true},
	}
	cell = measured.Choice.Capability.String()
	if got := capabilityCell(measured); !strings.Contains(got, cell) || !strings.Contains(got, "OVER-TIER") {
		t.Errorf("capabilityCell(measured over-tier) = %q, want the grade AND the waste flag", got)
	}
}

func TestRepeatedRungReasonsAreCountedNotHidden(t *testing.T) {
	got := tallyReasons([]string{
		modelroute.ReasonZoneUnmeasured,
		modelroute.ReasonZoneUnmeasured,
		modelroute.ReasonZoneUnderTier,
		modelroute.ReasonZoneUnmeasured,
	})
	want := []string{modelroute.ReasonZoneUnmeasured + " x3", modelroute.ReasonZoneUnderTier}
	if len(got) != len(want) {
		t.Fatalf("tallyReasons = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("tallyReasons = %q, want %q", got, want)
		}
	}
	// A rung that turned away three candidates and a rung that turned away one are
	// different facts about the operator's roster; deduplication would erase that.
	if single := tallyReasons([]string{modelroute.ReasonZoneUnmeasured}); len(single) != 1 || single[0] != modelroute.ReasonZoneUnmeasured {
		t.Errorf("a single reason grew a count: %q", single)
	}
	if empty := tallyReasons(nil); len(empty) != 0 {
		t.Errorf("tallyReasons(nil) = %q, want empty", empty)
	}
}
