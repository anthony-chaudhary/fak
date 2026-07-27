package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Witnesses for `fak route --place --serving FILE` — the liveness snapshot an operator
// can finally hand the placement ladder (epic #5416, track H).
//
// The gate itself is tested next door in internal/modelroute. What is under test HERE is
// everything that only exists once a snapshot meets a real roster on a real command line:
// that a probe filed under a model nothing binds is reported rather than silently gating
// nothing, that a misspelled key cannot switch fail-closed freshness off, and that a dead
// company host moves a delegated turn to the next company host rather than to a vendor.

// servingRoster has TWO fleet hosts, which is what makes a within-rung failover
// observable at all: with one company model, "the fleet rung is down" and "this host is
// down" are the same sentence, and the distinction that keeps money on the company's
// hardware could not be witnessed.
func servingRoster() modelroute.Roster {
	return modelroute.Roster{
		Version: "1",
		Accounts: []modelroute.Account{
			{ID: "box", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
			{ID: "corp-a", Kind: modelroute.KindFleet, BaseURL: "http://glm.infer.corp.internal:8000/v1", CredEnv: "FAK_CORP_TOKEN"},
			{ID: "corp-b", Kind: modelroute.KindFleet, BaseURL: "http://kimi.infer.corp.internal:8000/v1", CredEnv: "FAK_CORP_TOKEN"},
			{ID: "lab", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_API_KEY"},
		},
		Bindings: []modelroute.Binding{
			{Model: "rung-device", Account: "box", UpstreamModel: "qwen/qwen3.6-4b"},
			{Model: "rung-fleet-a", Account: "corp-a", UpstreamModel: "glm-5.2"},
			{Model: "rung-fleet-b", Account: "corp-b", UpstreamModel: "kimi-k3"},
			{Model: "rung-vendor", Account: "lab", UpstreamModel: "gpt-frontier"},
		},
		SpawnClasses: []modelroute.SpawnClass{
			{Type: "explore", Class: modelroute.ClassRoutine},
			{Type: "code-reviewer", Class: modelroute.ClassNormalImpl},
		},
	}
}

// servingCaps grades every rung, because an unmeasured candidate is barred from the cheap
// rungs before liveness is ever consulted — a serving test on an ungraded pool would be
// witnessing rule 2 of Place and calling it a gate.
const servingCaps = "rung-device=t2,rung-fleet-a=t1,rung-fleet-b=t1,rung-vendor=t0"

// writeServing drops a snapshot on disk and returns its path.
func writeServing(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "serving.json")
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// servingBlock returns just the snapshot half of the report.
func servingBlock(t *testing.T, out string) string {
	t.Helper()
	i := strings.Index(out, "SERVING SNAPSHOT")
	if i < 0 {
		t.Fatalf("no serving block was rendered:\n%s", out)
	}
	return out[i:]
}

// The headline. One of two company GPU hosts is not answering. The right answer is the
// OTHER company host — a fleet rung is several machines, and abandoning the rung because
// one box rebooted sends every token to a third-party lab to route around a neighbour that
// was idle. The bill and the diagnosis must also stay separable: this placement did not
// escalate, and something IS down, so failed-over is the bit that says so.
func TestADeadCompanyHostFailsOverToTheOtherOneNotToAVendor(t *testing.T) {
	r := servingRoster()
	snap := writeServing(t, `{
	  "schema": "fak.modelroute.serving.v1",
	  "covers": ["fleet"],
	  "models": {"rung-fleet-a": {"state": "down"}, "rung-fleet-b": {"state": "up"}}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "normal-impl"},
		placeOptions{CapSpec: servingCaps, ServingPath: snap})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, "placed       zone=fleet  model=rung-fleet-b") {
		t.Fatalf("a dead company host did not fail over to the other company host:\n%s", out)
	}
	if !strings.Contains(out, "self-hosted  yes") {
		t.Errorf("the work left the company's own hardware over one dead box:\n%s", out)
	}
	if !strings.Contains(out, "failed-over  yes") {
		t.Errorf("a liveness failover is not reported as one:\n%s", out)
	}
	if !strings.Contains(out, modelroute.ReasonZoneServingDown) {
		t.Errorf("the ladder does not record why the host was passed over:\n%s", out)
	}

	// The separation, witnessed by the counterfactual rather than asserted. This
	// placement is ALSO escalated — the laptop rung is genuinely too small for
	// normal-impl — and an operator who reads only that bit goes looking for a
	// capability problem while the dead GPU host stays dead. Running the identical
	// placement with no snapshot is what shows which bit the outage actually moved:
	// escalated is unchanged (a capability fact about the device rung, true either
	// way), failed-over appears only when something is not answering.
	_, control, _ := routePlaceRunOpts(t, &r, map[string]string{"work_class": "normal-impl"},
		placeOptions{CapSpec: servingCaps})
	if !strings.Contains(control, "placed       zone=fleet  model=rung-fleet-a") {
		t.Fatalf("without a snapshot the work was expected on the host that later dies:\n%s", control)
	}
	if !strings.Contains(control, "failed-over  no") {
		t.Errorf("a run with no liveness signal at all reported a failover:\n%s", control)
	}
	if !strings.Contains(control, "escalated  yes") || !strings.Contains(out, "escalated  yes") {
		t.Errorf("escalated should be set by the device rung's tier in BOTH runs; the outage must not\n"+
			"be what moves it.\nwith snapshot:\n%s\nwithout:\n%s", out, control)
	}
}

// THE FAIL-OPEN THIS SURFACE EXISTS TO CATCH. An observation filed under a model id the
// roster does not bind is honored nowhere: the ladder only ever asks about candidates, so
// the run is identical to one with no snapshot at all. The report validates cleanly, the
// operator believes they gated a dead host, and nothing anywhere says otherwise — unless
// the surface joins the two and reports it.
//
// The snapshot here declares no coverage, which is what makes the failure OPEN: with
// nothing claimed, silence about the real candidate gates nothing either, so both halves
// of the mistake are inert and the dead host keeps taking work. The covered case fails the
// other way and is witnessed below.
func TestAnObservationForAnUnboundModelGatesNothingAndIsNamed(t *testing.T) {
	r := servingRoster()
	// The operator wrote the UPSTREAM name (what their probe script dialled) instead of
	// the routed id the roster binds — the likeliest way to get this wrong.
	snap := writeServing(t, `{
	  "schema": "fak.modelroute.serving.v1",
	  "models": {"glm-5.2": {"state": "down"}, "rung-fleet-b": {"state": "up"}}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "normal-impl"},
		placeOptions{CapSpec: servingCaps, ServingPath: snap})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	// The host the operator believes they gated still takes the work, and nothing on
	// the ladder is different from a run with no snapshot at all.
	if !strings.Contains(out, "placed       zone=fleet  model=rung-fleet-a") {
		t.Fatalf("the fixture no longer places on the host the snapshot MEANT to gate; the point below is lost:\n%s", out)
	}
	if !strings.Contains(out, "failed-over  no") {
		t.Errorf("an observation that gates nothing was reported as a failover:\n%s", out)
	}
	block := servingBlock(t, out)
	for _, want := range []string{"UNBOUND", "glm-5.2", "gate NOTHING"} {
		if !strings.Contains(block, want) {
			t.Errorf("the unbound observation is not reported (%q missing):\n%s", want, block)
		}
	}
	// The fix, not just the diagnosis: this id is the upstream name of a candidate.
	if !strings.Contains(block, "glm-5.2  ->  rung-fleet-a") {
		t.Errorf("the join knows which candidate that id is dialled as and does not say so:\n%s", block)
	}
	// And the count is honest about the gate's real reach: 2 observations, 1 of which
	// can ever be consulted.
	if !strings.Contains(block, "observations 2, of which 1 name a model this roster binds") {
		t.Errorf("the snapshot's real reach is not stated:\n%s", block)
	}
}

// The SAME typo inside declared coverage fails the opposite way, and an operator who only
// knew about the fail-open would misread it completely. The real candidate is now silent
// on a rung the report claims to speak for, so it is passed over as unknown: fail-closed,
// safe, and it reads as an outage on a host that is up. The two halves of the join name it
// from both ends — the id nothing binds, and the candidate nothing observed.
func TestTheSameUnboundKeyInsideCoverageFailsClosedNotOpen(t *testing.T) {
	r := servingRoster()
	snap := writeServing(t, `{
	  "schema": "fak.modelroute.serving.v1",
	  "covers": ["fleet"],
	  "models": {"glm-5.2": {"state": "down"}, "rung-fleet-b": {"state": "up"}}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "normal-impl"},
		placeOptions{CapSpec: servingCaps, ServingPath: snap})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	// Passed over as UNKNOWN — not because anything was reported down about it.
	if !strings.Contains(out, "placed       zone=fleet  model=rung-fleet-b") {
		t.Fatalf("a candidate the snapshot is silent about inside its own coverage was assumed well:\n%s", out)
	}
	if !strings.Contains(out, modelroute.ReasonZoneServingUnknown) {
		t.Errorf("the ladder does not record that the host was passed over on a missing verdict:\n%s", out)
	}
	if strings.Contains(out, modelroute.ReasonZoneServingDown) {
		t.Errorf("a host nothing was reported about was recorded as DOWN; that sends an operator to reboot it:\n%s", out)
	}
	block := servingBlock(t, out)
	if !strings.Contains(block, "UNBOUND      glm-5.2") || !strings.Contains(block, "SILENT       rung-fleet-a") {
		t.Fatalf("one typo produced an unbound id and a silent candidate; the join reports only one of them:\n%s", block)
	}
	// Both lines are about the same host under two spellings, and the join says so.
	if !strings.Contains(block, "glm-5.2  ->  rung-fleet-a") {
		t.Errorf("nothing connects the id in UNBOUND to the candidate in SILENT:\n%s", block)
	}
}

// The reason a snapshot cannot simply be keyed by upstream name instead. Two accounts
// serving the SAME weights is how a company scales a rung — two GPU pools behind two
// base URLs, one model — and one observation about those weights cannot speak for both
// pools: one can be down while the other serves. So the hint names every candidate the
// id resolves to rather than picking one, and says why re-keying means splitting it.
func TestAnUpstreamNameServedByTwoAccountsResolvesToBothNotOne(t *testing.T) {
	r := servingRoster()
	r.Accounts = append(r.Accounts, modelroute.Account{
		ID: "corp-c", Kind: modelroute.KindFleet,
		BaseURL: "http://glm-pool-2.infer.corp.internal:8000/v1", CredEnv: "FAK_CORP_TOKEN"})
	r.Bindings = append(r.Bindings, modelroute.Binding{
		Model: "rung-fleet-c", Account: "corp-c", UpstreamModel: "glm-5.2"})
	snap := writeServing(t, `{
	  "schema": "fak.modelroute.serving.v1",
	  "models": {"glm-5.2": {"state": "down"}}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "normal-impl"},
		placeOptions{CapSpec: servingCaps + ",rung-fleet-c=t1", ServingPath: snap})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	block := servingBlock(t, out)
	if !strings.Contains(block, "glm-5.2  ->  rung-fleet-a rung-fleet-c") {
		t.Fatalf("an upstream name served by two accounts resolved to fewer than both:\n%s", block)
	}
	if !strings.Contains(block, "observe each routed id separately") {
		t.Errorf("nothing says that one observation cannot stand in for two pools:\n%s", block)
	}
}

// A misspelled key must not read as an absent one. `max_age_sec` decodes to a report with
// no freshness bound: fail-closed staleness silently OFF, on a file that still validates
// and still looks right. DisallowUnknownFields is what makes the whole shape closed.
func TestAMisspelledSnapshotKeyIsRefusedNotReadAsAbsent(t *testing.T) {
	r := servingRoster()
	snap := writeServing(t, `{
	  "schema": "fak.modelroute.serving.v1",
	  "as_of_unix": 1000,
	  "max_age_sec": 60,
	  "models": {"rung-fleet-a": {"state": "up", "observed_unix": 1}}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "normal-impl"},
		placeOptions{CapSpec: servingCaps, ServingPath: snap})
	if code == 0 {
		t.Fatalf("a snapshot with an unknown key was accepted; exit = %d", code)
	}
	if out != "" {
		t.Errorf("a refused snapshot still printed a placement:\n%s", out)
	}
	if !strings.Contains(errOut, "max_age_sec") {
		t.Errorf("the refusal does not name the offending key: %q", errOut)
	}
}

// A report that declares a freshness bound and carries no as-of stamp to measure ages
// against gates EVERY covered candidate as stale. That is fail-closed and working as
// designed, and it reads on the ladder exactly like a total outage — so the surface says
// which of the two it is. The CLI must not paper over it by stamping "now" into the file's
// clock: that would manufacture the freshest possible snapshot out of a file that says
// nothing about when it was taken.
func TestASnapshotWithNoAsOfStampReadsAsUncheckableNotAsAnOutage(t *testing.T) {
	r := servingRoster()
	snap := writeServing(t, `{
	  "schema": "fak.modelroute.serving.v1",
	  "max_age_seconds": 60,
	  "covers": ["device", "fleet"],
	  "models": {"rung-device": {"state": "up", "observed_unix": 1770000000},
	             "rung-fleet-a": {"state": "up", "observed_unix": 1770000000}}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: servingCaps, ServingPath: snap})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if !strings.Contains(out, modelroute.ReasonZoneServingStale) {
		t.Fatalf("an unmeasurable age was not treated as stale:\n%s", out)
	}
	block := servingBlock(t, out)
	for _, want := range []string{"STALE-ALL", "as_of_unix"} {
		if !strings.Contains(block, want) {
			t.Errorf("the uncheckable-freshness diagnosis is missing %q:\n%s", want, block)
		}
	}
	// The file's own clock is reported as absent, never filled in for it.
	if !strings.Contains(block, "as of        (none") {
		t.Errorf("the surface invented an as-of stamp the file does not carry:\n%s", block)
	}
	var rep placementReport
	_, jsonOut, _ := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: servingCaps, ServingPath: snap, JSON: true})
	if err := json.Unmarshal([]byte(jsonOut), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Serving == nil || rep.Serving.AsOfUnix != 0 || !rep.Serving.UncheckableFreshness {
		t.Errorf("the wire report does not carry the file's absent clock honestly: %+v", rep.Serving)
	}
}

// Silence inside declared coverage is not health — those candidates are passed over as
// unknown. That is the right refusal (a crashed prober must not read as a healthy fleet)
// and it is indistinguishable from an outage on the ladder alone, so the snapshot's own
// gap is named as a gap.
func TestSilenceInsideCoverageIsReportedAsAGapInTheProbe(t *testing.T) {
	r := servingRoster()
	// Covers the device rung and says nothing about the only model on it.
	snap := writeServing(t, `{
	  "schema": "fak.modelroute.serving.v1",
	  "covers": ["device"],
	  "models": {"rung-fleet-a": {"state": "up"}}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: servingCaps, ServingPath: snap})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	if strings.Contains(out, "placed       zone=device") {
		t.Fatalf("an unobserved candidate inside declared coverage was assumed well:\n%s", out)
	}
	block := servingBlock(t, out)
	for _, want := range []string{"SILENT", "rung-device", "gap in the probe"} {
		if !strings.Contains(block, want) {
			t.Errorf("the coverage gap is not reported (%q missing):\n%s", want, block)
		}
	}
}

// THE SHAPE THE EPIC IS ABOUT. A frontier parent turn delegates an "explore" child. With
// the laptop's server down, the child must not fall back to the parent's vendor rung: the
// company's own fleet is right there, and keeping delegated volume self-hosted is the
// whole thesis. The snapshot the parent walked is the snapshot the child walks.
func TestADeadLaptopMovesADelegatedTurnToTheFleetNotToTheVendor(t *testing.T) {
	r := servingRoster()
	snap := writeServing(t, `{
	  "schema": "fak.modelroute.serving.v1",
	  "covers": ["device", "fleet"],
	  "models": {"rung-device": {"state": "down"},
	             "rung-fleet-a": {"state": "up"}, "rung-fleet-b": {"state": "up"}}
	}`)
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "ultra-hard"},
		placeOptions{CapSpec: servingCaps, ServingPath: snap, SpawnType: "explore"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	block := spawnBlock(t, out)
	if !strings.Contains(block, "parent       zone=vendor") {
		t.Fatalf("the parent was expected on the vendor rung:\n%s", out)
	}
	if !strings.Contains(block, "placed       zone=fleet") {
		t.Fatalf("a dead laptop sent the delegated turn somewhere other than the company fleet:\n%s", block)
	}
	// Still a descent, still self-hosted: the event the epic counts survives the outage.
	if !strings.Contains(block, "self-hosted descent  yes") {
		t.Errorf("the delegated descent stopped being counted because one rung was down:\n%s", block)
	}
	if !strings.Contains(block, "failed-over  yes") {
		t.Errorf("the child's own block does not report the liveness failover:\n%s", block)
	}
}

// ABSENCE DISCIPLINE on the wire, and the flag's refusal to be a decoration.
func TestTheServingSummaryIsAbsentUnlessASnapshotWasGiven(t *testing.T) {
	r := servingRoster()
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: servingCaps, JSON: true})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(out), &raw); err != nil {
		t.Fatalf("--json is not valid JSON: %v\n%s", err, out)
	}
	if _, present := raw["serving"]; present {
		t.Errorf("a run with no --serving emitted a serving key:\n%s", out)
	}

	// A snapshot handed to a command that answers which MODEL, not which rung, would be
	// silently dropped — and everything on screen would look fine.
	code, stdout, errOut := runRT("--serving", "whatever.json")
	if code != 2 {
		t.Fatalf("--serving without --place: exit = %d, want 2 (usage)", code)
	}
	if !strings.Contains(errOut, "--place") {
		t.Errorf("the refusal does not name the missing flag: %q", errOut)
	}
	if strings.Contains(stdout, "SERVING") {
		t.Errorf("a serving block was printed without --place:\n%s", stdout)
	}
}

// A named file that cannot be read is an ERROR, not an empty report. Absence of the flag
// means "no liveness signal"; a named file that is missing means the operator's probe did
// not run, and placing work as though every host were healthy is the exact failure the
// gate exists to prevent.
func TestANamedSnapshotThatCannotBeReadIsRefusedNotTreatedAsEmpty(t *testing.T) {
	r := servingRoster()
	missing := filepath.Join(t.TempDir(), "the-probe-never-ran.json")
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "normal-impl"},
		placeOptions{CapSpec: servingCaps, ServingPath: missing})
	if code == 0 {
		t.Fatalf("a missing snapshot placed work anyway; exit = %d\n%s", code, out)
	}
	if out != "" {
		t.Errorf("a refused snapshot still printed a placement:\n%s", out)
	}
	if !strings.Contains(errOut, "the-probe-never-ran.json") {
		t.Errorf("the refusal does not name the file: %q", errOut)
	}
}
