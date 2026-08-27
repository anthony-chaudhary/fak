package selfinstall

import (
	"strings"
	"testing"
)

// spawnOK mirrors dispatchtick.PreflightOKVerdict. It is restated (not imported) because the
// gate vocabulary is the CALLER's — see ApplyGateSkew's okVerdict parameter — and a test that
// imported the tick package would be asserting on the import, not on the fold.
const spawnOK = "SPAWN_OK"

// TestDivergentCopiesCannotReturnSpawnOK is Done condition 5 of #6508, verbatim: create
// divergent copies on disk and prove the gate cannot return SPAWN_OK.
//
// The host it builds is the one the audit found — an `+uncommitted` repo-root binary
// adjudicating the spawn gate while a DIFFERENT, clean build fronts every worker it admits.
// Before this fold, that exact state was measured, reported as FAK_BIN_DISAGREEMENT, and
// admitted anyway: the tick answered SPAWN_OK, so unreviewed code kept deciding who may run.
func TestProvenanceFromCensusIgnoresNonAdmissionRoles(t *testing.T) {
	copies := []HotCopy{
		{Role: RolePath, Path: "path"}, {Role: RoleGoBin, Path: "gobin"}, {Role: RoleScheduled, Path: "scheduled"},
	}
	if got := ProvenanceFromCensus(copies); got != (GateProvenance{Probed: true}) {
		t.Fatalf("non-admission roles changed provenance: %+v", got)
	}
}

func TestDivergentCopiesCannotReturnSpawnOK(t *testing.T) {
	h, paths := hotHost(t, RoleGate, RoleWorker)
	const worker = "793e38a87d719a0b1c2d3e4f5a6b7c8d9e0f1a2b"
	probe := func(p string) (string, bool, bool) {
		if strings.EqualFold(p, paths[RoleGate]) {
			return "e5fc01af20cd0000000000000000000000000000", true, true // the dirty adjudicator
		}
		return worker, false, true
	}

	prov := ProvenanceFromCensus(Census(h, probe))
	verdict, reason := ApplyGateSkew(spawnOK, "", spawnOK, prov)
	if verdict == spawnOK {
		t.Fatalf("a +uncommitted adjudicator fronting a different worker build still returned %s: %+v", spawnOK, prov)
	}
	if verdict != RefuseBinSkew {
		t.Fatalf("verdict = %q, want %q", verdict, RefuseBinSkew)
	}
	if !strings.Contains(reason, "+uncommitted") || !strings.Contains(reason, paths[RoleGate]) {
		t.Errorf("refusal reason does not name the offending binary: %q", reason)
	}

	// Same two copies, CONVERGED on one clean build: the gate admits again. Without this the
	// refusal above would be indistinguishable from a fold that always refuses.
	same := func(string) (string, bool, bool) { return worker, false, true }
	if v, r := ApplyGateSkew(spawnOK, "", spawnOK, ProvenanceFromCensus(Census(h, same))); v != spawnOK || r != "" {
		t.Fatalf("a converged host was refused: verdict=%q reason=%q", v, r)
	}
}

// TestGateSkewRefusesAnUnpinnedOrDisagreeingAdjudicator covers Done condition 3's other two
// states — the gate binary that cannot say which commit it is, and the clean-but-different
// build — and pins the reason precedence so one host always yields one replayable reason.
func TestGateSkewRefusesAnUnpinnedOrDisagreeingAdjudicator(t *testing.T) {
	base := GateProvenance{
		Probed:   true,
		GatePath: "/repo/fak", GateResolved: true, GateAttested: true,
		GateBuild:      "0c96937b61ac2e1e9d1f0b3c4d5e6f7a8b9c0d1e",
		WorkerPath:     "/repo/tools/.bin/fak",
		WorkerResolved: true, WorkerAttested: true,
		WorkerBuild: "0c96937b61ac2e1e9d1f0b3c4d5e6f7a8b9c0d1e",
	}
	cases := []struct {
		name   string
		mut    func(*GateProvenance)
		refuse bool
		want   string
	}{
		{"converged", func(*GateProvenance) {}, false, ""},
		{"short rev of the same commit is NOT skew", func(p *GateProvenance) {
			p.WorkerBuild = p.GateBuild[:12]
		}, false, ""},
		{"unreviewed working-tree adjudicator", func(p *GateProvenance) { p.GateDirty = true }, true, "no commit reviews"},
		{"unpinned adjudicator", func(p *GateProvenance) { p.GateAttested, p.GateBuild = false, "" }, true, "unpinned adjudicator"},
		{"disagrees with the worker guard", func(p *GateProvenance) { p.WorkerBuild = "7298f8f2abbb" }, true, "different build than the one fronting the workers"},
		// Fail-open cases: nothing was measured, so nothing is claimed. A host with no fak
		// built already launches workers unwrapped; a refusal here would wedge it forever.
		{"no provenance collected", func(p *GateProvenance) { p.Probed, p.GateDirty = false, true }, false, ""},
		{"no gate binary on this host", func(p *GateProvenance) { p.GateResolved, p.GateDirty = false, true }, false, ""},
		{"no worker binary to disagree with", func(p *GateProvenance) {
			p.WorkerResolved, p.WorkerBuild = false, "7298f8f2abbb"
		}, false, ""},
		{"worker cannot attest", func(p *GateProvenance) {
			p.WorkerAttested, p.WorkerBuild = false, ""
		}, false, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := base
			c.mut(&p)
			refuse, why := GateSkewRefusal(p)
			if refuse != c.refuse {
				t.Fatalf("GateSkewRefusal refuse = %v, want %v (reason %q)", refuse, c.refuse, why)
			}
			if c.want != "" && !strings.Contains(why, c.want) {
				t.Errorf("reason %q does not name %q", why, c.want)
			}
			if !c.refuse && c.want == "" && why != "" {
				t.Errorf("a non-refusing gate produced a reason: %q", why)
			}
		})
	}

	// Precedence: a gate that is dirty AND disagreeing names the dirt, always.
	both := base
	both.GateDirty, both.WorkerBuild = true, "7298f8f2abbb"
	for i := 0; i < 4; i++ {
		if _, why := GateSkewRefusal(both); !strings.Contains(why, "+uncommitted") {
			t.Fatalf("reason is not stable across replays: %q", why)
		}
	}
}

// TestGateSkewOverrideAdmitsButAnnotates — the escape hatch must not be silent. A maintainer
// hand-testing a local build can still admit, but the tick that was let through says so, so a
// payload from an overridden tick is never mistaken later for one that passed clean.
func TestGateSkewOverrideAdmitsButAnnotates(t *testing.T) {
	p := GateProvenance{Probed: true, GatePath: "/repo/fak", GateResolved: true,
		GateAttested: true, GateBuild: "e5fc01af20cd", GateDirty: true, Allow: true}
	refuse, why := GateSkewRefusal(p)
	if refuse {
		t.Fatalf("an explicit override still refused: %s", why)
	}
	if !strings.Contains(why, "OVERRIDE") {
		t.Errorf("an overridden admission does not name the override: %q", why)
	}
	v, reason := ApplyGateSkew(spawnOK, "capacity available", spawnOK, p)
	if v != spawnOK {
		t.Fatalf("override did not admit: %q", v)
	}
	if !strings.Contains(reason, "capacity available") || !strings.Contains(reason, "OVERRIDE") {
		t.Errorf("overridden reason lost either the original cause or the caveat: %q", reason)
	}
	if SkewSummary(p) == "" {
		t.Errorf("SkewSummary must still describe the skew that was overridden")
	}
}

func TestRepositoryStalenessCannotReturnSpawnOK(t *testing.T) {
	base := GateProvenance{
		Probed: true, GatePath: "/repo/fak", GateResolved: true, GateAttested: true,
		GateBuild:  "1111111111111111111111111111111111111111",
		WorkerPath: "/repo/tools/.bin/fak", WorkerResolved: true, WorkerAttested: true,
		WorkerBuild: "1111111111111111111111111111111111111111",
		RepoHead:    "2222222222222222222222222222222222222222", RepoRelation: "BEHIND",
		ResolvedCount: 3,
	}
	verdict, reason := ApplyGateSkew(spawnOK, "capacity available", spawnOK, base)
	if verdict != RefuseBinStale {
		t.Fatalf("verdict = %q, want %q", verdict, RefuseBinStale)
	}
	for _, want := range []string{"111111111111", "222222222222", "fak self-update --force --root ."} {
		if !strings.Contains(reason, want) {
			t.Errorf("reason %q does not contain recovery evidence %q", reason, want)
		}
	}

	base.RepoRelation = "MATCH"
	if v, _ := ApplyGateSkew(spawnOK, "", spawnOK, base); v != spawnOK {
		t.Fatalf("matching repository build refused: %q", v)
	}
	for _, relation := range []string{"AHEAD", "DIVERGED", "UNKNOWN"} {
		base.RepoRelation = relation
		if v, r := ApplyGateSkew(spawnOK, "", spawnOK, base); v != RefuseBinProvenance || !strings.Contains(r, relation) {
			t.Errorf("relation %s = %q / %q, want typed provenance refusal", relation, v, r)
		}
	}
	base.RepoRelation, base.ResolvedCount = "BEHIND", 1
	if v, _ := ApplyGateSkew(spawnOK, "", spawnOK, base); v != spawnOK {
		t.Fatalf("one resolver observation claimed aggregate staleness: %q", v)
	}
	base.ResolvedCount = 3
	base.RepoRelation, base.Allow = "BEHIND", true
	if v, r := ApplyGateSkew(spawnOK, "capacity available", spawnOK, base); v != spawnOK || !strings.Contains(r, "OVERRIDE") {
		t.Fatalf("historical override = %q / %q, want annotated admission", v, r)
	}
}

// TestApplyGateSkewNeverOverwritesAHigherPrecedenceRefusal — the fold sits BELOW the capacity
// ladder. A preflight that already refused (at-cap, no-seat, host) keeps its verdict and its
// reason; replacing them would lose the actually-binding term and send an operator to converge
// binaries when the real problem is a full seat pool.
func TestApplyGateSkewNeverOverwritesAHigherPrecedenceRefusal(t *testing.T) {
	dirty := GateProvenance{Probed: true, GatePath: "/repo/fak", GateResolved: true,
		GateAttested: true, GateBuild: "e5fc01af20cd", GateDirty: true}
	v, r := ApplyGateSkew("REFUSE_AT_CAP", "live workers 6 >= cap 6", spawnOK, dirty)
	if v != "REFUSE_AT_CAP" || r != "live workers 6 >= cap 6" {
		t.Fatalf("an existing refusal was rewritten: verdict=%q reason=%q", v, r)
	}
	// A caller that names no admit token cannot have its verdict rewritten either.
	if v, _ := ApplyGateSkew(spawnOK, "", "", dirty); v != spawnOK {
		t.Errorf("fold fired with no okVerdict declared: %q", v)
	}
	// The zero provenance (nothing measured) never touches a verdict.
	if v, r := ApplyGateSkew(spawnOK, "ok", spawnOK, GateProvenance{}); v != spawnOK || r != "ok" {
		t.Errorf("unmeasured provenance changed the verdict: %q / %q", v, r)
	}
}

func TestApplyGateSkewPreservesPythonStaleRefusal(t *testing.T) {
	p := GateProvenance{Probed: true, GateResolved: true, GateAttested: true, GateDirty: true}
	for _, verdict := range []string{RefuseBinStale, RefuseBinProvenance} {
		reason := verdict + ": Python repository admission refused before worker creation"
		v, r := ApplyGateSkew(verdict, reason, spawnOK, p)
		if v != verdict || r != reason {
			t.Errorf("Python refusal %s changed at Go fold: %q / %q", verdict, v, r)
		}
	}
}
