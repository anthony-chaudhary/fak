package eveparity_test

import (
	"bytes"
	"encoding/json"
	"flag"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/eveparity"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	// Blank import wires the ABI (Ref resolver + engines + the rank-100 adjudicator
	// Default) so gateway.New can build a real proxy — the fak-routed arm runs through
	// genuine fak mediation, not a stand-in.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

var update = flag.Bool("update", false, "regenerate the golden witness artifact")

// The canonical documented command lines recorded in the witness. They mirror the raw
// vs fak arms in EVE-EVAL-PARITY-RUNBOOK.md and are fixed (no random httptest port) so
// the golden artifact is byte-deterministic.
func rawCommand(strict bool) string {
	if strict {
		return "eve eval --json --junit --strict --suite eve-fixture-parity"
	}
	return "eve eval --json --junit --suite eve-fixture-parity"
}

func fakCommand(strict bool) string {
	if strict {
		return "eve eval --json --junit --strict --base-url http://127.0.0.1:8080/v1 --suite eve-fixture-parity"
	}
	return "eve eval --json --junit --base-url http://127.0.0.1:8080/v1 --suite eve-fixture-parity"
}

// standUpFakGateway builds a REAL fak gateway proxying to upstreamURL and returns its
// base URL. It admits the fixture "search" tool through fak's floor (exactly what an
// operator routing an eval through fak would configure) and restores the default floor
// afterward, so the fak arm exercises fak's genuine session/policy/proxy path.
func standUpFakGateway(t *testing.T, upstreamURL string) string {
	t.Helper()
	adjudicator.Default.SetPolicy(adjudicator.Policy{Allow: map[string]bool{"search": true}})
	t.Cleanup(func() { adjudicator.Default.SetPolicy(adjudicator.DefaultPolicy()) })

	srv, err := gateway.New(gateway.Config{
		Model:    "fixture",
		Provider: "openai",
		// fak joins BaseURL + "/chat/completions" (internal/agent/adapters.go), so the
		// base must end in /v1 — the same shape the runbook's --base-url uses.
		BaseURL: upstreamURL + "/v1",
		VDSO:    true,
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts.URL
}

// TestEveParityRawVsFak is the #2605 acceptance witness: the fixture Eve suite runs raw
// (against the fixture model directly) and fak-routed (through a real gateway proxy), and
// the two arms must reach the SAME pass/fail on every case — including the deliberately
// failing hard gate, whose Eve reason must be byte-identical — under both --strict modes.
func TestEveParityRawVsFak(t *testing.T) {
	upstream := httptest.NewServer(eveparity.FixtureUpstream())
	defer upstream.Close()
	fakURL := standUpFakGateway(t, upstream.URL)
	suite := eveparity.FixtureSuite()

	for _, strict := range []bool{false, true} {
		raw := eveparity.RunArm("raw", rawCommand(strict), upstream.URL, suite, strict, nil)
		fak := eveparity.RunArm("fak", fakCommand(strict), fakURL, suite, strict, nil)
		w := eveparity.Compare(suite.Name, strict, raw, fak)

		if w.ParityVerdict != "pass" {
			t.Fatalf("strict=%v: parity_verdict = %q, want pass\nwitness: %s", strict, w.ParityVerdict, mustJSON(t, w))
		}
		if !w.GateFailurePreserved {
			t.Errorf("strict=%v: gate_failure_preserved = false (fak downgraded a hard gate)", strict)
		}
		if !w.StrictThresholdPreserved {
			t.Errorf("strict=%v: strict_threshold_preserved = false", strict)
		}
		if !w.SessionIDsPreserved {
			t.Errorf("strict=%v: session_ids_preserved = false", strict)
		}
		if !w.TokenMetadataPreserved {
			t.Errorf("strict=%v: token_metadata_preserved = false", strict)
		}

		byCase := map[string]eveparity.CaseParity{}
		for _, cp := range w.Cases {
			byCase[cp.CaseID] = cp
		}

		// The positive tool gate: the fixture "search" call must survive fak's floor so
		// BOTH arms pass t.calledTool — proving fak did not strip an admitted tool call.
		if cp := byCase["succeeded-and-tool"]; !cp.RawPassed || !cp.FakPassed {
			t.Errorf("strict=%v: succeeded-and-tool raw=%v fak=%v, want both pass (fak stripped the tool call?)", strict, cp.RawPassed, cp.FakPassed)
		}
		if cp := byCase["content-exact"]; !cp.RawPassed || !cp.FakPassed {
			t.Errorf("strict=%v: content-exact raw=%v fak=%v, want both pass", strict, cp.RawPassed, cp.FakPassed)
		}

		// The deliberately-failing hard gate: fails BOTH arms, same reason, NOT downgraded.
		gf := byCase["deliberate-gate-fail"]
		if gf.RawPassed || gf.FakPassed {
			t.Errorf("strict=%v: deliberate-gate-fail raw=%v fak=%v, want both FAIL", strict, gf.RawPassed, gf.FakPassed)
		}
		if gf.RawReason == "" || gf.RawReason != gf.FakReason {
			t.Errorf("strict=%v: deliberate-gate-fail reasons differ:\n raw=%q\n fak=%q", strict, gf.RawReason, gf.FakReason)
		}
		if gf.Downgraded {
			t.Errorf("strict=%v: deliberate-gate-fail marked downgraded on an honest parity run", strict)
		}

		// The strict-sensitive soft case: passes without --strict, fails WITH it — and the
		// two arms must agree either way (strict threshold preserved across the proxy).
		ss := byCase["soft-strict-fail"]
		if ss.RawPassed != ss.FakPassed {
			t.Errorf("strict=%v: soft-strict-fail raw=%v fak=%v, want equal", strict, ss.RawPassed, ss.FakPassed)
		}
		if want := !strict; ss.RawPassed != want {
			t.Errorf("strict=%v: soft-strict-fail passed=%v, want %v (soft threshold only enforced under --strict)", strict, ss.RawPassed, want)
		}
	}
}

// TestEveParityGoldenWitness pins the deterministic parity witness artifact (strict
// arm). The golden proves the adapter emits raw result, fak-routed result, per-case
// diff/parity verdict, and the command lines — and that gate_failure_preserved is true.
func TestEveParityGoldenWitness(t *testing.T) {
	upstream := httptest.NewServer(eveparity.FixtureUpstream())
	defer upstream.Close()
	fakURL := standUpFakGateway(t, upstream.URL)
	suite := eveparity.FixtureSuite()

	raw := eveparity.RunArm("raw", rawCommand(true), upstream.URL, suite, true, nil)
	fak := eveparity.RunArm("fak", fakCommand(true), fakURL, suite, true, nil)
	w := eveparity.Compare(suite.Name, true, raw, fak)

	got := mustJSON(t, w)
	goldenPath := filepath.Join("testdata", "parity-witness.golden.json")
	if *update {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote golden %s", goldenPath)
		return
	}
	want, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden (run `go test -run Golden -update` to create it): %v", err)
	}
	if !bytes.Equal(bytes.TrimSpace(got), bytes.TrimSpace(want)) {
		t.Errorf("witness artifact drifted from golden %s.\n--- got ---\n%s", goldenPath, got)
	}
}

// TestCompareCatchesSilentDowngrade is the adversarial witness the golden narrates: if a
// fak arm ever turned the deliberately-failing HARD gate into a pass (the exact
// silent-downgrade #2605 exists to catch), Compare must flag it — downgraded, parity
// divergent, gate-failure NOT preserved. A comparator that could not catch this would be
// worthless as a regression guard.
func TestCompareCatchesSilentDowngrade(t *testing.T) {
	suite := eveparity.FixtureSuite()
	gateCase := suite.Cases[3] // deliberate-gate-fail
	if gateCase.ID != "deliberate-gate-fail" {
		t.Fatalf("fixture drift: case[3] = %q", gateCase.ID)
	}
	// Raw: the model did not call "write" -> hard gate FAILS.
	rawTr := eveparity.Transcript{CaseID: gateCase.ID, SessionID: "s", Succeeded: true, PromptTokens: 1, CompletionTokens: 1, FinalText: "no"}
	rawOut := eveparity.Evaluate(gateCase, rawTr, true)
	if rawOut.Passed {
		t.Fatal("setup: expected the raw hard gate to fail")
	}
	// Fak (tampered): the SAME case reported as passed — a silent downgrade.
	fakOut := rawOut
	fakOut.Passed = true
	fakOut.FailReason = ""
	fakOut.Checks = nil

	raw := eveparity.ArmResult{Arm: "raw", Cases: []eveparity.CaseOutcome{rawOut}, SessionIDsPresent: true, TokenMetadataPresent: true}
	fak := eveparity.ArmResult{Arm: "fak", Cases: []eveparity.CaseOutcome{fakOut}, SessionIDsPresent: true, TokenMetadataPresent: true}
	w := eveparity.Compare(suite.Name, true, raw, fak)

	if w.ParityVerdict != "divergent" {
		t.Errorf("parity_verdict = %q, want divergent", w.ParityVerdict)
	}
	if w.GateFailurePreserved {
		t.Errorf("gate_failure_preserved = true, want false (a hard gate was downgraded)")
	}
	if len(w.Cases) != 1 || !w.Cases[0].Downgraded {
		t.Errorf("downgrade not flagged: %+v", w.Cases)
	}
}

// TestEvaluateStrictSoftThreshold is the pure-unit floor: a soft score below its
// threshold fails the case ONLY under --strict; a hard gate fails regardless.
func TestEvaluateStrictSoftThreshold(t *testing.T) {
	soft := eveparity.Case{ID: "s", ExpectSucceeded: true, Soft: &eveparity.SoftSpec{Name: "q", Score: 0.5, Threshold: 0.7}}
	tr := eveparity.Transcript{CaseID: "s", Succeeded: true}
	if out := eveparity.Evaluate(soft, tr, false); !out.Passed {
		t.Errorf("non-strict: soft-below-threshold should pass, got fail (%s)", out.FailReason)
	}
	if out := eveparity.Evaluate(soft, tr, true); out.Passed {
		t.Errorf("strict: soft-below-threshold should fail")
	} else if out.FailKind != eveparity.SoftScore {
		t.Errorf("strict soft fail kind = %v, want SoftScore", out.FailKind)
	}

	hard := eveparity.Case{ID: "h", ExpectSucceeded: true, ExpectToolCall: "write"}
	if out := eveparity.Evaluate(hard, eveparity.Transcript{CaseID: "h", Succeeded: true}, false); out.Passed || out.FailKind != eveparity.HardGate {
		t.Errorf("hard gate must fail regardless of strict: passed=%v kind=%v", out.Passed, out.FailKind)
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	return b
}
