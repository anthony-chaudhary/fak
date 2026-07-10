package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/assumecheck"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestAssumeWiringMatchesDeclaredStatus binds the shell's witness-dispatch table to
// the registry's declared per-row WitnessStatus: a row is marked wired if and only
// if this shell actually has a gatherer for it, and every gatherer key is a
// registered id. This is what keeps the declared marker from drifting into a lie
// when C3 (#3821) adds or moves drivers.
func TestAssumeWiringMatchesDeclaredStatus(t *testing.T) {
	for _, a := range assumecheck.Registry() {
		_, wired := assumeWitnessGatherers[a.ID]
		if wired != (a.WitnessStatus == assumecheck.WitnessWired) {
			t.Fatalf("assumption %q declares witness status %q but shell wiring is %v", a.ID, a.WitnessStatus, wired)
		}
	}
	for id := range assumeWitnessGatherers {
		if _, ok := assumecheck.Lookup(id); !ok {
			t.Fatalf("witness gatherer wired for unregistered assumption %q", id)
		}
	}
}

// TestAssumeCheckDeclaredOnlyIsUnverifiable proves the proof-by-default posture end
// to end: checking a registered-but-unwired row yields UNVERIFIABLE (exit 4) with
// the wiring gap named as the explanation — never a fabricated HOLDS. Runs entirely
// off the declared registry; a declared-only row touches no disk or roster.
func TestAssumeCheckDeclaredOnlyIsUnverifiable(t *testing.T) {
	for _, a := range assumecheck.Registry() {
		if a.WitnessStatus != assumecheck.WitnessDeclaredOnly {
			continue
		}
		var out, errBuf bytes.Buffer
		code := runAssume(&out, &errBuf, []string{"check", a.ID})
		if code != 4 {
			t.Fatalf("check %q: exit=%d (stderr=%q), want 4 (UNVERIFIABLE, fail-closed)", a.ID, code, errBuf.String())
		}
		s := out.String()
		if !strings.Contains(s, string(assumecheck.OutcomeUnverifiable)) {
			t.Fatalf("check %q output carries no UNVERIFIABLE outcome:\n%s", a.ID, s)
		}
		if strings.Contains(s, "outcome    : "+string(assumecheck.OutcomeHolds)) {
			t.Fatalf("check %q fabricated a HOLDS for a declared-only row:\n%s", a.ID, s)
		}
		if !strings.Contains(s, "declared-only") {
			t.Fatalf("check %q does not explain the declared-only wiring gap:\n%s", a.ID, s)
		}
	}
}

// TestAssumeListEnumeratesRegistry proves `fak assume list` shows every registered
// row and distinguishes wired from declared-only wiring.
func TestAssumeListEnumeratesRegistry(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runAssume(&out, &errBuf, []string{"list"}); code != 0 {
		t.Fatalf("list: exit=%d (stderr=%q)", code, errBuf.String())
	}
	s := out.String()
	for _, a := range assumecheck.Registry() {
		if !strings.Contains(s, a.ID) {
			t.Fatalf("list omits registered assumption %q:\n%s", a.ID, s)
		}
	}
	if !strings.Contains(s, string(assumecheck.WitnessWired)) || !strings.Contains(s, string(assumecheck.WitnessDeclaredOnly)) {
		t.Fatalf("list does not distinguish wired from declared-only wiring:\n%s", s)
	}
}

// TestAssumeListJSONCarriesSchemaAndWiring proves `fak assume list --json` emits
// the fak.assume.list.v1 schema with one row per registered assumption, in registry
// order, each carrying an honest per-row wired bit that matches the shell's
// dispatch table (a relation against Registry(), not a frozen count).
func TestAssumeListJSONCarriesSchemaAndWiring(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runAssume(&out, &errBuf, []string{"list", "--json"}); code != 0 {
		t.Fatalf("list --json: exit=%d (stderr=%q)", code, errBuf.String())
	}
	var rec struct {
		Schema      string `json:"schema"`
		Assumptions []struct {
			Assumption assumecheck.Assumption `json:"assumption"`
			Wired      bool                   `json:"wired"`
		} `json:"assumptions"`
	}
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("list --json is not valid JSON: %v\n%s", err, out.String())
	}
	if rec.Schema != "fak.assume.list.v1" {
		t.Fatalf("schema = %q, want fak.assume.list.v1", rec.Schema)
	}
	rows := assumecheck.Registry()
	if len(rec.Assumptions) != len(rows) {
		t.Fatalf("list --json rows = %d, want one per registered assumption (%d)", len(rec.Assumptions), len(rows))
	}
	for i, row := range rec.Assumptions {
		if row.Assumption.ID != rows[i].ID {
			t.Fatalf("row %d id = %q, want registry-order %q", i, row.Assumption.ID, rows[i].ID)
		}
		_, wired := assumeWitnessGatherers[row.Assumption.ID]
		if row.Wired != wired {
			t.Fatalf("row %q wired=%v diverges from the shell dispatch table (%v)", row.Assumption.ID, row.Wired, wired)
		}
	}
}

// TestAssumeWiredKindsResolveDrivers proves every driver-backed wired row's
// declared witness kind actually resolves a registered driver (#3821 C3) — the
// name-resolved dispatch can never silently fall through for a row the registry
// advertises as wired. WitnessLedgerRead rows are exempt: that kind is a bespoke
// per-assumption authority read (gatherSeatLaunchableEvidence), not a generic
// driver.
func TestAssumeWiredKindsResolveDrivers(t *testing.T) {
	for _, a := range assumecheck.Registry() {
		if a.WitnessStatus != assumecheck.WitnessWired || a.WitnessKind == assumecheck.WitnessLedgerRead {
			continue
		}
		d, ok := assumecheck.ResolveDriver(a.WitnessKind)
		if !ok {
			t.Fatalf("wired assumption %q declares kind %s but no driver is registered for it", a.ID, a.WitnessKind)
		}
		if d.Kind() != a.WitnessKind {
			t.Fatalf("driver resolved for %q stamps kind %s, want %s", a.ID, d.Kind(), a.WitnessKind)
		}
	}
}

// TestAssumeSeatPoolTriState proves the seat-pool probe's pure exit mapping:
// free seats hold (0), a depleted pool refutes (1), an unreadable roster cannot
// witness (err) — so the command-probe driver turns each into the right
// closed-vocabulary outcome.
func TestAssumeSeatPoolTriState(t *testing.T) {
	if _, _, err := assumeSeatPoolTriState(dispatchtick.SeatCheck{Error: "no roster"}); err == nil {
		t.Fatal("an unreadable seat pool must map to the cannot-witness err branch")
	}
	detail, code, err := assumeSeatPoolTriState(dispatchtick.SeatCheck{
		Total: dispatchtick.IntPtr(4), Free: dispatchtick.IntPtr(0), Leased: dispatchtick.IntPtr(4), Depleted: true,
	})
	if err != nil || code != 1 {
		t.Fatalf("depleted pool mapped to (code=%d, err=%v), want the witnessed-refute exit 1", code, err)
	}
	if !strings.Contains(detail, "depleted") {
		t.Fatalf("depleted detail %q does not say so", detail)
	}
	detail, code, err = assumeSeatPoolTriState(dispatchtick.SeatCheck{
		Total: dispatchtick.IntPtr(4), Free: dispatchtick.IntPtr(2), Leased: dispatchtick.IntPtr(2),
	})
	if err != nil || code != 0 {
		t.Fatalf("free pool mapped to (code=%d, err=%v), want the holds exit 0", code, err)
	}
	if !strings.Contains(detail, "free=2") {
		t.Fatalf("free-pool detail %q drops the counts", detail)
	}
}

// TestAssumeKernelLoopTriState proves the `dos loop --json` probe's pure exit
// mapping: a non-refusing verdict holds (0), a HALT/REFUSE verdict refutes (1),
// and a failed probe or a verdict-less answer cannot witness (err) — the dos CLI
// exits 0 whenever it can answer, so the verdict, not the exit code, carries
// liveness.
func TestAssumeKernelLoopTriState(t *testing.T) {
	if _, _, err := assumeKernelLoopTriState(nil, errors.New("dos not found")); err == nil {
		t.Fatal("a probe that could not run must map to the cannot-witness err branch")
	}
	if _, _, err := assumeKernelLoopTriState(map[string]any{"alive": 1}, nil); err == nil {
		t.Fatal("an answer without a verdict must map to the cannot-witness err branch")
	}
	detail, code, err := assumeKernelLoopTriState(map[string]any{"verdict": "AT_TARGET", "alive": 2, "target": 2}, nil)
	if err != nil || code != 0 {
		t.Fatalf("healthy verdict mapped to (code=%d, err=%v), want the holds exit 0", code, err)
	}
	if !strings.Contains(detail, "verdict=AT_TARGET") || !strings.Contains(detail, "alive=2") {
		t.Fatalf("healthy detail %q drops the loop state", detail)
	}
	for _, verdict := range []string{"REFUSE_HOST", "HALTED", "PROPOSED_HALT"} {
		_, code, err := assumeKernelLoopTriState(map[string]any{"verdict": verdict}, nil)
		if err != nil || code != 1 {
			t.Fatalf("refusing verdict %q mapped to (code=%d, err=%v), want the witnessed-refute exit 1", verdict, code, err)
		}
	}
}

// TestAssumeCheckConfigDirSeedEndToEnd proves the wired config-flag seed's WHOLE
// path — registry Lookup -> name-resolved driver dispatch -> os.Stat witness ->
// kernel verdict -> exit code — against a hermetic temp registry, both ways: a
// seat whose config dir exists HOLDS (exit 0), and a seat whose dir vanished is
// VIOLATED (exit 3) with the SEAT_CONFIG_DIR_MISSING refusal token surfaced.
// This is the #3821 acceptance capture as a regression test: a REAL witnessed
// outcome from a driver, never an UNVERIFIABLE fallthrough.
func TestAssumeCheckConfigDirSeedEndToEnd(t *testing.T) {
	tmp := t.TempDir()
	presentDir := filepath.Join(tmp, "seat-present")
	if err := os.MkdirAll(presentDir, 0o755); err != nil {
		t.Fatal(err)
	}
	ghostDir := filepath.Join(tmp, "seat-vanished")
	registry := filepath.Join(tmp, "registry.json")
	body := fmt.Sprintf(`{"version":"fak-config-homes/v1","homes":[{"name":"present","dir":%q},{"name":"ghost","dir":%q}]}`,
		filepath.ToSlash(presentDir), filepath.ToSlash(ghostDir))
	if err := os.WriteFile(registry, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errBuf bytes.Buffer
	code := runAssume(&out, &errBuf, []string{"check", "seat-config-dir-present", "--registry", registry, "--home", tmp, "--seat", "present"})
	if code != 0 || !strings.Contains(out.String(), string(assumecheck.OutcomeHolds)) {
		t.Fatalf("present config dir: exit=%d (stderr=%q), want 0 HOLDS:\n%s", code, errBuf.String(), out.String())
	}

	out.Reset()
	errBuf.Reset()
	code = runAssume(&out, &errBuf, []string{"check", "seat-config-dir-present", "--registry", registry, "--home", tmp})
	if code != 3 || !strings.Contains(out.String(), string(assumecheck.OutcomeViolated)) {
		t.Fatalf("vanished config dir: exit=%d (stderr=%q), want 3 VIOLATED:\n%s", code, errBuf.String(), out.String())
	}
	s := out.String()
	if !strings.Contains(s, "ghost") || !strings.Contains(s, "SEAT_CONFIG_DIR_MISSING") {
		t.Fatalf("VIOLATED report neither names the pruned seat nor carries the refusal token:\n%s", s)
	}
}

// TestAssumeCheckUnknownIDNamesTheMenu proves an unknown id is a usage error (exit
// 2) that names the known ids from the registry, not a guessed check.
func TestAssumeCheckUnknownIDNamesTheMenu(t *testing.T) {
	var out, errBuf bytes.Buffer
	if code := runAssume(&out, &errBuf, []string{"check", "no-such-assumption"}); code != 2 {
		t.Fatalf("unknown id: exit=%d, want 2", code)
	}
	for _, a := range assumecheck.Registry() {
		if !strings.Contains(errBuf.String(), a.ID) {
			t.Fatalf("unknown-id usage error omits known id %q:\n%s", a.ID, errBuf.String())
		}
	}
}
