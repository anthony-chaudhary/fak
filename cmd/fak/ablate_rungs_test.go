package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

// turntaxTraceArg is a repo-root turnbench (NOT bench) trace, addressed relative to this
// package dir (cmd/fak) — `go test` runs with CWD at the package, so the default
// testdata/turntax lookup does not resolve here; pass it explicitly like the tau2 suite.
const turntaxTraceArg = "../../testdata/turntax/turntax-airline.json"

// runRungs drives runAblate's --rungs mode with captured streams, always pinning the
// explicit turntax trace so the run never depends on CWD or resolveSuite (which os.Exits).
func runRungs(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	full := append([]string{"--trace", turntaxTraceArg, "--rungs"}, args...)
	code := runAblate(&out, &errb, full)
	return code, out.String(), errb.String()
}

// firstChainRung is the first real adjudicator rung the full sweep would flip (skipping the
// vdso fast-path lever), so a single-rung test names a rung that actually exists in the
// chain instead of hardcoding one that could be renamed.
func firstChainRung(t *testing.T) string {
	t.Helper()
	for _, n := range ablateRungCatalog() {
		if n != turnbench.VDSOLever {
			return n
		}
	}
	t.Fatalf("no chain rung in ablateRungCatalog(): %v", ablateRungCatalog())
	return ""
}

// AC1: `fak ablate --rungs --trace FILE` prints one row per rung (delta + witness), exit 0.
func TestRunAblateRungs_FullSweepTable(t *testing.T) {
	code, out, errb := runRungs()
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	for _, want := range []string{
		"fak ablate --rungs", "workload hash", "baseline (full chain", "rungs replayed",
		"realized", "present", "changed", "witness", turnbench.VDSOLever,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rung table missing %q:\n%s", want, out)
		}
	}
	// one row per rung: the vdso lever plus every chain rung must all appear.
	for _, rung := range ablateRungCatalog() {
		if !strings.Contains(out, rung) {
			t.Fatalf("rung %q dropped from table:\n%s", rung, out)
		}
	}
}

// AC2 (present half): `--rungs=<name>` flips ONLY that rung, and it is Present in the chain.
func TestRunAblateRungs_SingleRungFlipsOnlyThat(t *testing.T) {
	rung := firstChainRung(t)
	var out, errb bytes.Buffer
	code := runAblate(&out, &errb, []string{"--trace", turntaxTraceArg, "--rungs=" + rung, "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var rep turnbench.LeverFlipReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad --json: %v\n%s", err, out.String())
	}
	if len(rep.Levers) != 1 {
		t.Fatalf("want exactly 1 flipped rung, got %d: %+v", len(rep.Levers), rep.Levers)
	}
	if rep.Levers[0].Lever != rung {
		t.Fatalf("flipped %q, want %q", rep.Levers[0].Lever, rung)
	}
	if !rep.Levers[0].Present {
		t.Fatalf("known rung %q reported Present=false", rung)
	}
}

// AC2 (unknown half): an unknown rung is reported Present=false, never silently dropped.
func TestRunAblateRungs_UnknownRungPresentFalse(t *testing.T) {
	var out, errb bytes.Buffer
	code := runAblate(&out, &errb, []string{"--trace", turntaxTraceArg, "--rungs=__nonesuch_rung__", "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var rep turnbench.LeverFlipReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("bad --json: %v\n%s", err, out.String())
	}
	if len(rep.Levers) != 1 {
		t.Fatalf("unknown rung must still be reported as one row, got %d", len(rep.Levers))
	}
	if rep.Levers[0].Present {
		t.Fatalf("unknown rung reported Present=true, want false")
	}
}

// AC3: --json emits a parseable LeverFlipReport; --out writes the same report to a file.
func TestRunAblateRungs_JSONAndOutEmission(t *testing.T) {
	// --json to stdout
	var jout, jerr bytes.Buffer
	if code := runAblate(&jout, &jerr, []string{"--trace", turntaxTraceArg, "--rungs", "--json"}); code != 0 {
		t.Fatalf("json exit=%d stderr=%s", code, jerr.String())
	}
	var rep turnbench.LeverFlipReport
	if err := json.Unmarshal(jout.Bytes(), &rep); err != nil {
		t.Fatalf("bad --json: %v", err)
	}
	if rep.LeversReplayed != len(rep.Levers) || len(rep.Levers) == 0 {
		t.Fatalf("levers_replayed=%d vs %d levers", rep.LeversReplayed, len(rep.Levers))
	}

	// --out writes the report file and the human run notes it.
	outPath := filepath.Join(t.TempDir(), "rungs.json")
	var oout, oerr bytes.Buffer
	if code := runAblate(&oout, &oerr, []string{"--trace", turntaxTraceArg, "--rungs", "--out", outPath}); code != 0 {
		t.Fatalf("out exit=%d stderr=%s", code, oerr.String())
	}
	b, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("--out did not write %s: %v", outPath, err)
	}
	if err := json.Unmarshal(b, &turnbench.LeverFlipReport{}); err != nil {
		t.Fatalf("--out file is not a LeverFlipReport: %v", err)
	}
	if !strings.Contains(oout.String(), outPath) {
		t.Fatalf("--out run did not report the written path:\n%s", oout.String())
	}
}

// --list gains a rung section (human + JSON) sourced from ablateRungCatalog().
func TestPrintAblateCatalog_RungSection(t *testing.T) {
	var human bytes.Buffer
	printAblateCatalog(&human, false)
	if !strings.Contains(human.String(), "adjudicator rungs") {
		t.Fatalf("--list human output missing rung section:\n%s", human.String())
	}
	for _, rung := range ablateRungCatalog() {
		if !strings.Contains(human.String(), rung) {
			t.Fatalf("--list rung section dropped %q:\n%s", rung, human.String())
		}
	}

	var j bytes.Buffer
	printAblateCatalog(&j, true)
	var payload struct {
		Rungs []string `json:"rungs"`
	}
	if err := json.Unmarshal(j.Bytes(), &payload); err != nil {
		t.Fatalf("--list --json unparseable: %v", err)
	}
	if len(payload.Rungs) != len(ablateRungCatalog()) || len(payload.Rungs) == 0 {
		t.Fatalf("--list --json rungs=%v, want %v", payload.Rungs, ablateRungCatalog())
	}
}
