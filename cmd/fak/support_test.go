package main

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/covmatrix"
	"github.com/anthony-chaudhary/fak/internal/supportmaturity"
)

// TestSupportReadOutCitesRungWitness is the #1254 witness clause: every row cites the
// non-author witness that PROVES its rung. For each live (family x backend) cell we
// recompute the rung off the committed ladder (FromSupport) and the witness bound to that
// rung (WitnessFor), and assert the read-out row carries exactly that witness — so the
// "witness behind the rung" column can never drift from the binding it claims.
func TestSupportReadOutCitesRungWitness(t *testing.T) {
	rows := supportReadOut()
	if len(rows) == 0 {
		t.Fatal("supportReadOut returned no rows")
	}
	byCell := map[string]supportRow{}
	for _, r := range rows {
		byCell[r.Family+"\x00"+r.Backend] = r
	}
	for _, c := range covmatrix.Grid() {
		r, ok := byCell[c.Family+"\x00"+c.Backend]
		if !ok {
			t.Fatalf("read-out missing cell %s x %s", c.Family, c.Backend)
		}
		rung := supportmaturity.FromSupport(c.Support)
		wantWitness := supportmaturity.WitnessFor(rung).String()
		if r.Witness != wantWitness {
			t.Errorf("%s x %s: witness = %q, want %q (the witness bound to rung %s)",
				c.Family, c.Backend, r.Witness, wantWitness, rung)
		}
		if r.Witness == "" {
			t.Errorf("%s x %s: empty witness — every row must cite its rung-witness", c.Family, c.Backend)
		}
		if r.Rung != rung.String() || r.RungLabel != rung.Label() {
			t.Errorf("%s x %s: rung = %q/%q, want %q/%q", c.Family, c.Backend, r.Rung, r.RungLabel, rung.String(), rung.Label())
		}
	}
}

// TestSupportJSONRoundTrips is the #1254 --json round-trip witness: the JSON the verb
// emits decodes back into exactly the read-out it folded — no field lost, no field
// reshaped.
func TestSupportJSONRoundTrips(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSupport(&out, &errb, []string{"--json"}); code != 0 {
		t.Fatalf("runSupport --json returned %d\nstderr:\n%s", code, errb.String())
	}
	var got []supportRow
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("decode --json output: %v\n%s", err, out.String())
	}
	if want := supportReadOut(); !reflect.DeepEqual(got, want) {
		t.Fatalf("--json did not round-trip:\n got=%+v\nwant=%+v", got, want)
	}
}

// TestSupportDeterministic guards the fold + render: two builds at one commit are
// byte-identical, so the verb's output is golden-stable.
func TestSupportDeterministic(t *testing.T) {
	if !reflect.DeepEqual(supportReadOut(), supportReadOut()) {
		t.Fatal("supportReadOut is not deterministic across two calls")
	}
	if renderSupportReadOut(supportReadOut()) != renderSupportReadOut(supportReadOut()) {
		t.Fatal("renderSupportReadOut is not deterministic across two calls")
	}
}

// TestSupportTableRendersInstrument is the golden-content witness for the human view: the
// table carries every column of the per-cell instrument, and a known anchor cell renders
// its full rung . regime . target . next-action . witness line. The live grid classifies
// every Llama cell SUPPORTED (PreNorm + a CI oracle), so Llama lowers to M4 correct -> R2
// optimize, target M4 correct, witnessed by the CI oracle, routed to the rsiloop loop.
func TestSupportTableRendersInstrument(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSupport(&out, &errb, []string{}); code != 0 {
		t.Fatalf("runSupport returned %d\nstderr:\n%s", code, errb.String())
	}
	s := out.String()
	for _, want := range []string{"FAMILY", "BACKEND", "RUNG", "REGIME", "TARGET", "WITNESS", "NEXT-ACTION"} {
		if !strings.Contains(s, want) {
			t.Errorf("table missing column header %q", want)
		}
	}
	for _, want := range []string{"Llama", "M4 correct", "R2 optimize", "oracle-in-ci", "rsiloop+shipgate"} {
		if !strings.Contains(s, want) {
			t.Errorf("table missing expected Llama instrument token %q\n---\n%s", want, s)
		}
	}
}

// TestSupportFilter pins the "where is X" half: --backend narrows to one backend exactly,
// and --family narrows by case-insensitive substring.
func TestSupportFilter(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSupport(&out, &errb, []string{"--json", "--backend", "cpu"}); code != 0 {
		t.Fatalf("runSupport --backend cpu returned %d\nstderr:\n%s", code, errb.String())
	}
	var rows []supportRow
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(rows) == 0 {
		t.Fatal("--backend cpu returned no rows")
	}
	for _, r := range rows {
		if r.Backend != "cpu" {
			t.Errorf("--backend cpu leaked backend %q", r.Backend)
		}
	}

	out.Reset()
	errb.Reset()
	if code := runSupport(&out, &errb, []string{"--json", "--family", "LLAMA"}); code != 0 {
		t.Fatalf("runSupport --family LLAMA returned %d", code)
	}
	rows = nil
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(rows) == 0 {
		t.Fatal("--family LLAMA (case-insensitive) returned no rows")
	}
	for _, r := range rows {
		if !strings.Contains(strings.ToLower(r.Family), "llama") {
			t.Errorf("--family LLAMA leaked family %q", r.Family)
		}
	}
}

// TestSupportRejectsPositionalArg pins the closed-flag contract: a stray positional arg is
// a usage error (exit 2), never silently ignored.
func TestSupportRejectsPositionalArg(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSupport(&out, &errb, []string{"oops"}); code != 2 {
		t.Fatalf("runSupport with a positional arg returned %d, want 2", code)
	}
}

// TestSupportDeclaredTargets pins the target column to the scorecard's declared target
// table: an accelerated non-PreNorm fenced cell honestly targets M1, while a PreNorm hot
// path cell targets M4 correctness.
func TestSupportDeclaredTargets(t *testing.T) {
	rows := supportReadOut()
	byCell := map[string]supportRow{}
	for _, r := range rows {
		byCell[r.Family+"\x00"+r.Backend] = r
	}
	cases := []struct {
		family, backend string
		want            supportmaturity.Rung
	}{
		{"GPT-NeoX", "cuda", supportmaturity.M1Fenced},
		{"Llama", "cuda", supportmaturity.M4Correct},
	}
	for _, tc := range cases {
		row, ok := byCell[tc.family+"\x00"+tc.backend]
		if !ok {
			t.Fatalf("missing row for %s x %s", tc.family, tc.backend)
		}
		if row.Target != tc.want.String() || row.TargetName != tc.want.Label() {
			t.Errorf("%s x %s target = %s/%s, want %s/%s",
				tc.family, tc.backend, row.Target, row.TargetName, tc.want.String(), tc.want.Label())
		}
	}
}
