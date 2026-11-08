package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute/layoutcontract"
)

// layoutContractObservedFile renders an observed-layout JSON fixture from a
// registered contract so tests assert against the live registry, not copied
// constants. mutate may nil out or rewrite fields for the negative arms.
func layoutContractObservedFile(t *testing.T, c layoutcontract.Contract, mutate func(*observedLayout)) string {
	t.Helper()
	obs := observedLayout{
		StorageOrder:    strPtr(string(c.WeightOrder)),
		ScaleLayout:     strPtr(string(c.ScaleLayout)),
		InterleaveWidth: intPtr(c.InterleaveWidth),
		PaddingRule:     strPtr(c.PaddingRule),
		PackingStage:    strPtr(string(c.PackingPath)),
	}
	if mutate != nil {
		mutate(&obs)
	}
	b, err := json.Marshal(obs)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "observed.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func strPtr(s string) *string { return &s }

func intPtr(i int) *int { return &i }

func layoutContractVerdictFor(t *testing.T, rec *layoutContractCheckReceipt, field string) layoutContractVerdict {
	t.Helper()
	for _, v := range rec.Verdicts {
		if v.Field == field {
			return v
		}
	}
	t.Fatalf("verdict for %q missing in %+v", field, rec.Verdicts)
	return layoutContractVerdict{}
}

// fak-test:runtime fast est=5ms lane=default
func TestLayoutContractFamiliesStableOrder(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"layoutcontract", "families"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var rec layoutContractFamiliesReceipt
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("decode families receipt: %v\n%s", err, out.String())
	}
	if rec.Schema != layoutContractFamiliesSchema {
		t.Fatalf("schema=%q", rec.Schema)
	}
	want := layoutcontract.Families()
	if !reflect.DeepEqual(rec.Families, want) {
		t.Fatalf("families=%v want %v", rec.Families, want)
	}
	if !sort.StringsAreSorted(rec.Families) {
		t.Fatalf("families not stable-sorted: %v", rec.Families)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestLayoutContractShowClosedContracts(t *testing.T) {
	for _, fam := range layoutcontract.Families() {
		t.Run(fam, func(t *testing.T) {
			var out, errOut bytes.Buffer
			code := run(&out, &errOut, []string{"layoutcontract", "show", fam})
			if code != 0 {
				t.Fatalf("code=%d stderr=%s", code, errOut.String())
			}
			var rec layoutContractShowReceipt
			if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
				t.Fatalf("decode show receipt: %v\n%s", err, out.String())
			}
			c := layoutcontract.Must(fam)
			if rec.Schema != layoutContractShowSchema || rec.Family != fam {
				t.Fatalf("schema=%q family=%q", rec.Schema, rec.Family)
			}
			if rec.WeightOrder != string(c.WeightOrder) || rec.ScaleLayout != string(c.ScaleLayout) ||
				rec.PackingPath != string(c.PackingPath) || rec.PaddingRule != c.PaddingRule ||
				rec.InterleaveWidth != c.InterleaveWidth {
				t.Fatalf("receipt fields diverge from registry: %+v vs %+v", rec, c)
			}
			if rec.Source.Path != c.Source.Path || rec.Source.Line != c.Source.Line || rec.Source.SHA != c.Source.SHA {
				t.Fatalf("source ref diverges: %+v vs %+v", rec.Source, c.Source)
			}
			if !strings.Contains(rec.Source.Ref, ":") || !strings.Contains(rec.Source.Ref, "@") {
				t.Fatalf("ref %q is not path:line@sha form", rec.Source.Ref)
			}
			if rec.Transform.Present != c.Transform.Present() {
				t.Fatalf("transform present=%v want %v", rec.Transform.Present, c.Transform.Present())
			}
			if c.Transform.Present() && (rec.Transform.Name != c.Transform.Name || rec.Transform.Source == nil) {
				t.Fatalf("transform view incomplete: %+v", rec.Transform)
			}
			if !c.Transform.Present() && (rec.Transform.Name != "" || rec.Transform.Source != nil) {
				t.Fatalf("absent transform must carry no name/source: %+v", rec.Transform)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestLayoutContractShowRefusals(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"layoutcontract", "show", "NOT_A_FAMILY"}); code != 2 {
		t.Fatalf("unknown family code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "ErrUnknownFamily") {
		t.Fatalf("stderr must name ErrUnknownFamily: %s", errOut.String())
	}
	for _, argv := range [][]string{
		{"layoutcontract", "show"},
		{"layoutcontract", "show", "Q4_K", "extra"},
		{"layoutcontract", "show", "Q4_K", "--bogus"},
	} {
		var out, errOut bytes.Buffer
		if code := run(&out, &errOut, argv); code != 2 {
			t.Fatalf("argv=%v code=%d", argv, code)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestLayoutContractCheckVerdicts(t *testing.T) {
	q8 := layoutcontract.Must("Q8_0") // pinned interleave width: the match arm
	q4k := layoutcontract.Must("Q4_K")

	t.Run("all fields match exits 0", func(t *testing.T) {
		path := layoutContractObservedFile(t, q8, nil)
		var out, errOut bytes.Buffer
		code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", path})
		if code != 0 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
		var rec layoutContractCheckReceipt
		if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
			t.Fatalf("decode check receipt: %v\n%s", err, out.String())
		}
		if rec.Schema != layoutContractCheckSchema || rec.Result != "match" {
			t.Fatalf("schema=%q result=%q", rec.Schema, rec.Result)
		}
		if rec.Summary != (layoutContractCheckSummary{Match: 5, Mismatch: 0, Unknown: 0}) {
			t.Fatalf("summary=%+v", rec.Summary)
		}
		if rec.ObservedSHA256 == "" || !strings.HasPrefix(rec.ObservedSHA256, "sha256:") {
			t.Fatalf("observed_sha256=%q", rec.ObservedSHA256)
		}
		for _, v := range rec.Verdicts {
			if v.Verdict != layoutContractVerdictMatch {
				t.Fatalf("field %s verdict=%q want match", v.Field, v.Verdict)
			}
		}
	})

	t.Run("one mismatched field exits 1", func(t *testing.T) {
		path := layoutContractObservedFile(t, q8, func(o *observedLayout) {
			o.StorageOrder = strPtr("tile-swizzled")
		})
		var out, errOut bytes.Buffer
		code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", path})
		if code != 1 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
		var rec layoutContractCheckReceipt
		if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		if rec.Result != "mismatch" {
			t.Fatalf("result=%q", rec.Result)
		}
		v := layoutContractVerdictFor(t, &rec, "storage_order")
		if v.Verdict != layoutContractVerdictMismatch || v.Expected != string(q8.WeightOrder) || v.Observed != "tile-swizzled" {
			t.Fatalf("verdict=%+v", v)
		}
		if rec.Summary.Mismatch != 1 || rec.Summary.Match != 4 {
			t.Fatalf("summary=%+v", rec.Summary)
		}
	})

	t.Run("out-of-vocabulary value mismatches, never coerces", func(t *testing.T) {
		path := layoutContractObservedFile(t, q8, func(o *observedLayout) {
			o.StorageOrder = strPtr("diagonal")
		})
		var out, errOut bytes.Buffer
		if code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", path}); code != 1 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
		var rec layoutContractCheckReceipt
		if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		if v := layoutContractVerdictFor(t, &rec, "storage_order"); v.Verdict != layoutContractVerdictMismatch {
			t.Fatalf("verdict=%+v", v)
		}
	})

	t.Run("absent fields are unknown, exit stays 0", func(t *testing.T) {
		path := layoutContractObservedFile(t, q8, func(o *observedLayout) {
			o.PaddingRule = nil
			o.PackingStage = nil
		})
		var out, errOut bytes.Buffer
		code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", path})
		if code != 0 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
		var rec layoutContractCheckReceipt
		if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		for _, field := range []string{"padding_rule", "packing_stage"} {
			if v := layoutContractVerdictFor(t, &rec, field); v.Verdict != layoutContractVerdictUnknown {
				t.Fatalf("%s verdict=%+v", field, v)
			}
		}
		if rec.Summary.Unknown != 2 || rec.Summary.Match != 3 || rec.Result != "match" {
			t.Fatalf("summary=%+v result=%q", rec.Summary, rec.Result)
		}
	})

	t.Run("host-resolved width is unknown with a note", func(t *testing.T) {
		if q4k.InterleaveWidth != 0 {
			t.Fatalf("fixture precondition: Q4_K width=%d, want host-resolved 0", q4k.InterleaveWidth)
		}
		path := layoutContractObservedFile(t, q4k, func(o *observedLayout) {
			o.InterleaveWidth = intPtr(8)
		})
		var out, errOut bytes.Buffer
		code := run(&out, &errOut, []string{"layoutcontract", "check", "Q4_K", "--observed", path})
		if code != 0 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
		var rec layoutContractCheckReceipt
		if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
			t.Fatal(err)
		}
		v := layoutContractVerdictFor(t, &rec, "interleave_width")
		if v.Verdict != layoutContractVerdictUnknown || v.Note == "" || v.Observed != "8" {
			t.Fatalf("verdict=%+v", v)
		}
	})

	t.Run("pinned width mismatch exits 1", func(t *testing.T) {
		path := layoutContractObservedFile(t, q8, func(o *observedLayout) {
			o.InterleaveWidth = intPtr(8)
		})
		var out, errOut bytes.Buffer
		if code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", path}); code != 1 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
	})
}

// fak-test:runtime fast est=10ms lane=default
func TestLayoutContractCheckRefusals(t *testing.T) {
	q8 := layoutcontract.Must("Q8_0")

	t.Run("unknown family exits 2 naming ErrUnknownFamily", func(t *testing.T) {
		path := layoutContractObservedFile(t, q8, nil)
		var out, errOut bytes.Buffer
		code := run(&out, &errOut, []string{"layoutcontract", "check", "NOT_A_FAMILY", "--observed", path})
		if code != 2 || !strings.Contains(errOut.String(), "ErrUnknownFamily") {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
	})

	t.Run("closed vocabulary violation exits 2", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "observed.json")
		if err := os.WriteFile(path, []byte(`{"storage_order":"row-major","bogus_key":1}`), 0o644); err != nil {
			t.Fatal(err)
		}
		var out, errOut bytes.Buffer
		code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", path})
		if code != 2 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
	})

	t.Run("malformed json exits 2", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "observed.json")
		if err := os.WriteFile(path, []byte(`{"storage_order":`), 0o644); err != nil {
			t.Fatal(err)
		}
		var out, errOut bytes.Buffer
		if code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", path}); code != 2 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
	})

	t.Run("type mismatch exits 2", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "observed.json")
		if err := os.WriteFile(path, []byte(`{"interleave_width":"8"}`), 0o644); err != nil {
			t.Fatal(err)
		}
		var out, errOut bytes.Buffer
		if code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", path}); code != 2 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
	})

	t.Run("missing observed file exits 2", func(t *testing.T) {
		var out, errOut bytes.Buffer
		code := run(&out, &errOut, []string{"layoutcontract", "check", "Q8_0", "--observed", filepath.Join(t.TempDir(), "nope.json")})
		if code != 2 {
			t.Fatalf("code=%d stderr=%s", code, errOut.String())
		}
	})

	for _, argv := range [][]string{
		{"layoutcontract", "check"},
		{"layoutcontract", "check", "Q8_0"},
		{"layoutcontract", "check", "--observed", "x.json"},
		{"layoutcontract", "check", "Q8_0", "Q4_K", "--observed", "x.json"},
		{"layoutcontract", "check", "Q8_0", "--observed"},
		{"layoutcontract", "check", "Q8_0", "--observed", "x.json", "--bogus"},
	} {
		var out, errOut bytes.Buffer
		if code := run(&out, &errOut, argv); code != 2 {
			t.Fatalf("argv=%v code=%d", argv, code)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestLayoutContractOrphansDerivesCallSites(t *testing.T) {
	root := t.TempDir()
	writeFixture := func(rel, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	writeFixture("internal/model/loader.go", `package model

import "github.com/anthony-chaudhary/fak/internal/compute/layoutcontract"

func LoadWeights() { _ = layoutcontract.Get("Q4_K") }
`)
	// Test files are not production call sites.
	writeFixture("internal/ggufload/header_test.go", `package ggufload

import "github.com/anthony-chaudhary/fak/internal/compute/layoutcontract"

func TestNothing() { _ = layoutcontract.Must("Q8_0") }
`)
	// The registry's own package directory is excluded.
	writeFixture("internal/compute/layoutcontract/self.go", `package layoutcontract

func Self() { _ = All() }
`)
	// A seam reference that names no family tag attributes to no family.
	writeFixture("internal/compute/upload.go", `package compute

import "github.com/anthony-chaudhary/fak/internal/compute/layoutcontract"

func Upload() { _ = layoutcontract.All() }
`)
	// A family mention without a registry reference attributes nothing.
	writeFixture("internal/model/nomatch.go", `package model

const NVFP4Note = "the scale image is tiled"
`)
	// Outside the bounded seam roots: not scanned at all.
	writeFixture("internal/engine/outside.go", `package engine

import "github.com/anthony-chaudhary/fak/internal/compute/layoutcontract"

func Run() { _ = layoutcontract.Get("MXFP4") }
`)

	var out, errOut bytes.Buffer
	code := run(&out, &errOut, []string{"layoutcontract", "orphans", "--root", root})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var rec layoutContractOrphansReceipt
	if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
		t.Fatalf("decode orphans receipt: %v\n%s", err, out.String())
	}
	if rec.Schema != layoutContractOrphansSchema {
		t.Fatalf("schema=%q", rec.Schema)
	}
	if rec.FilesScanned != 3 {
		t.Fatalf("files_scanned=%d want 3 (loader.go, upload.go, nomatch.go)", rec.FilesScanned)
	}
	if len(rec.Families) != len(layoutcontract.Families()) {
		t.Fatalf("families=%d want %d", len(rec.Families), len(layoutcontract.Families()))
	}
	byFamily := map[string]layoutContractOrphanRow{}
	for _, row := range rec.Families {
		byFamily[row.Family] = row
	}
	q4k := byFamily["Q4_K"]
	if q4k.Orphan || !reflect.DeepEqual(q4k.CallSites, []string{"internal/model/loader.go"}) {
		t.Fatalf("Q4_K row=%+v", q4k)
	}
	for _, fam := range []string{"MXFP4", "NVFP4", "Q8_0"} {
		row := byFamily[fam]
		if !row.Orphan || len(row.CallSites) != 0 {
			t.Fatalf("%s row=%+v", fam, row)
		}
	}
	wantOrphans := []string{"MXFP4", "NVFP4", "Q8_0"}
	if !reflect.DeepEqual(rec.Orphans, wantOrphans) {
		t.Fatalf("orphans=%v want %v", rec.Orphans, wantOrphans)
	}
	if !sort.StringsAreSorted(rec.Orphans) {
		t.Fatalf("orphans not stable-sorted: %v", rec.Orphans)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestLayoutContractOrphansRefusals(t *testing.T) {
	for _, argv := range [][]string{
		{"layoutcontract", "orphans", "--root"},
		{"layoutcontract", "orphans", t.TempDir(), "extra"},
		{"layoutcontract", "orphans", "--root", filepath.Join(t.TempDir(), "missing")},
	} {
		var out, errOut bytes.Buffer
		if code := run(&out, &errOut, argv); code != 2 {
			t.Fatalf("argv=%v code=%d stderr=%s", argv, code, errOut.String())
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestLayoutContractHelpAndUsage(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"layoutcontract"}); code != 2 {
		t.Fatalf("no subcommand code=%d", code)
	}
	if code := run(&out, &errOut, []string{"layoutcontract", "bogus"}); code != 2 {
		t.Fatalf("unknown subcommand code=%d", code)
	}
	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, []string{"layoutcontract", "help"}); code != 0 {
		t.Fatalf("help code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{
		"usage: fak-dev layoutcontract <subcommand>",
		"families", "list registered quantized kernel families",
		"show FAMILY", "emit the full contract as JSON",
		"check FAMILY --observed FILE", "compare an observed layout description",
		"orphans [--root DIR]", "report registered families with no production call",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("help missing %q:\n%s", want, out.String())
		}
	}
}
