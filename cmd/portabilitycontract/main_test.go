package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	pc "github.com/anthony-chaudhary/fak/internal/portabilitycontract"
)

const fixtureDir = "../../internal/portabilitycontract/testdata"

func fixture(t *testing.T, name string) (string, pc.Package) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(fixtureDir, name))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	var p pc.Package
	if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatal(err)
	}
	return path, p
}

func TestRunUsageExitCodes(t *testing.T) {
	cases := map[string][]string{
		"no arguments": nil,
		"no package":   {"--check"},
		"two packages": {"--check", "a.json", "b.json"},
		"unknown flag": {"--bogus", "a.json"},
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			var out, errb bytes.Buffer
			if code := run(args, &out, &errb); code != 2 {
				t.Fatalf("exit=%d, want 2", code)
			}
			if out.Len() != 0 {
				t.Fatalf("stdout not empty: %q", out.String())
			}
		})
	}
}

func TestRunFailsClosed(t *testing.T) {
	good, _ := fixture(t, "representative.golden.json")
	invalid, _ := fixture(t, "invalid.schema.json")
	cases := map[string][]string{
		"missing file":   {"--check", filepath.Join(t.TempDir(), "absent.json")},
		"invalid schema": {"--check", invalid},
		"no action":      {good},
	}
	for name, args := range cases {
		t.Run(name, func(t *testing.T) {
			var out, errb bytes.Buffer
			if code := run(args, &out, &errb); code != 1 {
				t.Fatalf("exit=%d, want 1", code)
			}
			if out.Len() != 0 || errb.Len() == 0 {
				t.Fatalf("stdout=%q stderr=%q", out.String(), errb.String())
			}
		})
	}
}

func TestRunCheckReportsValidPackage(t *testing.T) {
	path, p := fixture(t, "representative.golden.json")
	var out, errb bytes.Buffer
	if code := run([]string{"--check", path}, &out, &errb); code != 0 {
		t.Fatalf("exit=%d stderr=%q", code, errb.String())
	}
	if got, want := strings.TrimSpace(out.String()), fmt.Sprintf("VALID %s objects=%d", p.PackageID, count(p)); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestRunWriteIdentitiesIsStableOnGoldenPackage(t *testing.T) {
	path, p := fixture(t, "representative.golden.json")
	var out, errb bytes.Buffer
	if code := run([]string{"--write-identities", path}, &out, &errb); code != 0 {
		t.Fatalf("exit=%d stderr=%q", code, errb.String())
	}
	if got := strings.TrimSpace(out.String()); got != p.PackageID {
		t.Fatalf("identity=%q, want %q", got, p.PackageID)
	}
	out.Reset()
	if code := run([]string{"--check", path}, &out, &errb); code != 0 {
		t.Fatalf("rewritten package no longer checks: exit=%d stderr=%q", code, errb.String())
	}
}
