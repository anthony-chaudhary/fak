package main

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func TestDefaultSuiteCmds(t *testing.T) {
	t.Run("named packages produce build then vet", func(t *testing.T) {
		got := defaultSuiteCmds("./internal/dojo/... ./internal/dojocal/...")
		want := [][]string{
			{"go", "build", "./internal/dojo/...", "./internal/dojocal/..."},
			{"go", "vet", "./internal/dojo/...", "./internal/dojocal/..."},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("defaultSuiteCmds mismatch:\n got=%v\nwant=%v", got, want)
		}
	})

	t.Run("empty pkg string falls back to ./...", func(t *testing.T) {
		got := defaultSuiteCmds("   ")
		want := [][]string{
			{"go", "build", "./..."},
			{"go", "vet", "./..."},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("defaultSuiteCmds empty mismatch:\n got=%v\nwant=%v", got, want)
		}
	})
}

func TestWslTestSuiteCmds(t *testing.T) {
	t.Run("named packages produce a single go test invocation", func(t *testing.T) {
		got := wslTestSuiteCmds("./internal/dojo/...")
		want := [][]string{
			{"go", "test", "-short", "-count=1", "./internal/dojo/..."},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("wslTestSuiteCmds mismatch:\n got=%v\nwant=%v", got, want)
		}
	})

	t.Run("empty pkg string falls back to ./...", func(t *testing.T) {
		got := wslTestSuiteCmds("")
		want := [][]string{
			{"go", "test", "-short", "-count=1", "./..."},
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("wslTestSuiteCmds empty mismatch:\n got=%v\nwant=%v", got, want)
		}
	})
}

func TestParseFloat(t *testing.T) {
	cases := []struct {
		name    string
		in      string
		want    float64
		wantErr bool
	}{
		{"plain decimal", "0.75", 0.75, false},
		{"leading/trailing whitespace trimmed", "  1.25  ", 1.25, false},
		{"integer", "3", 3, false},
		{"non-numeric errors", "not-a-number", 0, true},
		{"empty errors", "", 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseFloat(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseFloat(%q) = %v, want error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseFloat(%q) unexpected error: %v", tc.in, err)
			}
			if got != tc.want {
				t.Fatalf("parseFloat(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestCandidateFromRow(t *testing.T) {
	t.Run("parses a well-formed RECALIBRATE label", func(t *testing.T) {
		wc, err := candidateFromRow(rsiloop.Row{Candidate: "RECALIBRATE estimate/fold -> 0.42"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if wc.Lever != "estimate" {
			t.Errorf("Lever = %q, want %q", wc.Lever, "estimate")
		}
		if wc.Metric != "fold" {
			t.Errorf("Metric = %q, want %q", wc.Metric, "fold")
		}
		if wc.NewClaimed != 0.42 {
			t.Errorf("NewClaimed = %v, want %v", wc.NewClaimed, 0.42)
		}
	})

	t.Run("refuses a non-RECALIBRATE label", func(t *testing.T) {
		_, err := candidateFromRow(rsiloop.Row{Candidate: "REPROJECT estimate/fold -> 0.42"})
		if err == nil {
			t.Fatal("expected refusal for a non-RECALIBRATE row, got nil")
		}
	})

	t.Run("errors when the value arrow is missing", func(t *testing.T) {
		_, err := candidateFromRow(rsiloop.Row{Candidate: "RECALIBRATE estimate/fold 0.42"})
		if err == nil {
			t.Fatal("expected error when ' -> ' separator is absent, got nil")
		}
	})

	t.Run("errors when lever/metric is missing the slash", func(t *testing.T) {
		_, err := candidateFromRow(rsiloop.Row{Candidate: "RECALIBRATE estimatefold -> 0.42"})
		if err == nil {
			t.Fatal("expected error when lever/metric has no slash, got nil")
		}
	})

	t.Run("errors when the value is not a float", func(t *testing.T) {
		_, err := candidateFromRow(rsiloop.Row{Candidate: "RECALIBRATE estimate/fold -> not-a-number"})
		if err == nil {
			t.Fatal("expected error when value is non-numeric, got nil")
		}
	})
}
