package versionskew

import (
	"reflect"
	"testing"
)

func TestVersionedParity(t *testing.T) {
	t.Setenv("FAK_APP_VERSION", "env-1.0")

	input := map[string]any{"a": 1}
	got := Versioned(input, "")
	want := map[string]any{"a": 1, "version": "env-1.0"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Versioned() = %#v, want %#v", got, want)
	}
	if !reflect.DeepEqual(input, map[string]any{"a": 1}) {
		t.Fatalf("Versioned mutated input: %#v", input)
	}
}

func TestVersionedExplicitVersionAndExistingValueWin(t *testing.T) {
	if got := Versioned(map[string]any{"a": 1}, "1.2.3"); got["version"] != "1.2.3" {
		t.Fatalf("explicit version = %#v, want 1.2.3", got["version"])
	}
	if got := Versioned(map[string]any{"version": "9.9.9"}, "1.2.3"); got["version"] != "9.9.9" {
		t.Fatalf("existing version = %#v, want 9.9.9", got["version"])
	}
}

func TestVersionedRowsParity(t *testing.T) {
	rows := []map[string]any{{"a": 1}, {"b": 2}}
	got := VersionedRows(rows, "7")
	want := []map[string]any{{"a": 1, "version": "7"}, {"b": 2, "version": "7"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("VersionedRows() = %#v, want %#v", got, want)
	}
	if _, ok := rows[0]["version"]; ok {
		t.Fatalf("VersionedRows mutated input: %#v", rows)
	}
}

func TestBenchmarkConceptVersionParity(t *testing.T) {
	if BenchmarkConceptVersion != "fak.benchmark-concept.v1" {
		t.Fatalf("BenchmarkConceptVersion = %q", BenchmarkConceptVersion)
	}
}
