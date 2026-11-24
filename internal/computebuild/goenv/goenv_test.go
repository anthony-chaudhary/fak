package goenv

import (
	"path/filepath"
	"slices"
	"testing"
)

func mapLookup(m map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := m[k]; return v, ok }
}

func TestTrimpathFlags(t *testing.T) {
	for in, want := range map[string]string{
		"":                        "-trimpath",
		"  -mod=mod  ":            "-mod=mod -trimpath",
		"-trimpath":               "-trimpath",
		"--trimpath":              "--trimpath",
		"-trimpath=false -race":   "-trimpath=false -race",
		"-race -trimpath=true":    "-race -trimpath=true",
		"-ldflags=-X=a.trimpath=": "-ldflags=-X=a.trimpath= -trimpath",
	} {
		if got := TrimpathFlags(in); got != want {
			t.Errorf("TrimpathFlags(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestGoFlagsHonorsOptOut(t *testing.T) {
	if got := GoFlags("-race", mapLookup(nil)); got != "-race -trimpath" {
		t.Fatalf("GoFlags = %q", got)
	}
	if got := GoFlags("-race", mapLookup(map[string]string{NoTrimpathEnv: " 1 "})); got != "-race" {
		t.Fatalf("GoFlags with opt-out = %q", got)
	}
	if got := GoFlags("-race", mapLookup(map[string]string{NoTrimpathEnv: "0"})); got != "-race -trimpath" {
		t.Fatalf("GoFlags with %s=0 = %q", NoTrimpathEnv, got)
	}
}

func TestSharedCacheRequiresAbsolutePath(t *testing.T) {
	abs := filepath.Join(t.TempDir(), "cache")
	for _, tc := range []struct {
		val  *string
		want string
	}{
		{nil, ""},
		{ptr(""), ""},
		{ptr("off"), ""},
		{ptr("relative"), ""},
		{ptr(abs + "\x00"), ""},
		{ptr(" " + abs + " "), abs},
	} {
		m := map[string]string{}
		if tc.val != nil {
			m[SharedCacheEnv] = *tc.val
		}
		got, ok := SharedCache(mapLookup(m))
		if got != tc.want || ok != (tc.want != "") {
			t.Errorf("SharedCache(%v) = %q, %v; want %q", tc.val, got, ok, tc.want)
		}
	}
}

func TestEnvironLookupLastWins(t *testing.T) {
	lookup := EnvironLookup([]string{"GOFLAGS=-a", "junk", "GOFLAGS=-b"})
	if v, ok := lookup("GOFLAGS"); !ok || v != "-b" {
		t.Fatalf("lookup = %q, %v", v, ok)
	}
	if _, ok := lookup("GOCACHE"); ok {
		t.Fatal("missing key reported present")
	}
}

func TestTrimpathArgsHonorsOptOut(t *testing.T) {
	if got := TrimpathArgs(mapLookup(nil)); !slices.Equal(got, []string{"-trimpath"}) {
		t.Fatalf("TrimpathArgs default = %q", got)
	}
	if got := TrimpathArgs(mapLookup(map[string]string{NoTrimpathEnv: "1"})); got != nil {
		t.Fatalf("TrimpathArgs with %s=1 = %q, want nil", NoTrimpathEnv, got)
	}
}

func TestEnvironAppendsTrimpathOnce(t *testing.T) {
	env := []string{"PATH=/bin", "GOFLAGS=-race"}
	got := Environ(env)
	if v, _ := EnvironLookup(got)("GOFLAGS"); v != "-race -trimpath" {
		t.Fatalf("Environ GOFLAGS = %q (env %q)", v, got)
	}
	if len(env) != 2 || env[1] != "GOFLAGS=-race" {
		t.Fatalf("Environ mutated its input: %q", env)
	}
	if again := Environ(got); len(again) != len(got) {
		t.Fatalf("Environ re-appended an existing -trimpath: %q", again)
	}
	optOut := []string{NoTrimpathEnv + "=1", "GOFLAGS=-race"}
	if got := Environ(optOut); len(got) != len(optOut) {
		t.Fatalf("Environ ignored %s=1: %q", NoTrimpathEnv, got)
	}
}

func TestTrimsSubcommandExcludesVet(t *testing.T) {
	for sub, want := range map[string]bool{"test": true, "build": true, "list": true, "vet": false} {
		if got := TrimsSubcommand(sub); got != want {
			t.Errorf("TrimsSubcommand(%q) = %v, want %v", sub, got, want)
		}
	}
}

func ptr(s string) *string { return &s }
