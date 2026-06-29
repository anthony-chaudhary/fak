package main

import (
	"bytes"
	"io"
	"reflect"
	"strings"
	"testing"
)

func TestReleaseDefaultsToStatusAndPassesFlags(t *testing.T) {
	old := releaseRunScript
	defer func() { releaseRunScript = old }()

	var gotScript string
	var gotArgs []string
	releaseRunScript = func(root, script string, args []string, stdout, stderr io.Writer) int {
		gotScript = script
		gotArgs = append([]string(nil), args...)
		return 7
	}

	var out, errb bytes.Buffer
	rc := runRelease(&out, &errb, []string{"--json", "--skip-gh"})
	if rc != 7 {
		t.Fatalf("exit = %d, want 7", rc)
	}
	if gotScript != "release_status.py" {
		t.Fatalf("script = %q, want release_status.py", gotScript)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--json", "--skip-gh"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseDispatchesKnownHelper(t *testing.T) {
	old := releaseRunScript
	defer func() { releaseRunScript = old }()

	var gotScript string
	var gotArgs []string
	releaseRunScript = func(root, script string, args []string, stdout, stderr io.Writer) int {
		gotScript = script
		gotArgs = append([]string(nil), args...)
		return 0
	}

	rc := runRelease(io.Discard, io.Discard, []string{"publish", "--version", "1.2.3", "--json"})
	if rc != 0 {
		t.Fatalf("exit = %d, want 0", rc)
	}
	if gotScript != "release_publish.py" {
		t.Fatalf("script = %q, want release_publish.py", gotScript)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--version", "1.2.3", "--json"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseDispatchesShipHelper(t *testing.T) {
	old := releaseRunShip
	defer func() { releaseRunShip = old }()

	var gotArgs []string
	releaseRunShip = func(stdout, stderr io.Writer, args []string) int {
		gotArgs = append([]string(nil), args...)
		return 9
	}

	rc := runRelease(io.Discard, io.Discard, []string{"ship", "--execute", "--json"})
	if rc != 9 {
		t.Fatalf("exit = %d, want 9", rc)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--execute", "--json"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseDispatchesStableHelper(t *testing.T) {
	old := releaseRunScript
	defer func() { releaseRunScript = old }()

	var gotScript string
	var gotArgs []string
	releaseRunScript = func(root, script string, args []string, stdout, stderr io.Writer) int {
		gotScript = script
		gotArgs = append([]string(nil), args...)
		return 0
	}

	rc := runRelease(io.Discard, io.Discard, []string{"stable-context", "--codename", "2026-06-bedrock", "--json"})
	if rc != 0 {
		t.Fatalf("exit = %d, want 0", rc)
	}
	if gotScript != "stable_release_context.py" {
		t.Fatalf("script = %q, want stable_release_context.py", gotScript)
	}
	if !reflect.DeepEqual(gotArgs, []string{"--codename", "2026-06-bedrock", "--json"}) {
		t.Fatalf("args = %#v", gotArgs)
	}
}

func TestReleaseExecuteCutAddsSkipDryRun(t *testing.T) {
	args := releaseArgs("cut", []string{"--execute", "--json"})
	if !reflect.DeepEqual(args, []string{"--execute", "--json", "--skip-dry-run"}) {
		t.Fatalf("args = %#v", args)
	}
	already := releaseArgs("tag", []string{"--execute", "--skip-dry-run", "--json"})
	if !reflect.DeepEqual(already, []string{"--execute", "--skip-dry-run", "--json"}) {
		t.Fatalf("already = %#v", already)
	}
	dry := releaseArgs("cut", []string{"--json"})
	if !reflect.DeepEqual(dry, []string{"--json"}) {
		t.Fatalf("dry = %#v", dry)
	}
}

func TestReleaseUnknownSubcommand(t *testing.T) {
	var errb bytes.Buffer
	rc := runRelease(io.Discard, &errb, []string{"nope"})
	if rc != 2 {
		t.Fatalf("exit = %d, want 2", rc)
	}
	if !strings.Contains(errb.String(), "unknown subcommand") {
		t.Fatalf("missing unknown-subcommand error:\n%s", errb.String())
	}
	if !strings.Contains(errb.String(), "stable|stable-context") {
		t.Fatalf("help does not surface stable release helpers:\n%s", errb.String())
	}
}

func TestReleaseUsageSurfacesCanonicalPath(t *testing.T) {
	var out bytes.Buffer
	releaseUsage(&out)
	text := out.String()
	for _, want := range []string{
		"release_decide -> release_lock -> release_cut",
		"release_tag",
		"release_publish",
		"release-artifacts verification",
		"ship --execute",
		"staleness",
		"stable|stable-context",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("usage missing %q:\n%s", want, text)
		}
	}
}
