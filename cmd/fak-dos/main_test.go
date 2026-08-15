package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dosdecision"
)

func TestCommandAddReadCleanup(t *testing.T) {
	ws := t.TempDir()
	var out, errOut bytes.Buffer
	add := []string{"decisions", "add", "--workspace", ws, "--key", "bench-6349-blank", "--action", "OPEN_ISSUE", "--severity", "P1", "--payload", `{"case":"blank"}`, "--json"}
	if code := run(&out, &errOut, add); code != 0 {
		t.Fatalf("add code=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, add); code != 0 {
		t.Fatalf("repeat code=%d stderr=%s", code, errOut.String())
	}
	var repeated struct {
		Created bool `json:"created"`
	}
	if err := json.Unmarshal(out.Bytes(), &repeated); err != nil || repeated.Created {
		t.Fatalf("repeat=%s err=%v", out.String(), err)
	}
	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--native=false", "--json"}); code != 0 {
		t.Fatalf("list code=%d stderr=%s", code, errOut.String())
	}
	var rows []map[string]any
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil || len(rows) != 1 {
		t.Fatalf("list=%s err=%v", out.String(), err)
	}
	if rows[0]["key"] != "bench-6349-blank" || rows[0]["action"] != "OPEN_ISSUE" || rows[0]["severity"] != "P1" {
		t.Fatalf("wrong row: %+v", rows[0])
	}
	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, []string{"decisions", "remove", "--workspace", ws, "--key", "bench-6349-blank", "--json"}); code != 0 {
		t.Fatalf("remove code=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--native=false", "--json"}); code != 0 {
		t.Fatal(errOut.String())
	}
	if string(bytes.TrimSpace(out.Bytes())) != "[]" {
		t.Fatalf("residue: %s", out.String())
	}
}

func TestListDelegatesNativeReader(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fixture; exercised under WSL validation")
	}
	ws := t.TempDir()
	bin := t.TempDir()
	script := filepath.Join(bin, "dos")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nprintf '[{\"kind\":\"NATIVE\"}]'\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", bin+string(os.PathListSeparator)+os.Getenv("PATH"))
	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var rows []map[string]any
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil || len(rows) != 1 || rows[0]["kind"] != "NATIVE" {
		t.Fatalf("rows=%s err=%v", out.String(), err)
	}
}

func TestListRevalidatesReleasedNativeCollision(t *testing.T) {
	ws := t.TempDir()
	oldNative, oldLive := nativeDecisions, liveLanes
	t.Cleanup(func() { nativeDecisions, liveLanes = oldNative, oldLive })
	nativeDecisions = func(string, io.Writer) ([]dosdecision.Row, error) {
		return []dosdecision.Row{{
			"kind": "ARBITER_REFUSE", "resolver_kind": "HUMAN", "lane": "candidate",
			"reason_text": "lane 'cmd' is already held by a live loop — pick a different --lane or wait.",
			"evidence":    []any{"journal seq #1733"}, "resolved": false,
		}}, nil
	}
	liveLanes = func(string) dosdecision.LiveSet {
		return dosdecision.LiveSet{Lanes: []string{"docs"}, Known: true}
	}

	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--json"}); code != 0 {
		t.Fatalf("active list code=%d stderr=%s", code, errOut.String())
	}
	if got := string(bytes.TrimSpace(out.Bytes())); got != "[]" {
		t.Fatalf("active queue = %s, want [] after blocker release", got)
	}

	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--all", "--json"}); code != 0 {
		t.Fatalf("all list code=%d stderr=%s", code, errOut.String())
	}
	var all []dosdecision.Row
	if err := json.Unmarshal(out.Bytes(), &all); err != nil || len(all) != 1 {
		t.Fatalf("all=%s err=%v", out.String(), err)
	}
	if all[0]["resolved"] != true || all[0]["resolution"] != dosdecision.ResolutionLeaseReleased {
		t.Fatalf("resolved history = %#v", all[0])
	}

	out.Reset()
	errOut.Reset()
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--summary", "--json"}); code != 0 {
		t.Fatalf("summary code=%d stderr=%s", code, errOut.String())
	}
	var summary struct {
		Cleared int               `json:"cleared"`
		Active  []dosdecision.Row `json:"active"`
	}
	if err := json.Unmarshal(out.Bytes(), &summary); err != nil {
		t.Fatalf("summary=%s err=%v", out.String(), err)
	}
	if summary.Cleared != 1 || len(summary.Active) != 0 {
		t.Fatalf("summary=%s", out.String())
	}
}

func TestListKeepsCollisionWhenBlockerIsLive(t *testing.T) {
	ws := t.TempDir()
	oldNative, oldLive := nativeDecisions, liveLanes
	t.Cleanup(func() { nativeDecisions, liveLanes = oldNative, oldLive })
	nativeDecisions = func(string, io.Writer) ([]dosdecision.Row, error) {
		return []dosdecision.Row{{
			"kind": "ARBITER_REFUSE", "lane": "candidate",
			"reason_text": "lane 'cmd' is already held by a live loop",
		}}, nil
	}
	liveLanes = func(string) dosdecision.LiveSet {
		return dosdecision.LiveSet{Lanes: []string{"cmd"}, Known: true}
	}

	var out, errOut bytes.Buffer
	if code := run(&out, &errOut, []string{"decisions", "list", "--workspace", ws, "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var rows []dosdecision.Row
	if err := json.Unmarshal(out.Bytes(), &rows); err != nil || len(rows) != 1 {
		t.Fatalf("rows=%s err=%v", out.String(), err)
	}
}
