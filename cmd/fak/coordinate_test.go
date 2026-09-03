package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestCoordinateHarnessNeutralWholePath(t *testing.T) {
	var baseline coordinateReceipt
	for _, h := range []string{"claude", "codex", "opencode", "fak-native"} {
		got, err := coordinate(coordinateDemoInput(h))
		if err != nil {
			t.Fatal(err)
		}
		if !got.Accepted || got.ComputeEngine != "fak_native" || got.EvidenceSufficiency != "whole_path" {
			t.Fatalf("%s: %+v", h, got)
		}
		if h == "claude" {
			baseline = got
		} else if !reflect.DeepEqual(got, baseline) {
			t.Fatalf("harness changed neutral result\n%+v\n%+v", baseline, got)
		}
	}
}
func TestCoordinateChangesWholePathAndRejectsRawModel(t *testing.T) {
	base := coordinateDemoInput("codex")
	a, _ := coordinate(base)
	queue := base
	queue.QueueState = "saturated"
	b, _ := coordinate(queue)
	cache := base
	cache.CacheAction = "build_shared_prefix"
	c, _ := coordinate(cache)
	place := base
	place.Placement = "load_required"
	d, _ := coordinate(place)
	if a.Action == b.Action || a.Action == c.Action || a.Action == d.Action {
		t.Fatalf("actions did not change: %q %q %q %q", a.Action, b.Action, c.Action, d.Action)
	}
	raw := base
	raw.RawModelOnly = true
	r, _ := coordinate(raw)
	if r.Accepted || r.EvidenceSufficiency != "raw_model_insufficient" {
		t.Fatal(r)
	}
	off := base
	off.Coordination = false
	r, _ = coordinate(off)
	if !r.Delegated || r.Action != "delegate_existing_behavior" {
		t.Fatal(r)
	}
}
func TestCoordinateCLIAndCommittedRenderWitness(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runCoordinate(&out, &errout, []string{"--demo"}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
	want, err := os.ReadFile(filepath.Join("testdata", "coordinate-demo.golden.json"))
	if err != nil {
		t.Fatal(err)
	}
	var gotJSON, wantJSON any
	if json.Unmarshal(out.Bytes(), &gotJSON) != nil || json.Unmarshal(want, &wantJSON) != nil {
		t.Fatal("invalid render JSON")
	}
	if !reflect.DeepEqual(gotJSON, wantJSON) {
		t.Fatalf("render mismatch\ngot %s\nwant %s", out.Bytes(), want)
	}
	old := os.Stdin
	defer func() { os.Stdin = old }()
	f, err := os.CreateTemp("", "coordinate-input-*.json")
	if err != nil {
		t.Fatal(err)
	}
	name := f.Name()
	defer os.Remove(name)
	_, _ = f.WriteString(`{"schema":"fak.coordinate/1","harness":"codex","task_id":"x","workers":2,"coordination":true,"unknown":1}`)
	_ = f.Close()
	f, err = os.Open(name)
	if err != nil {
		t.Fatal(err)
	}
	os.Stdin = f
	out.Reset()
	errout.Reset()
	code := runCoordinate(&out, &errout, []string{"--json"})
	_ = f.Close()
	os.Stdin = old
	if code != 2 || !strings.Contains(errout.String(), "unknown field") {
		t.Fatalf("code=%d err=%s", code, errout.String())
	}
}
