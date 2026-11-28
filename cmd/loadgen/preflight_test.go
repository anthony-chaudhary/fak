package main

import (
	"errors"
	"testing"
	"time"
)

func TestPreflight(t *testing.T) {
	dir := t.TempDir()
	okDial := func(string, time.Duration) error { return nil }
	badDial := func(string, time.Duration) error { return errors.New("refused") }
	nofile := func(soft, hard uint64) func() (uint64, uint64, error) {
		return func() (uint64, uint64, error) { return soft, hard, nil }
	}
	statuses := func(cs []PreflightCheck) map[string]PreflightStatus {
		m := map[string]PreflightStatus{}
		for _, c := range cs {
			m[c.Name] = c.Status
		}
		return m
	}

	cs, ok := RunPreflight(PreflightConfig{ArtifactDir: dir, Endpoint: "h:1", Dial: okDial, WantNoFile: 100, NoFile: nofile(200, 200)})
	if !ok || statuses(cs)["artifact_dir"] != PreflightPass || statuses(cs)["nofile_limit"] != PreflightPass || statuses(cs)["endpoint"] != PreflightPass {
		t.Fatalf("all-green run = %+v ok=%v", cs, ok)
	}

	cs, ok = RunPreflight(PreflightConfig{ArtifactDir: dir, WantNoFile: 100, NoFile: nofile(10, 500)})
	if !ok || statuses(cs)["nofile_limit"] != PreflightWarn {
		t.Fatalf("raisable soft limit must warn without blocking: %+v ok=%v", cs, ok)
	}

	cs, ok = RunPreflight(PreflightConfig{ArtifactDir: dir, Endpoint: "h:1", Dial: badDial, WantNoFile: 100, NoFile: nofile(10, 50)})
	if ok || statuses(cs)["nofile_limit"] != PreflightFail || statuses(cs)["endpoint"] != PreflightFail {
		t.Fatalf("hard floor and dead endpoint must both fail and both report: %+v ok=%v", cs, ok)
	}

	if cs, ok = RunPreflight(PreflightConfig{}); ok || statuses(cs)["artifact_dir"] != PreflightFail {
		t.Fatalf("empty artifact dir must fail: %+v ok=%v", cs, ok)
	}
}
