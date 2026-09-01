package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/launchguard"
)

func TestLaunchguardStatusAndReset(t *testing.T) {
	t.Setenv("FAK_LAUNCHGUARD_DIR", t.TempDir())
	g, err := defaultLaunchguard()
	if err != nil {
		t.Fatal(err)
	}
	_, lease, err := g.Admit("service:test")
	if err != nil {
		t.Fatal(err)
	}
	if err := lease.Finish(false); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if code := runLaunchguard(&out, &stderr, []string{"status", "--identity", "service:test"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if strings.Contains(out.String(), "service:test") || !strings.Contains(out.String(), "attempts=1/3") {
		t.Fatalf("status=%q", out.String())
	}
	out.Reset()
	stderr.Reset()
	if code := runLaunchguard(&out, &stderr, []string{"reset", "--identity", "service:test"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(out.String(), launchguard.StableIdentity("service:test")) {
		t.Fatalf("reset=%q", out.String())
	}
	status, err := g.Inspect("service:test")
	if err != nil || status.Attempts != 0 || status.Quarantined {
		t.Fatalf("status=%+v err=%v", status, err)
	}
}

func TestLaunchguardRequiresIdentity(t *testing.T) {
	var out, stderr bytes.Buffer
	if code := runLaunchguard(&out, &stderr, []string{"status"}); code != 2 || !strings.Contains(stderr.String(), "--identity is required") {
		t.Fatalf("code=%d stderr=%q", code, stderr.String())
	}
}
