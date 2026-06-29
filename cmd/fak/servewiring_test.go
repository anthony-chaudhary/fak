package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestServeWiringCheckPassesOnRealTree proves the audited baseline matches the real serve.go:
// every Config-backed row is still fed by the gateway.New literal, and no Config feature is
// unaudited. This is the test that fails the moment a real wiring regression lands.
func TestServeWiringCheckPassesOnRealTree(t *testing.T) {
	root := repoRootFromTest(t)
	var out, errb bytes.Buffer
	code := runServeWiring(&out, &errb, []string{"--workspace", root, "--check"})
	if code != 0 {
		t.Fatalf("serve-wiring --check on the real tree returned %d, want 0\nstdout:\n%s\nstderr:\n%s", code, out.String(), errb.String())
	}
	if !strings.HasPrefix(out.String(), "OK") {
		t.Fatalf("serve-wiring --check OK output missing; got:\n%s", out.String())
	}
}

func TestServeRegistersFakReadEngineBeforeGatewayNew(t *testing.T) {
	root := repoRootFromTest(t)
	servePath := filepath.Join(root, "cmd", "fak", "serve.go")
	body, err := os.ReadFile(servePath)
	if err != nil {
		t.Fatalf("read serve.go: %v", err)
	}
	src := string(body)
	for _, want := range []string{
		`"github.com/anthony-chaudhary/fak/internal/agent"`,
		"func configureServeToolEngines()",
		`agent.RegisterReadEngine("")`,
		"configureServeToolEngines()",
		"srv, err := gateway.New",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("serve.go missing %q", want)
		}
	}
	call := "\n\tconfigureServeToolEngines()\n"
	if !strings.Contains(src, call) {
		t.Fatalf("serve.go missing startup call %q", strings.TrimSpace(call))
	}
	if strings.Index(src, call) > strings.Index(src, "srv, err := gateway.New") {
		t.Fatal("serve must register fak_read's engine before gateway.New")
	}
}

// TestServeWiringDetectsDroppedField proves the drift guard fires: a serve.go that stops
// setting a Config field flips its row to dead-wired and reds --check. This is the dead-wiring
// regression the verb exists to catch.
func TestServeWiringDetectsDroppedField(t *testing.T) {
	root := repoRootFromTest(t)
	gw, err := os.ReadFile(filepath.Join(root, "internal", "gateway", "gateway.go"))
	if err != nil {
		t.Fatalf("read real gateway.go: %v", err)
	}

	fake := t.TempDir()
	if err := os.MkdirAll(filepath.Join(fake, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(fake, "internal", "gateway"), 0o755); err != nil {
		t.Fatal(err)
	}
	// Real gateway.go (the Config fields still exist) but a serve.go that sets only Model:
	// every audited Config-backed feature should now read as dead-wired.
	if err := os.WriteFile(filepath.Join(fake, "internal", "gateway", "gateway.go"), gw, 0o644); err != nil {
		t.Fatal(err)
	}
	stub := "package main\nfunc x() {\n\tsrv, _ := gateway.New(gateway.Config{\n\t\tModel: *model,\n\t})\n\t_ = srv\n}\n"
	if err := os.WriteFile(filepath.Join(fake, "cmd", "fak", "serve.go"), []byte(stub), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runServeWiring(&out, &errb, []string{"--workspace", fake, "--check"})
	if code == 0 {
		t.Fatalf("serve-wiring --check passed on a serve.go that sets no features; want drift exit 1\n%s", out.String())
	}
	if !strings.Contains(out.String(), "dead-wired") {
		t.Fatalf("drift output should name dead-wired rows; got:\n%s", out.String())
	}
}

// TestServeWiringMarkdownRenders proves --md emits a table row per audited feature.
func TestServeWiringMarkdownRenders(t *testing.T) {
	root := repoRootFromTest(t)
	var out, errb bytes.Buffer
	if code := runServeWiring(&out, &errb, []string{"--workspace", root, "--md"}); code != 0 {
		t.Fatalf("serve-wiring --md returned %d\nstderr:\n%s", code, errb.String())
	}
	md := out.String()
	if !strings.Contains(md, "| Feature | Status |") {
		t.Fatalf("markdown header missing; got:\n%s", md)
	}
	for _, r := range servewiringData {
		if !strings.Contains(md, "`"+r.Feature+"`") {
			t.Errorf("markdown table missing row for %q", r.Feature)
		}
	}
}

// repoRootFromTest walks up from the test's working dir to the module root (the dir holding
// go.mod), so the test is independent of repoRoot()'s git assumptions.
func repoRootFromTest(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("go.mod not found walking up from test dir")
		}
		dir = parent
	}
}
