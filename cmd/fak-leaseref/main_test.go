package main

import (
	"bytes"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/leaserefcli"
)

// fak-test:runtime unit est=1ms lane=default
func TestStandaloneSessionPublishDispatch(t *testing.T) {
	for _, args := range [][]string{{"--help"}, {}, {"--session", "invalid/id"}, {"--ttl", "bad"}, {"unexpected"}} {
		var gotOut, gotErr, wantOut, wantErr bytes.Buffer
		got := run(&gotOut, &gotErr, append([]string{"session-publish"}, args...))
		want := leaserefcli.RunSessionPublish(&wantOut, &wantErr, args)
		if got != want || gotOut.String() != wantOut.String() || gotErr.String() != wantErr.String() {
			t.Fatalf("args=%v got=(%d,%q,%q) want=(%d,%q,%q)", args, got, gotOut.String(), gotErr.String(), want, wantOut.String(), wantErr.String())
		}
	}
	for _, args := range [][]string{nil, {"acquire"}, {"release"}, {"commit"}, {"publish"}, {"session-publish-other"}} {
		var out, stderr bytes.Buffer
		if got := run(&out, &stderr, args); got != 2 || out.Len() != 0 {
			t.Fatalf("unsupported %v: exit=%d out=%q", args, got, out.String())
		}
	}
	var out, stderr bytes.Buffer
	if code := run(&out, &stderr, []string{"--help"}); code != 0 || stderr.Len() != 0 || !strings.Contains(out.String(), "Local session descriptor only") {
		t.Fatalf("help: exit=%d out=%q err=%q", code, out.String(), stderr.String())
	}
}

// fak-test:runtime unit est=100ms lane=default
func TestStandaloneSessionPublishDependencyBoundary(t *testing.T) {
	root, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(root, "go.mod")); err == nil {
			break
		}
		parent := filepath.Dir(root)
		if parent == root {
			t.Fatal("module root not found")
		}
		root = parent
	}
	const prefix = "github.com/anthony-chaudhary/fak/"
	// This is a conservative source boundary: inspect every production Go file,
	// including other-platform/tagged files, not a serial or selected-platform stub.
	// A new dependency must be reviewed explicitly rather than silently relinking
	// inference, developer tools or the full command registry into this leaf.
	allowed := map[string]bool{
		"cmd/fak-leaseref": true, "internal/leaserefcli": true,
		"internal/sessionref": true, "internal/pathutil": true, "internal/refid": true,
		"internal/windowgate": true, "internal/strmatch": true, "pkg/sysproc": true,
	}
	queue := []string{"cmd/fak-leaseref"}
	seen := map[string]bool{}
	for len(queue) != 0 {
		pkg := queue[0]
		queue = queue[1:]
		if seen[pkg] {
			continue
		}
		if !allowed[pkg] {
			t.Fatalf("dependency boundary crossed: %s", pkg)
		}
		seen[pkg] = true
		entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(pkg)))
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			name := entry.Name()
			if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") || strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") {
				continue
			}
			parsed, err := parser.ParseFile(token.NewFileSet(), filepath.Join(root, filepath.FromSlash(pkg), name), nil, parser.ImportsOnly)
			if err != nil {
				t.Fatal(err)
			}
			for _, spec := range parsed.Imports {
				path, err := strconv.Unquote(spec.Path.Value)
				if err != nil {
					t.Fatal(err)
				}
				if strings.HasPrefix(path, prefix) {
					queue = append(queue, strings.TrimPrefix(path, prefix))
				}
			}
		}
	}
	pkgs := make([]string, 0, len(seen))
	for pkg := range seen {
		pkgs = append(pkgs, pkg)
	}
	sort.Strings(pkgs)
	t.Logf("all-platform repository dependency closure (%d): %v", len(pkgs), pkgs)
	for _, unrelated := range []string{"internal/leaseref", "internal/rehydrate", "internal/resume", "internal/model", "internal/compute", "internal/devcmd", "internal/agent", "internal/gateway", "cmd/fak"} {
		if seen[unrelated] {
			t.Fatalf("unrelated source package %s invalidates standalone tool", unrelated)
		}
	}
}
