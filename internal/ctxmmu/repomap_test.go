package ctxmmu

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestRepoMapBoundedGraph(t *testing.T) {
	files := []RepoMapFile{
		{Path: "main.go", Content: "package p\nfunc main(){ ResolveTarget(); ResolveTarget() }\n" + strings.Repeat("// BODY_NOT_IN_MAP\n", 512), Version: "first"},
		{Path: "target.go", Content: "package p\nfunc ResolveTarget() int { return 42 }\n", Version: "second"},
		{Path: "unused.go", Content: "package p\nfunc Unused(){}\n", Version: "third"},
	}
	opts := RepoMapOptions{Tokens: 256, Mentioned: []string{"ResolveTarget"}}
	got, err := BuildRepoMap(context.Background(), files, opts)
	if err != nil {
		t.Fatal(err)
	}
	if got.Files != 3 || got.Definitions != 3 || got.References == 0 || got.Edges == 0 ||
		got.Iterations != RepoMapIterations || got.EstimatedTokens > opts.Tokens || len(got.Text) > opts.Tokens*4 {
		t.Fatalf("bad bounds/coverage: %+v", got)
	}
	target, unused := strings.Index(got.Text, `"ResolveTarget"`), strings.Index(got.Text, `"Unused"`)
	if target < 0 || unused < 0 || target >= unused {
		t.Fatalf("referenced personalized definition was not ranked first: %s", got.Text)
	}
	if strings.Contains(got.Text, "BODY_NOT_IN_MAP") || len(got.Text) >= got.InputBytes {
		t.Fatal("render is not a bounded derived symbol view")
	}
	reversed := []RepoMapFile{files[2], files[1], files[0]}
	again, err := BuildRepoMap(context.Background(), reversed, opts)
	if err != nil || !reflect.DeepEqual(got, again) {
		t.Fatalf("nondeterministic input order: %v", err)
	}
	for _, tc := range []struct {
		name    string
		files   []RepoMapFile
		options RepoMapOptions
	}{
		{"duplicate", []RepoMapFile{files[0], files[0]}, opts},
		{"path escape", []RepoMapFile{{Path: "../x.go", Content: "package p"}}, opts},
		{"partial syntax", []RepoMapFile{{Path: "x.go", Content: "package p\nfunc incomplete("}}, opts},
		{"large input", []RepoMapFile{{Path: "x.go", Content: strings.Repeat(" ", RepoMapMaxFileBytes+1)}}, opts},
		{"output cap", files, RepoMapOptions{Tokens: RepoMapMaxTokens + 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if m, err := BuildRepoMap(context.Background(), tc.files, tc.options); err == nil || m.Text != "" {
				t.Fatal("invalid input produced a map")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := BuildRepoMap(ctx, files, opts); err != context.Canceled {
		t.Fatalf("cancel=%v", err)
	}
}
