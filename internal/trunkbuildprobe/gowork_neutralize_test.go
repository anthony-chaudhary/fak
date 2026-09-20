package trunkbuildprobe

import (
	"os"
	"path/filepath"
	"testing"
)

// TestGoWorkUsesEscapingPath covers the workspace-escape classifier that gates
// go.work neutralization in the committed-tree probe. A committed `go.work`
// naming a sibling outside the extract must be detected; an in-tree workspace
// must be left alone.
func TestGoWorkUsesEscapingPath(t *testing.T) {
	root := filepath.Join("C:", string(filepath.Separator), "work", "fak-private")
	cases := []struct {
		name string
		body string
		want bool
	}{
		{
			name: "live fak-private form escapes",
			body: "go 1.26.0\n\nuse (\n\t.\n\t../fak\n)\n",
			want: true,
		},
		{
			name: "single-line relative escape",
			body: "go 1.26.0\nuse ../fak\n",
			want: true,
		},
		{
			name: "in-tree block stays",
			body: "go 1.26.0\n\nuse (\n\t.\n\t./internal/foo\n)\n",
			want: false,
		},
		{
			name: "in-tree single stays",
			body: "go 1.26.0\nuse ./sub\n",
			want: false,
		},
		{
			name: "comments ignored",
			body: "go 1.26.0\n// use ../fak is only a comment\nuse .\n",
			want: false,
		},
		{
			name: "absolute escape",
			body: "go 1.26.0\nuse C:\\other\\repo\n",
			want: true,
		},
		{
			name: "deep relative escape",
			body: "go 1.26.0\nuse sub/../../..\n",
			want: true,
		},
		{
			name: `windows long-path namespace in-tree`,
			body: "go 1.26.0\nuse \\\\?\\C:\\work\\fak-private\\sub\n",
			want: false,
		},
		{
			name: "empty is inert",
			body: "",
			want: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := goWorkUsesEscapingPath(tc.body, root); got != tc.want {
				t.Fatalf("goWorkUsesEscapingPath(%q) = %v, want %v", tc.body, got, tc.want)
			}
		})
	}
}

// TestNeutralizeEscapingGoWork proves the probe's committed-tree extract drops
// an escaping go.work and leaves every other file (and an in-tree go.work)
// intact. This is the regression witness for the fleet-wide TREE_POISONED
// freeze: without it, `go build ./cmd/fak` fails with
// "cannot load module ../fak listed in go.work file" and dispatch refuses.
func TestNeutralizeEscapingGoWork(t *testing.T) {
	t.Run("escaping removed", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "go.work"), []byte("go 1.26.0\nuse ( .\n../fak )\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module x\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		neutralizeEscapingGoWork(dir)
		if _, err := os.Stat(filepath.Join(dir, "go.work")); !os.IsNotExist(err) {
			t.Fatalf("escaping go.work should be removed, stat err = %v", err)
		}
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err != nil {
			t.Fatalf("unrelated file must be untouched: %v", err)
		}
	})

	t.Run("in-tree kept", func(t *testing.T) {
		dir := t.TempDir()
		body := "go 1.26.0\nuse ( .\n./sub )\n"
		if err := os.WriteFile(filepath.Join(dir, "go.work"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		neutralizeEscapingGoWork(dir)
		got, err := os.ReadFile(filepath.Join(dir, "go.work"))
		if err != nil {
			t.Fatalf("in-tree go.work must survive: %v", err)
		}
		if string(got) != body {
			t.Fatalf("in-tree go.work mutated: %q", got)
		}
	})

	t.Run("absent is no-op", func(t *testing.T) {
		neutralizeEscapingGoWork(t.TempDir())
	})
}
