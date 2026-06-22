package pathlint

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"testing"
)

// parseSrc is a tiny helper to AST-parse a synthetic source string.
func parseSrc(t *testing.T, src string) *ast.File {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), "synthetic.go", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return f
}

// TestAnalyzeFiles verifies the witness itself (verify the verifier): a path flag that
// is expanded must NOT be flagged; one that is not expanded MUST be; and a non-path
// flag is ignored either way.
func TestAnalyzeFiles(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want []string // offending flag names, sorted
	}{
		{
			name: "expanded gguf is clean",
			src: `package main
import "x/pathutil"
func main() {
	gguf := flag.String("gguf", "", "")
	*gguf = pathutil.ExpandTilde(*gguf)
	_ = gguf
}`,
			want: nil,
		},
		{
			name: "unexpanded gguf is flagged",
			src: `package main
func main() {
	gguf := flag.String("gguf", "", "")
	_ = gguf
}`,
			want: []string{"gguf"},
		},
		{
			name: "bare ExpandTilde call counts",
			src: `package main
func main() {
	hf := flag.String("hf", "", "")
	*hf = ExpandTilde(*hf)
	_ = hf
}`,
			want: nil,
		},
		{
			name: "fs.String subcommand flags are seen",
			src: `package main
func run() {
	p := fs.String("tokenizer", "", "")
	_ = p
}`,
			want: []string{"tokenizer"},
		},
		{
			name: "non-path flag is ignored",
			src: `package main
func main() {
	n := flag.String("name", "", "")
	_ = n
}`,
			want: nil,
		},
		{
			name: "one expanded one not",
			src: `package main
func main() {
	gguf := flag.String("gguf", "", "")
	dir := flag.String("dir", "", "")
	*gguf = pathutil.ExpandTilde(*gguf)
	_, _ = gguf, dir
}`,
			want: []string{"dir"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := analyzeFiles("synthetic", []*ast.File{parseSrc(t, tc.src)})
			gotFlags := map[string]bool{}
			for _, o := range got {
				gotFlags[o.Flag] = true
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v offenses, want flags %v", got, tc.want)
			}
			for _, w := range tc.want {
				if !gotFlags[w] {
					t.Errorf("expected flag %q to be flagged; got %v", w, got)
				}
			}
		})
	}
}

// TestRepoCmdsExpandTilde is the live enforcement: every command in the real cmd/ tree
// that takes a model/tokenizer/dir path flag must route it through ExpandTilde. A new
// command that forgets fails here with the offending list — the boundary stays closed.
func TestRepoCmdsExpandTilde(t *testing.T) {
	root := repoRoot(t)
	offenses, err := ScanCmdTree(root)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(offenses) > 0 {
		t.Errorf("%d path flags never reach pathutil.ExpandTilde:", len(offenses))
		for _, o := range offenses {
			t.Errorf("  %s", o)
		}
		t.Errorf("fix: add `*<flag> = pathutil.ExpandTilde(*<flag>)` right after flag parsing")
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
