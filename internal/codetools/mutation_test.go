package codetools

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestWriteCreateOverwriteAndModes(t *testing.T) {
	ts, root := newTestToolset(t)
	out, bad := ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "dir/a.txt", Content: "one", Mode: "create"}))
	if bad {
		t.Fatalf("create: %s", out)
	}
	if b, _ := os.ReadFile(filepath.Join(root, "dir", "a.txt")); string(b) != "one" {
		t.Fatalf("created = %q", b)
	}
	if out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "dir/a.txt", Content: "two", Mode: "create"})); !bad || errCode(t, out) != CodeExists {
		t.Fatalf("second create = %s", out)
	}
	if out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "missing.txt", Content: "x", Mode: "overwrite"})); !bad || errCode(t, out) != CodeNotFound {
		t.Fatalf("missing overwrite = %s", out)
	}
	if out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "dir/a.txt", Content: "two", Mode: "overwrite"})); bad {
		t.Fatalf("overwrite: %s", out)
	}
}

func TestEditExactConflictAndReplaceAll(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "a.txt")
	mustWrite(t, p, "old old")
	out, bad := ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "new"}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("ambiguous = %s", out)
	}
	out, bad = ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "new", ReplaceAll: true}))
	if bad {
		t.Fatalf("replace all: %s", out)
	}
	if b, _ := os.ReadFile(p); string(b) != "new new" {
		t.Fatalf("edited = %q", b)
	}
	out, bad = ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "a.txt", OldString: "absent", NewString: "x"}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("mismatch = %s", out)
	}
}

func TestMutationTraversalSymlinkCancellationAndBound(t *testing.T) {
	ts, root := newTestToolset(t)
	for _, tc := range []struct {
		name string
		run  func() ([]byte, bool)
		code string
	}{
		{"write traversal", func() ([]byte, bool) {
			return ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "../escape", Content: "x", Mode: "create"}))
		}, CodePathEscape},
		{"edit traversal", func() ([]byte, bool) {
			return ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "../escape", OldString: "x", NewString: "y"}))
		}, CodePathEscape},
		{"canceled", func() ([]byte, bool) {
			return ts.write(deadCtx(), argsOf(t, WriteArgs{FilePath: "x", Content: "x", Mode: "create"}))
		}, CodeCanceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, bad := tc.run()
			if !bad || errCode(t, out) != tc.code {
				t.Fatalf("got %s", out)
			}
		})
	}
	bounded, _ := New(Config{Root: root, Limits: Limits{MaxWriteBytes: 3}})
	out, bad := bounded.write(context.Background(), argsOf(t, WriteArgs{FilePath: "big", Content: "four", Mode: "create"}))
	if !bad || errCode(t, out) != CodeTooLarge {
		t.Fatalf("bound = %s", out)
	}
	outside := t.TempDir()
	mustWrite(t, filepath.Join(outside, "secret"), "secret")
	link := filepath.Join(root, "link")
	if err := os.Symlink(filepath.Join(outside, "secret"), link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skip("symlink privilege")
		}
		t.Fatal(err)
	}
	out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "link", Content: "no", Mode: "overwrite"}))
	if !bad || !strings.Contains(string(out), CodeSymlinkEscape) {
		t.Fatalf("symlink write = %s", out)
	}
}

func TestMutationRungDefaultDenyAndCacheContract(t *testing.T) {
	ts, _ := newTestToolset(t)
	mk := func(tool string, meta map[string]string) *abi.ToolCall {
		return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: argsOf(t, WriteArgs{FilePath: "a", Content: "x", Mode: "create"})}, Meta: meta}
	}
	v := ts.Adjudicate(context.Background(), mk(ToolWrite, CallMeta(ToolWrite, "p")))
	if v.Kind != abi.VerdictAllow || v.By != RungName {
		t.Fatalf("allow verdict=%+v", v)
	}
	denied, _ := New(Config{Root: t.TempDir(), Policy: Policy{Allow: map[string]bool{ToolRead: true}}})
	v = denied.Adjudicate(context.Background(), mk(ToolWrite, CallMeta(ToolWrite, "p")))
	if v.Kind != abi.VerdictDeny || v.Meta["code"] != CodeDefaultDeny {
		t.Fatalf("default deny=%+v", v)
	}
	v = ts.Adjudicate(context.Background(), mk(ToolWrite, map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}))
	if v.Kind != abi.VerdictDeny || v.Meta["code"] != CodeCacheScope {
		t.Fatalf("cache scope=%+v", v)
	}
	escape := mk(ToolWrite, map[string]string{"readOnlyHint": "true", "idempotentHint": "true"})
	escape.Args.Inline = argsOf(t, WriteArgs{FilePath: "../escape", Content: "x", Mode: "create"})
	v = ts.Adjudicate(context.Background(), escape)
	if v.Meta["code"] != CodePathEscape {
		t.Fatalf("canonicalization did not precede cache matching: %+v", v)
	}
}
