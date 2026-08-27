package committedtree

import (
	"archive/tar"
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestExtractPreservesRawBlobBytes(t *testing.T) {
	for _, autocrlf := range []string{"true", "false"} {
		t.Run("autocrlf="+autocrlf, func(t *testing.T) {
			repo := t.TempDir()
			runGit(t, repo, "init", "--quiet")
			runGit(t, repo, "config", "user.name", "Committed Tree Test")
			runGit(t, repo, "config", "user.email", "committed-tree@example.invalid")
			runGit(t, repo, "config", "core.autocrlf", autocrlf)

			body := []byte("line one\nline two\n")
			if err := os.WriteFile(filepath.Join(repo, "sample.txt"), body, 0o644); err != nil {
				t.Fatal(err)
			}
			runGit(t, repo, "add", "sample.txt")
			runGit(t, repo, "commit", "--quiet", "-m", "fixture")

			want := runGit(t, repo, "cat-file", "blob", "HEAD:sample.txt")
			if !bytes.Equal(want, body) {
				t.Fatalf("fixture blob = %q, want %q", want, body)
			}
			dir, err := Extract(repo, "HEAD")
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = os.RemoveAll(dir) })
			got, err := os.ReadFile(filepath.Join(dir, "sample.txt"))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("extracted bytes = %q, raw blob bytes = %q", got, want)
			}
		})
	}
}

func runGit(t *testing.T, repo string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return out
}

func TestUntarRejectsTraversal(t *testing.T) {
	var b bytes.Buffer
	tw := tar.NewWriter(&b)
	if err := tw.WriteHeader(&tar.Header{Name: "../escape", Mode: 0o644, Size: 1, Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := untar(&b, t.TempDir()); err == nil {
		t.Fatal("expected traversal rejection")
	}
}

func TestUntarDoesNotMaterializeSymlink(t *testing.T) {
	var b bytes.Buffer
	tw := tar.NewWriter(&b)
	if err := tw.WriteHeader(&tar.Header{Name: "link", Linkname: "../escape", Mode: 0o777, Typeflag: tar.TypeSymlink}); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	if err := untar(&b, root); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Lstat(filepath.Join(root, "link")); !os.IsNotExist(err) {
		t.Fatalf("symlink was materialized: %v", err)
	}
}

func TestUntarMaterializesRegularFile(t *testing.T) {
	var b bytes.Buffer
	tw := tar.NewWriter(&b)
	body := []byte("package x\n")
	if err := tw.WriteHeader(&tar.Header{Name: "x/x.go", Mode: 0o644, Size: int64(len(body)), Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	_, _ = tw.Write(body)
	_ = tw.Close()
	root := t.TempDir()
	if err := untar(&b, root); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(root, "x", "x.go"))
	if err != nil || !bytes.Equal(got, body) {
		t.Fatalf("got=%q err=%v", got, err)
	}
}
