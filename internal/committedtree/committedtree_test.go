package committedtree

import (
	"archive/tar"
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

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
