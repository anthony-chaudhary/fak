package computetrace

import (
	"archive/zip"
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestTorchPickleCPUFixture(t *testing.T) {
	v, err := ReadTorchPickle("testdata/sglang-dumper-tiny/reader/reader.pt")
	if err != nil {
		t.Fatal(err)
	}
	list := v.(map[string]any)["value"].([]any)
	for i, want := range [][]float32{{1, 3, 5, 7, 9, 11}, {1, -2, .5}, {1, -2, .5}} {
		if got := list[i].(TorchTensor).Values; !reflect.DeepEqual(got, want) {
			t.Fatalf("tensor %d: %v", i, got)
		}
	}
	if got := list[3].(TorchTensor).IDs; !reflect.DeepEqual(got, []int64{1, 16777217, -2}) {
		t.Fatal(got)
	}
}

func torchRewrite(t *testing.T, edit func(string, []byte) []byte) string {
	return torchRewriteFile(t, "testdata/sglang-dumper-tiny/reader/reader.pt", edit)
}

func torchRewriteFile(t *testing.T, source string, edit func(string, []byte) []byte) string {
	t.Helper()
	input, err := zip.OpenReader(source)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Close()
	path := filepath.Join(t.TempDir(), "variant.pt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := zip.NewWriter(f)
	for _, entry := range input.File {
		r, err := entry.Open()
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			t.Fatal(err)
		}
		out, err := w.Create(entry.Name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = out.Write(edit(entry.Name, b)); err != nil {
			t.Fatal(err)
		}
	}
	if err = w.Close(); err != nil {
		t.Fatal(err)
	}
	if err = f.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// fak-test:runtime fast est=50ms lane=default
func TestTorchPickleRefusesUnsafe(t *testing.T) {
	for _, kind := range []string{"global", "opcode", "byteorder", "truncated"} {
		t.Run(kind, func(t *testing.T) {
			path := torchRewrite(t, func(name string, b []byte) []byte {
				if filepath.Base(name) == "byteorder" && kind == "byteorder" {
					return []byte("big")
				}
				if filepath.Base(name) != "data.pkl" {
					return b
				}
				switch kind {
				case "global":
					return bytes.ReplaceAll(b, []byte("torch._utils\n_rebuild_tensor_v2"), []byte("os\nsystem"))
				case "opcode":
					return []byte{0x80, 2, 'P', '.'}
				case "truncated":
					return b[:len(b)/2]
				}
				return b
			})
			if _, err := ReadTorchPickle(path); !errors.Is(err, ErrTorchPickle) {
				t.Fatalf("want sentinel, got %v", err)
			}
		})
	}
}
