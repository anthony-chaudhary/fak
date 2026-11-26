package inklingmetal

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestPrefixReuseChunks(t *testing.T) {
	const rows, window = 2048, 2048
	for _, tc := range []struct {
		name                                   string
		shared, n, oldest, chunks, draft, want int
	}{
		{"nothing shared", 0, 5000, 1, 2, 0, 0},
		{"diverges at the first token", 0, 2050, 1, 0, 0, 0},
		{"less than a chunk shared", 2000, 2050, 1, 0, 0, 0},
		{"session never filled a chunk", 2000, 2050, 1, 0, 0, 0},
		{"one chunk", 2048, 2100, 1, 1, 0, 1},
		{"shared past the retained chunks", 6000, 6100, 1, 2, 0, 2},
		{"shared inside the retained chunks", 3000, 9000, 1, 4, 0, 1},
		{"prompt ends on its last shared chunk", 4096, 4096, 1, 2, 0, 1},
		{"prompt is one chunk the session holds", 2048, 2048, 1, 1, 0, 0},
		{"needed chunk evicted", 3000, 9000, 2, 4, 0, 0},
		{"oldest retained chunk", 4100, 9000, 2, 4, 0, 2},
		{"no chunk retained", 9000, 9000, 5, 4, 0, 0},
		{"shared longer than the prompt", 5000, 4000, 1, 2, 0, 0},
		{"draft steps back a chunk", 4196, 4300, 1, 2, window, 1},
		{"draft with one shared chunk", 2100, 2200, 1, 1, window, 0},
		{"draft below its window", 2000, 2000, 1, 0, window, 0},
		{"draft whose new rows fill the window", 4196, 6144, 1, 2, window, 2},
		{"draft one row short of the window", 4196, 6143, 1, 2, window, 1},
		{"draft with the stepped-back chunk evicted", 8300, 8400, 4, 4, window, 0},
	} {
		if got := reuseChunks(tc.shared, tc.n, tc.oldest, tc.chunks, rows, tc.draft); got != tc.want {
			t.Errorf("%s: kept %d chunks, want %d", tc.name, got, tc.want)
		}
	}
	for _, bad := range [][6]int{{4096, 5000, 1, 2, 0, 0}, {4096, 5000, 0, 2, rows, 0}, {4096, 5000, 1, 2, rows, -1}} {
		if got := reuseChunks(bad[0], bad[1], bad[2], bad[3], bad[4], bad[5]); got != 0 {
			t.Errorf("invalid geometry %v kept %d chunks", bad, got)
		}
	}
	// Whatever is kept leaves a row to prefill and, with a draft, its window.
	for shared := 0; shared <= 3*rows+5; shared += 97 {
		for extra := 0; extra <= 2*rows+5; extra += 211 {
			for _, draft := range []int{0, window} {
				n := shared + extra
				keep := reuseChunks(shared, n, 1, 8, rows, draft)
				if keep*rows > shared || keep > 0 && n-keep*rows < max(1, draft) {
					t.Fatalf("shared %d of %d with window %d kept %d chunks", shared, n, draft, keep)
				}
			}
		}
	}
}

func TestPrefixReserve(t *testing.T) {
	for _, tc := range [][3]int{{1, 1 << 20, 1}, {2562, 1 << 20, 4096}, {4096, 1 << 20, 4096}, {4097, 1 << 20, 8192}, {70000, 100000, 100000}, {100000, 100000, 100000}, {200000, 100000, 200000}, {0, 100, 0}} {
		if got := prefixReserve(tc[0], tc[1]); got != tc[2] {
			t.Errorf("reserve for %d under %d is %d, want %d", tc[0], tc[1], got, tc[2])
		}
	}
}

func TestPrefixKeySeparatesEveryField(t *testing.T) {
	base := PrefixKey{Model: "/m.gguf", Draft: "/d", ModelBytes: 1, ModelModified: 2, DraftModified: 3, Layers: 48, KVFormat: 1, StateFormat: 2, ChunkRows: 2048, DraftWindow: 2048, Scope: "run"}
	if base.String() != base.String() || len(base.String()) != 64 {
		t.Fatalf("unstable key %q", base.String())
	}
	seen := map[string]string{base.String(): "base"}
	v := reflect.ValueOf(&base).Elem()
	for i := 0; i < v.NumField(); i++ {
		changed := base
		f := reflect.ValueOf(&changed).Elem().Field(i)
		switch f.Kind() {
		case reflect.String:
			f.SetString(f.String() + "x")
		case reflect.Int64:
			f.SetInt(f.Int() + 1)
		case reflect.Uint32:
			f.SetUint(f.Uint() + 1)
		default:
			t.Fatalf("field %s has an unhandled kind", v.Type().Field(i).Name)
		}
		name := v.Type().Field(i).Name
		if other, dup := seen[changed.String()]; dup {
			t.Errorf("changing %s gives the key of %s", name, other)
		}
		seen[changed.String()] = name
	}
}

func TestPrefixFilesFollowTheCheckpoint(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "m.gguf")
	draft := filepath.Join(dir, "draft")
	if err := os.WriteFile(model, []byte("weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(draft, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(draft, "config.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	first, err := prefixFiles(model, draft)
	if err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(dir, "link.gguf")
	if err := os.Symlink(model, link); err != nil {
		t.Fatal(err)
	}
	if linked, err := prefixFiles(link, draft); err != nil || linked != first {
		t.Fatalf("a link to the checkpoint keyed %+v, want %+v: %v", linked, first, err)
	}
	alone, err := prefixFiles(model, "")
	if err != nil || alone.String() == first.String() {
		t.Fatalf("target-only model shares the draft model's key: %v", err)
	}
	if err := os.WriteFile(model, []byte("other weights"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(model, time.Now(), time.Now().Add(time.Hour)); err != nil {
		t.Fatal(err)
	}
	if rewritten, err := prefixFiles(model, draft); err != nil || rewritten.String() == first.String() {
		t.Fatalf("rewritten checkpoint kept its key: %v", err)
	}
	if _, err := prefixFiles(filepath.Join(dir, "absent.gguf"), ""); err == nil {
		t.Fatal("missing checkpoint was keyed")
	}
}
