package workerworktree

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fakeDisambiguationTree installs the list and read seams over an in-memory
// tree. Each record is "mode blob oid size<TAB>path" in ls-tree -z form.
func fakeDisambiguationTree(t *testing.T, blobs map[string]string, modes map[string]string, read func(ctx context.Context, oid string) ([]byte, error)) {
	t.Helper()
	var listing strings.Builder
	for path, body := range blobs {
		mode := "100644"
		if m, ok := modes[path]; ok {
			mode = m
		}
		fmt.Fprintf(&listing, "%s blob %s %d\t%s\x00", mode, "oid:"+path, len(body), path)
	}
	oldList, oldRead := listDisambiguationTree, readDisambiguationObject
	t.Cleanup(func() {
		listDisambiguationTree = oldList
		readDisambiguationObject = oldRead
	})
	raw := listing.String()
	listDisambiguationTree = func(context.Context, string, string) ([]byte, error) {
		return []byte(raw), nil
	}
	readDisambiguationObject = func(ctx context.Context, _ string, objectID string) ([]byte, error) {
		return read(ctx, objectID)
	}
}

func bodyByObjectID(blobs map[string]string) func(context.Context, string) ([]byte, error) {
	return func(_ context.Context, oid string) ([]byte, error) {
		path := strings.TrimPrefix(oid, "oid:")
		return []byte(blobs[path]), nil
	}
}

func TestMaterializeDisambiguationTreeWritesCorpusWithContentAndMode(t *testing.T) {
	saved := disambiguationWriteWorkers
	t.Cleanup(func() { disambiguationWriteWorkers = saved })
	disambiguationWriteWorkers = 8

	blobs := map[string]string{
		"README.md": "root readme\n",
		"tools/concept_disambiguation_scorecard.data/_meta.json": "{}\n",
		"internal/app/run.go":      "package app\n",
		"internal/app/run_test.go": "not corpus\n",
		"vendor/huge/blob.go":      "skipped dir\n",
	}
	modes := map[string]string{"internal/app/run.go": "100755"}
	for d := 0; d < 20; d++ {
		for f := 0; f < 10; f++ {
			path := fmt.Sprintf("docs/d%02d/sub/doc%02d.md", d, f)
			blobs[path] = fmt.Sprintf("content of %s\n", path)
		}
	}
	fakeDisambiguationTree(t, blobs, modes, bodyByObjectID(blobs))

	dst := filepath.Join(t.TempDir(), "tree")
	if err := materializeDisambiguationTree(context.Background(), "/repo", "tree", dst); err != nil {
		t.Fatalf("materialize: %v", err)
	}

	for path, want := range blobs {
		if strings.HasPrefix(path, "vendor/") || strings.HasSuffix(path, "_test.go") {
			if _, err := os.Stat(filepath.Join(dst, filepath.FromSlash(path))); !os.IsNotExist(err) {
				t.Fatalf("%s: non-corpus path materialized (stat err %v)", path, err)
			}
			continue
		}
		got, err := os.ReadFile(filepath.Join(dst, filepath.FromSlash(path)))
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		if string(got) != want {
			t.Fatalf("%s: content %q, want %q", path, got, want)
		}
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(filepath.Join(dst, "internal", "app", "run.go"))
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != 0755 {
			t.Fatalf("executable blob mode %v, want 0755", info.Mode().Perm())
		}
		info, err = os.Stat(filepath.Join(dst, "README.md"))
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode().Perm() != 0644 {
			t.Fatalf("regular blob mode %v, want 0644", info.Mode().Perm())
		}
	}
}

func TestMaterializeDisambiguationTreeFailsWholeCallOnOneReadError(t *testing.T) {
	saved := disambiguationWriteWorkers
	t.Cleanup(func() { disambiguationWriteWorkers = saved })
	disambiguationWriteWorkers = 4

	blobs := map[string]string{}
	for i := 0; i < 64; i++ {
		blobs[fmt.Sprintf("docs/f%02d.md", i)] = "x\n"
	}
	sentinel := errors.New("object store unavailable")
	fakeDisambiguationTree(t, blobs, nil, func(ctx context.Context, oid string) ([]byte, error) {
		if oid == "oid:docs/f17.md" {
			return nil, sentinel
		}
		return []byte("x\n"), nil
	})

	err := materializeDisambiguationTree(context.Background(), "/repo", "tree", t.TempDir())
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want wrapped %v", err, sentinel)
	}
}

func TestMaterializeDisambiguationTreeCancellationReturnsPromptly(t *testing.T) {
	saved := disambiguationWriteWorkers
	t.Cleanup(func() { disambiguationWriteWorkers = saved })
	disambiguationWriteWorkers = 4

	blobs := map[string]string{}
	for i := 0; i < 200; i++ {
		blobs[fmt.Sprintf("docs/c%03d.md", i)] = "x\n"
	}
	var inFlight atomic.Int32
	fakeDisambiguationTree(t, blobs, nil, func(ctx context.Context, oid string) ([]byte, error) {
		inFlight.Add(1)
		defer inFlight.Add(-1)
		<-ctx.Done()
		return nil, ctx.Err()
	})

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, cancel)
	defer cancel()

	start := time.Now()
	err := materializeDisambiguationTree(ctx, "/repo", "tree", t.TempDir())
	elapsed := time.Since(start)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("cancellation took %s, want prompt return", elapsed)
	}
	if n := inFlight.Load(); n != 0 {
		t.Fatalf("%d reads still running after return; workers must exit before materialize returns", n)
	}
}
