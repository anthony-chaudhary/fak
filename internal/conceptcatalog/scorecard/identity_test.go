package scorecard

import (
	"os"
	"testing"
	"testing/fstest"
)

func TestSourceDigestMatchesPackageSources(t *testing.T) {
	onDisk, err := SourceDigestFS(os.DirFS("."))
	if err != nil {
		t.Fatal(err)
	}
	if SourceDigest() != onDisk {
		t.Fatalf("embedded digest %s != on-disk digest %s", SourceDigest(), onDisk)
	}
}

func TestSourceDigestTracksSourceEdits(t *testing.T) {
	base := fstest.MapFS{
		"fold.go":         {Data: []byte("package scorecard\n")},
		"render.go":       {Data: []byte("package scorecard\n\nfunc f() {}\n")},
		"fold_test.go":    {Data: []byte("package scorecard\n")},
		"testdata/x.json": {Data: []byte("{}")},
	}
	digest := func(fsys fstest.MapFS) string {
		t.Helper()
		d, err := SourceDigestFS(fsys)
		if err != nil {
			t.Fatal(err)
		}
		return d
	}
	want := digest(base)

	edited := fstest.MapFS{}
	for k, v := range base {
		edited[k] = v
	}
	edited["render.go"] = &fstest.MapFile{Data: []byte("package scorecard\n\nfunc f() { _ = 1 }\n")}
	if digest(edited) == want {
		t.Fatal("editing an analyzer source did not change the digest")
	}

	renamed := fstest.MapFS{"fold.go": base["fold.go"], "render2.go": base["render.go"], "fold_test.go": base["fold_test.go"]}
	if digest(renamed) == want {
		t.Fatal("renaming an analyzer source did not change the digest")
	}

	testOnly := fstest.MapFS{}
	for k, v := range base {
		testOnly[k] = v
	}
	testOnly["fold_test.go"] = &fstest.MapFile{Data: []byte("package scorecard\n\n// changed\n")}
	testOnly["testdata/x.json"] = &fstest.MapFile{Data: []byte("[]")}
	if digest(testOnly) != want {
		t.Fatal("test-only edits changed the analyzer digest")
	}

	if _, err := SourceDigestFS(fstest.MapFS{"a_test.go": base["fold_test.go"]}); err == nil {
		t.Fatal("digest of a source-less tree succeeded")
	}
}
