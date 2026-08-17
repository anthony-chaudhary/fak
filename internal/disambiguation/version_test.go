package disambiguation

import (
	"reflect"
	"slices"
	"testing"
)

func TestIndexVersionStableOnReorderOnlyInput(t *testing.T) {
	forward := []Entry{cloneEntry(publicEntries[0]), cloneEntry(publicEntries[1])}
	reverse := []Entry{cloneEntry(publicEntries[1]), cloneEntry(publicEntries[0])}
	slices.Reverse(reverse[0].Identity.Aliases)
	slices.Reverse(reverse[0].Contrasts)
	slices.Reverse(reverse[0].Sources)
	want, err := VersionIndex(forward)
	if err != nil {
		t.Fatal(err)
	}
	got, err := VersionIndex(reverse)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reorder changed version:\ngot  %+v\nwant %+v", got, want)
	}
}

func TestIndexVersionChangesOnSemanticContentChange(t *testing.T) {
	baseline := []Entry{cloneEntry(publicEntries[0]), cloneEntry(publicEntries[1])}
	changed := []Entry{cloneEntry(publicEntries[0]), cloneEntry(publicEntries[1])}
	changed[0].Definition += " Public semantic revision."
	before, err := VersionIndex(baseline)
	if err != nil {
		t.Fatal(err)
	}
	after, err := VersionIndex(changed)
	if err != nil {
		t.Fatal(err)
	}
	if before.SourceRevision == after.SourceRevision {
		t.Fatal("semantic change preserved source revision")
	}
	if before.ContentSHA256 == after.ContentSHA256 {
		t.Fatal("semantic change preserved content digest")
	}
	if before.EntryCount != after.EntryCount {
		t.Fatal("semantic edit changed entry count")
	}
}

func TestPublicIndexVersionMatchesGeneratedBytes(t *testing.T) {
	version, err := CurrentIndexVersion()
	if err != nil {
		t.Fatal(err)
	}
	if version.Schema != IndexVersionSchema || version.IndexSchema != GeneratedIndexSchemaVersion || version.EntryCount != len(publicEntries) {
		t.Fatalf("unexpected version: %+v", version)
	}
	if len(version.ContentSHA256) != 64 {
		t.Fatalf("content digest=%q", version.ContentSHA256)
	}
}
