package disambiguation

import (
	"bytes"
	"crypto/sha256"
	"slices"
	"testing"
)

func TestGeneratePublicIndexIsByteDeterministic(t *testing.T) {
	first, err := GeneratePublicIndex()
	if err != nil {
		t.Fatal(err)
	}
	second, err := GeneratePublicIndex()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("same source generated different bytes: %x != %x", sha256.Sum256(first), sha256.Sum256(second))
	}
}

func TestGenerateIndexCanonicalizesInputOrdering(t *testing.T) {
	forward := []Entry{cloneEntry(publicEntries[0]), cloneEntry(publicEntries[1])}
	reverse := []Entry{cloneEntry(publicEntries[1]), cloneEntry(publicEntries[0])}
	slices.Reverse(reverse[0].Identity.Aliases)
	slices.Reverse(reverse[0].Contrasts)
	slices.Reverse(reverse[0].Sources)
	gotForward, err := GenerateIndex(forward)
	if err != nil {
		t.Fatal(err)
	}
	gotReverse, err := GenerateIndex(reverse)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(gotForward, gotReverse) {
		t.Fatalf("reordered source generated different bytes")
	}
}

func TestGenerateIndexRejectsInvalidEntry(t *testing.T) {
	invalid := cloneEntry(publicEntries[0])
	invalid.Definition = ""
	if _, err := GenerateIndex([]Entry{invalid}); err == nil {
		t.Fatal("expected invalid entry rejection")
	}
}
