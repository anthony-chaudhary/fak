package studymonitor

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestSelfInventoryDeterministicAndIndependentOfCheckoutPath(t *testing.T) {
	a := t.TempDir()
	b := t.TempDir()
	for _, root := range []string{a, b} {
		writeSelfFixture(t, root, "README.md", "hello\n")
		writeSelfFixture(t, root, "cmd/demo/main.go", "package main\n")
		writeSelfFixture(t, root, DefaultSelfInventoryPath, "ignored self artifact\n")
	}
	one, err := BuildSelfInventory(a, "anthony-chaudhary/fak", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	two, err := BuildSelfInventory(b, "anthony-chaudhary/fak", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(one, two) {
		t.Fatalf("identical committed bytes at different checkout paths differ:\n%+v\n%+v", one, two)
	}
	if one.ContentRoot == "" || one.TrackedFiles != 2 || one.Entries[0].Path != "README.md" {
		t.Fatalf("unexpected manifest: %+v", one)
	}
}

func TestSelfInventoryFailRefreshPassWithTypedMutation(t *testing.T) {
	root := t.TempDir()
	writeSelfFixture(t, root, "README.md", "before\n")
	fresh, err := BuildSelfInventory(root, "anthony-chaudhary/fak", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	writeSelfManifest(t, root, fresh)

	verified, err := VerifySelfInventory(root, DefaultSelfInventoryPath, "anthony-chaudhary/fak")
	if err != nil || !verified.OK {
		t.Fatalf("fresh verify = %+v, err=%v", verified, err)
	}
	writeSelfFixture(t, root, "README.md", "after\n")
	drifted, err := VerifySelfInventory(root, DefaultSelfInventoryPath, "anthony-chaudhary/fak")
	if err != nil {
		t.Fatal(err)
	}
	if drifted.OK || len(drifted.Drift) != 1 || drifted.Drift[0].Kind != SelfDriftContentChanged || drifted.Drift[0].Path != "README.md" {
		t.Fatalf("mutation diagnostics = %+v", drifted)
	}

	refreshed, err := BuildSelfInventory(root, "anthony-chaudhary/fak", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	writeSelfManifest(t, root, refreshed)
	passed, err := VerifySelfInventory(root, DefaultSelfInventoryPath, "anthony-chaudhary/fak")
	if err != nil || !passed.OK {
		t.Fatalf("verify after explicit refresh = %+v, err=%v", passed, err)
	}
	if refreshed.RefreshWitness.MutationVerdict != "typed changed-path drift" || refreshed.RefreshWitness.FreshVerdict != "verified" {
		t.Fatalf("manifest omitted refresh witness: %+v", refreshed.RefreshWitness)
	}
}

func TestSelfInventoryReportsAddedAndRemovedPathsInOrder(t *testing.T) {
	root := t.TempDir()
	writeSelfFixture(t, root, "b.txt", "b\n")
	writeSelfFixture(t, root, "z.txt", "z\n")
	manifest, err := BuildSelfInventory(root, "repo", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	writeSelfManifest(t, root, manifest)
	if err := os.Remove(filepath.Join(root, "z.txt")); err != nil {
		t.Fatal(err)
	}
	writeSelfFixture(t, root, "a.txt", "a\n")
	result, err := VerifySelfInventory(root, DefaultSelfInventoryPath, "repo")
	if err != nil {
		t.Fatal(err)
	}
	want := []SelfInventoryDriftKind{SelfDriftPathAdded, SelfDriftPathRemoved}
	if len(result.Drift) != 2 || result.Drift[0].Path != "a.txt" || result.Drift[1].Path != "z.txt" ||
		!reflect.DeepEqual([]SelfInventoryDriftKind{result.Drift[0].Kind, result.Drift[1].Kind}, want) {
		t.Fatalf("drift = %+v", result.Drift)
	}
}

func TestSelfInventoryIgnoresManifestAndWorktreeControlTrees(t *testing.T) {
	root := t.TempDir()
	writeSelfFixture(t, root, "kept.txt", "kept\n")
	writeSelfFixture(t, root, DefaultSelfInventoryPath, "one\n")
	writeSelfFixture(t, root, "worktrees/peer/dirty.go", "dirty\n")
	writeSelfFixture(t, root, ".git/objects/fake", "object\n")
	one, err := BuildSelfInventory(root, "repo", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	writeSelfFixture(t, root, DefaultSelfInventoryPath, "two\n")
	writeSelfFixture(t, root, "worktrees/peer/dirty.go", "changed\n")
	two, err := BuildSelfInventory(root, "repo", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(one, two) || one.TrackedFiles != 1 {
		t.Fatalf("control or self paths changed root: one=%+v two=%+v", one, two)
	}
}

func writeSelfFixture(t *testing.T, root, rel, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeSelfManifest(t *testing.T, root string, manifest SelfInventory) {
	t.Helper()
	var data bytes.Buffer
	if err := WriteSelfInventory(&data, manifest); err != nil {
		t.Fatal(err)
	}
	writeSelfFixture(t, root, DefaultSelfInventoryPath, data.String())
}
