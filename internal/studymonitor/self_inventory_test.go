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

func TestSelfInventoryMutationQAMatrix(t *testing.T) {
	type wantDrift struct {
		kind SelfInventoryDriftKind
		path string
	}
	tests := []struct {
		name            string
		withoutManifest bool
		mutate          func(*testing.T, string, *SelfInventory)
		want            []wantDrift
	}{
		{
			name: "empty default", withoutManifest: true,
			want: []wantDrift{{SelfDriftManifestMissing, DefaultSelfInventoryPath}},
		},
		{
			name: "add", mutate: func(t *testing.T, root string, _ *SelfInventory) {
				writeSelfFixture(t, root, "b.go", "package b\n")
			},
			want: []wantDrift{{SelfDriftPathAdded, "b.go"}},
		},
		{
			name: "delete", mutate: func(t *testing.T, root string, _ *SelfInventory) {
				if err := os.Remove(filepath.Join(root, "a.md")); err != nil {
					t.Fatal(err)
				}
			},
			want: []wantDrift{{SelfDriftPathRemoved, "a.md"}},
		},
		{
			name: "rename", mutate: func(t *testing.T, root string, _ *SelfInventory) {
				if err := os.Rename(filepath.Join(root, "a.md"), filepath.Join(root, "z.md")); err != nil {
					t.Fatal(err)
				}
			},
			want: []wantDrift{{SelfDriftPathRemoved, "a.md"}, {SelfDriftPathAdded, "z.md"}},
		},
		{
			name: "content change", mutate: func(t *testing.T, root string, _ *SelfInventory) {
				writeSelfFixture(t, root, "a.md", "changed\n")
			},
			want: []wantDrift{{SelfDriftContentChanged, "a.md"}},
		},
		{
			name: "reclassification", mutate: func(t *testing.T, root string, manifest *SelfInventory) {
				manifest.Entries[0].Classification = "runtime_source"
				writeSelfManifest(t, root, *manifest)
			},
			want: []wantDrift{{SelfDriftClassChanged, "a.md"}},
		},
		{
			name: "unknown class", mutate: func(t *testing.T, root string, manifest *SelfInventory) {
				manifest.Entries[0].Classification = "unknown_future_class"
				writeSelfManifest(t, root, *manifest)
			},
			want: []wantDrift{{SelfDriftClassChanged, "a.md"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			root := t.TempDir()
			writeSelfFixture(t, root, "a.md", "original\n")
			manifest, err := BuildSelfInventory(root, "repo", DefaultSelfInventoryPath)
			if err != nil {
				t.Fatal(err)
			}
			if !tt.withoutManifest {
				writeSelfManifest(t, root, manifest)
			}
			if tt.mutate != nil {
				tt.mutate(t, root, &manifest)
			}
			got, err := VerifySelfInventory(root, DefaultSelfInventoryPath, "repo")
			if err != nil {
				t.Fatal(err)
			}
			if got.OK || len(got.Drift) != len(tt.want) {
				t.Fatalf("verification = %+v, want drift %+v", got, tt.want)
			}
			for i, want := range tt.want {
				if got.Drift[i].Kind != want.kind || got.Drift[i].Path != want.path {
					t.Fatalf("drift[%d] = %+v, want kind=%s path=%s", i, got.Drift[i], want.kind, want.path)
				}
			}
		})
	}
}

func TestSelfInventoryDeterministicRerunBytes(t *testing.T) {
	root := t.TempDir()
	writeSelfFixture(t, root, "README.md", "stable\n")
	writeSelfFixture(t, root, "cmd/tool/main.go", "package main\n")
	one, err := BuildSelfInventory(root, "repo", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	two, err := BuildSelfInventory(root, "repo", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	var a, b bytes.Buffer
	if err := WriteSelfInventory(&a, one); err != nil {
		t.Fatal(err)
	}
	if err := WriteSelfInventory(&b, two); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) || one.ContentRoot != two.ContentRoot {
		t.Fatalf("rerun differs: roots %s/%s", one.ContentRoot, two.ContentRoot)
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
