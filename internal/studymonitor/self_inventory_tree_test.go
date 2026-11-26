package studymonitor

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

// The streamed builder must produce the manifest the on-disk builder produces for the
// same committed bytes, including the control-tree and manifest exclusions.
func TestSelfInventoryFromTreeMatchesOnDiskBuilder(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"README.md":                    "hello\n",
		"cmd/demo/main.go":             "package main\n",
		"docs/worktrees/notes.md":      "nested worktrees dir is content\n",
		"sub/.git/config":              "nested git dir is content\n",
		".git/HEAD":                    "ref: refs/heads/main\n",
		"worktrees/ticket-x/README.md": "control tree\n",
		DefaultSelfInventoryPath:       "stale manifest\n",
		"scripts/run.sh":               "#!/bin/sh\n",
	}
	for rel, content := range files {
		writeSelfFixture(t, root, rel, content)
	}
	onDisk, err := BuildSelfInventory(root, "anthony-chaudhary/fak", DefaultSelfInventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	streamed, err := BuildSelfInventoryFromTree("anthony-chaudhary/fak", DefaultSelfInventoryPath, func(visit func(string, []byte) error) error {
		rels := make([]string, 0, len(files))
		for rel := range files {
			rels = append(rels, rel)
		}
		sort.Strings(rels)
		for _, rel := range rels {
			data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
			if err != nil {
				return err
			}
			if err := visit(rel, data); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(onDisk, streamed) {
		t.Fatalf("streamed manifest differs from on-disk manifest:\non-disk %+v\nstreamed %+v", onDisk, streamed)
	}
	want := []string{"README.md", "cmd/demo/main.go", "docs/worktrees/notes.md", "scripts/run.sh", "sub/.git/config"}
	got := make([]string, 0, len(streamed.Entries))
	for _, entry := range streamed.Entries {
		got = append(got, entry.Path)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("streamed paths = %v, want %v", got, want)
	}
}
