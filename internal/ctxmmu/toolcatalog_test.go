package ctxmmu

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolcatalog"
)

func TestToolCatalogSnapshotPageRoundTrip(t *testing.T) {
	reg, err := toolcatalog.CompileSkill([]byte("---\nname: repo_search\ndescription: Search the repository\n---\n```fak-program\n{\"version\":\"fak.skill-program/v1\",\"name\":\"repo_search\",\"description\":\"Search the repository\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}}},\"executor\":{\"argv\":[\"fak\",\"code\",\"search\"]}}\n```"), "skills/search/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := toolcatalog.Expose([]toolcatalog.Registration{reg}, []string{"repo_search"}, "")
	if err != nil {
		t.Fatal(err)
	}
	pages := NewToolPageTable(New())
	pinned, err := pages.RegisterToolCatalogSnapshot(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	got, err := pages.FaultToolCatalogSnapshot(context.Background(), pinned.PageHash, pinned.SnapshotDigest)
	if err != nil {
		t.Fatal(err)
	}
	if got.Digest != snapshot.Digest || len(got.Tools) != 1 || got.Tools[0].Name != "repo_search" {
		t.Fatalf("faulted snapshot = %#v", got)
	}
	second, err := pages.RegisterToolCatalogSnapshot(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if !second.Deduplicated || second.PageHash != pinned.PageHash {
		t.Fatalf("second pin = %#v; first = %#v", second, pinned)
	}
}

func TestToolCatalogSnapshotPageRefusesWrongDigest(t *testing.T) {
	snapshot, err := toolcatalog.Expose(nil, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	pages := NewToolPageTable(New())
	pinned, err := pages.RegisterToolCatalogSnapshot(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pages.FaultToolCatalogSnapshot(context.Background(), pinned.PageHash, "sha256:stale"); err == nil {
		t.Fatal("wrong snapshot digest accepted")
	}
}
