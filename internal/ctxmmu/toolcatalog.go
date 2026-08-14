package ctxmmu

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/toolcatalog"
)

const toolCatalogPageName = "fak.toolcatalog.snapshot/v1"

// ToolCatalogPage pins one verified model-facing tool snapshot in the ctxmmu
// content-addressed tool-page store. SnapshotDigest identifies the catalog
// selection; PageHash identifies the exact stored bytes.
type ToolCatalogPage struct {
	SnapshotDigest string `json:"snapshot_digest"`
	PageHash       string `json:"page_hash"`
	Deduplicated   bool   `json:"deduplicated"`
}

// RegisterToolCatalogSnapshot verifies and stores a model-facing snapshot. The
// installed registry is deliberately not accepted here: only an already-selected
// snapshot may cross into the provider-request path.
func (t *ToolPageTable) RegisterToolCatalogSnapshot(snapshot toolcatalog.Snapshot) (ToolCatalogPage, error) {
	if t == nil {
		return ToolCatalogPage{}, fmt.Errorf("ctxmmu: nil tool page table")
	}
	if err := toolcatalog.ValidateSnapshot(snapshot); err != nil {
		return ToolCatalogPage{}, err
	}
	body, err := json.Marshal(snapshot)
	if err != nil {
		return ToolCatalogPage{}, fmt.Errorf("ctxmmu: encode tool catalog snapshot: %w", err)
	}
	hash, dedup := t.Register(toolCatalogPageName, body)
	return ToolCatalogPage{SnapshotDigest: snapshot.Digest, PageHash: hash, Deduplicated: dedup}, nil
}

// FaultToolCatalogSnapshot reads and re-verifies a pinned snapshot. Callers pass
// both identities so a stale page substitution and a stale selection both fail
// closed before a request is built.
func (t *ToolPageTable) FaultToolCatalogSnapshot(ctx context.Context, pageHash, snapshotDigest string) (toolcatalog.Snapshot, error) {
	if t == nil {
		return toolcatalog.Snapshot{}, fmt.Errorf("ctxmmu: nil tool page table")
	}
	body, err := t.Fault(ctx, pageHash)
	if err != nil {
		return toolcatalog.Snapshot{}, err
	}
	var snapshot toolcatalog.Snapshot
	if err := json.Unmarshal(body, &snapshot); err != nil {
		return toolcatalog.Snapshot{}, fmt.Errorf("ctxmmu: decode tool catalog snapshot: %w", err)
	}
	if err := toolcatalog.ValidateSnapshot(snapshot); err != nil {
		return toolcatalog.Snapshot{}, err
	}
	if snapshot.Digest != snapshotDigest {
		return toolcatalog.Snapshot{}, fmt.Errorf("ctxmmu: tool catalog snapshot digest mismatch: got %s want %s", snapshot.Digest, snapshotDigest)
	}
	return snapshot, nil
}
