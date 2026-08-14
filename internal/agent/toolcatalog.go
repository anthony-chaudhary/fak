package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/toolcatalog"
)

// ToolCatalogAudit is the request-side evidence for the exact selected tool
// surface shown to a model. Omissions explain why installed tools were absent.
type ToolCatalogAudit struct {
	SnapshotDigest string                 `json:"snapshot_digest"`
	PageHash       string                 `json:"page_hash"`
	Omissions      []toolcatalog.Omission `json:"omissions,omitempty"`
}

// ToolDefsFromCatalogPage faults a digest-pinned selected snapshot and lowers it
// to the provider-neutral declarations consumed by every transcript adapter.
// It cannot expose an installed-but-unselected registration because registry
// state is not an input to this seam.
func ToolDefsFromCatalogPage(ctx context.Context, pages *ctxmmu.ToolPageTable, pageHash, snapshotDigest string) ([]ToolDef, ToolCatalogAudit, error) {
	snapshot, err := pages.FaultToolCatalogSnapshot(ctx, pageHash, snapshotDigest)
	if err != nil {
		return nil, ToolCatalogAudit{}, err
	}
	tools := make([]ToolDef, len(snapshot.Tools))
	for i, tool := range snapshot.Tools {
		if !json.Valid(tool.InputSchema) {
			return nil, ToolCatalogAudit{}, fmt.Errorf("agent: tool %q has invalid input schema", tool.Name)
		}
		tools[i] = ToolDef{
			Type: "function",
			Function: ToolDefFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  append(json.RawMessage(nil), tool.InputSchema...),
			},
		}
	}
	audit := ToolCatalogAudit{
		SnapshotDigest: snapshot.Digest,
		PageHash:       pageHash,
		Omissions:      append([]toolcatalog.Omission(nil), snapshot.Omitted...),
	}
	return tools, audit, nil
}

// CompleteCatalog resolves a pinned selected-tool snapshot immediately before
// building the provider request. The returned audit belongs to that request.
func (p *HTTPPlanner) CompleteCatalog(ctx context.Context, messages []Message, pages *ctxmmu.ToolPageTable, pageHash, snapshotDigest string, opts ...SampleOpt) (*Completion, ToolCatalogAudit, error) {
	tools, audit, err := ToolDefsFromCatalogPage(ctx, pages, pageHash, snapshotDigest)
	if err != nil {
		return nil, ToolCatalogAudit{}, err
	}
	completion, err := p.Complete(ctx, messages, tools, opts...)
	return completion, audit, err
}

// CompleteCatalogStream is the streaming equivalent of CompleteCatalog.
func (p *HTTPPlanner) CompleteCatalogStream(ctx context.Context, sink StreamSink, messages []Message, pages *ctxmmu.ToolPageTable, pageHash, snapshotDigest string, opts ...SampleOpt) (*Completion, ToolCatalogAudit, error) {
	tools, audit, err := ToolDefsFromCatalogPage(ctx, pages, pageHash, snapshotDigest)
	if err != nil {
		return nil, ToolCatalogAudit{}, err
	}
	completion, err := p.CompleteStream(ctx, sink, messages, tools, opts...)
	return completion, audit, err
}
