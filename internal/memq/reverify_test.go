package memq

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
)

func TestRecallBackendReportsStaleArtifactRefusal(t *testing.T) {
	ctx := context.Background()
	r := recall.NewRecorder("memq-stale")
	r.Record(ctx, "status", []byte("commit deadbee fixed the refund fee path"))
	dir := t.TempDir()
	if err := r.Persist(dir); err != nil {
		t.Fatal(err)
	}
	s, err := recall.Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	s.WithArtifactVerifier(func(_ context.Context, claims []recall.ArtifactClaim) []recall.ArtifactFinding {
		out := make([]recall.ArtifactFinding, 0, len(claims))
		for _, c := range claims {
			out = append(out, recall.ArtifactFinding{Claim: c, Status: recall.ArtifactStale, Detail: "missing in git"})
		}
		return out
	})

	res, err := Run(ctx, NewRecallBackend(s, ""), Query{
		Intent: "refund fee",
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpRender},
		},
	}, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Rendered) != 0 {
		t.Fatalf("stale recall artifact rendered into context: %+v", res.Rendered)
	}
	if len(res.Refused) != 1 || res.Refused[0].Reason != "stale_recall_artifact" {
		t.Fatalf("refusals = %+v, want stale_recall_artifact", res.Refused)
	}
}
