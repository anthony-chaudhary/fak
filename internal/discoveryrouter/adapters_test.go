package discoveryrouter

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func TestDocsAdapterRetainsOwnerRevisionAndReason(t *testing.T) {
	cat, err := devindex.Load(filepath.Clean(filepath.Join("..", "..")))
	if err != nil {
		t.Fatal(err)
	}
	hits, watermark, err := (DocsAdapter{Catalog: cat, Revision: "g123"}).Search("native inference goal", 5)
	if err != nil || watermark != "g123" || len(hits) == 0 {
		t.Fatalf("hits=%+v watermark=%q err=%v", hits, watermark, err)
	}
	if hits[0].Source != "docs" || hits[0].Revision != "g123" || hits[0].Owner == "" || hits[0].Reason == "" {
		t.Fatalf("hit=%+v", hits[0])
	}
}

func TestFleetAdapterClassifierIsBounded(t *testing.T) {
	a := FleetAdapter{}
	if !a.Relevant("which active worker session crashed") {
		t.Fatal("runtime query excluded")
	}
	if a.Relevant("where is the architecture guide") {
		t.Fatal("docs query incorrectly routed to sessions")
	}
}
