package dispatchconservation

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadTailLinesBoundsAndPreservesOrder(t *testing.T) {
	path := filepath.Join(t.TempDir(), "progress.jsonl")
	var b strings.Builder
	for i := 0; i < 25; i++ {
		fmt.Fprintf(&b, "row-%02d\n", i)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0644); err != nil {
		t.Fatal(err)
	}
	got := ReadTailLines(path, 10)
	if len(got) != 10 || got[0] != "row-15" || got[9] != "row-24" {
		t.Fatalf("got=%v", got)
	}
}
