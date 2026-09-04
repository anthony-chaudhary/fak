package edittx

import (
	"context"
	"fmt"
	"testing"
)

// BenchmarkEditTx exercises transactional editing in a loop across multiple files.
func BenchmarkEditTx(b *testing.B) {
	root := b.TempDir()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		content1 := fmt.Sprintf("file 1 content at iteration %d\n", i)
		content2 := fmt.Sprintf("file 2 content at iteration %d\n", i)
		spec := Spec{
			Edits: []Edit{
				{Path: "bench1.txt", Content: &content1},
				{Path: "bench2.txt", Content: &content2},
			},
		}
		res := Apply(ctx, Options{
			Root: root,
			Spec: spec,
		})
		if !res.OK {
			b.Fatalf("Apply failed at iteration %d: %s", i, res.Reason)
		}
	}
}
