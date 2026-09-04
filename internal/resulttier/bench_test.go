package resulttier

import (
	"testing"
)

// BenchmarkResultTier measures performance of standard tier assignment, slicing, and cursor pagination.
func BenchmarkResultTier(b *testing.B) {
	items := make([]int, 100)
	for i := range items {
		items[i] = i
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Exercise default tier assignment
		req := PaginationRequest{Limit: 0}
		lim, off, cont, err := ResolvePagination(req, len(items))
		if err != nil || lim != int(DefaultTier) || off != 0 || !cont.HasMore {
			b.Fatalf("unexpected ResolvePagination failure: %v", err)
		}

		// Exercise pagination slicing across standard tiers
		for _, t := range []int{int(Tier1), int(Tier3), int(Tier5), int(Tier10)} {
			sliceReq := PaginationRequest{Limit: t, Offset: i % 50}
			sliced, sCont, sErr := Slice(items, sliceReq)
			if sErr != nil || len(sliced) > t || sCont.Tier != t {
				b.Fatalf("unexpected Slice failure: %v", sErr)
			}
		}

		// Exercise cursor parsing and continuation
		cursorReq := PaginationRequest{Cursor: cont.NextCursor, Limit: 5}
		_, cOff, _, cErr := ResolvePagination(cursorReq, len(items))
		if cErr != nil || cOff != 5 {
			b.Fatalf("unexpected cursor ResolvePagination failure: %v", cErr)
		}
	}
}
