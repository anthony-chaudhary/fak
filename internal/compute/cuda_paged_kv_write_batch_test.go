package compute

import (
	"fmt"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestPagedKVAppendBatchesMappingAndFailureOrder(t *testing.T) {
	type submission struct {
		ids            []int32
		src, off, rows int
	}
	t.Run("single row tail and full page use one ID", func(t *testing.T) {
		for _, tc := range [][2]int{{0, 1}, {14, 2}, {16, 16}} {
			var events []string
			pagedKVAppendBatches(tc[0], tc[1], 16,
				func(blk int) bool {
					events = append(events, fmt.Sprintf("before %d", blk))
					return true // e.g. first-page growth or shared-tail COW
				}, func(blk int) int32 {
					events = append(events, fmt.Sprintf("prepare %d", blk))
					return int32(20 + blk)
				}, func(ids []int32, src, off, rows int) {
					if len(ids) != 1 || cap(ids) != 1 || ids[0] != int32(20+tc[0]/16) ||
						src != 0 || off != tc[0]%16 || rows != tc[1] {
						t.Fatalf("single-page staging: ids=%v cap=%d src=%d off=%d rows=%d", ids, cap(ids), src, off, rows)
					}
					events = append(events, "write")
				})
			want := []string{fmt.Sprintf("before %d", tc[0]/16), fmt.Sprintf("prepare %d", tc[0]/16), "write"}
			if !reflect.DeepEqual(events, want) {
				t.Fatalf("single-page order=%v, want %v", events, want)
			}
		}
	})
	t.Run("single-page errors stop without replay", func(t *testing.T) {
		for _, prepareFails := range []bool{false, true} {
			marker := &struct{}{}
			writes, committed := 0, false
			var caught any
			func() {
				defer func() { caught = recover() }()
				pagedKVAppendBatches(15, 1, 16, func(int) bool { return true },
					func(int) int32 {
						if prepareFails {
							panic(marker)
						}
						return 7
					}, func([]int32, int, int, int) { writes++; panic(marker) })
				committed = true
			}()
			wantWrites := 1
			if prepareFails {
				wantWrites = 0
			}
			if caught != marker || committed || writes != wantWrites {
				t.Fatalf("single-page failure: caught=%v committed=%v writes=%d", caught, committed, writes)
			}
		}
	})
	t.Run("ragged noncontiguous and 64-page cap", func(t *testing.T) {
		const startRow, totalRows, pageRows = 14, 1050, 16
		firstRows := cudaPagedKVWriteMaxPages*pageRows - startRow
		remainingRows := totalRows - firstRows
		var got []submission
		pagedKVAppendBatches(startRow, totalRows, pageRows, func(int) bool { return false },
			func(blk int) int32 { return int32(200 - blk) },
			func(ids []int32, src, off, rows int) {
				got = append(got, submission{append([]int32(nil), ids...), src, off, rows})
			})
		if len(got) != 2 || len(got[0].ids) != cudaPagedKVWriteMaxPages || len(got[1].ids) != (remainingRows+pageRows-1)/pageRows ||
			got[0].src != 0 || got[0].off != startRow || got[0].rows != firstRows ||
			got[1].src != firstRows || got[1].off != 0 || got[1].rows != remainingRows {
			t.Fatalf("unexpected groups: %+v", got)
		}
		for _, g := range got {
			for row := 0; row < g.rows; row++ {
				logical := startRow + g.src + row
				physical := g.ids[(g.off+row)/pageRows]
				if physical != int32(200-logical/pageRows) || (g.off+row)%pageRows != logical%pageRows {
					t.Fatalf("row %d mapped to page %d", logical, physical)
				}
			}
		}
	})
	t.Run("flush before growth and COW", func(t *testing.T) {
		var events []string
		pagedKVAppendBatches(14, 38, 16,
			func(blk int) bool { return blk == 1 || blk == 3 },
			func(blk int) int32 {
				events = append(events, fmt.Sprintf("prepare %d", blk))
				return int32(20 + blk)
			}, func(ids []int32, src, off, rows int) {
				events = append(events, fmt.Sprintf("write %v %d %d %d", ids, src, off, rows))
			})
		want := []string{"prepare 0", "write [20] 0 14 2", "prepare 1", "prepare 2", "write [21 22] 2 0 32", "prepare 3", "write [23] 34 0 4"}
		if !reflect.DeepEqual(events, want) {
			t.Fatalf("events=%v, want %v", events, want)
		}
	})
	for _, failure := range []string{"write before preparation boundary", "preparation", "full group write"} {
		t.Run(failure, func(t *testing.T) {
			marker := &struct{ name string }{failure}
			prepared, writes, committed := 0, 0, false
			var caught any
			func() {
				defer func() { caught = recover() }()
				pagedKVAppendBatches(0, 65*16, 16,
					func(blk int) bool { return failure != "full group write" && blk == 1 },
					func(blk int) int32 {
						if failure == "preparation" && blk == 1 {
							panic(marker)
						}
						prepared++
						return int32(blk)
					}, func([]int32, int, int, int) {
						writes++
						if failure != "preparation" {
							panic(marker)
						}
					})
				committed = true
			}()
			wantPrepared := 1
			if failure == "full group write" {
				wantPrepared = 64
			}
			if caught != marker || committed || writes != 1 || prepared != wantPrepared {
				t.Fatalf("caught=%v committed=%v writes=%d prepared=%d", caught, committed, writes, prepared)
			}
		})
	}
	t.Run("invalid range fails before callbacks", func(t *testing.T) {
		for _, tc := range [][3]int{{-1, 1, 16}, {0, 0, 16}, {0, 1, 0}, {int(^uint(0) >> 1), 1, 16}} {
			called := false
			var caught any
			func() {
				defer func() { caught = recover() }()
				pagedKVAppendBatches(tc[0], tc[1], tc[2], func(int) bool { called = true; return false },
					func(int) int32 { called = true; return 0 }, func([]int32, int, int, int) { called = true })
			}()
			if caught == nil || called {
				t.Fatalf("range %v: panic=%v called=%v", tc, caught, called)
			}
		}
	})
}
