package model

import (
	"math"
	"reflect"
	"testing"
)

// Two scores suffice to make MaxInt+1 overflow int64 on a 64-bit host. The old
// check then permits invalid int32 row IDs; no large allocation is involved.
// The boundary controls also pin the existing policy for zero visible rows.
// fak-test:runtime fast est=1ms lane=default
func TestV41SelectIndexRowsOffsetOverflow(t *testing.T) {
	t.Parallel()
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name            string
		scores          []float32
		rows, k, offset int
		want            []int32
		wantErr         bool
	}{
		{"machine max plus last row", []float32{3, 2}, 2, 2, maxInt, nil, true},
		{"int32 max plus last row", []float32{3, 2}, 2, 2, math.MaxInt32, nil, true},
		{"last int32 row", []float32{3}, 1, 1, math.MaxInt32, []int32{math.MaxInt32}, false},
		{"last two int32 rows", []float32{3, 2}, 2, 2, math.MaxInt32 - 1, []int32{math.MaxInt32 - 1, math.MaxInt32}, false},
		{"negative offset", []float32{3}, 1, 1, -1, nil, true},
		{"zero visible boundary", []float32{3, 2}, 0, 2, math.MaxInt32, []int32{-1, -1}, false},
		{"empty boundary", nil, 0, 2, math.MaxInt32, []int32{}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := V41SelectIndexRows(tc.scores, tc.rows, tc.k, tc.offset, nil)
			if (err != nil) != tc.wantErr || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("rows=%v err=%v, want rows=%v error=%t", got, err, tc.want, tc.wantErr)
			}
		})
	}
	if maxInt > math.MaxInt32 {
		// No visible rows has always required a representable offset too.
		if got, err := V41SelectIndexRows(nil, 0, 0, maxInt, nil); got != nil || err == nil {
			t.Fatal("empty selection accepted a nonrepresentable offset")
		}
	}
}
