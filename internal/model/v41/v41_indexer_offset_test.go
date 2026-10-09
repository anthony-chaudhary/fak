package v41

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41SelectIndexRowsOffsetDoesNotOverflow(t *testing.T) {
	t.Parallel()

	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                string
		offset, compressLen int
		want                []int32
		wantErr             bool
	}{
		// On 64-bit int, the old addition wraps MaxInt+1 to MinInt and
		// admits invalid rows. On 32-bit int, this is an ordinary range check.
		{name: "max_int_two_rows", offset: maxInt, compressLen: 2, wantErr: true},
		{name: "past_int32_last_row", offset: math.MaxInt32, compressLen: 2, wantErr: true},
		{name: "last_two_int32_rows", offset: math.MaxInt32 - 1, compressLen: 2, want: []int32{math.MaxInt32 - 1, math.MaxInt32}},
		{name: "last_int32_row", offset: math.MaxInt32, compressLen: 1, want: []int32{math.MaxInt32, -1}},
		{name: "no_visible_rows", offset: math.MaxInt32, want: []int32{-1, -1}},
		{name: "no_visible_rows_max_int", offset: maxInt, want: []int32{-1, -1}, wantErr: maxInt > math.MaxInt32},
		{name: "negative_empty_offset", offset: -1, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows, err := V41SelectIndexRows([]float32{9, 8}, tc.compressLen, 2, tc.offset, nil)
			if tc.wantErr {
				if err == nil || rows != nil {
					t.Fatalf("got rows %v, error %v; want nil rows and an error", rows, err)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(rows, tc.want) {
				t.Fatalf("got rows %v, error %v; want %v and no error", rows, err, tc.want)
			}
		})
	}
}
