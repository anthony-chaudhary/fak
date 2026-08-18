package strictjson

import "testing"

func TestRows(t *testing.T) {
	type row struct {
		N int `json:"n"`
	}
	for _, tc := range []struct {
		raw  string
		want int
	}{{`[{"n":1},{"n":2}]`, 2}, {`{"n":1}`, 1}, {``, 0}, {`no`, 0}} {
		if got := len(Rows[row](tc.raw)); got != tc.want {
			t.Errorf("Rows(%q) length = %d, want %d", tc.raw, got, tc.want)
		}
	}
}
