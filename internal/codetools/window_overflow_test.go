package codetools

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestReadWindowBoundsAvoidOverflow(t *testing.T) {
	t.Parallel()
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name          string
		offset, limit int
		truncated     bool
		want          string
		wantTruncated bool
	}{
		{"large limit after first line", 2, maxInt, false, "two\nthree", false},
		{"large limit after second line", 3, maxInt, false, "three", false},
		{"large limit from start", 0, maxInt, false, "one\ntwo\nthree", false},
		{"bounded page", 2, 1, false, "two", true},
		{"exact remaining lines", 2, 2, false, "two\nthree", false},
		{"zero limit", 2, 0, false, "two\nthree", false},
		{"preserve byte truncation", 2, maxInt, true, "two\nthree", true},
		{"offset beyond content", maxInt, maxInt, false, "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := ReadArgs{FilePath: "sample.go", Offset: tc.offset, Limit: tc.limit}
			if refusal := args.Validate(); refusal != nil {
				t.Fatalf("valid read window refused: %+v", refusal)
			}
			got, truncated := window("one\ntwo\nthree", tc.offset, tc.limit, tc.truncated)
			if got != tc.want || truncated != tc.wantTruncated {
				t.Fatalf("window = (%q, %v), want (%q, %v)", got, truncated, tc.want, tc.wantTruncated)
			}
		})
	}
}
