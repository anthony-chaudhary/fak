package suiteverify

import "testing"

func TestOptionalSchema(t *testing.T) {
	for _, tc := range []struct {
		name, got string
		wantErr   bool
	}{
		{name: "omitted"},
		{name: "matching", got: "v1"},
		{name: "mismatch", got: "v2", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := OptionalSchema("demo", tc.got, "v1")
			if (err != nil) != tc.wantErr {
				t.Fatalf("OptionalSchema() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}
