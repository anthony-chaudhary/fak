package jsonlledger

import (
	"strings"
	"testing"
)

func TestFirst(t *testing.T) {
	type row struct {
		Schema string `json:"schema"`
	}
	tests := []struct {
		name       string
		input      string
		wantSchema string
		wantFound  bool
		wantErr    bool
	}{
		{name: "first nonblank row", input: "\n\n{\"schema\":\"v1\"}\n{\"schema\":\"v2\"}\n", wantSchema: "v1", wantFound: true},
		{name: "empty", input: "\n\n"},
		{name: "whitespace is a malformed row", input: "  \n{\"schema\":\"v1\"}\n", wantErr: true},
		{name: "malformed first row", input: "not-json\n{\"schema\":\"v1\"}\n", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, found, err := First[row](strings.NewReader(tt.input))
			if (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, wantErr %v", err, tt.wantErr)
			}
			if found != tt.wantFound {
				t.Fatalf("found = %v, want %v", found, tt.wantFound)
			}
			if got.Schema != tt.wantSchema {
				t.Fatalf("schema = %q, want %q", got.Schema, tt.wantSchema)
			}
		})
	}
}

func TestFirstAcceptsLongRow(t *testing.T) {
	padding := strings.Repeat("x", 128*1024)
	got, found, err := First[map[string]string](strings.NewReader(`{"schema":"v1","padding":"` + padding + `"}`))
	if err != nil || !found || got["schema"] != "v1" {
		t.Fatalf("First long row = (%q, %v, %v)", got["schema"], found, err)
	}
}
