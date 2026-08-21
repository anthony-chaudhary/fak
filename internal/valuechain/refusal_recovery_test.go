package valuechain

import (
	"strings"
	"testing"
)

func TestVerticalValueChainRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name     string
		manifest Manifest
		input    Input
		want     []string
	}{
		{name: "schema", manifest: Manifest{}, want: []string{"schema", Schema}},
		{name: "manifest name", manifest: Manifest{Schema: Schema}, input: Input{Schema: Schema}, want: []string{"name", "required"}},
		{name: "stage identity", manifest: Manifest{Schema: Schema, Name: "chain", Stages: []Stage{{}}}, input: Input{Schema: Schema}, want: []string{"stage", "id", "required"}},
		{
			name: "observation key",
			manifest: Manifest{
				Schema: Schema, Name: "chain", Stages: []Stage{{ID: "s", Kind: "stage"}}, Arms: []Arm{{ID: "a"}},
			},
			input: Input{Schema: Schema, Observations: []Observation{{
				ID: "observation", TraceID: "trace", StageID: "missing", Arm: "a", Provenance: "test",
			}}},
			want: []string{"unknown stage", "missing"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Audit(tc.manifest, tc.input)
			if err == nil {
				t.Fatal("Audit accepted refusal input")
			}
			for _, want := range tc.want {
				if !strings.Contains(err.Error(), want) {
					t.Fatalf("error %q omits recovery text %q", err, want)
				}
			}
		})
	}
}
