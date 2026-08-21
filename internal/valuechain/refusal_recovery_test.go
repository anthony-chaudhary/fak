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
		{name: "schema", manifest: Manifest{}, want: []string{"schema", "fak-value-chain/1"}},
		{name: "chain id", manifest: Manifest{Schema: "fak-value-chain/1"}, want: []string{"chain_id", "required"}},
		{name: "stage identity", manifest: Manifest{Schema: "fak-value-chain/1", ChainID: "c", Outcome: Outcome{Unit: "tasks"}, Stages: []Stage{{}}}, want: []string{"stage", "id", "name", "required"}},
		{name: "observation key", manifest: Manifest{Schema: "fak-value-chain/1", ChainID: "c", Outcome: Outcome{Unit: "tasks"}, Stages: []Stage{{ID: "s", Name: "Stage"}}, Arms: []Arm{{ID: "a", Label: "Arm"}}}, input: Input{Observations: []Observation{{StageID: "missing", ArmID: "a"}}}, want: []string{"unknown stage", "missing"}},
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
