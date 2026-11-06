package benchauthority

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestValidateArtifactsEscapedEnvelopeCannotFallBackToLegacy(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, raw string
		refuse    bool
	}{
		{
			name:   "escaped first character with invalid simulation evidence",
			raw:    `{"\u0062enchmark_artifact":{"schema":"fak-benchmark-artifact/1","run_id":"fixture","simulation_evidence":"not-an-object"}}`,
			refuse: true,
		},
		{
			name:   "escaped underscore with null payload",
			raw:    `{"benchmark\u005fartifact":null}`,
			refuse: true,
		},
		{
			name:   "escaped last character with incomplete payload",
			raw:    `{"benchmark_artifac\u0074":{"run_id":"fixture"}}`,
			refuse: true,
		},
		{
			name:   "literal key with invalid payload",
			raw:    `{"benchmark_artifact":null}`,
			refuse: true,
		},
		{
			name:   "malformed JSON with literal key",
			raw:    `{"benchmark_artifact":`,
			refuse: true,
		},
		{
			name: "escaped valid envelope retains existing acceptance",
			raw:  `{"\u0062enchmark_artifact":{"schema":"fak-benchmark-artifact/1","run_id":"fixture"}}`,
		},
		{
			name: "legacy JSON without envelope",
			raw:  `{"lineage":{"lineage_schema":"fak-bench-lineage/1"}}`,
		},
		{
			name: "legacy text without envelope",
			raw:  "legacy benchmark log",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			if err := os.WriteFile(filepath.Join(root, "fixture.json"), []byte(tc.raw), 0o600); err != nil {
				t.Fatal(err)
			}
			errs := ValidateArtifacts(root, []Claim{{ID: "fixture", Artifact: "fixture.json", Status: Measured, Competitive: true}})
			if tc.refuse {
				if len(errs) != 1 || !strings.Contains(errs[0].Error(), "malformed benchmark_artifact envelope") {
					t.Fatalf("expected malformed envelope refusal, got %v", errs)
				}
			} else if len(errs) != 0 {
				t.Fatalf("existing valid or legacy artifact was refused: %v", errs)
			}
		})
	}
}
