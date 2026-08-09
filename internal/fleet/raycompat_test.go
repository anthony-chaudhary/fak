package fleet

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestRayRecoveryProbeReceipts(t *testing.T) {
	const upstream = "ray-project/ray@ray-2.56.1+g936f0d7d49d9"
	cases := []struct {
		name string
		got  RecoveryProbe
	}{
		{"known-positive", ProbeRecovery("known-positive", upstream, true, 0, 0, 3)},
		{"cancelled-retry-exhausted", ProbeRecovery("cancelled-retry-exhausted", upstream, false, 143, 3, 3)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := os.ReadFile(filepath.Join("testdata", "ray-"+tc.name+".json"))
			if err != nil {
				t.Fatal(err)
			}
			var want RecoveryProbe
			if err := json.Unmarshal(b, &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(tc.got, want) {
				t.Fatalf("receipt mismatch\n got: %#v\nwant: %#v", tc.got, want)
			}
		})
	}
}

func TestRayRecoveryProbeCannotPassVacuously(t *testing.T) {
	got := ProbeRecovery("cancelled-retry-exhausted", "ray-project/ray@ray-2.56.1+g936f0d7d49d9", false, 143, 3, 3)
	if got.Recoverable || got.Action != "ESCALATE" || got.FakState != "FAILED" {
		t.Fatalf("exhausted cancellation must fail closed: %#v", got)
	}
}
