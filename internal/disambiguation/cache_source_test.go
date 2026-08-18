package disambiguation

import "testing"

func TestRunCacheSourceSelfTestReturnsFourPairwiseConcepts(t *testing.T) {
	report, err := RunCacheSourceSelfTest()
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != CacheSourceSelfTestSchemaVersion || report.IndexVersion != PublicIndexVersion {
		t.Fatalf("versions=%#v", report)
	}
	if len(report.Resolutions) != 4 || !report.Pairwise {
		t.Fatalf("report=%#v", report)
	}
	want := map[string]string{"vDSO cache": "tool-result cache", "KV cache": "model KV cache", "radix cache": "radix prefix cache", "provider cache": "provider prompt cache"}
	owners := map[string]bool{}
	for _, resolution := range report.Resolutions {
		if resolution.CanonicalTerm != want[resolution.Input] {
			t.Errorf("%q resolved to %q", resolution.Input, resolution.CanonicalTerm)
		}
		if resolution.SourcePath == "" || resolution.Scope == "" || resolution.ContrastCount != 3 {
			t.Errorf("incomplete resolution=%#v", resolution)
		}
		owners[resolution.OwnerLeaf] = true
	}
	if len(owners) != 4 {
		t.Fatalf("owners=%v, want four independent owners", owners)
	}
}

func TestCacheSourcePairwiseContrastsNameDistinctObjects(t *testing.T) {
	pairs := [][2]string{{"tool-result cache", "model KV cache"}, {"model KV cache", "radix prefix cache"}, {"radix prefix cache", "provider prompt cache"}}
	for _, pair := range pairs {
		left, err := Query(pair[0])
		if err != nil {
			t.Fatal(err)
		}
		contrast, ok := contrastTo(left.Entry, pair[1])
		if !ok || contrast.ForbiddenConflation == nil || !*contrast.ForbiddenConflation {
			t.Errorf("missing forbidden pair %q/%q", pair[0], pair[1])
		}
	}
}
