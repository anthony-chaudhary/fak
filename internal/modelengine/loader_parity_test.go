package modelengine

import (
	"strings"
	"testing"
)

func TestLoaderParityFaithfulPathsReproduceReference(t *testing.T) {
	ref := loaderReferenceModel()
	baseline := loaderFingerprint(ref)
	want := loaderDecode(ref)
	for _, p := range loaderFaithfulPaths() {
		got := p.load(ref)
		if err := loaderServeGate(got, baseline); err != nil {
			t.Fatalf("faithful path %q was refused before serve: %v", p.name, err)
		}
		prov := loaderProvenanceOf("loader-parity/"+p.name, got, p.backend, baseline)
		if !prov.complete() {
			t.Fatalf("faithful path %q has incomplete provenance: %+v", p.name, prov)
		}
		v := loaderJudge(want, loaderDecode(got), prov)
		if !v.Pass {
			t.Fatalf("faithful path %q diverged from reference: %s", p.name, v.Detail)
		}
	}
}

func TestLoaderParityPlantedDefectsFailBeforeServeAndLocalize(t *testing.T) {
	ref := loaderReferenceModel()
	baseline := loaderFingerprint(ref)
	want := loaderDecode(ref)

	defects := []struct {
		name string
		load func(loaderModel) loaderModel
	}{
		{"changed-config-default", loaderConfigDefaultDefect},
		{"reordered-shard", loaderReorderShardDefect},
		{"dropped-shard", loaderDroppedShardDefect},
	}

	for _, tc := range defects {
		got := tc.load(ref)

		// Acceptance: a changed config default or tensor shard fails BEFORE serve.
		if err := loaderServeGate(got, baseline); err == nil {
			t.Fatalf("defect %q passed the pre-serve gate — it must fail before serve", tc.name)
		}

		// The defect must actually perturb generation (else it is not a witness).
		engTok := loaderDecode(got)
		wantIdx := loaderFirstDiff(want, engTok)
		if wantIdx < 0 {
			t.Fatalf("defect %q did not change the token stream — not a representative defect", tc.name)
		}

		prov := loaderProvenanceOf("loader-parity/"+tc.name, got, "defect", baseline)
		v := loaderJudge(want, engTok, prov)
		if v.Pass {
			t.Fatalf("defect %q was judged pass — a planted defect must fail", tc.name)
		}
		if v.Artifact == nil || v.Artifact.Divergence == nil {
			t.Fatalf("defect %q produced no replay artifact", tc.name)
		}
		if got := v.Artifact.Divergence.Index; got != wantIdx {
			t.Fatalf("defect %q first divergence reported at %d, want %d (detail: %s)", tc.name, got, wantIdx, v.Detail)
		}
		// The artifact renders provenance and scrubbed tensor names, never weights.
		s := v.Artifact.String()
		if !strings.Contains(s, "tensors=blk.") || strings.Contains(s, "data:") {
			t.Fatalf("defect %q replay artifact is not scrubbed/renderable: %s", tc.name, s)
		}
	}
}

func TestLoaderParityInconclusiveIsNeverPass(t *testing.T) {
	ref := loaderReferenceModel()
	baseline := loaderFingerprint(ref)
	prov := loaderProvenanceOf("loader-parity/empty", ref, "empty-engine", baseline)
	v := loaderJudge(loaderDecode(ref), nil, prov)
	if v.Pass {
		t.Fatalf("an empty engine trace was judged pass — inconclusive evidence must never pass")
	}
	if v.Artifact == nil {
		t.Fatalf("inconclusive case produced no replay artifact")
	}
}

func TestLoaderParityProvenanceComplete(t *testing.T) {
	ref := loaderReferenceModel()
	baseline := loaderFingerprint(ref)
	for _, p := range loaderFaithfulPaths() {
		prov := loaderProvenanceOf("loader-parity/"+p.name, p.load(ref), p.backend, baseline)
		if !prov.complete() {
			t.Fatalf("path %q incomplete provenance: %+v", p.name, prov)
		}
		if prov.Tier != "PR" {
			t.Fatalf("path %q tier=%q, want PR", p.name, prov.Tier)
		}
	}
}
