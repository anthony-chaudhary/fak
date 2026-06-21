package cachemeta

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestFromRefComputesInlineDigestAndPreservesSecurity(t *testing.T) {
	ref := abi.Ref{
		Kind:   abi.RefInline,
		Inline: []byte(`{"ok":true}`),
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeFleet,
	}

	e := FromRef(ref, WithAdmission(AdmissionAllow, "unit-test"))
	if e.ID.Digest != DigestBytes(ref.Inline) {
		t.Fatalf("digest %q, want computed inline digest", e.ID.Digest)
	}
	if e.ID.Length != int64(len(ref.Inline)) || e.ID.MediaType != MediaBytes || e.ID.Unit != UnitBytes {
		t.Fatalf("bad identity: %+v", e.ID)
	}
	if e.Security.Taint != abi.TaintTrusted || e.Security.Scope != abi.ScopeFleet {
		t.Fatalf("security did not preserve Ref taint/scope: %+v", e.Security)
	}
	if e.Security.AdmissionVerdict != AdmissionAllow || e.Security.AdmittedBy != "unit-test" {
		t.Fatalf("admission option not applied: %+v", e.Security)
	}
}

func TestFromVDSOKeyDescribesToolResult(t *testing.T) {
	ref := abi.Ref{Kind: abi.RefBlob, Digest: "payload-digest", Len: 42, Taint: abi.TaintTrusted, Scope: abi.ScopeFleet}
	e, err := FromVDSOKey("search_flights:argdigest:1.2", ref, WithWitness("git:abc123"))
	if err != nil {
		t.Fatalf("FromVDSOKey: %v", err)
	}
	if e.Plane != PlaneToolResult || e.Derivation.Producer != "vdso" {
		t.Fatalf("wrong plane/producer: %+v", e)
	}
	if e.Derivation.Tool != "search_flights" || e.Derivation.ArgsDigest != "argdigest" {
		t.Fatalf("wrong tool key axes: %+v", e.Derivation)
	}
	if e.Validity.AdmittedAtEpoch != "1.2" || e.Validity.Witness != "git:abc123" {
		t.Fatalf("wrong validity: %+v", e.Validity)
	}
	if e.Coherence.InvalidationMode != InvalidationExternalRefutation {
		t.Fatalf("witnessed tool entry should invalidate on refutation, got %q", e.Coherence.InvalidationMode)
	}
}

func TestFromVDSOKeyRejectsMalformedKey(t *testing.T) {
	_, err := FromVDSOKey("not-enough-parts", abi.Ref{})
	if !errors.Is(err, ErrBadVDSOKey) {
		t.Fatalf("err=%v, want ErrBadVDSOKey", err)
	}
}

func TestFromContextPageMarksQuarantine(t *testing.T) {
	e := FromContextPage(ContextPage{
		SessionID:   "s1",
		Step:        7,
		Role:        "read_webpage",
		Descriptor:  "read_webpage: [sealed]",
		Digest:      "page-digest",
		Len:         128,
		Taint:       abi.TaintTainted,
		Quarantined: true,
		QID:         "q2",
		Reason:      "TRUST_VIOLATION",
		Witness:     "etag:old",
		TrustEpoch:  3,
	})
	if e.Plane != PlaneContextPage || e.ID.MediaType != MediaRecallPage {
		t.Fatalf("bad context-page identity: %+v", e)
	}
	if e.Security.Taint != abi.TaintQuarantined || e.Security.AdmissionVerdict != AdmissionQuarantine {
		t.Fatalf("quarantine not reflected in security: %+v", e.Security)
	}
	if e.Labels["session_id"] != "s1" || e.Labels["step"] != "7" || e.Labels["qid"] != "q2" {
		t.Fatalf("page labels missing: %+v", e.Labels)
	}
}

func TestFromKVPrefixUsesTokenDigestAndPrefixMode(t *testing.T) {
	tokens := []int{1, 2, 3, 5, 8}
	e := FromKVPrefix(KVPrefix{Tokens: tokens, ModelID: "m", TokenizerID: "tok", Owner: "radixkv"})
	if e.ID.Digest != DigestTokenIDs(tokens) || e.ID.Length != int64(len(tokens)) {
		t.Fatalf("bad KV identity: %+v", e.ID)
	}
	if e.ID.MediaType != MediaKVSpan || e.ID.Unit != UnitPositions {
		t.Fatalf("bad KV media/unit: %+v", e.ID)
	}
	if e.Derivation.ModelID != "m" || e.Derivation.TokenizerID != "tok" || e.Derivation.PositionMode != PositionPrefixAligned {
		t.Fatalf("bad derivation: %+v", e.Derivation)
	}
	if e.Security.Scope != abi.ScopeFleet || e.Security.Taint != abi.TaintTrusted {
		t.Fatalf("KV prefix should default to fleet/trusted, got %+v", e.Security)
	}
}

func TestFromMemoryViewDescribesDerivedContextView(t *testing.T) {
	src := FromContextPage(ContextPage{
		SessionID: "s1",
		Step:      2,
		Role:      "search_flights",
		Digest:    "source-digest",
		Len:       64,
		Taint:     abi.TaintTrusted,
	})
	e := FromMemoryView(MemoryView{
		ViewID:            "view-s1-2",
		ViewType:          "snippet",
		Length:            48,
		SourceRefs:        []EntryID{src.ID},
		Producer:          "contextq",
		PolicyVersion:     "policy-v1",
		Scope:             abi.ScopeAgent,
		Taint:             abi.TaintTrusted,
		Coverage:          0.75,
		FaithfulnessProbe: 1.0,
		Witness:           "git:abc123",
		TTLMillis:         1000,
	})
	if e.Plane != PlaneMemoryView || e.ID.MediaType != MediaMemoryView {
		t.Fatalf("bad memory-view identity: %+v", e)
	}
	if e.Labels["view_id"] != "view-s1-2" || e.Labels["view_type"] != "snippet" {
		t.Fatalf("view labels missing: %+v", e.Labels)
	}
	if len(e.Derivation.SourceRefs) != 1 || e.Derivation.SourceRefs[0] != src.ID {
		t.Fatalf("source refs missing: %+v", e.Derivation.SourceRefs)
	}
	if e.Validity.PolicyVersion != "policy-v1" || e.Validity.Witness != "git:abc123" || e.Validity.TTLMillis != 1000 {
		t.Fatalf("validity missing: %+v", e.Validity)
	}
	if e.Metrics.Coverage != 0.75 || e.Metrics.FaithfulnessProbe != 1.0 {
		t.Fatalf("view metrics missing: %+v", e.Metrics)
	}
}

func TestLookupVerdictKeepsMissReasonsTyped(t *testing.T) {
	m := Miss(ReasonScopeDenied)
	if m.Kind != LookupMiss || m.CanServe() {
		t.Fatalf("miss should not be serveable: %+v", m)
	}
	e := FromKVPrefix(KVPrefix{Tokens: []int{9}})
	h := Hit(e)
	if !h.CanServe() || h.Handle != e.ID {
		t.Fatalf("hit should carry serveable handle: %+v", h)
	}
}
