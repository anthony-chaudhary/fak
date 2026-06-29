package cachemeta

import "testing"

func TestPlanExternalInvalidationsDropsRemoteKVAndReferencingAttentionIndex(t *testing.T) {
	remoteKV := FromKVPrefix(
		KVPrefix{Tokens: []int{10, 20, 30, 40}, ModelID: "glm-5.2", TokenizerID: "glm-tokenizer", Owner: "radixkv"},
		WithResidency(TierProvider, "sglang", "session-7"),
		WithAdmission(AdmissionQuarantine, "l3-referee"),
		WithDeletionCertificate(DeletionCertificate{Schema: "fak.deletioncert/v1", Subject: "kv-subject", Digest: "cert-digest"}),
		WithLabel("provider", "sglang"),
		WithLabel("engine", "glm-moe-dsa"),
	)
	idx := FromAttentionIndex(
		AttentionIndex{
			Tokens:         []int{10, 20, 30, 40},
			ModelID:        "glm-5.2",
			TokenizerID:    "glm-tokenizer",
			IndexerID:      "glm52-dsa-indexer:v1",
			LayerGroup:     "layers-0-3",
			Layers:         []int{0, 1, 2, 3},
			DecisionDigest: DigestBytes([]byte("topk")),
			ParentKV:       remoteKV.ID,
			Owner:          "sglang-dsa",
			Lease:          "session-7",
			Causal:         true,
		},
		WithResidency(TierProvider, "sglang", "session-7"),
		WithLabel("provider", "sglang"),
		WithLabel("engine", "glm-moe-dsa"),
	)
	other := FromKVPrefix(
		KVPrefix{Tokens: []int{99}, ModelID: "glm-5.2", TokenizerID: "glm-tokenizer"},
		WithResidency(TierProvider, "sglang", "session-8"),
	)
	telemetry := FromProviderCache(ProviderCache{
		Provider:     "sglang",
		ModelID:      "glm-5.2",
		CachedTokens: 128,
		PromptTokens: 256,
	})

	dirs := PlanExternalInvalidations(remoteKV.ID, []Entry{remoteKV, idx, other, telemetry})
	if len(dirs) != 2 {
		t.Fatalf("directives = %d, want remote K/V + attention index: %+v", len(dirs), dirs)
	}
	byKind := map[ExternalInvalidationKind]ExternalInvalidationDirective{}
	for _, d := range dirs {
		byKind[d.Kind] = d
		if d.Provider != "sglang" || d.Engine != "glm-moe-dsa" {
			t.Fatalf("directive lost provider/engine labels: %+v", d)
		}
		if d.Residency.Tier != TierProvider || d.Residency.Owner != "sglang" || d.Residency.Lease != "session-7" {
			t.Fatalf("directive lost external residency: %+v", d)
		}
	}
	if d := byKind[ExternalInvalidateKVSpan]; d.Entry != remoteKV.ID || d.Reason != "poisoned_kv" {
		t.Fatalf("bad K/V directive: %+v", d)
	}
	if d := byKind[ExternalInvalidateKVSpan]; d.Governance.Security.AdmissionVerdict != AdmissionQuarantine ||
		d.Governance.Security.AdmittedBy != "l3-referee" ||
		d.Governance.Lease != "session-7" ||
		d.Governance.DeletionCertificate.Digest != "cert-digest" {
		t.Fatalf("K/V directive lost governance attestation: %+v", d.Governance)
	}
	if d := byKind[ExternalInvalidateAttentionIndex]; d.Entry != idx.ID || d.Reason != "parent_kv_poisoned" {
		t.Fatalf("bad attention-index directive: %+v", d)
	}
	if d := byKind[ExternalInvalidateAttentionIndex]; d.Governance.DeletionCertificate.Digest != "cert-digest" ||
		d.Governance.Security.AdmittedBy != "l3-referee" ||
		d.Governance.Lease != "session-7" {
		t.Fatalf("dependent attention-index directive did not inherit K/V governance: %+v", d.Governance)
	}
	for _, d := range dirs {
		if d.Entry == telemetry.ID {
			t.Fatalf("provider telemetry must not become an invalidation directive: %+v", dirs)
		}
	}
}

func TestPlanExternalInvalidationsRejectsEmptyPoisonedKV(t *testing.T) {
	if dirs := PlanExternalInvalidations(EntryID{}, []Entry{FromKVPrefix(KVPrefix{Tokens: []int{1}})}); len(dirs) != 0 {
		t.Fatalf("empty poisoned K/V should produce no directives: %+v", dirs)
	}
}

func TestExactSpanTargetsProjectsNamedKVAndAttentionIndex(t *testing.T) {
	remoteKV := FromKVPrefix(
		KVPrefix{Tokens: []int{10, 20, 30, 40}, ModelID: "glm-5.2", TokenizerID: "glm-tokenizer", Owner: "radixkv"},
		WithResidency(TierProvider, "sglang", "session-7"),
	)
	idx := FromAttentionIndex(
		AttentionIndex{
			Tokens:         []int{10, 20, 30, 40},
			ModelID:        "glm-5.2",
			TokenizerID:    "glm-tokenizer",
			IndexerID:      "glm52-dsa-indexer:v1",
			LayerGroup:     "layers-0-3",
			Layers:         []int{0, 1, 2, 3},
			DecisionDigest: DigestBytes([]byte("topk")),
			ParentKV:       remoteKV.ID,
			Causal:         true,
		},
		WithResidency(TierProvider, "sglang", "session-7"),
	)

	dirs := PlanExternalInvalidations(remoteKV.ID, []Entry{remoteKV, idx})
	targets := ExactSpanTargets(dirs)
	if len(targets) != 2 {
		t.Fatalf("targets = %d, want kv span + attention index: %+v", len(targets), targets)
	}
	byKind := map[ExternalInvalidationKind]ExactSpanTarget{}
	for _, tg := range targets {
		byKind[tg.Kind] = tg
		if tg.Digest == "" || tg.Unit == "" || tg.Length <= 0 {
			t.Fatalf("exact-span target lost content-addressed identity: %+v", tg)
		}
	}
	if tg := byKind[ExternalInvalidateKVSpan]; tg.Digest != remoteKV.ID.Digest || tg.MediaType != remoteKV.ID.MediaType || tg.Reason != "poisoned_kv" {
		t.Fatalf("bad K/V span target: %+v", tg)
	}
	if tg := byKind[ExternalInvalidateKVSpan]; tg.Governance.Lease != "session-7" {
		t.Fatalf("exact-span target lost governance descriptor: %+v", tg.Governance)
	}
	if tg := byKind[ExternalInvalidateAttentionIndex]; tg.Digest != idx.ID.Digest || tg.MediaType != idx.ID.MediaType || tg.Reason != "parent_kv_poisoned" {
		t.Fatalf("bad attention-index target: %+v", tg)
	}
}

func TestAttestInvalidationsSurfacesDegradedGovernedOutcome(t *testing.T) {
	kv := FromKVPrefix(
		KVPrefix{Tokens: []int{1, 2}, ModelID: "m", Owner: "kvmmu"},
		WithResidency(TierProvider, "vllm", "lease-27"),
		WithAdmission(AdmissionQuarantine, "l3-referee"),
		WithDeletionCertificate(DeletionCertificate{Schema: "fak.deletioncert/v1", Subject: "span-27", Digest: "cert-27"}),
	)
	dirs := []ExternalInvalidationDirective{{
		Kind:       ExternalInvalidateKVSpan,
		Entry:      kv.ID,
		Residency:  kv.Residency,
		Reason:     "poisoned_kv",
		Governance: GovernanceFromEntry(kv),
	}}

	att := AttestInvalidations(dirs, KVEvictionScopeWholePrefixCache, false, "exact_span_unsupported_whole_prefix_flush")
	if len(att) != 1 {
		t.Fatalf("attestations = %d, want 1", len(att))
	}
	if !att[0].Degraded || att[0].Scope != KVEvictionScopeWholePrefixCache || att[0].ExactSpanSupported {
		t.Fatalf("degraded whole-prefix outcome not surfaced: %+v", att[0])
	}
	if !att[0].RefereeAdmitted || att[0].RefereeReason != KVRefereeAdmitted {
		t.Fatalf("degraded outcome did not pass through the referee: %+v", att[0])
	}
	if att[0].Governance.Security.AdmissionVerdict != AdmissionQuarantine ||
		att[0].Governance.Lease != "lease-27" ||
		att[0].Governance.DeletionCertificate.Digest != "cert-27" {
		t.Fatalf("governance not carried in attestation: %+v", att[0].Governance)
	}
}

func TestExactSpanTargetsSkipsDirectivesWithoutSpanIdentity(t *testing.T) {
	// A coarse, identity-less whole-cache directive (the proxy-quarantine shape:
	// Kind set but no Entry) yields no exact-span target, so a caller that requires
	// exact-span eviction fails closed rather than "precisely evicting nothing".
	dirs := []ExternalInvalidationDirective{{
		Kind:   ExternalInvalidateKVSpan,
		Plane:  PlaneKVPrefix,
		Reason: "proxy_tool_result_quarantine",
	}}
	if targets := ExactSpanTargets(dirs); len(targets) != 0 {
		t.Fatalf("identity-less directive must not project to an exact-span target: %+v", targets)
	}
	if targets := ExactSpanTargets(nil); targets != nil {
		t.Fatalf("nil directives must project to nil targets: %+v", targets)
	}
}
