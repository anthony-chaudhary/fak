package cachemeta

import "testing"

// baseLease is a decode-side lease over a named span, granted at t=1000 and expiring
// at t=5000. Individual tests set Event / clock to exercise a transition.
func baseLease() NIXLLease {
	return NIXLLease{
		LeaseID:         "lease-abc",
		TraceID:         "trace-1",
		RequestID:       "req-1",
		RemoteEngineID:  "vllm-prefill-0",
		SpanDigest:      DigestBytes([]byte("kv-span-xyz")),
		Tokens:          128,
		ModelID:         "m",
		TokenizerID:     "t",
		Role:            NIXLRoleDecodePull,
		GrantedAtMillis: 1000,
		ExpiresAtMillis: 5000,
	}
}

// Acceptance #1: lease create, heartbeat, transfer-complete, and expiry fold into the
// expected cachemeta residency verdicts.
func TestNIXLLeaseVerdictFoldsLifecycle(t *testing.T) {
	tests := []struct {
		name       string
		event      NIXLLeaseEvent
		now        int64
		wantKind   LookupKind
		wantReason LookupReason
		wantServe  bool
	}{
		{"create-active", NIXLLeaseCreate, 1000, LookupHit, ReasonNone, true},
		{"heartbeat-active", NIXLLeaseHeartbeat, 3000, LookupHit, ReasonNone, true},
		{"transfer-complete-released", NIXLLeaseTransferComplete, 3000, LookupMiss, ReasonLeaseReleased, false},
		{"explicit-expiry", NIXLLeaseExpiry, 3000, LookupMiss, ReasonExpiredTTL, false},
		{"abort-fault", NIXLLeaseAbort, 3000, LookupFault, ReasonResidencyFault, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := baseLease()
			l.Event = tt.event
			v := NIXLLeaseVerdict(l, tt.now)
			if v.Kind != tt.wantKind {
				t.Fatalf("kind = %q, want %q", v.Kind, tt.wantKind)
			}
			if v.Reason != tt.wantReason {
				t.Fatalf("reason = %q, want %q", v.Reason, tt.wantReason)
			}
			if got := v.CanServe(); got != tt.wantServe {
				t.Fatalf("CanServe() = %v, want %v", got, tt.wantServe)
			}
		})
	}
}

// Acceptance #2: an expired lease demotes from the warm set and NO cache-aware route
// is taken on that entry. Same lease, still Event=heartbeat (never explicitly
// expired) — only the clock advances past ExpiresAtMillis, and the verdict must flip
// from serveable to a demote/miss purely from time.
func TestNIXLExpiredLeaseDemotesWarmSetAndBlocksRoute(t *testing.T) {
	l := baseLease()
	l.Event = NIXLLeaseHeartbeat

	// Before expiry: warm, routable.
	if !l.Warm(4999) {
		t.Fatalf("lease should be warm at t=4999 (expiry=5000)")
	}
	if v := NIXLLeaseVerdict(l, 4999); !v.CanServe() {
		t.Fatalf("pre-expiry verdict must be serveable (a cache-aware route is valid)")
	}

	// At/after expiry: demoted, and a router gating on CanServe takes no route.
	if l.Warm(5000) {
		t.Fatalf("lease must NOT be warm at t=5000 (expiry=5000); warm set must demote it")
	}
	v := NIXLLeaseVerdict(l, 5000)
	if v.Kind != LookupMiss || v.Reason != ReasonExpiredTTL {
		t.Fatalf("expired verdict = %q/%q, want miss/expired_ttl", v.Kind, v.Reason)
	}
	if v.CanServe() {
		t.Fatalf("expired lease must not serve: a cache-aware route was taken on a demoted entry")
	}
	if l.StateAt(6000) != NIXLStateExpired {
		t.Fatalf("state well past expiry = %q, want expired", l.StateAt(6000))
	}
}

// An unidentified lease (no span digest) can never back a warm hit — fail-closed.
func TestNIXLUnidentifiedLeaseNeverWarm(t *testing.T) {
	l := baseLease()
	l.Event = NIXLLeaseCreate
	l.SpanDigest = ""
	if l.Warm(1000) {
		t.Fatalf("a lease naming no span must never be warm")
	}
	v := NIXLLeaseVerdict(l, 1000)
	if v.Kind != LookupMiss || v.Reason != ReasonAbsent {
		t.Fatalf("unidentified verdict = %q/%q, want miss/absent", v.Kind, v.Reason)
	}
}

// FromNIXLLease must place the lease on the kv_transfer plane at TierRemote, held
// under the lease id and owned by the remote engine, with the witness labels intact.
func TestFromNIXLLeaseLowersToRemoteResidency(t *testing.T) {
	l := baseLease()
	l.Event = NIXLLeaseHeartbeat
	e := FromNIXLLease(l, 3000)

	if e.Plane != PlaneKVTransfer {
		t.Fatalf("plane = %q, want kv_transfer", e.Plane)
	}
	if e.Residency.Tier != TierRemote {
		t.Fatalf("tier = %q, want remote", e.Residency.Tier)
	}
	if e.Residency.Owner != "vllm-prefill-0" || e.Residency.Lease != "lease-abc" {
		t.Fatalf("residency owner/lease = %q/%q, want vllm-prefill-0/lease-abc", e.Residency.Owner, e.Residency.Lease)
	}
	if e.ID.Digest != l.SpanDigest || e.ID.MediaType != MediaKVSpan {
		t.Fatalf("entry identity not bound to the KV span")
	}
	if e.Coherence.InvalidationMode != InvalidationExternalRefutation {
		t.Fatalf("invalidation mode = %q, want external_refutation", e.Coherence.InvalidationMode)
	}
	for k, want := range map[string]string{
		"nixl_lease":       "lease-abc",
		"nixl_role":        string(NIXLRoleDecodePull),
		"nixl_event":       string(NIXLLeaseHeartbeat),
		"nixl_state":       string(NIXLStateActive),
		"remote_engine":    "vllm-prefill-0",
		"trace_id":         "trace-1",
		"request_id":       "req-1",
		"lease_expires_at": "5000",
	} {
		if got := e.Labels[k]; got != want {
			t.Fatalf("label %q = %q, want %q", k, got, want)
		}
	}

	// An aborted lease lowers to a typed FAULT outcome, not silent OK residency.
	l.Event = NIXLLeaseAbort
	fe := FromNIXLLease(l, 3000)
	if fe.Labels["outcome"] != string(KVTransferFault) {
		t.Fatalf("aborted lease outcome = %q, want fault", fe.Labels["outcome"])
	}
	if fe.Labels["nixl_state"] != string(NIXLStateFailed) {
		t.Fatalf("aborted lease state label = %q, want failed", fe.Labels["nixl_state"])
	}
}

// Acceptance #3: a deletion/clear event records exact-span proof, whole-prefix reset,
// or no proof — and the no-proof case fails closed with no valid eviction scope.
func TestClassifyNIXLClearRecordsProof(t *testing.T) {
	span := DigestBytes([]byte("kv-span-xyz"))
	tests := []struct {
		name      string
		clear     NIXLClear
		wantProof NIXLClearProof
		wantScope KVEvictionScope
		wantOK    bool
	}{
		{
			"exact-span-confirmed",
			NIXLClear{SpanDigest: span, ExactSpanConfirmed: true},
			NIXLClearExactSpan, KVEvictionScopeExactSpan, true,
		},
		{
			"whole-prefix-reset",
			NIXLClear{SpanDigest: span, WholePrefixReset: true},
			NIXLClearWholePrefix, KVEvictionScopeWholePrefixCache, true,
		},
		{
			"no-receipt-no-proof",
			NIXLClear{SpanDigest: span},
			NIXLClearNoProof, "", false,
		},
		{
			// "exact" claimed but no span identity -> cannot be precise -> no proof.
			"exact-without-span-degrades-to-none",
			NIXLClear{ExactSpanConfirmed: true},
			NIXLClearNoProof, "", false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ClassifyNIXLClear(tt.clear)
			if got != tt.wantProof {
				t.Fatalf("proof = %q, want %q", got, tt.wantProof)
			}
			scope, ok := got.EvictionScope()
			if ok != tt.wantOK || scope != tt.wantScope {
				t.Fatalf("EvictionScope() = %q,%v want %q,%v", scope, ok, tt.wantScope, tt.wantOK)
			}
		})
	}
}
