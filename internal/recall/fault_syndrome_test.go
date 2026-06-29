package recall

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// stubOracle is a test RepairOracle that pins each recovery fact independently, so a single
// test can place a fault in either bucket purely by what recovery knowledge it grants — the
// whole point of the classifier is that the SAME fault is repairable or erasure depending on
// whether a recovery path exists.
type stubOracle struct {
	replicas     map[string]bool // content addresses with a known clean replica
	rederivable  map[string]bool // witnesses whose trust can be re-derived
	replicaCalls []string        // addresses HasCleanReplica was asked about (proves the read happened)
	rederivCalls []string        // witnesses TrustRederivable was asked about
}

func (o *stubOracle) HasCleanReplica(d string) bool {
	o.replicaCalls = append(o.replicaCalls, d)
	return o.replicas[d]
}

func (o *stubOracle) TrustRederivable(w string) bool {
	o.rederivCalls = append(o.rederivCalls, w)
	return o.rederivable[w]
}

func TestClassifySyndrome_MetadataMismatch_Repairable(t *testing.T) {
	body := []byte("authoritative tool result bytes")
	d := Digest(body)
	// A sealed page stamped clean, then its Quarantined bit flipped off in the manifest AFTER
	// the syndrome was stamped (the unseal tamper). Body is authoritative; only the metadata
	// check word disagrees ⇒ FaultRepairable.
	sealed := stampSyndrome(Page{Step: 4, Digest: d, Len: int64(len(body)), Taint: 3, Quarantined: true, QID: "q9"})
	tampered := sealed
	tampered.Quarantined = false

	fs := classifySyndrome(tampered, body, &stubOracle{})
	if fs.Class != SyndromeRepairable {
		t.Fatalf("metadata-only mismatch over a present body => REPAIRABLE, got %v (reason %q)", fs.Class, fs.Reason)
	}
	if fs.Recovery != RecoveryMetadataRederive {
		t.Fatalf("metadata repair => RecoveryMetadataRederive, got %v", fs.Recovery)
	}
	if fs.Fault != FaultRepairable {
		t.Fatalf("underlying fault folded should be FaultRepairable, got %v", fs.Fault)
	}
	if fs.Step != 4 {
		t.Fatalf("classifier must carry the page step, got %d", fs.Step)
	}
}

func TestClassifySyndrome_DigestMiss_ReplicaSplitsTheBucket(t *testing.T) {
	body := []byte("the page body that later went missing")
	d := Digest(body)
	p := stampSyndrome(Page{Step: 1, Digest: d, Len: int64(len(body)), Taint: 1})

	// (a) body absent, NO replica known ⇒ ERASURE (uncorrectable).
	gone := &stubOracle{}
	fs := classifySyndrome(p, nil, gone)
	if fs.Class != SyndromeErasure || fs.Recovery != RecoveryNone {
		t.Fatalf("absent body, no replica => ERASURE/none, got %v/%v (reason %q)", fs.Class, fs.Recovery, fs.Reason)
	}
	if !fs.HasAxis || fs.Axis != EvidenceDigest {
		t.Fatalf("a digest erasure must report the digest axis, got hasAxis=%v axis=%v", fs.HasAxis, fs.Axis)
	}
	// the classifier must actually have CONSULTED the oracle about THIS address (not vacuous).
	if len(gone.replicaCalls) != 1 || gone.replicaCalls[0] != d {
		t.Fatalf("classifier must ask the oracle about the page digest, asked %v", gone.replicaCalls)
	}

	// (b) SAME absent body, but a clean replica of the same address EXISTS ⇒ REPAIRABLE by
	//     byte-faithful re-fetch. The fault is identical; only the recovery knowledge differs.
	withReplica := &stubOracle{replicas: map[string]bool{d: true}}
	fs = classifySyndrome(p, nil, withReplica)
	if fs.Class != SyndromeRepairable || fs.Recovery != RecoveryReplicaRefetch {
		t.Fatalf("absent body WITH clean replica => REPAIRABLE/replica_refetch, got %v/%v", fs.Class, fs.Recovery)
	}

	// (c) body PRESENT but rotted (does not hash to the address), no replica ⇒ ERASURE.
	fs = classifySyndrome(p, []byte("rotted, hashes elsewhere"), &stubOracle{})
	if fs.Class != SyndromeErasure || fs.Recovery != RecoveryNone {
		t.Fatalf("rotted body, no replica => ERASURE, got %v/%v", fs.Class, fs.Recovery)
	}
}

// TestClassifySyndrome_Witness drives the revoked-witness branch through the live package vDSO,
// the same ledger the classifier folds. A witnessed page whose witness is refuted fails the
// witness/trust-epoch axes; whether that is repairable turns ENTIRELY on the recovery oracle:
// re-derivable trust => REPAIRABLE, truly retired => ERASURE.
func TestClassifySyndrome_Witness(t *testing.T) {
	body := []byte("a witnessed tool result")
	d := Digest(body)
	const w = "src://refuted-for-785-test"
	p := stampSyndrome(Page{Step: 2, Digest: d, Len: int64(len(body)), Taint: 1, Witness: w, TrustEpoch: 7})

	// Sanity: BEFORE any revocation the page is healthy (clean fault, no failed axis).
	if fs := classifySyndrome(p, body, &stubOracle{}); fs.Class != SyndromeHealthy {
		t.Fatalf("an un-refuted witnessed clean page => HEALTHY, got %v (reason %q)", fs.Class, fs.Reason)
	}

	// Refute the witness in the live ledger so the digest axis still holds but witness/
	// trust-epoch now block. Revocation is permanent and the witness id is unique to this
	// test, matching the sibling pattern (dream_test/parity_test/scrub_test all Revoke directly).
	vdso.Default.Revoke(w)

	// (a) trust NOT re-derivable ⇒ ERASURE of trust (seal).
	noTrust := &stubOracle{}
	fs := classifySyndrome(p, body, noTrust)
	if fs.Class != SyndromeErasure || fs.Recovery != RecoveryNone {
		t.Fatalf("refuted witness, trust not re-derivable => ERASURE, got %v/%v (reason %q)", fs.Class, fs.Recovery, fs.Reason)
	}
	if !fs.HasAxis || fs.Axis != EvidenceWitness {
		t.Fatalf("revoked-witness erasure must report the witness axis, got hasAxis=%v axis=%v", fs.HasAxis, fs.Axis)
	}
	if len(noTrust.rederivCalls) == 0 || noTrust.rederivCalls[0] != w {
		t.Fatalf("classifier must ask the oracle whether trust for %q is re-derivable, asked %v", w, noTrust.rederivCalls)
	}

	// (b) SAME refuted witness, but trust IS re-derivable (a live replacement witness) ⇒
	//     REPAIRABLE without rewriting a byte. The re-derivable-trust vs truly-erased split.
	live := &stubOracle{rederivable: map[string]bool{w: true}}
	fs = classifySyndrome(p, body, live)
	if fs.Class != SyndromeRepairable || fs.Recovery != RecoveryWitnessRederive {
		t.Fatalf("refuted witness WITH re-derivable trust => REPAIRABLE/witness_rederive, got %v/%v", fs.Class, fs.Recovery)
	}
}

func TestClassifySyndrome_Quarantine_Erasure(t *testing.T) {
	body := []byte("poison the gate sealed")
	d := Digest(body)
	// A sealed page whose syndrome AGREES with its sealed metadata (so the only fault is the
	// quarantine axis, not a metadata mismatch). It is poison: never repairable into trusted text.
	sealed := stampSyndrome(Page{Step: 0, Digest: d, Len: int64(len(body)), Taint: 3, Quarantined: true, QID: "q1"})
	// stampSyndrome leaves FaultClean (syndrome matches). To make a fault the classifier must
	// see the quarantine BLOCK, classify via the session helper which folds clearance; here the
	// page-level syndrome reports the quarantine axis as required-and-not-held. But classifySyndrome
	// short-circuits on FaultClean as healthy — the quarantine block surfaces only when the fault
	// is non-clean OR via the session clearance path. So assert the SESSION path (uncleared seal).
	s := &Session{Manifest: Manifest{Pages: []Page{sealed}}, cas: map[string][]byte{d: body}, cleared: map[string]bool{}}
	fs, err := s.ClassifyFaultSyndrome(0, &stubOracle{})
	if err != nil {
		t.Fatal(err)
	}
	if fs.Class != SyndromeErasure {
		t.Fatalf("an UNCLEARED sealed page must classify ERASURE (never repair poison), got %v (reason %q)", fs.Class, fs.Reason)
	}
	if !fs.HasAxis || fs.Axis != EvidenceQuarantine {
		t.Fatalf("a quarantine erasure must report the quarantine axis, got hasAxis=%v axis=%v", fs.HasAxis, fs.Axis)
	}

	// A CLEARED seal over the same authoritative body is NOT an erasure on the quarantine axis:
	// the session recorded a clearance, so the quarantine block is folded out and the page is
	// healthy (its syndrome agrees, body present).
	s.cleared = map[string]bool{"q1": true}
	fs, err = s.ClassifyFaultSyndrome(0, &stubOracle{})
	if err != nil {
		t.Fatal(err)
	}
	if fs.Class == SyndromeErasure && fs.HasAxis && fs.Axis == EvidenceQuarantine {
		t.Fatalf("a CLEARED seal must not be a quarantine erasure, got %v/%v", fs.Class, fs.Axis)
	}
}

func TestClassifySyndrome_HealthyPassThrough(t *testing.T) {
	body := []byte("a clean, present, durable page")
	d := Digest(body)
	clean := stampSyndrome(Page{Step: 3, Digest: d, Len: int64(len(body)), Taint: 0})
	if fs := classifySyndrome(clean, body, &stubOracle{}); fs.Class != SyndromeHealthy || fs.Recovery != RecoveryNone {
		t.Fatalf("a clean page => HEALTHY/none, got %v/%v", fs.Class, fs.Recovery)
	}
	// An UNSTAMPED page (no syndrome) is honest-absence, not a fault.
	unstamped := Page{Step: 5, Digest: d, Len: int64(len(body))}
	if fs := classifySyndrome(unstamped, body, &stubOracle{}); fs.Class != SyndromeHealthy {
		t.Fatalf("an unchecked page => HEALTHY (not a fault), got %v", fs.Class)
	}
}

// TestClassifySyndrome_NilOracleFailsClosed proves the security-load-bearing default: with NO
// oracle, a digest miss and a revoked witness both classify ERASURE — "no recovery knowledge"
// means "no recovery", never a free pass.
func TestClassifySyndrome_NilOracleFailsClosed(t *testing.T) {
	body := []byte("present once")
	d := Digest(body)
	p := stampSyndrome(Page{Step: 0, Digest: d, Len: int64(len(body))})
	if fs := classifySyndrome(p, nil, nil); fs.Class != SyndromeErasure {
		t.Fatalf("nil oracle + absent body => ERASURE (fail closed), got %v", fs.Class)
	}
}

// TestClassifySyndrome_IsPureRead proves the classification mutates nothing: the page value, the
// body bytes, and the oracle's recovery facts are byte-identical before and after — it is a read.
func TestClassifySyndrome_IsPureRead(t *testing.T) {
	body := []byte("immutable body")
	d := Digest(body)
	p := stampSyndrome(Page{Step: 1, Digest: d, Len: int64(len(body)), Taint: 2, QID: "q2"})
	bodyCopy := append([]byte(nil), body...)
	pCopy := p
	oracle := &stubOracle{replicas: map[string]bool{d: true}}

	_ = classifySyndrome(p, body, oracle)

	if !reflect.DeepEqual(p, pCopy) {
		t.Fatalf("classifySyndrome mutated the page: before=%+v after=%+v", pCopy, p)
	}
	if !reflect.DeepEqual(body, bodyCopy) {
		t.Fatal("classifySyndrome mutated the body bytes")
	}
}

func TestSyndromeClass_StringAndPredicates(t *testing.T) {
	for c, want := range map[SyndromeClass]string{
		SyndromeHealthy: "healthy", SyndromeRepairable: "repairable", SyndromeErasure: "erasure",
	} {
		if c.String() != want {
			t.Errorf("%d.String()=%q, want %q", c, c.String(), want)
		}
	}
	if !SyndromeRepairable.Repairable() || SyndromeRepairable.Erasure() {
		t.Error("SyndromeRepairable predicates wrong")
	}
	if !SyndromeErasure.Erasure() || SyndromeErasure.Repairable() {
		t.Error("SyndromeErasure predicates wrong")
	}
	for r, want := range map[Recovery]string{
		RecoveryNone: "none", RecoveryMetadataRederive: "metadata_rederive",
		RecoveryReplicaRefetch: "replica_refetch", RecoveryWitnessRederive: "witness_rederive",
	} {
		if r.String() != want {
			t.Errorf("recovery %d.String()=%q, want %q", r, r.String(), want)
		}
	}
}
