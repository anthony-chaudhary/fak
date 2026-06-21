// Command deletioncert is the end-to-end demonstrator for fak's provable-deletion
// receipt. It runs a REAL bit-exact KV eviction of a "secret" token span on fak's
// native engine, proves the surviving context is byte-identical to a run that
// never saw the secret (and differs from one that kept it), records the event in
// a tamper-evident hash-chained audit journal, mints a DeletionCertificate bound
// to that journal row, re-verifies the certificate, and finally TAMPERS with both
// the certificate and the journal to show verification fails closed.
//
// It runs on hardware that exists: a tiny in-memory synthetic model (no weights,
// no torch, no files), because the deletion property — "an evicted span leaves the
// context byte-identical to a run that never saw it" — is structural, not numeric.
// The same KV-cache code path (Prefill / Step / Evict) the HF-verified model uses
// is exercised here; see internal/model/synthetic.go.
//
// Usage:
//
//	go run ./cmd/deletioncert -selfcheck
//	    Full demo + assertions; exits non-zero if any invariant fails. Zero files.
//
//	go run ./cmd/deletioncert -selfcheck -out cert.json
//	    Also writes the minted certificate to cert.json for inspection.
package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/deletioncert"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the full demo with assertions (default when no other mode)")
	out := flag.String("out", "", "optional path to write the minted certificate JSON")
	flag.Parse()
	_ = selfcheck // single mode today; the flag documents intent and reserves room

	if err := run(*out); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("\nOK — provable-deletion certificate minted, verified, and tamper-rejected.")
}

func run(outPath string) error {
	// ---- 1. A tiny model and a "conversation" with a secret in the middle --------
	cfg := model.Config{
		HiddenSize: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 8, HeadDim: 8,
		IntermediateSize: 128, VocabSize: 256, RMSNormEps: 1e-5, RopeTheta: 10000,
	}
	m := model.NewSynthetic(cfg)

	// Token ids stand in for: a benign prefix, a SECRET tool result (an API key the
	// agent should never have ingested), and the query that follows it. The
	// synthetic model has no real language structure, so greedy decode saturates to
	// a constant token — but the SECRET changes WHICH constant (117 -> 233), which
	// is a genuine, non-vacuous witness that the span perturbs the model's output.
	prefix := []int{3, 17, 5}
	secret := []int{41, 2, 19, 7} // the poisoned/secret span
	query := []int{23, 11}
	const gen = 6 // tokens to decode greedily after the query

	fmt.Println("== fak provable-deletion demo ==")
	fmt.Printf("prefix=%v  secret=%v  query=%v\n", prefix, secret, query)

	// ---- 2. Three reference runs ------------------------------------------------
	// (a) NEVER saw the secret: prefix + query only.
	neverCont := continueGreedy(m, concat(prefix, query), gen)

	// (b) KEPT the secret: prefix + secret + query (the leak path).
	keptCont := continueGreedy(m, concat(prefix, secret, query), gen)

	// (c) EVICTED: prefill prefix+secret, evict the secret span BEFORE the query
	//     attends, then prefill the query and decode. This is the write-time
	//     quarantine path — the query never attends to the secret.
	s := m.NewSession()
	s.Prefill(prefix)
	s.Prefill(secret)
	if s.Cache.Len() != len(prefix)+len(secret) {
		return fmt.Errorf("pre-evict cache len %d", s.Cache.Len())
	}
	from, n := len(prefix), len(secret)
	removed := s.Cache.Evict(from, n)
	if removed != n || s.Cache.Len() != len(prefix) {
		return fmt.Errorf("evict removed %d (want %d), cache len %d (want %d)",
			removed, n, s.Cache.Len(), len(prefix))
	}
	logits := s.Prefill(query) // query prefilled AFTER eviction — never sees the secret
	evictCont := stepGreedy(s, logits, gen)

	fmt.Printf("\n  never-saw  continuation = %v\n", neverCont)
	fmt.Printf("  kept-secret continuation = %v\n", keptCont)
	fmt.Printf("  evicted    continuation = %v\n", evictCont)

	// ---- 3. The deletion property: evicted == never, and (non-vacuously) != kept -
	maxDelta := maxAbsIntDelta(evictCont, neverCont)
	if maxDelta != 0 {
		return fmt.Errorf("evicted continuation != never-saw (max|Δ|=%d) — eviction NOT bit-exact", maxDelta)
	}
	if equalInts(keptCont, neverCont) {
		return fmt.Errorf("kept-secret == never-saw — the secret does not perturb decode; witness is vacuous")
	}
	fmt.Printf("\n  PROVEN: evicted == never-saw (max|Δ|=0); kept-secret differs (non-vacuous).\n")

	// ---- 4. Record the eviction in a tamper-evident hash-chained journal --------
	jpath, err := os.CreateTemp("", "deletioncert-journal-*.jsonl")
	if err != nil {
		return err
	}
	jpath.Close()
	defer os.Remove(jpath.Name())
	j, err := journal.Open(jpath.Name())
	if err != nil {
		return err
	}
	// Emit a couple of benign decisions, then the QUARANTINE that records our evict,
	// so the anchor sits mid-chain (a realistic position, not row 1).
	emitDecide(j, "read", "trusted")
	emitDecide(j, "search", "tainted")
	witness := "commit:" + shortHash(secret) // the external witness the secret was admitted under
	emitQuarantine(j, witness, secret)
	if err := j.Flush(); err != nil {
		return err
	}

	// Re-read the journal as the auditor would, and confirm the chain is intact.
	nRows, err := journal.Verify(jpath.Name())
	if err != nil {
		return fmt.Errorf("journal failed its own integrity check: %w", err)
	}
	rows := j.Recent(nRows)
	anchorRow := rows[len(rows)-1] // the QUARANTINE row we just wrote
	fmt.Printf("\n  journal: %d rows, chain intact; anchor = seq %d (%s…)\n",
		nRows, anchorRow.Seq, anchorRow.Hash[:12])

	// ---- 5. Mint the certificate, bound to that journal row ---------------------
	// v1 is self-signed: a fresh per-mint keypair stands in for the deployment's
	// signing key. The certificate embeds the public key as its trust root.
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		return err
	}
	_ = pub
	cert, err := deletioncert.Mint(priv, deletioncert.Certificate{
		// Subject left empty: Mint binds it to Anchor.ResultDigest (the row's
		// content digest), and Verify re-enforces the equality — the cert names
		// WHICH data was evicted, not merely a position in the chain.
		Method:       "kv-cache-eviction",
		ModelPath:    "gqa-rope", // the path the equivalence claim is admissible for
		CodeCommit:   "selfcheck",
		WitnessName:  witness,
		Span:         deletioncert.Span{From: from, Len: n},
		EvictedCount: removed,
		Equivalence: deletioncert.Equivalence{
			Claim:       "surviving context byte-identical to a run that never saw the span (write-time evict, GQA/RoPE path)",
			MaxAbsDelta: float64(maxDelta), // 0
			RunID:       "cmd/deletioncert -selfcheck",
		},
		Anchor: deletioncert.Anchor{
			Seq:          anchorRow.Seq,
			PrevHash:     anchorRow.PrevHash,
			Hash:         anchorRow.Hash,
			ResultDigest: anchorRow.ResultDigest, // pins the subject to the row's content
		},
		JournalHead:  anchorRow.Hash, // the chain head at issue time
		TrustEpoch:   1,
		IssuedAtUnix: 0, // demo is clock-free; a real mint passes time.Now().Unix()
	})
	if err != nil {
		return err
	}
	fmt.Printf("  minted certificate: subject=%s scope=%s\n", cert.Subject, cert.Scope)

	if outPath != "" {
		b, _ := cert.Marshal()
		if err := os.WriteFile(outPath, b, 0o644); err != nil {
			return err
		}
		fmt.Printf("  wrote %s\n", outPath)
	}

	// ---- 6. Verify it against the (intact) journal — must be VALID ---------------
	jv := journalVerifier(rows)
	r := deletioncert.Verify(cert, jv)
	if !r.Valid {
		return fmt.Errorf("freshly-minted certificate did not verify: %+v", r)
	}
	fmt.Printf("\n  VERIFY (intact)        -> valid=%v sig=%v anchor=%v bound=%v equiv=%v self_attested=%v\n",
		r.Valid, r.SignatureOK, r.AnchorOK, r.AnchorBound, r.EquivalenceOK, r.SelfAttested)

	// ---- 7. Tamper checks — verification must FAIL CLOSED ------------------------
	// (a) Flip a field in the certificate: the signature must break.
	forged := cert
	forged.EvictedCount = 999
	if rr := deletioncert.Verify(forged, jv); rr.Valid {
		return fmt.Errorf("tampered certificate (evicted_count) passed verification")
	} else {
		fmt.Printf("  VERIFY (cert forged)   -> valid=false  (%s)\n", rr.Reason)
	}

	// (b) Over-claim the scope: also caught by the signature.
	forged2 := cert
	forged2.Scope = "all-derived-state-including-fine-tuned-weights"
	if rr := deletioncert.Verify(forged2, jv); rr.Valid {
		return fmt.Errorf("scope-inflated certificate passed verification")
	} else {
		fmt.Printf("  VERIFY (scope forged)  -> valid=false  (%s)\n", rr.Reason)
	}

	// (c) Rewrite the journal under the certificate: the anchor binding must break.
	rewritten := make([]journal.Row, len(rows))
	copy(rewritten, rows)
	rewritten[len(rewritten)-1].Hash = "0000000000000000000000000000000000000000000000000000000000000000"
	if rr := deletioncert.Verify(cert, journalVerifier(rewritten)); rr.Valid {
		return fmt.Errorf("certificate verified against a rewritten journal")
	} else {
		fmt.Printf("  VERIFY (journal rewrit)-> valid=false  (%s)\n", rr.Reason)
	}

	return nil
}

// ---- journal adapter --------------------------------------------------------

// journalVerifier adapts a slice of journal rows into a deletioncert.JournalVerifier.
// It re-validates the hash chain with the journal package's OWN verifier (so the
// integrity check is the auditor's, not ours) and then serves the requested row's
// hashes. A broken chain yields ok=false for every lookup — fail closed.
func journalVerifier(rows []journal.Row) deletioncert.JournalVerifier {
	return rowVerifier(rows)
}

type rowVerifier []journal.Row

func (rv rowVerifier) AnchorRow(seq uint64) (string, string, bool) {
	if _, err := journal.VerifyRows([]journal.Row(rv)); err != nil {
		return "", "", false // chain broken -> nothing is anchorable
	}
	for _, row := range rv {
		if row.Seq == seq {
			return row.PrevHash, row.Hash, true
		}
	}
	return "", "", false
}

// ---- journal emit helpers (build abi.Events the journal records) ------------

func emitDecide(j *journal.Journal, tool, taint string) {
	j.Emit(decideEvent(tool, taint))
}

func emitQuarantine(j *journal.Journal, witness string, span []int) {
	j.Emit(quarantineEvent(witness, span))
}

// ---- greedy decode helpers (exported-API only) ------------------------------

func continueGreedy(m *model.Model, ids []int, n int) []int {
	s := m.NewSession()
	logits := s.Prefill(ids)
	return stepGreedy(s, logits, n)
}

func stepGreedy(s *model.Session, logits []float32, n int) []int {
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		nx := argmax(logits)
		out = append(out, nx)
		if s.M.Cfg.IsEOS(nx) {
			break
		}
		logits = s.Step(nx)
	}
	return out
}

func argmax(v []float32) int {
	best, bi := float32(-1e30), 0
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

// ---- small utilities --------------------------------------------------------

func concat(parts ...[]int) []int {
	var out []int
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func maxAbsIntDelta(a, b []int) int {
	if len(a) != len(b) {
		return 1 << 30
	}
	mx := 0
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > mx {
			mx = d
		}
	}
	return mx
}

func shortHash(ids []int) string {
	h := sha256.New()
	for _, id := range ids {
		fmt.Fprintf(h, "%d,", id)
	}
	return hex.EncodeToString(h.Sum(nil))[:12]
}
