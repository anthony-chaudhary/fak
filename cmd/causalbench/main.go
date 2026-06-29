// Command causalbench is the end-to-end demonstrator for fak's CAUSAL
// invalidation-on-external-write: a tool RESULT cached under an external
// world-state witness (a git commit / blob hash / etag) is evicted byte-exact —
// and only it — the moment a later external write REFUTES that witness, while
// every sibling cached under an unrefuted witness stays warm.
//
// This is the causal sibling of cmd/deletioncert. deletioncert proves
// "a span I CHOSE to evict leaves the surviving context byte-identical to a run
// that never saw it" (single-stream, operator-driven). causalbench proves the
// harder, autonomous property the right-sizing plan's matrix row 6 names:
//
//	the system ITSELF discovers WHICH cached reads depended on a now-stale
//	external world-state and evicts exactly those, byte-exact, refusing
//	re-admission — the MESI-invalidate analogue in the integrity direction.
//
// It drives the REAL process-global vDSO (vdso.Default — the same instance every
// kernel syscall routes through), the same Lookup/Emit/Revoke code path live
// `fak serve` traffic uses (internal/gateway/coherence.go). No model, no weights,
// no files: the causal-coherence property is structural over cache identity and
// the witness ledger, not numeric.
//
// The chain it witnesses, with assertions (exits non-zero on any failure):
//
//  1. ADMIT   two read-only tool results under two DIFFERENT external witnesses
//     (w1=commit-A, w2=commit-B). Both then serve as byte-exact tier-2 hits.
//  2. WRITE   an external write refutes w1 (the file the w1 read observed changed).
//     Revoke(w1) evicts EXACTLY the w1 entry, leaves w2 warm, refuses
//     re-admission under w1, bumps the trust epoch, and broadcasts a
//     Revocation on the coherence bus.
//  3. EXACT   the w1 entry's served payload (pre-write) is byte-identical to a
//     fresh engine call — eviction lost the stale BINDING, never the
//     bytes; and the w2 sibling stays bit-identical across the write.
//  4. TARGET  refuting an UNRELATED witness evicts nothing (causal targeting, not
//     a blunt world flush); a re-fill under the refuted w1 is refused (the
//     durable content-addressed store cannot silently repopulate it).
//
// Usage:
//
//	go run ./cmd/causalbench -selfcheck
//	    Full demo + assertions; exits non-zero if any invariant fails. Zero files.
//
//	go run ./cmd/causalbench -selfcheck -out witness.json
//	    Also writes the machine-readable witness record to witness.json (the
//	    BENCHMARK-AUTHORITY artifact).
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the full demo with assertions (default when no other mode)")
	out := flag.String("out", "", "optional path to write the machine-readable witness record JSON")
	flag.Parse()
	_ = selfcheck // single mode today; the flag documents intent and reserves room

	rec, err := run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
		os.Exit(1)
	}
	if *out != "" {
		b, _ := benchcli.MarshalReport(rec)
		if werr := os.WriteFile(*out, b, 0o644); werr != nil {
			fmt.Fprintf(os.Stderr, "FAIL: write %s: %v\n", *out, werr)
			os.Exit(1)
		}
		fmt.Printf("  wrote %s\n", *out)
	}
	fmt.Println("\nOK — external write causally evicted exactly the dependent read, byte-exact, " +
		"siblings warm, re-admission refused.")
}

// witnessRecord is the machine-readable artifact this demo emits — the row
// BENCHMARK-AUTHORITY.md binds. Every field is an asserted invariant, not a
// narration: a consumer can re-check the claim from the record alone.
type witnessRecord struct {
	Version string `json:"version"`
	Demo    string `json:"demo"`
	// The causal-coherence invariants, all proven true by run() before this is written.
	W1HitBeforeWrite      bool `json:"w1_hit_before_write"`       // w1 read served byte-exact from cache
	W2HitBeforeWrite      bool `json:"w2_hit_before_write"`       // w2 read served byte-exact from cache
	W1ServedByteExact     bool `json:"w1_served_byte_exact"`      // cached w1 bytes == fresh engine bytes
	W1EvictedByWrite      int  `json:"w1_evicted_by_write"`       // entries Revoke(w1) stranded (==1: targeted)
	W2WarmAfterWrite      bool `json:"w2_warm_after_write"`       // w2 still a byte-exact hit after the write
	W2ByteIdenticalAcross bool `json:"w2_byte_identical_across"`  // w2 served bytes unchanged across the write
	W1MissAfterWrite      bool `json:"w1_miss_after_write"`       // w1 read now misses (-> engine, fresh)
	W1ReadmissionRefused  bool `json:"w1_readmission_refused"`    // re-fill under refuted w1 does not repopulate
	UnrelatedEvicts       int  `json:"unrelated_witness_evicts"`  // refuting an unrelated witness evicts 0
	CoherenceBroadcast    bool `json:"coherence_broadcast_fired"` // a Revocation reached a bus subscriber
	TrustEpochAdvanced    bool `json:"trust_epoch_advanced"`      // integrity clock bumped on refutation
	MaxAbsDelta           int  `json:"max_abs_delta"`             // 0 — the byte-exact headline
}

// run executes the full causal-invalidation chain against the real vdso.Default
// and returns the asserted witness record. Any violated invariant returns an
// error (the demo exits non-zero).
func run() (witnessRecord, error) {
	ctx := context.Background()
	v := vdso.Default

	// A clean root epoch for this run so a prior process-global write cannot alias
	// our keys. BumpWorld bumps only the consistency epoch (worldVer) and does NOT
	// touch the bus; trustEpoch and the revoked-witness ledger persist on the
	// process-global vdso.Default, which is exactly why the assertions below compare
	// trustEpoch RELATIVELY (before+1), never against a hardcoded 0.
	v.BumpWorld()

	// Subscribe to the integrity bus so we can prove the cross-agent broadcast
	// fired — the half that makes invalidation CAUSAL across processes, not just
	// a local evict. Mirrors gateway/coherence.go's newCoherenceFeed subscription.
	var (
		busMu      sync.Mutex
		broadcasts []vdso.Revocation
	)
	cancel := v.SubscribeRevocations(func(rv vdso.Revocation) {
		busMu.Lock()
		broadcasts = append(broadcasts, rv)
		busMu.Unlock()
	})
	defer cancel()

	// The external world-state witnesses. In a live agent loop these are the git
	// commit / blob hash / etag the orchestration substrate already holds; here
	// they are two distinct commits, each observed by one read.
	const (
		w1 = "commit:aaaaaaaaaaaa" // the world-state read #1 observed
		w2 = "commit:bbbbbbbbbbbb" // the world-state read #2 observed (independent)
	)

	fmt.Println("== fak causal-invalidation demo ==")
	fmt.Printf("witnesses: read#1 admitted under %s, read#2 under %s\n", w1, w2)

	// Two read-only, idempotent tool calls. read_config observed world-state w1;
	// read_policy observed w2. Each returns a deterministic payload (the "engine"
	// answer) — distinct, so a cross-wire would be caught.
	call1 := readCall("read_config", `{"path":"/etc/app.conf"}`, w1)
	call2 := readCall("read_policy", `{"path":"/etc/policy.json"}`, w2)
	fresh1 := enginePayload("read_config", w1)
	fresh2 := enginePayload("read_policy", w2)

	// ---- 1. ADMIT: a miss, then the engine completion FILLS the cache under the witness.
	if _, hit := v.Lookup(ctx, call1); hit {
		return witnessRecord{}, fmt.Errorf("read#1 unexpectedly hit before any fill")
	}
	v.Emit(complete(call1, fresh1)) // engine produced read#1's result under w1
	v.Emit(complete(call2, fresh2)) // engine produced read#2's result under w2

	// Both now serve as byte-exact tier-2 hits.
	got1, hit1, err := lookupBytes(ctx, v, call1)
	if err != nil {
		return witnessRecord{}, err
	}
	got2, hit2, err := lookupBytes(ctx, v, call2)
	if err != nil {
		return witnessRecord{}, err
	}
	if !hit1 || !hit2 {
		return witnessRecord{}, fmt.Errorf("post-fill: read#1 hit=%v read#2 hit=%v (want both true)", hit1, hit2)
	}
	w1Exact := bytes.Equal(got1, fresh1)
	if !w1Exact {
		return witnessRecord{}, fmt.Errorf("cached read#1 bytes != fresh engine bytes — hit is not equal-to-a-fresh-call")
	}
	if !bytes.Equal(got2, fresh2) {
		return witnessRecord{}, fmt.Errorf("cached read#2 bytes != fresh engine bytes")
	}
	fmt.Printf("\n  ADMIT: read#1 + read#2 both serve byte-exact from cache (hit==fresh call).\n")

	// ---- 2. WRITE: an external write changes the file read#1 observed → its witness
	// w1 is refuted. Revoke(w1) is the causal eviction: it strands exactly the w1
	// consumer-set, bumps the trust epoch, and broadcasts on the coherence bus.
	trustBefore := v.TrustEpoch()
	evicted := v.Revoke(w1)
	trustAfter := v.TrustEpoch()
	fmt.Printf("\n  WRITE: external write refutes %s → Revoke evicted %d entr%s; "+
		"trustEpoch %d→%d.\n", w1, evicted, plural(evicted), trustBefore, trustAfter)
	if evicted != 1 {
		return witnessRecord{}, fmt.Errorf("Revoke(w1) evicted %d entries, want exactly 1 (the read#1 consumer) — "+
			"eviction is not causally TARGETED", evicted)
	}
	if trustAfter != trustBefore+1 {
		return witnessRecord{}, fmt.Errorf("trustEpoch did not advance on refutation (%d→%d)", trustBefore, trustAfter)
	}

	// ---- 3. EXACT + TARGET: read#1 now misses (→ engine, fresh); read#2 stays a
	// byte-IDENTICAL hit (the write did not strand the sibling — causal, not blunt).
	if _, hit := v.Lookup(ctx, call1); hit {
		return witnessRecord{}, fmt.Errorf("read#1 STILL served from cache after its witness was refuted — STALE SERVE")
	}
	got2b, hit2b, err := lookupBytes(ctx, v, call2)
	if err != nil {
		return witnessRecord{}, err
	}
	if !hit2b {
		return witnessRecord{}, fmt.Errorf("read#2 was evicted by an UNRELATED write — eviction over-fired (blunt flush)")
	}
	w2Identical := bytes.Equal(got2b, got2)
	if !w2Identical {
		return witnessRecord{}, fmt.Errorf("read#2 served DIFFERENT bytes across the write (%x vs %x)", got2b, got2)
	}
	fmt.Printf("  EXACT: read#1 now MISSES (→ fresh engine); read#2 stays a byte-identical hit (max|Δ|=0).\n")

	// ---- 4. RE-ADMISSION REFUSED: the durable content-addressed store would gladly
	// re-serve read#1's identical bytes — but a re-fill under the refuted witness must
	// NOT repopulate it, or the eviction would be cosmetic.
	v.Emit(complete(call1, fresh1)) // engine "re-reads" — but still under the refuted w1
	if _, hit := v.Lookup(ctx, call1); hit {
		return witnessRecord{}, fmt.Errorf("read#1 RE-ADMITTED under a refuted witness — the CAS silently repopulated it")
	}
	readmissionRefused := true
	fmt.Printf("  REFUSE: a re-fill under the refuted witness does not repopulate read#1 (CAS cannot resurrect it).\n")

	// ---- 5. UNRELATED-WITNESS TARGETING: refuting a witness NOTHING was admitted under
	// strands nothing locally (it still bumps the epoch + broadcasts for remote holders).
	unrelated := v.Revoke("commit:cccccccccccc")
	if unrelated != 0 {
		return witnessRecord{}, fmt.Errorf("refuting an unrelated witness evicted %d local entries, want 0", unrelated)
	}
	fmt.Printf("  TARGET: refuting an unrelated witness evicts 0 local entries (targeted, not a flush).\n")

	// ---- 6. COHERENCE BUS: the refutation reached a subscriber (the cross-agent /
	// cross-process propagation — what makes the invalidation causal beyond this pool).
	busMu.Lock()
	gotW1Broadcast := false
	for _, rv := range broadcasts {
		if rv.Witness == w1 && rv.Evicted == 1 {
			gotW1Broadcast = true
		}
	}
	nBroadcast := len(broadcasts)
	busMu.Unlock()
	if !gotW1Broadcast {
		return witnessRecord{}, fmt.Errorf("no Revocation{witness=%s, evicted=1} reached the coherence bus (%d total) — "+
			"cross-agent propagation did not fire", w1, nBroadcast)
	}
	fmt.Printf("  BUS:    a Revocation(%s, evicted=1) was broadcast to subscribers (%d bus events).\n", w1, nBroadcast)

	return witnessRecord{
		Version:               "fak-causalbench-v1",
		Demo:                  "cmd/causalbench -selfcheck",
		W1HitBeforeWrite:      hit1,
		W2HitBeforeWrite:      hit2,
		W1ServedByteExact:     w1Exact,
		W1EvictedByWrite:      evicted,
		W2WarmAfterWrite:      hit2b,
		W2ByteIdenticalAcross: w2Identical,
		W1MissAfterWrite:      true,
		W1ReadmissionRefused:  readmissionRefused,
		UnrelatedEvicts:       unrelated,
		CoherenceBroadcast:    gotW1Broadcast,
		TrustEpochAdvanced:    trustAfter == trustBefore+1,
		MaxAbsDelta:           0,
	}, nil
}

// ---- vDSO driving helpers (exported-API only) -------------------------------

// readCall builds a read-only, idempotent tool call observing world-state `witness`.
// The readOnlyHint+idempotentHint meta is what routes it to the vDSO fast path; the
// "witness" meta is the external world-state the read observed (revoke.go's witness).
func readCall(tool, args, witness string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args), Len: int64(len(args)), Taint: abi.TaintTrusted},
		Meta: map[string]string{
			"readOnlyHint":   "true",
			"idempotentHint": "true",
			"witness":        witness,
		},
	}
}

// enginePayload is the deterministic "engine" answer for a (tool, witness) pair —
// distinct per tool so a cross-wire between read#1 and read#2 would be caught.
func enginePayload(tool, witness string) []byte {
	return []byte(fmt.Sprintf(`{"tool":%q,"observed":%q,"value":%q}`, tool, witness, digest(tool+"@"+witness)))
}

// complete builds the EvComplete event the vDSO observes to fill its cache — the
// same event an engine result raises in the live loop. The result carries the
// read's witness forward (defaultWitnessOf falls back to the call's meta anyway).
func complete(c *abi.ToolCall, payload []byte) abi.Event {
	r := &abi.Result{
		Call:    c,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: payload, Len: int64(len(payload)), Taint: abi.TaintTrusted},
		Status:  abi.StatusOK,
	}
	return abi.Event{Kind: abi.EvComplete, Call: c, Result: r}
}

// lookupBytes performs a Lookup and materializes the served payload bytes.
func lookupBytes(ctx context.Context, v *vdso.VDSO, c *abi.ToolCall) ([]byte, bool, error) {
	res, hit := v.Lookup(ctx, c)
	if !hit {
		return nil, false, nil
	}
	if res.Payload.Kind != abi.RefInline {
		return nil, true, fmt.Errorf("served payload is not inline (kind=%d) — cannot byte-compare", res.Payload.Kind)
	}
	return res.Payload.Inline, true, nil
}

// ---- small utilities --------------------------------------------------------

func digest(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])[:12]
}

func plural(n int) string {
	if n == 1 {
		return "y"
	}
	return "ies"
}
