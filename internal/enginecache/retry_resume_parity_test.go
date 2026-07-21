package enginecache

// retry_resume_parity_test.go is the executable, captured proof for issue #4538
// (test(quality): prove retry and mid-generation resume preserve semantics),
// one contract in the engine-mode parity cohort under the quality-ladder epic
// #4509. It is a sibling of proofs_witness_test.go and follows the same
// discipline (fak/docs/proofs/00-METHOD.md): model faithfully the ONE contract
// the property depends on, assert a real metamorphic/round-trip property, and
// carry a planted-defect CONTRAST that proves the assertion is non-vacuous.
//
// Contract under test — retry/resume semantic parity:
//   When a decode is interrupted at ANY lifecycle boundary (right after prefill,
//   at any mid-generation step, or just before the terminal EOS) and then either
//   retried from scratch or resumed from its committed checkpoint under a PINNED
//   identity and seed, the resulting token stream is byte-identical to the
//   uninterrupted reference: no token is duplicated, none is omitted, and the
//   prompt/completion accounting matches.
//
// Why this belongs next to the cache proofs: greedy decode from a KV/prefix
// cache is a deterministic function of the COMMITTED token prefix. A retry or a
// mid-generation resume replays that function from the committed checkpoint, so
// the cache-consistent engine MUST reproduce the same continuation. The two
// failure modes a real resume exhibits are (a) re-emitting the boundary token
// from the committed buffer (a DUPLICATE) and (b) advancing the position counter
// past the boundary token without emitting it (an OMISSION). Both are planted
// here as representative defects and both must trip the parity gate.
//
// Evidence provenance (acceptance criteria of #4538): every case records model,
// tokenizer, engine/backend, seed, code/module revision, and the baseline +
// tolerance it was judged against; a divergence localizes the FIRST actionable
// position and emits a scrubbed replay artifact (token ids and positions only,
// never decoded text); missing/inconclusive evidence is never scored as pass.
//
// Tier: PR (deterministic oracle, no network, no model weights). Runtime cost is
// a few milliseconds — see caseRecord.RuntimeCostNote.

import (
	"encoding/binary"
	"encoding/json"
	"hash/fnv"
	"io"
	"runtime/debug"
	"strings"
	"testing"
)

// decodeIdentity is the pinned provenance a quality case is replayed under. Two
// runs sharing a decodeIdentity MUST produce the same token stream; this is the
// "seed or deterministic oracle" the acceptance criteria require.
type decodeIdentity struct {
	ModelID     string `json:"model"`
	TokenizerID string `json:"tokenizer"`
	Engine      Engine `json:"engine"`
	Seed        uint64 `json:"seed"`
	Revision    string `json:"module_revision"`
	Baseline    string `json:"baseline"`
	Tolerance   string `json:"tolerance"`
}

const (
	// eosTokenID is the terminal stop token appended after the generated tokens;
	// it makes the "interrupt just before EOS" boundary explicit.
	eosTokenID = 50256
	vocabFloor = 10
	vocabCeil  = 50000
	maxNew     = 8 // generated tokens before EOS; small but covers every boundary
)

// genToken is the deterministic greedy decode oracle: the next token is a pure
// function of (seed, model, tokenizer, committed prefix). This is the single
// contract the parity property rests on — a cache-consistent engine resuming
// from the same committed prefix derives the same token.
func genToken(id decodeIdentity, prefix []int) int {
	h := fnv.New64a()
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], id.Seed)
	_, _ = h.Write(b[:])
	_, _ = io.WriteString(h, id.ModelID)
	_, _ = h.Write([]byte{0})
	_, _ = io.WriteString(h, id.TokenizerID)
	_, _ = h.Write([]byte{0})
	for _, t := range prefix {
		binary.LittleEndian.PutUint64(b[:], uint64(t))
		_, _ = h.Write(b[:])
	}
	return vocabFloor + int(h.Sum64()%(vocabCeil-vocabFloor))
}

// decodeReference runs the uninterrupted decode: the baseline every interrupted
// run is compared against. The returned stream is [g0..g(maxNew-1), EOS].
func decodeReference(id decodeIdentity, prompt []int) []int {
	committed := append([]int(nil), prompt...)
	gen := make([]int, 0, maxNew+1)
	for i := 0; i < maxNew; i++ {
		t := genToken(id, committed)
		gen = append(gen, t)
		committed = append(committed, t)
	}
	return append(gen, eosTokenID)
}

// resumeBug selects which retry/resume implementation to exercise: the correct
// one, or one of the two planted representative defects.
type resumeBug int

const (
	resumeCorrect           resumeBug = iota // faithful resume from the checkpoint
	resumeDuplicateBoundary                  // re-emits the boundary token (dup)
	resumeSkipBoundary                       // drops the boundary token (omission)
)

// resumeFromBoundary models an interrupt after k committed generated tokens
// (0 = right after prefill, maxNew = just before EOS) followed by a resume that
// re-runs the deterministic oracle from the committed checkpoint. `ref` is the
// uninterrupted reference generated tokens WITHOUT the trailing EOS. The correct
// path reproduces `ref`+EOS exactly; each bug injects one representative defect.
func resumeFromBoundary(id decodeIdentity, prompt, ref []int, k int, bug resumeBug) []int {
	committed := append([]int(nil), prompt...)
	committed = append(committed, ref[:k]...) // the k correctly-committed tokens
	gen := append([]int(nil), ref[:k]...)

	switch bug {
	case resumeDuplicateBoundary:
		// BUG: the resume re-emits the last committed token from its buffer
		// before continuing, duplicating the boundary token. Requires k>=1.
		dup := ref[k-1]
		gen = append(gen, dup)
		committed = append(committed, dup)
	case resumeSkipBoundary:
		// BUG: the resume double-advances the position counter — it commits the
		// boundary token into the KV buffer (so the continuation is derived as
		// if it were present) but never emits it into the stream, omitting it.
		// Requires k < maxNew (there is a boundary token to skip).
		committed = append(committed, ref[k])
	}

	// Continue deterministic decode to the pinned budget of maxNew generated
	// tokens, then append the terminal EOS.
	for len(gen) < maxNew {
		t := genToken(id, committed)
		gen = append(gen, t)
		committed = append(committed, t)
	}
	return append(gen, eosTokenID)
}

// firstDivergence returns the index of the first position at which got differs
// from ref, or -1 when they are identical. A length mismatch diverges at the
// first surplus/missing index.
func firstDivergence(ref, got []int) int {
	n := len(ref)
	if len(got) < n {
		n = len(got)
	}
	for i := 0; i < n; i++ {
		if ref[i] != got[i] {
			return i
		}
	}
	if len(ref) != len(got) {
		return n
	}
	return -1
}

// replayArtifact is the scrubbed, schema-valid failure bundle emitted on a
// divergence. It carries token ids and positions and full identity provenance —
// never decoded text or payload bytes — so it can be replayed and triaged
// without leaking content. "Missing or inconclusive evidence is never pass":
// a run that cannot produce a conclusive comparison emits Inconclusive=true.
type replayArtifact struct {
	Schema            string         `json:"schema"`
	Identity          decodeIdentity `json:"identity"`
	Tier              string         `json:"tier"`
	Prompt            []int          `json:"prompt_tokens"`
	InterruptBoundary int            `json:"interrupt_boundary"`
	FirstDivergentPos int            `json:"first_divergent_pos"`
	ExpectedTokenID   int            `json:"expected_token_id"`
	GotTokenID        int            `json:"got_token_id"`
	RefLen            int            `json:"ref_len"`
	GotLen            int            `json:"got_len"`
	Classification    string         `json:"classification"`
	Inconclusive      bool           `json:"inconclusive"`
}

// caseRecord is the per-case evidence row: the pinned identity plus the tier and
// runtime-cost documentation the acceptance criteria require.
type caseRecord struct {
	Identity        decodeIdentity `json:"identity"`
	Tier            string         `json:"tier"`
	RuntimeCostNote string         `json:"runtime_cost"`
}

// buildArtifact localizes the first actionable divergence between ref and got
// and classifies it as a duplicate, an omission, or a length/accounting
// mismatch. It never inspects text — only the token-id sequences.
func buildArtifact(id decodeIdentity, prompt, ref, got []int, boundary int) replayArtifact {
	pos := firstDivergence(ref, got)
	art := replayArtifact{
		Schema:            "fak.quality.retry-resume-parity/v1",
		Identity:          id,
		Tier:              "PR",
		Prompt:            append([]int(nil), prompt...),
		InterruptBoundary: boundary,
		FirstDivergentPos: pos,
		RefLen:            len(ref),
		GotLen:            len(got),
	}
	if pos < 0 {
		art.Inconclusive = true // no divergence: not a failure artifact
		art.Classification = "none"
		art.FirstDivergentPos = -1
		return art
	}
	if pos < len(ref) {
		art.ExpectedTokenID = ref[pos]
	} else {
		art.ExpectedTokenID = -1
	}
	if pos < len(got) {
		art.GotTokenID = got[pos]
	} else {
		art.GotTokenID = -1
	}
	switch {
	case pos > 0 && pos < len(got) && got[pos] == got[pos-1]:
		art.Classification = "duplicate"
	case pos+1 < len(ref) && pos < len(got) && got[pos] == ref[pos+1]:
		art.Classification = "omission"
	case pos == len(ref)-2 && pos < len(got) &&
		got[pos] == genToken(id, append(append([]int(nil), prompt...), ref[:pos+1]...)):
		// Terminal boundary: the omitted token was the LAST generated token, so
		// the faithful continuation shifts a token BEYOND the reference budget
		// into pos while the reference truncated to EOS there. ref[pos+1] is that
		// EOS (not a real generated token), so the shift-by-one match above cannot
		// fire; re-derive the faithful successor from the committed prefix (which
		// includes the boundary token) to recognize the omission.
		art.Classification = "omission"
	case len(ref) != len(got):
		art.Classification = "accounting_mismatch"
	default:
		art.Classification = "divergence"
	}
	return art
}

// moduleRevision records the code revision the case ran at. It prefers the
// embedded VCS revision, falling back to the module version and finally to an
// explicit "uncommitted" marker so the provenance field is never silently empty.
func moduleRevision() string {
	if bi, ok := debug.ReadBuildInfo(); ok {
		for _, s := range bi.Settings {
			if s.Key == "vcs.revision" && s.Value != "" {
				return s.Value
			}
		}
		if bi.Main.Version != "" {
			return bi.Main.Version
		}
	}
	return "uncommitted"
}

// parityIdentities are the pinned cases: the two engines fak has a cache-control
// identity for, each under a fixed seed and deterministic-oracle tolerance.
func parityIdentities() []decodeIdentity {
	rev := moduleRevision()
	return []decodeIdentity{
		{
			ModelID: "glm-5.2", TokenizerID: "glm-tokenizer", Engine: EngineSGLang,
			Seed: 0x5eed_1538, Revision: rev,
			Baseline: "uninterrupted-reference-decode", Tolerance: "exact-match (deterministic oracle, tolerance=0)",
		},
		{
			ModelID: "glm-5.2", TokenizerID: "glm-tokenizer", Engine: EngineVLLM,
			Seed: 0x5eed_1538, Revision: rev,
			Baseline: "uninterrupted-reference-decode", Tolerance: "exact-match (deterministic oracle, tolerance=0)",
		},
	}
}

var parityPrompt = []int{101, 102, 103, 104}

// requireRecordedProvenance fails the case if any required evidence field is
// empty. Under the acceptance criteria, a case that does not record its
// provenance cannot be scored as a pass regardless of the token comparison.
func requireRecordedProvenance(t *testing.T, id decodeIdentity) {
	t.Helper()
	for name, v := range map[string]string{
		"model":     id.ModelID,
		"tokenizer": id.TokenizerID,
		"engine":    string(id.Engine),
		"revision":  id.Revision,
		"baseline":  id.Baseline,
		"tolerance": id.Tolerance,
	} {
		if strings.TrimSpace(v) == "" {
			t.Fatalf("provenance field %q is empty; a case without recorded provenance is never a pass", name)
		}
	}
	if id.Seed == 0 {
		t.Fatalf("seed is unset; a stochastic case without a pinned seed or deterministic oracle is never a pass")
	}
}

// TestRetryReproducesIdenticalStream proves the whole-request RETRY property: a
// second decode under the pinned identity/seed reproduces the reference stream
// exactly, with matching prompt/completion accounting.
func TestRetryReproducesIdenticalStream(t *testing.T) {
	for _, id := range parityIdentities() {
		requireRecordedProvenance(t, id)
		ref := decodeReference(id, parityPrompt)
		retry := decodeReference(id, parityPrompt)
		if pos := firstDivergence(ref, retry); pos != -1 {
			art := buildArtifact(id, parityPrompt, ref, retry, -1)
			t.Fatalf("retry diverged from reference for %s: %+v", id.Engine, art)
		}
		// Accounting: prompt tokens + completion tokens must be exact and stable.
		if got, want := len(retry), maxNew+1; got != want {
			t.Fatalf("completion accounting = %d tokens, want %d (maxNew+EOS)", got, want)
		}
		if retry[len(retry)-1] != eosTokenID {
			t.Fatalf("retry did not terminate on EOS: %d", retry[len(retry)-1])
		}
	}
}

// TestResumeAtEveryLifecycleBoundaryPreservesSemantics is the core proof: for
// EVERY interrupt boundary (0 = post-prefill .. maxNew = pre-EOS), a correct
// resume reproduces the reference stream with no duplicated or omitted token and
// matching accounting.
func TestResumeAtEveryLifecycleBoundaryPreservesSemantics(t *testing.T) {
	for _, id := range parityIdentities() {
		requireRecordedProvenance(t, id)
		full := decodeReference(id, parityPrompt)
		ref := full[:maxNew] // generated tokens without EOS
		for k := 0; k <= maxNew; k++ {
			resumed := resumeFromBoundary(id, parityPrompt, ref, k, resumeCorrect)
			if pos := firstDivergence(full, resumed); pos != -1 {
				art := buildArtifact(id, parityPrompt, full, resumed, k)
				t.Fatalf("resume at boundary %d diverged for %s: %+v", k, id.Engine, art)
			}
			if len(resumed) != maxNew+1 {
				t.Fatalf("resume at boundary %d changed accounting: %d tokens, want %d", k, len(resumed), maxNew+1)
			}
			if hasAdjacentDuplicate(resumed[:maxNew]) {
				t.Fatalf("resume at boundary %d introduced a duplicate token", k)
			}
		}
	}
}

// hasAdjacentDuplicate reports whether two identical tokens sit adjacent — the
// signature of a boundary-token re-emission.
func hasAdjacentDuplicate(toks []int) bool {
	for i := 1; i < len(toks); i++ {
		if toks[i] == toks[i-1] {
			return true
		}
	}
	return false
}

// TestPlantedDuplicateResumeIsCaught is the non-vacuity CONTRAST for the
// duplicate failure mode: a resume that re-emits the boundary token MUST trip
// the parity gate at the boundary and emit a scrubbed replay artifact classified
// as "duplicate". If this ever passes silently, the correct-path proof above is
// vacuous.
func TestPlantedDuplicateResumeIsCaught(t *testing.T) {
	id := parityIdentities()[0]
	full := decodeReference(id, parityPrompt)
	ref := full[:maxNew]
	caught := 0
	for k := 1; k <= maxNew; k++ { // k>=1: a boundary token exists to duplicate
		got := resumeFromBoundary(id, parityPrompt, ref, k, resumeDuplicateBoundary)
		pos := firstDivergence(full, got)
		if pos == -1 {
			t.Fatalf("planted duplicate at boundary %d was NOT caught (gate is vacuous)", k)
		}
		if pos != k {
			t.Fatalf("duplicate at boundary %d localized to pos %d, want first divergence at %d", k, pos, k)
		}
		art := buildArtifact(id, parityPrompt, full, got, k)
		if art.Classification != "duplicate" {
			t.Fatalf("boundary %d misclassified as %q, want duplicate: %+v", k, art.Classification, art)
		}
		if art.Inconclusive {
			t.Fatalf("a caught defect must not be inconclusive: %+v", art)
		}
		caught++
	}
	if caught == 0 {
		t.Fatal("no duplicate boundary was exercised")
	}
}

// TestPlantedOmittedResumeIsCaught is the non-vacuity CONTRAST for the omission
// failure mode: a resume that drops the boundary token MUST trip the gate at the
// boundary and classify as "omission".
func TestPlantedOmittedResumeIsCaught(t *testing.T) {
	id := parityIdentities()[0]
	full := decodeReference(id, parityPrompt)
	ref := full[:maxNew]
	caught := 0
	for k := 0; k < maxNew; k++ { // k<maxNew: a boundary token exists to skip
		got := resumeFromBoundary(id, parityPrompt, ref, k, resumeSkipBoundary)
		pos := firstDivergence(full, got)
		if pos == -1 {
			t.Fatalf("planted omission at boundary %d was NOT caught (gate is vacuous)", k)
		}
		if pos != k {
			t.Fatalf("omission at boundary %d localized to pos %d, want first divergence at %d", k, pos, k)
		}
		art := buildArtifact(id, parityPrompt, full, got, k)
		if art.Classification != "omission" {
			t.Fatalf("boundary %d misclassified as %q, want omission: %+v", k, art.Classification, art)
		}
		caught++
	}
	if caught == 0 {
		t.Fatal("no omission boundary was exercised")
	}
}

// TestReplayArtifactIsScrubbedAndSchemaValid proves the emitted failure bundle
// is machine-readable, schema-stamped, carries full identity provenance, and is
// scrubbed of decoded text (only token ids and positions). It also asserts the
// per-case record documents an explicit tier and runtime cost.
func TestReplayArtifactIsScrubbedAndSchemaValid(t *testing.T) {
	id := parityIdentities()[0]
	full := decodeReference(id, parityPrompt)
	ref := full[:maxNew]
	got := resumeFromBoundary(id, parityPrompt, ref, 3, resumeDuplicateBoundary)
	art := buildArtifact(id, parityPrompt, full, got, 3)

	blob, err := json.Marshal(art)
	if err != nil {
		t.Fatalf("replay artifact must be JSON-serializable: %v", err)
	}
	s := string(blob)
	for _, want := range []string{
		`"schema":"fak.quality.retry-resume-parity/v1"`,
		`"model":"glm-5.2"`,
		`"tokenizer":"glm-tokenizer"`,
		`"engine":"sglang"`,
		`"module_revision":`,
		`"first_divergent_pos":3`,
		`"classification":"duplicate"`,
	} {
		if !strings.Contains(s, want) {
			t.Fatalf("artifact JSON missing %q:\n%s", want, s)
		}
	}
	// Scrubbed: the reference/candidate prompt text must never appear; the
	// artifact carries token ids, not decoded strings.
	if strings.Contains(strings.ToLower(s), "glm-5.2 says") || strings.Contains(s, "\"text\"") {
		t.Fatalf("artifact leaked decoded text; it must be scrubbed to token ids: %s", s)
	}

	rec := caseRecord{
		Identity:        id,
		Tier:            "PR",
		RuntimeCostNote: "deterministic oracle; no network or weights; ~milliseconds per case",
	}
	recBlob, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("case record must be JSON-serializable: %v", err)
	}
	if !strings.Contains(string(recBlob), `"tier":"PR"`) {
		t.Fatalf("case record must document an explicit tier: %s", recBlob)
	}
}

// TestParityGateNeverPassesOnInconclusive pins the "missing or inconclusive
// evidence is never pass" rule: buildArtifact marks a no-divergence comparison
// inconclusive rather than emitting it as a failure, and a genuine divergence is
// never inconclusive.
func TestParityGateNeverPassesOnInconclusive(t *testing.T) {
	id := parityIdentities()[0]
	full := decodeReference(id, parityPrompt)
	// Identical streams: not a failure artifact -> inconclusive as a failure row.
	if art := buildArtifact(id, parityPrompt, full, full, 0); !art.Inconclusive || art.FirstDivergentPos != -1 {
		t.Fatalf("identical streams must be inconclusive-as-failure, got %+v", art)
	}
	// A real divergence is conclusive.
	ref := full[:maxNew]
	got := resumeFromBoundary(id, parityPrompt, ref, 2, resumeSkipBoundary)
	if art := buildArtifact(id, parityPrompt, full, got, 2); art.Inconclusive {
		t.Fatalf("a real divergence must be conclusive, got %+v", art)
	}
}
