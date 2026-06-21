export const meta = {
  name: 'fak-math-proofs',
  description: 'Prove-or-refute each fak sub-module math obligation with a deterministic witness, then adversarially verify each verdict',
  phases: [
    { title: 'Prove', detail: 'one agent per obligation: read code+tests, run the witness, draft the proof doc, emit a verdict' },
    { title: 'Verify', detail: 'independent agent re-runs each witness, checks file:line + non-vacuity, can only downgrade' },
  ],
}

// ---------------------------------------------------------------------------
// The obligation roster. Regime: N numerical | A algebraic | C crypto | D decision.
// `modules` are internal/ dirs. `theorems` are the properties to prove (hints —
// ground them in the ACTUAL code+tests). Model/compute are decomposed so no single
// agent owns the 17k-line core.
// ---------------------------------------------------------------------------
const ROSTER = [
  { id: 'N1', regime: 'N', modules: ['model'], title: 'model/attention', theorems: [
    'softmax over attention scores is row-stochastic (entries >=0, sum=1) and shift-invariant (softmax(x+c)=softmax(x))',
    'the causal mask makes the attention weight matrix strictly lower-triangular (position i attends only to j<=i)',
    'attention-sink renormalization: a learned sink logit is included in the denominator but dropped from the output (TestAttentionSinkSoftmaxDropsSink)'],
    testHint: "(go test -run 'Softmax|Attention|Causal|Mask|Sink' ./internal/model/ -count=1 -timeout 120s)" },
  { id: 'N2', regime: 'N', modules: ['model'], title: 'model/norm', theorems: [
    'RMSNorm computes x * gain / sqrt(mean(x^2)+eps) matching the definition',
    'LayerNorm is shift+scale equivariant and matches the reference within float tolerance',
    'normalization is numerically stable (no overflow/NaN on large-magnitude inputs)'],
    testHint: "(go test -run 'Norm|RMS|LayerNorm' ./internal/model/ -count=1 -timeout 120s)" },
  { id: 'N3', regime: 'N', modules: ['model'], title: 'model/rope', theorems: [
    'RoPE preserves the per-(head,pair) vector norm (it is a rotation)',
    'the rotated dot product depends only on the relative position (m-n)',
    'rope scaling variants (linear / NTK / YaRN) match their reference formula'],
    testHint: "(go test -run 'Rope|RoPE|Rotary|Scaling' ./internal/model/ -count=1 -timeout 120s)" },
  { id: 'N4', regime: 'N', modules: ['model'], title: 'model/mlp+residual', theorems: [
    'SwiGLU / GeGLU MLP computes the exact functional form down(act(gate(x)) * up(x))',
    'the residual stream is exact addition (no scaling/clipping unless the arch defines it)',
    'MoE routing selects top-k experts and the combine is a correct weighted sum'],
    testHint: "(go test -run 'MLP|SwiGLU|GeGLU|Residual|MoE|Expert' ./internal/model/ -count=1 -timeout 120s)" },
  { id: 'N5', regime: 'N', modules: ['model'], title: 'model/quant', theorems: [
    'Q4_K / Q8_0 dequant is affine-correct: value = scale*(q - zero) per the block layout',
    'AWQ dequant matches its reference',
    'the integer SDOT/int8 reduction path is BIT-IDENTICAL to the integer reference sum (float error only in the final de-affine combine)'],
    testHint: "(go test -run 'Quant|Q4K|Q4_K|Q8|AWQ|SDOT|Dequant' ./internal/model/ -count=1 -timeout 120s)" },
  { id: 'N6', regime: 'N', modules: ['model'], title: 'model/kv', theorems: [
    'KV append places key/value at the correct (layer,pos,head) slot; reload is bit-exact (TestHiddenStateRoundTripBitExact / kvlayout)',
    'span-exact eviction yields a context byte-identical to one that never saw the span (max|delta|=0, "proven == never-saw-it")',
    'sliding-window attention (SWA) masks exactly the out-of-window positions; prefix splice (SessionFromPrefix) == recompute'],
    testHint: "(go test -run 'KV|Evict|Kvlayout|SWA|Sliding|Prefix|RoundTripBitExact' ./internal/model/ -count=1 -timeout 180s)" },
  { id: 'N7', regime: 'N', modules: ['model','modelengine'], title: 'model/forward-parity', theorems: [
    'end-to-end forward pass matches the PyTorch/HF oracle: hidden/logits shapes, per-position argmax, and greedy ids (oracle_test.go)',
    'the parity holds across the supported arch families (qwen, glm, gpt_oss, ...)'],
    testHint: "(go test -run 'Oracle|Parity|Greedy|Argmax|Forward' ./internal/model/ -count=1 -timeout 240s) ; note: oracle rungs SKIP without .cache/oracle-* — if skipped, mark OPEN (witness present but fixture absent), do NOT mark PROVEN" },
  { id: 'N8', regime: 'N', modules: ['compute'], title: 'compute/gemm', theorems: [
    'the GEMM/HAL computes C=A*B equal to the naive triple-loop reference within float tolerance',
    'GEMM is bilinear: A(B+C)=AB+AC and (alpha A)B = alpha(AB)',
    'all backends (CPU / Metal / CUDA, by build tag) agree with the CPU reference (cosine~1 / argmax-stable)'],
    testHint: "(go test -run 'GEMM|Gemm|Matmul|Parity|Backend' ./internal/compute/ -count=1 -timeout 120s)" },
  { id: 'N9', regime: 'N', modules: ['metalgemm','compute'], title: 'metalgemm', theorems: [
    'Metal MPS GEMM matches the CPU reference (the metal first-light cosine=1.0 parity)',
    'the stub (non-metal build) and the metal build present the same interface (no math divergence by build tag)'],
    testHint: "(go test -run 'Metal|MPS|Gemm' ./internal/metalgemm/ ./internal/compute/ -count=1 -timeout 120s) ; metal rungs need -tags metal + Apple GPU — if not built, mark SCOPED-OUT or OPEN with reason" },
  { id: 'N10', regime: 'N', modules: ['tokenizer'], title: 'tokenizer', theorems: [
    'encode then decode round-trips to the original text (or ids), byte-identical where the vocab is lossless',
    'BPE merges apply in deterministic rank order (same input -> same token ids)',
    'tokenization matches the HF reference (oracle_qwen_test.go)'],
    testHint: "(go test -run 'RoundTrip|Oracle|BPE|Merge|Determinis' ./internal/tokenizer/ -count=1 -timeout 120s)" },
  { id: 'N11', regime: 'N', modules: ['ggufload'], title: 'ggufload', theorems: [
    'GGUF tensor offsets/strides are parsed to address the correct bytes (layout correctness)',
    'per-dtype dequant reads the right block layout (no off-by-block / wrong-stride)',
    'metadata round-trips (parse then re-read is consistent)'],
    testHint: "(go test ./internal/ggufload/ -count=1 -timeout 120s)" },

  { id: 'A1', regime: 'A', modules: ['radixkv'], title: 'radixkv', theorems: [
    'a new request discovers the LONGEST cached prefix by tree walk; reuse+suffix == full recompute',
    'reference counts are conserved (insert/clone/evict leave no dangling or leaked node)',
    'under pressure the LRU leaf is evicted (parents collapse upward), hot prefixes retained'],
    testHint: "(go test ./internal/radixkv/ -count=1 -timeout 120s)" },
  { id: 'A2', regime: 'A', modules: ['kvmmu'], title: 'kvmmu', theorems: [
    'the KV page mapping is a bijection over live spans (no two logical positions alias one slot; no slot lost)',
    'span addressing is exact (a named span maps to exactly its bytes)'],
    testHint: "(go test ./internal/kvmmu/ -count=1 -timeout 120s)" },
  { id: 'A3', regime: 'A', modules: ['cachemeta'], title: 'cachemeta', theorems: [
    'the cache key / attention index is deterministic (same entry -> same key)',
    'the kv-transfer + attention-index round-trip preserves the entry',
    'collision/eviction behavior is well-defined'],
    testHint: "(go test ./internal/cachemeta/ -count=1 -timeout 120s)" },
  { id: 'A4', regime: 'A', modules: ['recall'], title: 'recall', theorems: [
    'a query against a completed session returns the SAME answer as replaying the session (no false recall, nothing missed)',
    'recall is deterministic and input-driven'],
    testHint: "(go test ./internal/recall/ -count=1 -timeout 120s)" },
  { id: 'A5', regime: 'A', modules: ['contextq'], title: 'contextq', theorems: [
    'on-demand materialization reconstructs a context byte-identical to the original CDB image',
    'materialization is deterministic'],
    testHint: "(go test ./internal/contextq/ -count=1 -timeout 120s)" },
  { id: 'A6', regime: 'A', modules: ['blob'], title: 'blob', theorems: [
    'content-address put/get round-trips byte-identical (small inline and large blob)',
    'identical content dedupes to one digest (the address IS the hash)'],
    testHint: "(go test ./internal/blob/ -count=1 -timeout 120s)" },
  { id: 'A7', regime: 'A', modules: ['preflight'], title: 'preflight', theorems: [
    'the pre-flight ladder evaluates rungs cheapest-first (order == FoldRank)',
    'a deny at a cheap rung short-circuits the expensive ones (no wasted evaluation)'],
    testHint: "(go test ./internal/preflight/ -count=1 -timeout 120s)" },
  { id: 'A8', regime: 'A', modules: ['abi','architest'], title: 'abi+architest', theorems: [
    'every verdict-fold site orders by abi.FoldRank, a TOTAL order (architest gate, just-added)',
    'the internal package graph is a DAG with no upward imports; hot path has no os/exec',
    'the frozen wave-0 ABI spine is stable (abi_test round-trips)'],
    testHint: "(go test ./internal/architest/ ./internal/abi/ -count=1 -timeout 120s)" },

  { id: 'C1', regime: 'C', modules: ['journal'], title: 'journal', theorems: [
    'the journal is append-only and hash-chained: row N commits to row N-1, so mutating any past row breaks verification (tamper-evidence)',
    'each decision is durably flushed per write (a crash loses nothing already returned)'],
    testHint: "(go test ./internal/journal/ -count=1 -timeout 120s)" },
  { id: 'C2', regime: 'C', modules: ['deletioncert'], title: 'deletioncert', theorems: [
    'a DeletionCertificate binds the eviction count + span + the max|delta|=0 equivalence + the journal anchor row + integrity epoch',
    'mint then verify accepts a genuine certificate; any tampered field is rejected (mint/verify round-trip + negative)'],
    testHint: "(go test ./internal/deletioncert/ -count=1 -timeout 120s)" },
  { id: 'C3', regime: 'C', modules: ['provenance'], title: 'provenance', theorems: [
    'trust authorship belongs to the kernel: a model-supplied Meta["provenance"]="trusted" tag is IGNORED (cannot self-mint trust)',
    'the kernel-authored provenance label is the single source consulted by the gates'],
    testHint: "(go test ./internal/provenance/ -count=1 -timeout 120s)" },
  { id: 'C4', regime: 'C', modules: ['ifc'], title: 'ifc', theorems: [
    'the abi.Ref.Taint lattice join is monotone and associative/commutative (a join-semilattice)',
    'non-interference: tainted (untrusted-derived) data is barred from a sink unless declassified — a paraphrase cannot launder taint (provenance-keyed, not content-keyed)'],
    testHint: "(go test ./internal/ifc/ -count=1 -timeout 120s)" },

  { id: 'D1', regime: 'D', modules: ['adjudicator'], title: 'adjudicator', theorems: [
    'verdicts fold in abi.FoldRank order; the composition is fail-closed (the zero/absent value resolves to Deny, never Allow)',
    'the reference monitor mediates every decision (no path bypasses the fold)'],
    testHint: "(go test ./internal/adjudicator/ -count=1 -timeout 120s)" },
  { id: 'D2', regime: 'D', modules: ['policy'], title: 'policy', theorems: [
    'the loaded capability floor is SOUND: the effective deny set is a superset of the declared floor (you can tighten, never loosen below the floor)',
    'policy load is deterministic and round-trips'],
    testHint: "(go test ./internal/policy/ -count=1 -timeout 120s)" },
  { id: 'D3', regime: 'D', modules: ['ctxmmu'], title: 'ctxmmu', theorems: [
    'page-out is idempotent (paging out an already-stubbed payload is a no-op)',
    'a benign result round-trips byte-identical (no false page-out of clean bytes)'],
    testHint: "(go test ./internal/ctxmmu/ -count=1 -timeout 120s)" },
  { id: 'D4', regime: 'D', modules: ['normgate'], title: 'normgate', theorems: [
    'a benign page round-trips byte-identical (TestBenignPageRoundTripsByteIdentical)',
    'detection over the canonical view is a SUPERSET of the raw-regex detection (normalize-and-rescan catches obfuscations the raw gate missed)',
    'HONEST LIMIT: a semantic paraphrase with no marker word evades by design (TestParaphraseEvadesByDesign) — record as a stated boundary, not a PROVEN catch'],
    testHint: "(go test ./internal/normgate/ -count=1 -timeout 120s)" },
  { id: 'D5', regime: 'D', modules: ['canon'], title: 'canon', theorems: [
    'canonicalization is idempotent: canon(canon(x)) == canon(x) (a normal form)',
    'the canonical form folds the obfuscation classes it claims (homoglyph/fullwidth/zero-width/bidi/spacing)'],
    testHint: "(go test ./internal/canon/ -count=1 -timeout 120s)" },
  { id: 'D6', regime: 'D', modules: ['plancfi'], title: 'plancfi', theorems: [
    'the plan automaton admits a tool call ONLY if it is an approved transition from the current plan state (soundness — an off-plan gadget is denied)',
    'the state machine is deterministic'],
    testHint: "(go test ./internal/plancfi/ -count=1 -timeout 120s)" },
  { id: 'D7', regime: 'D', modules: ['grammar'], title: 'grammar', theorems: [
    'a malformed-but-repairable call is transformed to an arity-matched well-formed call (positional args zipped into named params) preserving the intended invocation',
    'no grammar for a tool => Defer (FAIL-OPEN, never over-refuse)'],
    testHint: "(go test ./internal/grammar/ -count=1 -timeout 120s)" },
  { id: 'D8', regime: 'D', modules: ['witness'], title: 'witness', theorems: [
    'the require-witness rung is sound: a call requiring a witness is denied unless the witness is present (TestVDSOSoundness and friends)'],
    testHint: "(go test ./internal/witness/ -count=1 -timeout 120s)" },
  { id: 'D9', regime: 'D', modules: ['ratelimit'], title: 'ratelimit', theorems: [
    'the token-bucket / quota bounds throughput: at most cap calls (or cap cost) per key per window; the (cap+1)-th emits Deny(RATE_LIMITED)',
    'the budget is conserved (accounting never double-spends or leaks credit)'],
    testHint: "(go test ./internal/ratelimit/ -count=1 -timeout 120s)" },
  { id: 'D10', regime: 'D', modules: ['gateway'], title: 'gateway', theorems: [
    'the wire-fronted kernel returns the SAME verdict as the in-process kernel (no bypass via the network seam)',
    'a tool-call that fails adjudication is DROPPED fail-closed (anthropic_exfil_floor_test); the drop is solid'],
    testHint: "(go test -run 'Floor|Exfil|Drop|Parity|Verdict|Adjud' ./internal/gateway/ -count=1 -timeout 180s)" },
  { id: 'D11', regime: 'D', modules: ['steward'], title: 'steward', theorems: [
    'each steward enforces exactly one checkable invariant predicate (single responsibility, composable)',
    'the steward population is deterministic and order-independent where it claims to be'],
    testHint: "(go test ./internal/steward/ -count=1 -timeout 120s)" },
  { id: 'D12', regime: 'D', modules: ['shipgate'], title: 'shipgate', theorems: [
    'RSI ship-gate keeps a candidate ONLY if the measured metric strictly improves; otherwise it reverts (keep-or-revert monotonicity)',
    'the measurement is deterministic given the same inputs'],
    testHint: "(go test ./internal/shipgate/ -count=1 -timeout 120s)" },
  { id: 'D13', regime: 'D', modules: ['vdso'], title: 'vdso', theorems: [
    'the vDSO fast-path answer is IDENTICAL to the engine answer it short-circuits (soundness: caching never changes the result)',
    'the 3-tier lookup is deterministic and the integrity epoch advances monotonically'],
    testHint: "(go test -run 'Soundness|RoundTrip|Determinis|Epoch' ./internal/vdso/ -count=1 -timeout 120s)" },
  { id: 'D14', regime: 'D', modules: ['metrics'], title: 'metrics', theorems: [
    'histogram percentiles are monotonic in q (p50 <= p90 <= p99) — TestHistPercentilesMonotonic',
    'the A/B KPI fold is correct (paired aggregation matches the definition)'],
    testHint: "(go test ./internal/metrics/ -count=1 -timeout 120s)" },
  { id: 'D15', regime: 'D', modules: ['gpulease'], title: 'gpulease', theorems: [
    'the advisory lease grants at most ONE holder machine-wide (mutual exclusion) ',
    'release is idempotent (TestReleaseIdempotent); a crashed holder lease is reclaimable'],
    testHint: "(go test ./internal/gpulease/ -count=1 -timeout 120s)" },

  { id: 'I1', regime: 'A', modules: ['engine','enginecache','modelengine'], title: 'engine-seam', theorems: [
    'the EngineDriver seam is deterministic for a fixed engine (same request -> same response shape)',
    'enginecache binds remote invalidation directives correctly (an invalidated entry is not served stale)'],
    testHint: "(go test ./internal/engine/ ./internal/enginecache/ ./internal/modelengine/ -count=1 -timeout 120s)" },
  { id: 'I2', regime: 'A', modules: ['bench','turnbench'], title: 'bench-ab-isolation', theorems: [
    'an A/B ablation is a PAIRED replay: same seed/trace -> same trajectory, with ONLY the toggled variable differing (TestParity / TestFanoutDeterministic)',
    'the turn-tax isolation attributes the delta to the toggled axis, not noise'],
    testHint: "(go test ./internal/bench/ ./internal/turnbench/ -count=1 -timeout 180s)" },
]

// ---------------------------------------------------------------------------
const PROVE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'title', 'regime', 'theorems', 'doc_markdown'],
  properties: {
    id: { type: 'string' },
    title: { type: 'string' },
    regime: { type: 'string', enum: ['N', 'A', 'C', 'D'] },
    theorems: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['key', 'statement', 'verdict', 'witness_cmd', 'witness_tests', 'evidence'],
        properties: {
          key: { type: 'string', description: 'short kebab id e.g. softmax-row-stochastic' },
          statement: { type: 'string', description: 'precise falsifiable theorem' },
          proof: { type: 'string', description: 'the argument with file:line mechanism refs' },
          verdict: { type: 'string', enum: ['PROVEN', 'REFUTED', 'OPEN', 'SCOPED-OUT'] },
          witness_cmd: { type: 'string', description: 'exact go test -run command actually run' },
          witness_tests: { type: 'array', items: { type: 'string' }, description: 'test func names that bear on it' },
          mechanism_refs: { type: 'array', items: { type: 'string' }, description: 'file:line of the implementing code' },
          evidence: { type: 'string', description: 'tail of the run output / PASS-FAIL-SKIP / why OPEN or SCOPED-OUT' },
        },
      },
    },
    doc_markdown: { type: 'string', description: 'the full <module>.md body, following the proof-object format in 00-METHOD.md §2' },
    summary: { type: 'string' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'checks', 'overall'],
  properties: {
    id: { type: 'string' },
    overall: { type: 'string', enum: ['CONFIRMED', 'DOWNGRADED', 'MIXED'] },
    checks: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['key', 'reverified_verdict', 'witness_nonvacuous', 'mechanism_refs_exist', 'agrees', 'note'],
        properties: {
          key: { type: 'string' },
          reverified_verdict: { type: 'string', enum: ['PROVEN', 'REFUTED', 'OPEN', 'SCOPED-OUT'] },
          witness_nonvacuous: { type: 'boolean', description: 'does the cited test actually ASSERT the theorem (not a vacuous/always-pass test)?' },
          mechanism_refs_exist: { type: 'boolean', description: 'do the cited file:line actually contain the claimed mechanism?' },
          agrees: { type: 'boolean', description: 'does your re-run agree with the prove-stage verdict?' },
          note: { type: 'string' },
        },
      },
    },
    corrected_doc_markdown: { type: 'string', description: 'OPTIONAL: corrected <module>.md body if the prove-stage doc over-claimed or mis-cited; empty if no correction' },
  },
}

const PROVE_PROMPT = (o) => `You are proving the MATHEMATICAL CORRECTNESS of fak sub-module obligation ${o.id} (${o.title}), regime ${o.regime}.

Working dir is the fleet repo ROOT. The Go module is in fak/ (run tests as: ${o.testHint}).
Modules in scope: ${o.modules.map(m => 'fak/internal/' + m).join(', ')}.

The discipline is fak/docs/proofs/00-METHOD.md (read it). For EACH theorem below you must PROVE or REFUTE it with a DETERMINISTIC WITNESS — a real test you actually run. Never mark PROVEN on argument alone.

THEOREMS TO DISCHARGE:
${o.theorems.map((t, i) => `  (${i + 1}) ${t}`).join('\n')}

DO THIS, GROUNDED IN THE REAL CODE:
1. Read the implementing source in the scope module(s) and find the file:line where each theorem's mechanism lives.
2. Find the existing test(s) that witness each theorem (grep the *_test.go in scope). Then ACTUALLY RUN the witness with the command above (adapt the -run filter to the real test names you found). Capture PASS / FAIL / SKIP and the output tail.
3. Assign each theorem a verdict from ACTUAL EVIDENCE:
   - PROVEN: a real test asserts the theorem AND it ran green here. Cite the test name + the file:line of the mechanism.
   - REFUTED: a test asserts it and ran RED, or you can exhibit a counterexample. Record the counterexample.
   - OPEN: the theorem is true-looking but NO existing test actually asserts it (or the witness SKIPPED, e.g. an oracle fixture is absent). Say exactly which test would close it. Do NOT promote to PROVEN.
   - SCOPED-OUT: discharging it needs a tool/build we don't run here (e.g. -tags metal on non-Apple-GPU, Gobra). Give the reason + upgrade path.
4. CRITICAL HONESTY: a green package run does NOT mean a specific theorem is witnessed — verify the SPECIFIC test actually asserts THAT property (read its body). A test that compiles and passes but doesn't check the theorem leaves it OPEN, not PROVEN. The model package is heavy — run TARGETED -run filters, never the whole package; if a weight-backed/oracle rung would OOM or SKIP, mark OPEN/SCOPED with the reason rather than forcing it.

Then write doc_markdown: the full body of fak/docs/proofs/${o.title.replace(/\//g, '-')}.md following 00-METHOD.md §2 (one block per theorem: THEOREM / REGIME / PROOF with file:line / WITNESS exact cmd / VERDICT dated / and a DOS line left as "bound at ship"). Start with an H1 "# ${o.id} · ${o.title}" and a one-paragraph intro of what the module computes and what "correct" means for it. Be precise and honest; an OPEN is a fine outcome, a falsely-PROVEN is not.

Return the structured object. Your text IS the data — no prose outside the schema.`

const VERIFY_PROMPT = (o, prove) => `You are the ADVERSARIAL VERIFIER for fak math-proof obligation ${o.id} (${o.title}). A prior agent produced verdicts; your job is to REFUTE them where you can. You may only DOWNGRADE confidence (PROVEN->OPEN/REFUTED), never inflate it.

Working dir is the fleet repo root; tests run via: ${o.testHint}

The prove-stage claimed these per-theorem verdicts:
${prove.theorems.map(t => `  - [${t.verdict}] ${t.key}: ${t.statement}\n      witness_cmd: ${t.witness_cmd}\n      witness_tests: ${(t.witness_tests || []).join(', ')}\n      mechanism_refs: ${(t.mechanism_refs || []).join(', ')}`).join('\n')}

For EACH theorem, independently:
1. RE-RUN the witness_cmd yourself (adapt if the -run filter is wrong). Confirm it really PASSES (or SKIPS/FAILS).
2. READ the cited test body and confirm it is NON-VACUOUS — it must actually ASSERT the theorem (real comparison against the property, not a smoke test that always passes). A PROVEN backed by a vacuous or unrelated test must be downgraded to OPEN.
3. CHECK the mechanism_refs (file:line) actually exist and contain the claimed mechanism. If a citation is wrong/hallucinated, flag it.
4. Set reverified_verdict: keep PROVEN only if the re-run is green AND the test is non-vacuous AND the mechanism citation is real. Otherwise downgrade (OPEN if just unwitnessed, REFUTED if the property actually fails).

If you downgraded anything OR found a mis-citation, return corrected_doc_markdown with the fixed version of the prove-stage doc (same format), correcting verdicts and citations. Otherwise leave it empty.

Be skeptical and concrete. Default to downgrading when uncertain. Return the structured object only.`

// ---------------------------------------------------------------------------
phase('Prove')
log(`Proving ${ROSTER.length} obligations, each verified independently`)

const results = await pipeline(
  ROSTER,
  (o) => agent(PROVE_PROMPT(o), { label: `prove:${o.id}`, phase: 'Prove', schema: PROVE_SCHEMA, effort: 'high' })
    .then(p => ({ o, prove: p })),
  ({ o, prove }) => {
    if (!prove) return { o, prove: null, verify: null }
    return agent(VERIFY_PROMPT(o, prove), { label: `verify:${o.id}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' })
      .then(v => ({ o, prove, verify: v }))
  },
)

// Fold: final verdict per theorem = prove verdict, downgraded by verify where they disagree.
const RANK = { 'PROVEN': 3, 'SCOPED-OUT': 2, 'OPEN': 1, 'REFUTED': 0 }
const out = results.filter(Boolean).filter(r => r.prove).map(({ o, prove, verify }) => {
  const vmap = {}
  if (verify && verify.checks) for (const c of verify.checks) vmap[c.key] = c
  const theorems = (prove.theorems || []).map(t => {
    const c = vmap[t.key]
    let finalVerdict = t.verdict
    if (c) {
      // verify can only downgrade: take the lower rank, and force-downgrade if vacuous/missing-citation
      const dgr = (!c.witness_nonvacuous || !c.mechanism_refs_exist) && t.verdict === 'PROVEN' ? 'OPEN' : c.reverified_verdict
      if (RANK[dgr] < RANK[t.verdict]) finalVerdict = dgr
    }
    return { ...t, finalVerdict, verify: c || null }
  })
  return {
    id: o.id, title: o.title, regime: o.regime, modules: o.modules,
    docPath: o.title.replace(/\//g, '-') + '.md',
    doc_markdown: (verify && verify.corrected_doc_markdown) ? verify.corrected_doc_markdown : prove.doc_markdown,
    theorems,
    overall_verify: verify ? verify.overall : 'NO-VERIFY',
  }
})

const tally = { PROVEN: 0, 'SCOPED-OUT': 0, OPEN: 0, REFUTED: 0 }
for (const r of out) for (const t of r.theorems) tally[t.finalVerdict] = (tally[t.finalVerdict] || 0) + 1
log(`Done. theorem verdicts: PROVEN=${tally.PROVEN} OPEN=${tally.OPEN} REFUTED=${tally.REFUTED} SCOPED-OUT=${tally['SCOPED-OUT']}`)

return { obligations: out, tally }
