export const meta = {
  name: 'fak-close-open-witnesses',
  description: 'Close OPEN math obligations by ADDING a sound deterministic witness test per package, directly on master',
  phases: [ { title: 'Close', detail: 'one agent per package adds proofs_witness_test.go, runs it, reports PROVEN/REFUTED/SKIP' } ],
}

const ROSTER = [
 {
  "package": "model",
  "docPath": "model-attention.md",
  "opens": [
   {
    "id": "N1",
    "key": "softmax-row-stochastic-shift-invariant",
    "statement": "For any finite score vector x, softmaxInPlace(x) yields entries y_i >= 0 with sum_i y_i = 1 (row-stochastic), and softmax(x + c*1) = softmax(x) for any scalar c (shift-invariance).",
    "mechanism_refs": [
     "fak/internal/model/forward.go:403"
    ],
    "existing_tests": []
   },
   {
    "id": "N1",
    "key": "causal-strictly-lower-triangular",
    "statement": "The dense scaled-dot-product attention weight matrix is strictly lower-triangular: query position i receives zero weight from any key position j > i (it attends only to j <= i).",
    "mechanism_refs": [
     "fak/internal/model/forward.go:234",
     "fak/internal/model/forward.go:242",
     "fak/internal/model/swa_test.go:268"
    ],
    "existing_tests": [
     "TestSWAWindowMasksOldKeys",
     "TestDSATopKIndicesAreCausalAndPrefixReusable"
    ]
   },
   {
    "id": "N2",
    "key": "layernorm-shift-scale-equivariant",
    "statement": "LayerNorm is invariant to affine input transforms on the normalized axis: for a>0,b, layernorm(a*x+b, w, bias, eps')[i] == layernorm(x, w, bias, eps)[i] in the eps->0 limit (mean-subtraction removes b; division by stddev removes a). RMSNorm is invariant to positive input scaling up to the learned gain.",
    "mechanism_refs": [
     "fak/internal/model/arch.go:457",
     "fak/internal/model/forward.go:356"
    ],
    "existing_tests": [
     "TestLayerNormAxis",
     "TestNormGain1p"
    ]
   },
   {
    "id": "N2",
    "key": "norm-numerically-stable-large-inputs",
    "statement": "For large-magnitude finite inputs (e.g. |x| ~ 1e15..1e20), rmsnorm/layernorm produce only finite outputs (no NaN, no +/-Inf) \u2014 the sum-of-squares does not overflow f32 and the 1/sqrt does not divide by zero/inf.",
    "mechanism_refs": [
     "fak/internal/model/forward.go:356",
     "fak/internal/model/arch.go:457",
     "fak/internal/model/arch.go:245"
    ],
    "existing_tests": [
     "TestGemmaStackChangesOutput"
    ]
   },
   {
    "id": "N3",
    "key": "rope-preserves-pair-norm",
    "statement": "For the unscaled (Llama) rotary map, applyRopeRow acts on each pair (h[j], h[j+half]) as a Givens rotation by angle p\u00b7inv_freq[j]; since cos\u00b2+sin\u00b2=1 it preserves the per-pair Euclidean norm (and hence the per-head vector norm) for every position p and every input vector.",
    "mechanism_refs": [
     "fak/internal/model/kv.go:473",
     "fak/internal/model/kv.go:465",
     "fak/internal/model/kv.go:451"
    ],
    "existing_tests": [
     "TestForwardRopeAppliesSharedRotation",
     "TestRopeRowsShareInvFreqBitExact"
    ]
   },
   {
    "id": "N3",
    "key": "rope-dot-relative-position",
    "statement": "For unscaled RoPE, the dot product of a query rotated at position m with a key rotated at position n equals the dot product of the same vectors rotated at positions (m\u2212n) and 0 \u2014 i.e. <R_m q, R_n k> depends on m,n only through the relative offset (m\u2212n), the defining property of RoPE.",
    "mechanism_refs": [
     "fak/internal/model/kv.go:467",
     "fak/internal/model/kv.go:473"
    ],
    "existing_tests": []
   },
   {
    "id": "N5",
    "key": "awq-matches-reference",
    "statement": "AWQ dequant weight[o*in+i] = scale[o]*(unpack4bit(code) - 8) matches its reference: (a) the affine arithmetic is bit-exact to an independently-computed expected value, and (b) it matches the HuggingFace AutoAWQ reference the format claims compatibility with.",
    "mechanism_refs": [
     "fak/internal/model/awq.go:57",
     "fak/internal/model/awq_scalar.go:32",
     "fak/internal/model/awq.go:49"
    ],
    "existing_tests": [
     "TestAWQUnpack4bit",
     "TestAWQDequantRowScalar",
     "TestAWQDotProductScalar",
     "TestAWQMatRows",
     "TestAWQOracleThreshold"
    ]
   },
   {
    "id": "N7",
    "key": "forward-parity-across-arch-families",
    "statement": "The forward-pass HF-oracle parity (hidden cosine\u22481, per-position argmax, greedy ids) holds not only for llama but across the supported arch families: qwen3 (QK-norm), qwen3moe (hybrid dense/sparse), gpt_oss (MXFP4 MoE), gemma3 (local/global attn, sandwich norm), glm_moe_dsa (MLA+DSA+MoE), mistral (SWA), llama3 (rope scaling), phi3 (longrope), deepseek-v2 (MLA).",
    "mechanism_refs": [
     "internal/model/forward.go:60",
     "internal/model/forward.go:130",
     "internal/model/oracle_test.go:1011",
     "internal/model/oracle_test.go:304",
     "internal/model/oracle_test.go:814",
     "internal/model/safetensors_stream_test.go:488"
    ],
    "existing_tests": [
     "TestOptionalQwen3OracleCoversQKNorm",
     "TestOptionalQwen3MoEOracleCoversHybridDenseSparseLayers",
     "TestOptionalGPTOSSMXFP4ForwardMatchesHFOracle",
     "TestOptionalGemma3OracleCoversLocalGlobalAttention",
     "TestOptionalGLMMoeDsaOracleForwardMatchesHFCacheless",
     "TestOptionalGLMMoeDsaOracleSessionCacheMatchesHF",
     "TestOptionalMistralSWAOracleNonVacuous",
     "TestOptionalLlama3OracleCoversScalingAndEOSList",
     "TestOptionalPhi3LongropeOracleCoversLongFactor",
     "TestOptionalDeepSeekV2OracleDocumentsMLABoundary"
    ]
   }
  ]
 },
 {
  "package": "compute",
  "docPath": "compute-gemm.md",
  "opens": [
   {
    "id": "N8",
    "key": "gemm-bilinear",
    "statement": "GEMM is bilinear: A(B+C) = AB + AC and (\u03b1A)B = \u03b1(AB), checked against the naive computation up to float tolerance (the metamorphic relation named in 00-METHOD \u00a73.2).",
    "mechanism_refs": [
     "fak/internal/compute/cpuref.go:119",
     "fak/internal/compute/cpuref.go:376"
    ],
    "existing_tests": []
   }
  ]
 },
 {
  "package": "metalgemm",
  "docPath": "metalgemm.md",
  "opens": [
   {
    "id": "N9",
    "key": "stub-metal-interface-parity",
    "statement": "The stub (default) build and the metal build of package metalgemm present the same exported interface, and the stub introduces no math divergence \u2014 it does no GPU math but degrades cleanly to 'unavailable' (Available()=false, Compiled()=false, Prefill ok=false) so callers fall back to the pure-Go CPU path.",
    "mechanism_refs": [
     "fak/internal/metalgemm/metalgemm_stub.go:1",
     "fak/internal/metalgemm/metalgemm.go:1",
     "fak/internal/metalgemm/metalgemm_stub.go:9",
     "fak/internal/metalgemm/metalgemm_stub.go:45",
     "fak/internal/model/metal_prefill.go:152"
    ],
    "existing_tests": []
   }
  ]
 },
 {
  "package": "radixkv",
  "docPath": "radixkv.md",
  "opens": [
   {
    "id": "A1",
    "key": "refcount-conservation",
    "statement": "Reference counts are conserved: insert/clone(split)/evict leave no dangling node (a removed node is unreachable, never matched) and no leaked lease (after every Lookup\u2192Insert\u2192Done request cycle, all leases taken net to zero \u2014 \u03a3 node.refs equals the number of in-flight requests, and is 0 once all are Done).",
    "mechanism_refs": [
     "fak/internal/radixkv/radixkv.go:170",
     "fak/internal/radixkv/radixkv.go:201",
     "fak/internal/radixkv/radixkv.go:210",
     "fak/internal/radixkv/radixkv.go:261"
    ],
    "existing_tests": [
     "TestLRURespectsLease"
    ]
   },
   {
    "id": "A1",
    "key": "lru-leaf-evicted-hot-retained",
    "statement": "Under a token budget (maxTokens>0), when tokens exceed budget the tree evicts the least-recently-used unlocked LEAF (and, since removing a leaf can make its parent a leaf, repeats \u2014 the upward collapse) until within budget; a hot/recently-touched prefix and a leased (refs>0) prefix are retained even when oldest.",
    "mechanism_refs": [
     "fak/internal/radixkv/radixkv.go:227",
     "fak/internal/radixkv/radixkv.go:239",
     "fak/internal/radixkv/radixkv.go:166",
     "fak/internal/radixkv/radixkv.go:261"
    ],
    "existing_tests": [
     "TestLRUEviction",
     "TestLRURespectsLease"
    ]
   }
  ]
 },
 {
  "package": "recall",
  "docPath": "recall.md",
  "opens": [
   {
    "id": "A4",
    "key": "recall-deterministic-input-driven",
    "statement": "Recall is deterministic and input-driven: for fixed (loaded session, query, k) the assembled working set \u2014 its membership, order, and bytes \u2014 is identical on every invocation, depending only on the inputs (the page table, the persisted CAS, the query string, the revocation/clearance state) and on nothing nondeterministic (no RNG, wall-clock, network, or map-iteration-order dependence in the output).",
    "mechanism_refs": [
     "fak/internal/recall/recall.go:376",
     "fak/internal/recall/recall.go:389",
     "fak/internal/recall/recall.go:478",
     "fak/internal/recall/recall.go:484",
     "fak/internal/recall/recall.go:331"
    ],
    "existing_tests": []
   }
  ]
 },
 {
  "package": "contextq",
  "docPath": "contextq.md",
  "opens": [
   {
    "id": "A5",
    "key": "materialize-byte-identical",
    "statement": "On-demand materialization (contextq.Query) reconstructs a context byte-identical to the original CDB image: every materialized benign page's bytes equal the bytes the CDB image holds for that page.",
    "mechanism_refs": [
     "fak/internal/contextq/contextq.go:366",
     "fak/internal/contextq/contextq.go:678",
     "fak/internal/cdb/cdb.go:173"
    ],
    "existing_tests": [
     "TestQueryMaterializesTypedWorkingSet",
     "TestAllFiveMaterializationVerdictsReachable"
    ]
   },
   {
    "id": "A5",
    "key": "materialization-deterministic",
    "statement": "contextq.Query is deterministic: for a fixed CDB image and a fixed Request, repeated calls produce an equal Result (same slices, views, verdicts, omissions, render plan, stats).",
    "mechanism_refs": [
     "fak/internal/contextq/contextq.go:263",
     "fak/internal/contextq/contextq.go:678",
     "fak/internal/contextq/contextq.go:722"
    ],
    "existing_tests": [
     "TestAllFiveMaterializationVerdictsReachable",
     "TestViewCacheCopiesPayloadBytes"
    ]
   }
  ]
 },
 {
  "package": "journal",
  "docPath": "journal.md",
  "opens": [
   {
    "id": "C1",
    "key": "per-write-durable-flush",
    "statement": "For a file-backed journal, each Emit that produces an audit Row flushes that row's bytes to the OS file before returning, so a process crash (not power loss) after Emit returns loses no row already committed \u2014 Verify(path) called WITHOUT any intervening Close()/Flush() recovers every emitted row.",
    "mechanism_refs": [
     "fak/internal/journal/journal.go:149-153",
     "fak/internal/journal/journal.go:296-308",
     "fak/internal/journal/journal.go:11-14"
    ],
    "existing_tests": [
     "TestFileJournalReopensAndContinuesChain",
     "TestVerifyDetectsTampering"
    ]
   }
  ]
 },
 {
  "package": "ifc",
  "docPath": "ifc.md",
  "opens": [
   {
    "id": "C4",
    "key": "taint-join-semilattice",
    "statement": "The abi.Ref.Taint join (implemented as taintRank-max via Ledger.Raise) is a join-semilattice: monotone (a Raise never lowers a trace's mark; an unseen trace is the identity Trusted), commutative (Raise(a) then Raise(b) yields the same mark as Raise(b) then Raise(a)), and associative \u2014 over the closed 3-element order Trusted<Tainted<Quarantined.",
    "mechanism_refs": [
     "fak/internal/ifc/ifc.go:65",
     "fak/internal/ifc/ifc.go:122",
     "fak/internal/abi/types.go:82"
    ],
    "existing_tests": [
     "TestLedgerIsBoundedByLRUTraceMarks",
     "TestNewLedgerUsesDefaultLimit"
    ]
   }
  ]
 },
 {
  "package": "ctxmmu",
  "docPath": "ctxmmu.md",
  "opens": [
   {
    "id": "D3",
    "key": "page-out-idempotent",
    "statement": "For any result r whose Payload is already a ctxmmu page-out/quarantine stub (the JSON {\"_quarantined\":true,...} or {\"_paged\":true,...} that Admit substitutes in-place), a second Admit(ctx,c,r) is a no-op: it returns VerdictAllow and does not page the stub out again (paged/quarantine counters and r.Payload unchanged by the re-admission).",
    "mechanism_refs": [
     "fak/internal/ctxmmu/mmu.go:76",
     "fak/internal/ctxmmu/mmu.go:84",
     "fak/internal/ctxmmu/mmu.go:98",
     "fak/internal/ctxmmu/mmu.go:114",
     "fak/internal/ctxmmu/mmu.go:255"
    ],
    "existing_tests": [
     "TestAdmitBenignAllows",
     "TestAdmitPoisonFixture",
     "TestAdmitOversizeBenignTransforms"
    ]
   },
   {
    "id": "D3",
    "key": "benign-byte-identical",
    "statement": "For any clean result body (no secret pattern, no injection marker, no degenerate repeat, len <= OversizeBytes=4096), Admit(ctx,c,r) returns VerdictAllow and leaves r.Payload byte-identical to the input \u2014 i.e. no false page-out / no mutation of clean bytes.",
    "mechanism_refs": [
     "fak/internal/ctxmmu/mmu.go:67",
     "fak/internal/ctxmmu/mmu.go:84",
     "fak/internal/ctxmmu/mmu.go:100"
    ],
    "existing_tests": [
     "TestAdmitBenignAllows",
     "TestAdmitPoisonFixture"
    ]
   }
  ]
 },
 {
  "package": "normgate",
  "docPath": "normgate.md",
  "opens": [
   {
    "id": "D4",
    "key": "benign-page-round-trips-byte-identical",
    "statement": "For a benign result (no secret shape, no injection marker on the canonical view), normgate.Admit returns VerdictDefer AND leaves r.Payload byte-identical to the input \u2014 it does not page out, stub, or otherwise mutate the payload bytes.",
    "mechanism_refs": [
     "internal/normgate/normgate.go:99",
     "internal/normgate/normgate.go:101",
     "internal/normgate/normgate_test.go:105"
    ],
    "existing_tests": [
     "TestBenignDefers"
    ]
   },
   {
    "id": "D4",
    "key": "canon-detection-superset-of-raw-regex",
    "statement": "Detection over the canonical view is a SUPERSET of the raw-substring detection: for every body the legacy raw gate (ctxmmu.hasInjection over strings.ToLower(raw)) flags, canon.Scan also flags it (Injection=true), AND canon additionally flags obfuscated variants (char-spacing, base64, homoglyph, zero-width, fullwidth, bidi-reverse) that the raw gate misses.",
    "mechanism_refs": [
     "internal/ctxmmu/mmu.go:220",
     "internal/canon/canon.go:98",
     "internal/canon/canon.go:197",
     "internal/canon/canon.go:177"
    ],
    "existing_tests": [
     "TestObfuscatedInjectionCaught",
     "TestObfuscatedSecretCaught"
    ]
   }
  ]
 },
 {
  "package": "canon",
  "docPath": "canon.md",
  "opens": [
   {
    "id": "D5",
    "key": "canonicalization-idempotent",
    "statement": "For all input strings x, Normalize(Normalize(x)) == Normalize(x): Normalize is a projection onto a canonical normal form, so re-canonicalizing a canonical body is a no-op.",
    "mechanism_refs": null,
    "existing_tests": []
   }
  ]
 },
 {
  "package": "steward",
  "docPath": "steward.md",
  "opens": [
   {
    "id": "D11",
    "key": "steward-population-deterministic-order-independent",
    "statement": "The steward population is deterministic and order-independent where it claims to be: for a fixed set of stewards and a fixed environment, (a) Sweep produces the same fired set and the same per-name fire tallies, and Prune removes exactly the stewards that never fired and keeps the rest, regardless of the ORDER in which the stewards were added to the Population.",
    "mechanism_refs": [
     "fak/internal/steward/steward.go:55",
     "fak/internal/steward/steward.go:61",
     "fak/internal/steward/steward.go:70",
     "fak/internal/steward/steward.go:87"
    ],
    "existing_tests": [
     "TestPrunePopulation",
     "TestNewStewardAndSweepReportsWitness"
    ]
   }
  ]
 },
 {
  "package": "shipgate",
  "docPath": "shipgate.md",
  "opens": [
   {
    "id": "D12",
    "key": "measurement-deterministic",
    "statement": "Evaluate is deterministic: for any fixed Witness w, repeated evaluations Evaluate(w) yield the identical (Decision, keep-bit) \u2014 no dependence on RNG, wall-clock, goroutine scheduling, map-iteration order, or mutable global state.",
    "mechanism_refs": [
     "fak/internal/shipgate/shipgate.go:54",
     "fak/internal/shipgate/shipgate.go:64"
    ],
    "existing_tests": []
   }
  ]
 },
 {
  "package": "vdso",
  "docPath": "vdso.md",
  "opens": [
   {
    "id": "D13",
    "key": "integrity-epoch-advances-monotonically",
    "statement": "The integrity (trust) epoch advances monotonically: a refutation (Revoke of a non-empty witness) strictly increases TrustEpoch by 1, an empty-witness Revoke is a no-op (epoch unchanged), and across a sequence of N refutations the epoch is strictly increasing and never decreases.",
    "mechanism_refs": [
     "fak/internal/vdso/revoke.go:91",
     "fak/internal/vdso/revoke.go:136",
     "fak/internal/vdso/vdso.go:85"
    ],
    "existing_tests": [
     "TestRevoke_OrthogonalToWorldVersion",
     "TestRevoke_EmptyWitnessNoOp",
     "TestRevoke_PublishesOnCoherenceBus"
    ]
   }
  ]
 },
 {
  "package": "metrics",
  "docPath": "metrics.md",
  "opens": [
   {
    "id": "D14",
    "key": "ab-kpi-fold-correct",
    "statement": "The A/B KPI fold is correct: the five KPI fields aggregate the paired On/Off Arm counters per their definitions (vdso_hit_rate=VDSOHits/Calls, context_pollution_rate=Quarantines/Calls, tokens_per_task=(InTokens+OutTokens)/Calls, tool_call p50/p99 = On arm percentiles; token_delta_pct = 100*(offTok-onTok)/offTok).",
    "mechanism_refs": [
     "fak/internal/metrics/metrics.go:175",
     "fak/internal/metrics/metrics.go:162",
     "fak/internal/bench/bench.go:237"
    ],
    "existing_tests": [
     "TestReportJSONAndTokenDelta",
     "TestValidateWorkloadHash",
     "TestComputeGate"
    ]
   }
  ]
 },
 {
  "package": "enginecache",
  "docPath": "engine-seam.md",
  "opens": [
   {
    "id": "I1",
    "key": "enginecache-end-to-end-not-served-stale",
    "statement": "After Invalidate succeeds against a live serving engine, a subsequent request for the invalidated prefix/span observes a cache MISS (is recomputed), not the pre-invalidation cached value.",
    "mechanism_refs": [
     "fak/internal/enginecache/enginecache.go:57",
     "fak/internal/enginecache/enginecache.go:130"
    ],
    "existing_tests": []
   }
  ]
 }
]

const SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['package','filename','pkg_clause','file_content','pkg_test_pass','ran_cmd','witnesses'],
  properties: {
    package: { type: 'string' },
    filename: { type: 'string', description: 'internal/<pkg>/proofs_witness_test.go' },
    pkg_clause: { type: 'string', description: 'the exact `package X` line used' },
    file_content: { type: 'string', description: 'the FULL Go test file content' },
    pkg_test_pass: { type: 'boolean', description: 'did the whole package go test pass WITH the new file present?' },
    ran_cmd: { type: 'string' },
    output_tail: { type: 'string' },
    witnesses: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['key','verdict','test_name','note'],
      properties: {
        key: { type: 'string' },
        verdict: { type: 'string', enum: ['PROVEN','REFUTED','SKIP'] },
        test_name: { type: 'string', description: 'the Test func that witnesses it (empty if SKIP)' },
        note: { type: 'string' },
      } } },
  },
}

const PROMPT = (g) => `You are CLOSING open math-proof obligations for fak package internal/${g.package} by WRITING a new deterministic witness test.

You are on the LIVE master working tree (this repo FORBIDS worktrees/branches — operator law master-is-single-source; do not attempt one). STRICT RULES: create ONLY the single new file internal/${g.package}/proofs_witness_test.go; do NOT edit, move, or delete ANY other file; do NOT run ANY git command (no add/commit/stash/checkout/clean/reset/branch/worktree). Another agent owns each other package's new file, so paths never collide. The integrator commits afterward.

The Go module is at fak/ (run: (go test ...)). The proof discipline is fak/docs/proofs/00-METHOD.md.

These theorems are currently OPEN (true-looking but NO existing test directly asserts them). Close as many as you SOUNDLY can:
${g.opens.map((o,i)=>`  (${i+1}) [${o.key}] ${o.statement}\n        mechanism: ${(o.mechanism_refs||[]).join(', ')||'(find it)'}`).join('\n')}

DO THIS:
1. READ internal/${g.package}/*.go and the existing *_test.go to learn the EXACT function signatures, the package clause used by existing tests, and whether the functions you must call are exported. Do NOT guess signatures — a test that doesn't compile fails the whole package.
2. Write ONE new file internal/${g.package}/proofs_witness_test.go containing a Go test per OPEN you can close. Use the SAME package clause as the package's existing internal tests if you must call unexported funcs (e.g. \`package ${g.package}\`). Zero new dependencies — stdlib \`testing\` (and \`testing/quick\`, \`math\`, \`math/rand\` with a FIXED seed) only.
3. Each test must be DETERMINISTIC and NON-VACUOUS — it must actually ASSERT the property (a real metamorphic/round-trip/invariant comparison), not a smoke test. Prefer:
   - metamorphic relations (e.g. softmax(x+c)==softmax(x); Σsoftmax==1; rope preserves per-pair norm; rope dot depends only on (m-n); GEMM A(B+C)==AB+AC; canon(canon(x))==canon(x); join idempotent/commutative/associative/monotone),
   - round-trip / byte-identity (encode∘decode==id; pageOut idempotent; benign body unchanged),
   - determinism (two runs on a fixed seed are bit-identical),
   - numerical stability (finite output, no NaN/Inf, on large-magnitude inputs).
   Use a tolerance of 1e-5..1e-6 for float metamorphic checks; use exact equality for integer/byte properties.
4. RUN it: (go test -run '<your new TestNames>' ./internal/${g.package}/ -count=1 -v) AND confirm the whole package still passes: (go test ./internal/${g.package}/ -count=1).
5. Per OPEN: verdict PROVEN if your new asserting test PASSES; REFUTED if the property genuinely FAILS (keep the test — that's a real finding, report the counterexample in note); SKIP if you cannot write a SOUND asserting test (needs absent fixtures / unclear API) — leave it OPEN, say why. NEVER fake a PROVEN with a vacuous/always-pass test; an honest SKIP is strictly better.

Return file_content = the FULL final file you wrote (so the integrator can re-create it), pkg_test_pass = whether the package go test passed with your file present, and one witnesses[] entry per OPEN. Structured object only.`

phase('Close')
log(`Closing OPENs across ${ROSTER.length} packages, directly on master`)
const results = await parallel(ROSTER.map(g => () =>
  agent(PROMPT(g), { label: `close:${g.package}`, phase: 'Close', schema: SCHEMA, effort: 'high' })))

const out = results.filter(Boolean)
const tally = { PROVEN:0, REFUTED:0, SKIP:0 }
for (const r of out) for (const w of (r.witnesses||[])) tally[w.verdict]=(tally[w.verdict]||0)+1
log(`closed: PROVEN=${tally.PROVEN} REFUTED=${tally.REFUTED} SKIP=${tally.SKIP}`)
return { packages: out, tally }
