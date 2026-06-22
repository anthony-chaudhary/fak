---
title: "fak proofs: math-correctness master ledger"
description: "The master ledger of fak's per-module correctness proofs: each theorem, its deterministic witness test, its live verdict, and its DOS git-evidence binding."
---

# fak math-proof section — master ledger

This is fak's **dedicated proof of mathematical correctness**, sub-module by sub-module.
Read [`00-METHOD.md`](00-METHOD.md) first: it defines the discipline — every module gets a
*theorem*, a *proof*, a **deterministic witness** (a test the toolchain re-runs), a
**verdict** (`PROVEN` / `REFUTED` / `OPEN` / `SCOPED-OUT`), and a **DOS binding** that grounds
"the proof shipped" in git evidence rather than an author's say-so.

This section is the math-correctness companion to
[`../../SUBSYSTEM-CHECKS.md`](../../SUBSYSTEM-CHECKS.md): that ledger proves a boundary is
*alive*; this one proves the *math it computes is correct*, and refuses to call anything
`PROVEN` on self-report.

- **`00-METHOD.md`** — the method: regimes (N/A/C/D), the witness taxonomy, the verdict
  vocabulary + honesty rule, DOS as meta-witness, the SOTA tooling survey, reproduction.
- **`<module>.md`** — one proof file per module/obligation: every theorem, its proof, its
  witness command, its live verdict, its DOS binding.

## How to read a verdict

A `PROVEN` here means: a named, deterministic test exists; it ran **green** on this node
(`go test`, native on macOS / via WSL on Windows); the green corroborates the stated
theorem; and the commit that shipped it is `diff-witnessed` by `dos commit-audit`. An
`OPEN` means the theorem is stated but no deterministic witness discharges it *yet* — the
file says which witness would. A `REFUTED` is a recorded finding with its counterexample,
not a deletion. Nothing is `PROVEN` by prose.

## Master ledger

> Verdicts are filled from **actual test runs** during the proof build, not asserted.
> Regime: **N** numerical · **A** algebraic/structural · **C** crypto/integrity · **D**
> decision-procedure. `Witness` names the load-bearing test(s); the per-module doc carries
> the exact `-run` target and `file:line`.

<!-- LEDGER-START -->
_Generated from the per-module proof files (**94 PROVEN · 2 OPEN · 1 REFUTED · 1 SCOPED-OUT** across 98 theorems / 40 obligations). Verdicts reflect an actual `go test -count=1` run on this node (darwin/arm64, go1.26), cross-checked by an independent full `./internal/...` suite pass (45 packages green, 0 failures). OPEN obligations closed in the witness pass (commit 3cb8ff9) show ✅ with their added witness test. The one REFUTED row (N7, Qwen3.6 hybrid-GDN multi-token greedy parity) is a recorded finding with its counterexample — token-3 decode drift — not a deletion (00-METHOD §4)._

### N — Numerical / linear-algebra

| Obl | Module | Theorem | Witness | Verdict |
|---|---|---|---|---|
| N1 | [model/attention](model-attention.md) | For any finite score vector x, softmaxInPlace(x) yields entries y_i >= 0 with sum_i y_i = 1 (row-stochastic), and soft | `TestProofSoftmaxRowStochasticAndShiftInvariant` | ✅ PROVEN |
|  | ↳ | The dense scaled-dot-product attention weight matrix is strictly lower-triangular: query position i receives zero weig | `TestProofCausalStrictlyLowerTriangular` | ✅ PROVEN |
|  | ↳ | When a learned per-head sink logit is present, softmaxDropSinkInPlace includes it in the softmax denominator but exclu | `TestAttentionSinkSoftmaxDropsSink` | ✅ PROVEN |
| N2 | [model/norm](model-norm.md) | For all x,w of equal length and eps>0, rmsnorm(x,w,eps)[i] == x[i] * w[i] / sqrt(mean(x^2)+eps) within 1e-6 (plain pat | `TestNormGain1p` | ✅ PROVEN |
|  | ↳ | For all x,w,bias of equal length and eps>0, layernorm(x,w,bias,eps)[i] == (x[i]-mean)*w[i]/sqrt(var+eps) + bias[i] wit | `TestLayerNormAxis` | ✅ PROVEN |
|  | ↳ | LayerNorm is invariant to affine input transforms on the normalized axis: for a>0,b, layernorm(a*x+b, w, bias, eps')[i | `TestProofLayerNormShiftScaleEquivariant`, `TestProofRMSNormPositiveScaleInvariant` | ✅ PROVEN |
|  | ↳ | For large-magnitude finite inputs (e.g. \|x\| ~ 1e15..1e20), rmsnorm/layernorm produce only finite outputs (no NaN, no | `TestProofNormNumericallyStableLargeInputs` | ✅ PROVEN |
| N3 | [model/rope](model-rope.md) | For the unscaled (Llama) rotary map, applyRopeRow acts on each pair (h[j], h[j+half]) as a Givens rotation by angle p· | `TestProofRopePreservesPairNorm` | ✅ PROVEN |
|  | ↳ | For unscaled RoPE, the dot product of a query rotated at position m with a key rotated at position n equals the dot pr | `TestProofRopeDotRelativePosition` | ✅ PROVEN |
|  | ↳ | The implemented RoPE scaling variants rescale inv_freq / cos·sin per their reference formula: llama3 piecewise low/hig | `TestRopeScalingLlama3`, `TestYarnRopeScalesCosSin` … | ✅ PROVEN |
| N4 | [model/mlp+residual](model-mlp+residual.md) | For every layer and every normalized input x, the dense MLP delta equals down_proj( act(gate_proj(x)) * up_proj(x) ) e | `TestMoEDenseNoOpIdentical`, `TestDenseActivationMLPWithBias` | ✅ PROVEN |
|  | ↳ | The residual stream update is exact elementwise addition of the sub-layer delta into the stream — x += body(norm(x)) f | `TestBlockTopologyComposition`, `TestSandwichNormUsesDistinctFeedForwardNorms` | ✅ PROVEN |
|  | ↳ | MoE routing computes logits=router(x), probs=softmax over ALL experts, selects the top-k experts by prob (torch.topk s | `TestMoERoutingHandComputed`, `TestMoEWiring` … | ✅ PROVEN |
| N5 | [model/quant](model-quant.md) | For every Q4_K super-block, q4kDequantSuperBlock reproduces value = d*sc_s*nibble - min*m_s (the affine per-sub-block  | `TestQ4KDequantSuperBlockMatchesRef`, `TestQ4KMatRowsMatchesF32` … | ✅ PROVEN |
|  | ↳ | AWQ dequant weight[o*in+i] = scale[o]*(unpack4bit(code) - 8) matches its reference: (a) the affine arithmetic is bit-e | `TestProofAWQMatchesReference` | ✅ PROVEN |
|  | ↳ | The arch-dispatched integer reduction q4kReduceRow (NEON SDOT on arm64) produces the per-sub-block int32 pairs IS=sum( | `TestQ4KReduceAsmMatchesScalar`, `TestQ4KInt8DotMatchesF32` | ✅ PROVEN |
| N6 | [model/kv](model-kv.md) | Per-position K/V written during a decoder block lands in the correct (layer, position, head) slot of the kernel-owned  | `TestStandardLayoutNoOp`, `TestSWAWindowUnsetIsNoOp` … | ✅ PROVEN |
|  | ↳ | KVCache.Evict(from,n) removes a contiguous position span from every layer and re-rotates each survivor's post-RoPE K t | `TestKVQuarantineEqualsNeverSaw`, `TestEvictRepositionsWithLayerSpecificRopeTheta` | ✅ PROVEN |
|  | ↳ | Sliding-window attention masks EXACTLY the out-of-window positions — query at absolute position p attends only keys wi | `TestSWAWindowMasksOldKeys`, `TestSWAWindowUnsetIsNoOp` … | ✅ PROVEN |
| N7 | [model/forward-parity](model-forward-parity.md) | For the loaded HF oracle (smollm2-135m, model_type=llama), the pure-Go Forward pass reproduces HF's per-layer hidden s | `TestForwardMatchesHFOracle`, `TestForwardMatchesHFOracle/smollm2-135m` | ✅ PROVEN |
|  | ↳ | Generate(prompt.Ids, n) on weights Go-decoded directly from raw safetensors (no torch in the load path) reproduces HF' | `TestForwardOnGoDecodedWeights` | ✅ PROVEN |
|  | ↳ | The forward-pass HF-oracle parity (hidden cosine≈1, per-position argmax, greedy ids) holds not only for llama but acro | `TestOptionalQwen3OracleCoversQKNorm`, `TestOptionalQwen3MoEOracleCoversHybridDenseSparseLayers` … | 🔸 OPEN |
|  | ↳ | Qwen3.6 (hybrid Gated-DeltaNet) in-kernel greedy decode reproduces the llama.cpp reference token-for-token on the same q4_k_m GGUF — **token-3 drift**: fak/oracle agree on tokens 0–1 then diverge at token 2 (fak `8160` vs oracle `90700`, a near-tie argmax flip; first-token parity holds) | `cmd/qwen35check` + captured oracle artifact pair (`experiments/qwen36/*multitoken*`) | ❌ REFUTED |
| N8 | [compute/gemm](compute-gemm.md) | For w[out,in], x[in], the HAL F32 MatMul/BatchedMatMul computes y[o]=Σ_i w[o,i]·x[i] equal to the package's canonical  | `TestMatMulDelegatesVerbatim`, `TestReductionOrderIsTheModelTree` | ✅ PROVEN |
|  | ↳ | GEMM is bilinear: A(B+C) = AB + AC and (αA)B = α(AB), checked against the naive computation up to float tolerance (the | `TestGEMMBilinear` | ✅ PROVEN |
|  | ↳ | Every registered backend (CPU / Metal / CUDA / Vulkan, by build tag) agrees with the CPU Reference: an Approx backend' | `TestQ8DispatchIsApproxAndGated`, `TestCorrectnessClassEnforcement` … | ⚪ SCOPED-OUT |
| N9 | [metalgemm](metalgemm.md) | For random W[out,in] and X[P,in], the Metal f16 GEMM Y=X·Wᵀ (MPSMatrixMultiplication / runtime-compiled MSL) equals th | `TestMatMulMatchesReference`, `TestResetReclaimsTable` … | ✅ PROVEN |
|  | ↳ | The stub (default) build and the metal build of package metalgemm present the same exported interface, and the stub in | `TestStubInterfaceParity_CompilesAndScalarContract` | ✅ PROVEN |
| N10 | [tokenizer](tokenizer.md) | For text x drawn from the lossless ByteLevel domain, Decode(Encode(x)) == x byte-identically; and for any id list, Dec | `TestEncodeSmallByteLevelBPEFixture`, `TestOptionalSmolLM2EncodeMatchesHFCorpus` … | ✅ PROVEN |
|  | ↳ | BPE merges apply in deterministic merge-rank order: at each step the adjacent symbol pair with the SMALLEST tokenizer. | `TestEncodeSmallByteLevelBPEFixture`, `TestOptionalSmolLM2EncodeMatchesHFCorpus` … | ✅ PROVEN |
|  | ↳ | Tokenization matches the independent HuggingFace / llama.cpp reference: Encode(text) equals the reference token ids by | `TestQwenOracleGolden`, `TestOptionalQwen36ChatMLPromptMatchesLlamaCpp` … | ✅ PROVEN |
| N11 | [ggufload](ggufload.md) | For every tensor in a parsed GGUF, FileOffset = align(headerEnd) + declared Offset (with Offset required to be alignme | `TestReadParsesMetadataTensorDirectoryAndConfig`, `TestWeightSourceReadsAndDequantizesSimpleTensors` | ✅ PROVEN |
|  | ↳ | For each supported dtype (F32, F16, BF16, Q8_0, Q4_K, Q6_K, Q5_0, Q5_1, Q5_K, Q2_K, Q3_K), dequantF32 reads the correc | `TestWeightSourceReadsAndDequantizesSimpleTensors` | ✅ PROVEN |
|  | ↳ | Parsing GGUF metadata is consistent with the encoding: for every metadata value type (string, uint8/16/32/64, int8/16/ | `TestReadParsesMetadataTensorDirectoryAndConfig`, `TestReadRejectsBadAlignmentAndBool` | ✅ PROVEN |

### A — Algebraic / structural

| Obl | Module | Theorem | Witness | Verdict |
|---|---|---|---|---|
| A1 | [radixkv](radixkv.md) | For any new request whose token sequence shares a prefix with a cached sequence, the tree discovers the LONGEST cached | `TestReuseThroughSplitMatchesRecompute`, `TestFewShotHitRate` … | ✅ PROVEN |
|  | ↳ | Reference counts are conserved: insert/clone(split)/evict leave no dangling node (a removed node is unreachable, never | `TestRefcountConservationCycleNetsZero, TestRemovedNodeUnreachableNeverMatched` | ✅ PROVEN |
|  | ↳ | Under a token budget (maxTokens>0), when tokens exceed budget the tree evicts the least-recently-used unlocked LEAF (a | `TestLRUEvictsOldestRetainsHotAndLeased, TestLRUUpwardCollapse` | ✅ PROVEN |
| A2 | [kvmmu](kvmmu.md) | For any sequence of segment appends and span evictions, the mapping from live logical positions to physical KV cache s | `TestLedgerRenumberAfterMiddleEvict`, `TestWriteTimeEvictEqualsNeverSaw` | ✅ PROVEN |
|  | ↳ | A named span maps to exactly its bytes: evicting segment by logical id removes exactly its recorded [From,From+Len) ca | `TestLedgerRenumberAfterMiddleEvict`, `TestEvictionIsContentDrivenNotPositional` … | ✅ PROVEN |
| A3 | [cachemeta](cachemeta.md) | For a fixed Entry's binding axes, the cache key (ManifestBindingDigest / AttentionIndexBindingDigest) is a determinist | `TestManifestBindingDigestIsDeterministicOverBindingAxes`, `TestAttentionIndexDigestIncludesIndexShareLayerSet` | ✅ PROVEN |
|  | ↳ | Lowering an event into an Entry and reading it back preserves the binding: FromKVTransfer->KVTransferVerdict recovers  | `TestFromKVTransferRecordsResidencyTransition`, `TestKVTransferRestoreFaultIsTypedNotSilent` … | ✅ PROVEN |
|  | ↳ | Collision and eviction are well-defined: distinct bindings produce distinct digests (no false hit), a lookup whose axe | `TestAttentionIndexLookupRequiresPrefixDecisionAndCausality`, `TestCheckResidentClaimRefusesBindingMismatch` … | ✅ PROVEN |
| A4 | [recall](recall.md) | For a completed session persisted as a core image, a query against the reloaded session returns the SAME answer as rep | `TestBenignPageRoundTripsByteIdentical`, `TestSessionIsSelfContained` … | ✅ PROVEN |
|  | ↳ | Recall is deterministic and input-driven: for fixed (loaded session, query, k) the assembled working set — its members | `TestRecallIsDeterministicAcrossRepeatedCalls, TestRecallIsIdenticalAcrossIndependentReloads, TestRecallDependsOnlyOnQueryAndK, TestRecallWorkingSetNeverContainsQuarantinedBytes` | ✅ PROVEN |
| A5 | [contextq](contextq.md) | On-demand materialization (contextq.Query) reconstructs a context byte-identical to the original CDB image: every mate | `TestMaterializeByteIdentical` | ✅ PROVEN |
|  | ↳ | contextq.Query is deterministic: for a fixed CDB image and a fixed Request, repeated calls produce an equal Result (sa | `TestMaterializationDeterministic` | ✅ PROVEN |
| A6 | [blob](blob.md) | For every payload b, Resolve(Put(b)) returns bytes byte-identical to b — for both the inline path (len(b) <= InlineMax | `TestPutSmallInlineRoundTrip`, `TestPutLargeBlobRoundTrip` … | ✅ PROVEN |
|  | ↳ | Two Puts of byte-identical content (even from distinct backing arrays) produce the same digest (the address IS the sha | `TestContentDedup`, `TestDefaultStoreViaABI` | ✅ PROVEN |
| A7 | [preflight](preflight.md) | For every ToolCall, the preflight ladder evaluates rung 0 (static JSON parse) strictly before rung 1 (schema validatio | `TestRung0FailureNeverReachesRung1`, `TestNegativesRowFields` | ✅ PROVEN |
|  | ↳ | A Deny at a cheaper rung short-circuits all more-expensive rungs: when rung 0 denies, rung 1 (JSON-Schema validation)  | `TestRung0FailureNeverReachesRung1` | ✅ PROVEN |
| A8 | [abi+architest](abi+architest.md) | Every internal package that folds a verdict chain most-restrictive-wins (kernel, kvmmu, recall, agent) orders that fol | `TestFoldSitesOrderByFoldRank`, `TestFoldRankOrdering` | ✅ PROVEN |
|  | ↳ | The internal package graph is a layered DAG: Go forbids import cycles (acyclicity), and the architest tier rule forbid | `TestNoUpwardImports`, `TestHotPathHasNoExec` … | ✅ PROVEN |
|  | ↳ | The closed-enum wire contract of the frozen wave-0 ABI (VerdictKind, Status, Outcome, TaintLabel, ShareScope, RefKind, | `TestABIGoldenFreeze`, `TestClosedReasonVocabulary` | ✅ PROVEN |
| I1 | [engine-seam](engine-seam.md) | For a fixed registered engine, EngineDriver.Complete is deterministic in (tool, args): the same request yields the sam | `TestDecodeIsDeterministicAndInputDriven`, `TestCompleteRunsRealDecode` … | ✅ PROVEN |
|  | ↳ | An external invalidation directive set is bound to the correct documented engine reset endpoint and never silently ser | `TestInvalidateSGLangFlushesRadixCache`, `TestInvalidateVLLMResetsPrefixCache` … | ✅ PROVEN |
|  | ↳ | After Invalidate succeeds against a live serving engine, a subsequent request for the invalidated prefix/span observes | `TestEndToEndNotServedStaleSGLang` | ✅ PROVEN |
| I2 | [bench-ab-isolation](bench-ab-isolation.md) | An A/B ablation is a PAIRED replay over the SAME seed/trace: each arm runs the identical call sequence through one fre | `TestFanoutDeterministic`, `TestStochastic_Determinism` … | ✅ PROVEN |
|  | ↳ | The turn-tax isolation attributes the measured A/B delta to the TOGGLED axis, not to noise: the on-arm-minus-off-arm t | `TestRun_VDSOAblationIsARealPathSwap`, `TestRun_HappyPathSavesNothing` … | ✅ PROVEN |

### C — Crypto / integrity

| Obl | Module | Theorem | Witness | Verdict |
|---|---|---|---|---|
| C1 | [journal](journal.md) | For a file-backed journal, every committed Row N carries PrevHash = (hash of row N-1) and Hash = sha256(PrevHash ‖ con | `TestVerifyDetectsTampering`, `TestFileJournalReopensAndContinuesChain` … | ✅ PROVEN |
|  | ↳ | For a file-backed journal, each Emit that produces an audit Row flushes that row's bytes to the OS file before returni | `TestPerWriteDurableFlush_VerifyWithoutCloseRecoversEveryEmittedRow` | ✅ PROVEN |
| C2 | [deletioncert](deletioncert.md) | A DeletionCertificate cryptographically binds, under one ed25519 detached signature, the eviction count (EvictedCount) | `TestTamperDetected`, `TestNonBitExactRejected` … | ✅ PROVEN |
|  | ↳ | Verify(Mint(priv, c)) accepts a genuine certificate (Valid=true with every rung green: SignatureOK, AnchorOK, AnchorBo | `TestMintVerifyRoundTrip`, `TestTamperDetected` … | ✅ PROVEN |
| C3 | [provenance](provenance.md) | For every ToolCall c and Result r, a model-supplied Meta["provenance"]="trusted_local" tag does NOT raise the provenan | `TestModelCannotAuthorTrust`, `TestTaintBySource` … | ✅ PROVEN |
|  | ↳ | The kernel-authored provenance label is the single definition the trust gates consult: the ifc information-flow gate a | `TestTaintBySource`, `TestKernelStampedResultState` … | ✅ PROVEN |
| C4 | [ifc](ifc.md) | The abi.Ref.Taint join (implemented as taintRank-max via Ledger.Raise) is a join-semilattice: monotone (a Raise never  | `TestTaintJoinLedgerRealizesSpec,TestTaintJoinIdentity,TestTaintJoinMonotone,TestTaintJoinIdempotent,TestTaintJoinCommutative,TestTaintJoinAssociative` | ✅ PROVEN |
|  | ↳ | Non-interference: tainted (untrusted-derived) data is barred from a sensitive sink unless explicitly declassified — an | `TestParaphrasedExfilBlockedByProvenance`, `TestForgedSelfTrustCannotEvadeTaint` … | ✅ PROVEN |

### D — Decision-procedure soundness

| Obl | Module | Theorem | Witness | Verdict |
|---|---|---|---|---|
| D1 | [adjudicator](adjudicator.md) | Verdicts compose in abi.FoldRank order and the composition is fail-closed: the zero/absent value (empty Policy, unmatc | `TestEmptyPolicyDefaultDeny`, `TestDefaultPolicyUnknownToolDefaultDeny` … | ✅ PROVEN |
|  | ↳ | The adjudicator is a reference monitor that mediates every tool-call decision: it self-registers into the defconfig ad | `TestRequestPathLeavesRegistered`, `TestDefaultAllowsAllowedTool` … | ✅ PROVEN |
| D2 | [policy](policy.md) | The capability floor loaded from a manifest is SOUND: the effective deny set of the resolved adjudicator.Policy is a s | `TestLoadedPolicyIsLoadBearing`, `TestEmptyManifestIsFailClosed` … | ✅ PROVEN |
|  | ↳ | Policy load is deterministic and round-trips: for any Policy p built from a manifest, FromPolicy(p).ToPolicy() == p, a | `TestRoundTrip`, `TestParseFromDumpBytes` … | ✅ PROVEN |
| D3 | [ctxmmu](ctxmmu.md) | For any result r whose Payload is already a ctxmmu page-out/quarantine stub (the JSON {"_quarantined":true,...} or {"_ | `TestProofPageOutIdempotent` | ✅ PROVEN |
|  | ↳ | For any clean result body (no secret pattern, no injection marker, no degenerate repeat, len <= OversizeBytes=4096), A | `TestProofBenignByteIdentical` | ✅ PROVEN |
| D4 | [normgate](normgate.md) | For a benign result (no secret shape, no injection marker on the canonical view), normgate.Admit returns VerdictDefer  | `TestBenignPageRoundTripsByteIdentical` | ✅ PROVEN |
|  | ↳ | Detection over the canonical view is a SUPERSET of the raw-substring detection: for every body the legacy raw gate (ct | `TestCanonInjectionSupersetOfRaw_Quick` | ✅ PROVEN |
|  | ↳ | HONEST LIMIT (stated boundary, not a catch): a pure-semantic paraphrase of an injection that contains NO lexical marke | `TestParaphraseEvadesByDesign` | ✅ PROVEN |
| D5 | [canon](canon.md) | For all input strings x, Normalize(Normalize(x)) == Normalize(x): Normalize is a projection onto a canonical normal fo | `TestNormalizeIdempotent_Deterministic` | ✅ PROVEN |
|  | ↳ | The canonical view folds each obfuscation class it claims — homoglyph (Cyrillic/Greek look-alikes), fullwidth ASCII, z | `TestObfuscatedInjectionCaught`, `TestNormalizeUndoesObfuscation` | ✅ PROVEN |
| D6 | [plancfi](plancfi.md) | For every ToolCall c with a declared plan on c.TraceID, Adjudicate raises NO objection (returns VerdictDefer) ONLY if  | `TestDeviationEscalates`, `TestStrictModeDenies` … | ✅ PROVEN |
|  | ↳ | The plan automaton is deterministic: the verdict and the next state (Sequence pos) are a pure function of (declared pl | `TestSequenceMode` | ✅ PROVEN |
| D7 | [grammar](grammar.md) | For a tool with a loaded grammar of N required params, a malformed call carrying _positional with exactly N values (an | `TestAdjudicatePositionalRepairable`, `TestAdjudicatePositionalUnrepairable` | ✅ PROVEN |
|  | ↳ | For a tool with NO grammar registered, Adjudicate returns VerdictDefer (By="grammar") regardless of how malformed the  | `TestAdjudicateNoGrammarDefers` | ✅ PROVEN |
| D8 | [witness](witness.md) | A tool call whose adjudicated verdict is VerdictRequireWitness is denied (never dispatched) unless an independently-au | `TestAncestorClaim`, `TestCommittedAndGrep` … | ✅ PROVEN |
| D9 | [ratelimit](ratelimit.md) | For a key with MaxCalls=N (resp. MaxCost=B), the first N admitted calls (resp. calls whose cumulative cost stays ≤ B)  | `TestQuotaDeniesOverCap`, `TestCostBudgetDeniesOverBudget` … | ✅ PROVEN |
|  | ↳ | Accounting never double-spends or leaks credit: a call's cost is added to the per-key counter exactly when (and only w | `TestDeniedCallConsumesNoBudget`, `TestResetClearsBudget` … | ✅ PROVEN |
| D10 | [gateway](gateway.md) | For every (tool, args) pair, the verdict the gateway returns over the network seam (POST /v1/fak/{adjudicate,syscall}, | `TestServedAdjudicateEmitsDecisionEvents`, `TestHTTPSyscallDenyIsValueNot5xx` … | ✅ PROVEN |
|  | ↳ | A proposed tool-call whose folded verdict is anything other than ALLOW or TRANSFORM (DENY, QUARANTINE, REQUIRE_WITNESS | `TestAnthropicProxyResultTaintGatesProposedExfil`, `TestChatProxyResultTaintGatesProposedExfil` … | ✅ PROVEN |
| D11 | [steward](steward.md) | Each steward enforces exactly one checkable invariant predicate: a FuncSteward carries one Check func returning (viola | `TestSecretInContext`, `TestLeaseDisjointness` … | ✅ PROVEN |
|  | ↳ | The steward population is deterministic and order-independent where it claims to be: for a fixed set of stewards and a | `TestStewardSweepFiredSetOrderIndependent` | ✅ PROVEN |
| D12 | [shipgate](shipgate.md) | For any candidate measurement Witness w, Evaluate(w) returns KEEP iff w.improved() (a STRICT metric gain under w.Lower | `TestEvaluateKeepsStrictGain`, `TestEvaluateRevertsNoGain` … | ✅ PROVEN |
|  | ↳ | Evaluate is deterministic: for any fixed Witness w, repeated evaluations Evaluate(w) yield the identical (Decision, ke | `TestEvaluateDeterministicRepeat` | ✅ PROVEN |
| D13 | [vdso](vdso.md) | For a tier-1 pure tool, the vDSO fast-path result is identical to the direct pure recomputation (the ground-truth engi | `TestUnit38_SoundnessTier1EqualsRecompute` | ✅ PROVEN |
|  | ↳ | A tier-2 cache hit returns the same answer a fresh engine call would: the cache re-serves the exact engine-produced pa | `TestUnit26_27_Tier2CacheAndCanonicalization`, `TestUnit28_BumpWorldInvalidates` … | ✅ PROVEN |
|  | ↳ | The 3-tier lookup is deterministic: Lookup consults a FIXED order (tier-1 pure → tier-3 static → tier-2 cache), the ti | `TestUnit25_Tier1Pure`, `TestUnit29_Tier3Static` … | ✅ PROVEN |
|  | ↳ | The integrity (trust) epoch advances monotonically: a refutation (Revoke of a non-empty witness) strictly increases Tr | `TestProof_IntegrityEpochMonotonicSequence` | ✅ PROVEN |
| D14 | [metrics](metrics.md) | For Hist.pct(p)=sorted[int(p/100*(n-1))] over the ascending-sorted sample slice, the percentiles are monotonic non-dec | `TestHistPercentilesMonotonic` | ✅ PROVEN |
|  | ↳ | The A/B KPI fold is correct: the five KPI fields aggregate the paired On/Off Arm counters per their definitions (vdso_ | `TestReportJSONAndTokenDelta`, `TestValidateWorkloadHash` … | 🔸 OPEN |
| D15 | [gpulease](gpulease.md) | At any instant at most one process holds the gpulease advisory lease: while one Acquire's Lease is live, a second Acqu | `TestNoWaitBusyThenFree`, `TestWaitTimesOut` … | ✅ PROVEN |
|  | ↳ | Release is idempotent (a second Release, and a Release on a nil *Lease, are no-ops that do not panic), AND a lease hel | `TestReleaseIdempotent`, `TestReleaseOnProcessExit` | ✅ PROVEN |
<!-- LEDGER-END -->

## DOS binding — the proof section is itself witnessed

The math above proves the *modules*. "I wrote these proofs and ran their witnesses" is
itself a claim — so the proof section is grounded one level up by the
[DOS](00-METHOD.md#5-dos-as-the-meta-witness) trust substrate (`dos commit-audit` /
`dos review`, run from the fleet repo root). Every proof-shipping commit was audited; each
is **`diff-witnessed`** — its diff does the *kind* of thing its subject claims, so the
"done" rests on git evidence, not narration:

| Commit | Claim | DOS verdict |
|---|---|---|
| `e75c9c1` | proof-section spine (method + ledger) | ✅ `OK` · `diff-witnessed` (doc) |
| `68dc0a9` | 40 grounded per-module proofs + ledger | ✅ `OK` · `diff-witnessed` (doc) |
| `3cb8ff9` | deterministic witnesses closing 25 OPEN obligations | ✅ `OK` · `diff-witnessed` (**test** — 16 added test files) |
| `7126a6a` | ledger 94/97 PROVEN, fold in closures | ✅ `OK` · `diff-witnessed` (doc) |
| `b720af7` | N7 Qwen3.6 token-3 parity **REFUTED** + MLX-bar / swap measurement caveats | ✅ `OK` · `diff-witnessed` (doc) — audited this run |

`dos review` over the full proof range reports **`has_residual: false`, `cleared_rate: 1`**
— every checkable claim in the range was corroborated by its own diff; there is no residual
a human must re-verify. The recursion is closed: **the math proves the modules; DOS proves
the math actually shipped.** Neither layer trusts the author.

## Obligation roster

The proof obligations, grouped by regime. Each links to its proof file once landed.

### N — Numerical / linear-algebra
- **N1 · model/attention** — softmax row-stochastic & shift-invariant; causal mask strictly lower-triangular; attention-sink renormalization.
- **N2 · model/norm** — RMSNorm / LayerNorm match definition and are numerically stable.
- **N3 · model/rope** — RoPE preserves per-head norm; depends only on relative position; scaling variants match reference.
- **N4 · model/mlp** — SwiGLU/GeGLU functional form exact; residual is exact addition.
- **N5 · model/quant** — Q4_K / Q8_0 / AWQ dequant affine-correct; integer `SDOT` reduction bit-identical.
- **N6 · model/kv** — KV append correct; eviction equivalence `max|Δ|=0`; SWA window mask; prefix splice == recompute.
- **N7 · model/forward-parity** — end-to-end forward parity vs PyTorch oracle (cosine≈1, argmax, greedy ids) across families.
- **N8 · compute/gemm** — GEMM/HAL correctness + backend parity (CPU vs Metal vs CUDA).
- **N9 · metalgemm** — Metal MPS GEMM parity with CPU reference.
- **N10 · tokenizer** — BPE encode/decode round-trip; merge order deterministic; oracle parity.
- **N11 · ggufload** — tensor layout/stride + dtype dequant offsets correct.

### A — Algebraic / structural
- **A1 · radixkv** — longest-prefix reuse correct; ref-count conservation; LRU leaf eviction; reuse == recompute.
- **A2 · kvmmu** — KV page mapping bijection / span-exact addressing.
- **A3 · cachemeta** — cache-key / attention-index determinism + collision behavior.
- **A4 · recall** — query result == replay result (no false/missed recall).
- **A5 · contextq** — on-demand materialization fidelity (materialized == original).
- **A6 · blob** — content-address round-trip byte-identical; dedup by digest.
- **A7 · preflight** — cheapest-first ladder order = `FoldRank` monotone.
- **A8 · abi+architest** — frozen-spine fold-rank total order; package DAG (no upward imports).

### C — Crypto / integrity
- **C1 · journal** — append-only hash-chain tamper-evidence (mutation breaks the chain).
- **C2 · deletioncert** — certificate binds evict-count + journal anchor + epoch; `mint∘verify` round-trips; `max|Δ|=0`.
- **C3 · provenance** — kernel-authored trust is unforgeable (model cannot mint a "trusted" tag).
- **C4 · ifc** — taint lattice: monotone join; non-interference (tainted data never reaches a sink).

### D — Decision-procedure soundness
- **D1 · adjudicator** — verdict fold ordered by `FoldRank`; fail-closed (zero value = `Deny`).
- **D2 · policy** — capability-floor soundness: `deny ⊇ declared floor`.
- **D3 · ctxmmu** — page-out idempotent; benign result round-trips byte-identical.
- **D4 · normgate** — canonicalize-and-rescan: benign round-trip byte-identical; `detection ⊇ raw-regex`; paraphrase-evades-by-design (honest limit).
- **D5 · canon** — de-obfuscation canonical form idempotent (`canon∘canon = canon`).
- **D6 · plancfi** — plan-CFI automaton admits only approved transitions (soundness).
- **D7 · grammar** — repair transform preserves arity; fail-open when no grammar.
- **D8 · witness** — require-witness soundness.
- **D9 · ratelimit** — token-bucket rate bound + budget conservation; emits `RATE_LIMITED` at cap.
- **D10 · gateway** — wire adjudication == in-process verdict (no bypass); tool-call drop fail-closed.
- **D11 · steward** — each steward a single checkable invariant predicate.
- **D12 · shipgate** — keep-or-revert measure monotone (keep only on strict metric gain).
- **D13 · vdso** — 3-tier fast-path soundness (cached answer == engine answer).
- **D14 · metrics** — histogram percentile monotonicity; KPI fold correctness.
- **D15 · gpulease** — machine-wide mutual exclusion (≤1 holder); release idempotent. *(Gobra upgrade path — see method §6.)*

### I — Infra determinism (lighter, grouped)
- **I1 · engine/enginecache/modelengine** — driver-seam determinism + cache-invalidation correctness.
- **I2 · bench/turnbench/swebench/webbench** — A/B paired-replay isolation (same seed → same trajectory; only the toggled variable differs).
