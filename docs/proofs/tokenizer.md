# N10 · tokenizer

The tokenizer leaf converts between UTF-8 text and the model's integer token ids for a HuggingFace fast `tokenizer.json` with a ByteLevel-BPE model (Qwen2.5 / Qwen3.6 / SmolLM2 / GPT-2 families). It loads the BPE vocab, the ordered merge-rank table, the added/special tokens, and a ByteLevel decoder (`ParseJSON`, `fak/internal/tokenizer/tokenizer.go:103`), then `Encode` (`:200`) splits text by the model-specific pre-tokenizer and greedily applies merges in rank order, while `Decode` (`:242`) inverts the ByteLevel byte↔unicode map back to the original bytes. "Correct" here is **regime N (numerical / exact integer path)**: the produced ids must be **bit-exact** to the HuggingFace / llama.cpp reference, the byte↔token map must be a **lossless bijection** (so `Decode∘Encode = id`), and the merge selection must be a **deterministic lowest-rank-first** function so the same text always yields the same ids. There is no float tolerance — every theorem here is checked by byte-identical / exact-id equality.

---

## Theorem 1 — encode/decode round-trips byte-identically (ByteLevel is lossless)

**THEOREM.** For text `x` in the lossless ByteLevel domain, `Decode(Encode(x)) == x` byte-for-byte; and for any id list `Decode` reconstructs the exact original bytes, because the ByteLevel byte→unicode map is a bijection over `0..255`.

**REGIME.** N (round-trip / involution, 00-METHOD.md §3.4).

**PROOF.** `Encode` (tokenizer.go:200) maps text→ids; `Decode` (tokenizer.go:242) maps each token's runes back to raw bytes via `byteLevelDecode[r]` (`:266-270`) and accumulates a `[]byte` across token boundaries before stringifying. The map is a bijection: `makeByteLevelDecode` (`:293`) assigns each of the 256 byte values a distinct rune and `makeByteLevelEncode` (`:326`) is its exact inverse, so `byteLevelEncode` (`:343`) and the per-rune `Decode` are inverse functions. Special tokens decode literally through `t.special` (`:260`). Hence the byte stream is reconstructed exactly, and a UTF-8 char split across two tokens reassembles (bytes accumulate before decode).

**WITNESS.** `go test -run 'TestEncodeSmallByteLevelBPEFixture|TestOptionalSmolLM2EncodeMatchesHFCorpus|TestDecodePreservesSplitUTF8Bytes|TestDecodeHFByteLevelFixtures' -v ./internal/tokenizer/ -count=1 -timeout 120s`
`TestEncodeSmallByteLevelBPEFixture` asserts `Decode(Encode(...))==` original (tokenizer_test.go:106-112); `TestOptionalSmolLM2EncodeMatchesHFCorpus` asserts the round-trip over 17 diverse cases (tokenizer_test.go:151-157); `TestDecodePreservesSplitUTF8Bytes` confirms split-UTF-8 reassembly.

**VERDICT.** PROVEN — 2026-06-20, macOS node (native go1.26 darwin/arm64). All `--- PASS`; SmolLM2 fixture present (run did not skip).

**DOS.** bound at ship.

---

## Theorem 2 — BPE merges apply in deterministic rank order

**THEOREM.** At each step the adjacent symbol pair with the **smallest** `tokenizer.json` merge rank is merged first; the same input always produces the same token ids.

**REGIME.** N (deterministic exact integer path; cf. 00-METHOD.md §3.5 determinism).

**PROOF.** `bpe` (tokenizer.go:351) seeds `bestRank = len(mergeRank)` (`:360`), scans every adjacent pair, and keeps the one with strictly smallest rank (`rank < bestRank`, `:366`), then merges all non-overlapping occurrences of that one pair (`:375-384`), looping until no pair has a rank. `mergeRank` is built once from the ordered `Merges` slice with `rank == index` (`:167-178`), i.e. the canonical tokenizer.json order. The selection is a pure function of `(symbols, mergeRank)` over an index-ordered loop with no map iteration in the decision, so it is deterministic. The exact-id goldens pin this: `'Hello'→9` and `'Ġworld'→18` in `TestEncodeSmallByteLevelBPEFixture` are reachable only if merges fire in rank order — a wrong order would change the ids. **Honesty caveat:** the witnesses assert exact ids (a wrong order breaks them) but there is no dedicated run-twice-compare stability test; determinism rests on the pure-function argument plus the re-run-stable goldens.

**WITNESS.** `go test -run 'TestEncodeSmallByteLevelBPEFixture|TestOptionalSmolLM2EncodeMatchesHFCorpus|TestQwenOracleGolden' -v ./internal/tokenizer/ -count=1 -timeout 120s`

**VERDICT.** PROVEN — 2026-06-20, macOS node. Exact-id assertions for the small fixture, the 17-case SmolLM2 corpus, and the 12-line Qwen2.5 llama.cpp golden all `--- PASS`.

**DOS.** bound at ship.

---

## Theorem 3 — tokenization matches the HF / llama.cpp reference (oracle parity)

**THEOREM.** `Encode(text)` equals the independent HuggingFace / llama.cpp reference token ids **byte-exactly** for the Qwen2.5, Qwen3.6, and SmolLM2 vocabularies over a diverse corpus.

**REGIME.** N (exact oracle parity, 00-METHOD.md §3.1 — integer path, bit-exact not tolerance-bounded).

**PROOF.** The witnesses compare fak `Encode` (tokenizer.go:200) against ids produced by a **separate engine** — `llama-tokenize` for Qwen (golden `testdata/qwen25_golden.tsv`, base64-text\\tids) and HF `tokenizers` for SmolLM2 — via `reflect.DeepEqual` (oracle_qwen_test.go:74; tokenizer_test.go:148). Parity holds because the pre-tokenizer is dispatched per model: `preTokenizeQwen` (tokenizer.go:526) implements the explicit Qwen `Split` regex alternatives and is chosen when `pre_tokenizer.hasSplit()` (`:180-185`), otherwise the GPT-2 ByteLevel default `preTokenizeByteLevel` (`:391`). The two families differ on whether a non-space/non-alnum char attaches to a following letter run; the wrong choice mis-tokenizes one family. The fix `c362d5e` added this dispatch plus the llama.cpp oracle gate. Both oracle tests SKIP (CI-safe) if the model cache is absent — here neither skipped.

**WITNESS.** `go test -run 'TestQwenOracleGolden|TestOptionalQwen36ChatMLPromptMatchesLlamaCpp|TestOptionalSmolLM2EncodeMatchesHFCorpus' -v ./internal/tokenizer/ -count=1 -timeout 120s`
`TestQwenOracleGolden` logged `llama.cpp oracle gate: 12 lines byte-exact (Qwen2.5)` against the real 7.0 MB qwen2.5 tokenizer.json; `TestOptionalQwen36ChatMLPromptMatchesLlamaCpp` matched the 21-id Qwen3.6 ChatML prompt (12.8 MB fixture); the SmolLM2 corpus matched HF reference ids.

**VERDICT.** PROVEN — 2026-06-20, macOS node. All `--- PASS`; all three oracle fixtures present so the run is a real parity pass, not a skip.

**DOS.** bound at ship.

---

### Reproduce

```bash
go test -run 'RoundTrip|Oracle|BPE|Merge|Determinis|Decode|Encode|Qwen|SmolLM2|StageT2|ParseJSON' ./internal/tokenizer/ -count=1 -timeout 120s
```
Native `go test` on the macOS fleet node (go1.26 darwin/arm64). Oracle fixtures live under `~/.cache/fak-models/tokenizers/{qwen2.5,qwen3.6}/tokenizer.json` and `~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/`; absent caches downgrade the oracle theorems to a SKIP (recorded as such), never a false PROVEN.
