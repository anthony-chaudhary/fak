# Explicit llama-SPM tokenizer profile
This is an explicit FAK JSON interchange for the llama.cpp scored adjacent-pair SentencePiece family, adapted from ggml-org/llama.cpp cd8427d018c07eefab6c110f7f115c31ec4c6157 (MIT). It is NOT Hugging Face Unigram/Viterbi, general SentencePiece, binary .model or embedded GGUF support. No automatic format conversion is provided.

Supply --tokenizer /path/to/tokenizer.json (or its directory) on the ordinary fak serve invocation. The selected JSON contains exactly six fields:
- fak_tokenizer_profile: "llama-spm-v1"
- normalization: "identity"
- tokens: ordered nonempty unique string pieces
- scores: matching finite float32 scores
- types: matching integer types 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte
- options: exactly bos_id, eos_id (integer IDs), add_bos, add_eos, add_space_prefix (booleans).
All fields are required. Unknown/duplicate/null fields, nonfinite or invalid scores, incomplete byte fallback and unsupported normalization refuse. Supply all 256 typed <0xHH> byte pieces. IDs must match the loaded model's actual BOS/EOS configuration; admission refuses mismatches rather than changing model stop identity. -1 special IDs are permitted by the core only when disabled, but ordinary raw serving requires concrete matching IDs. JSON scores are explicitly the llama scored-pair semantics, never Viterbi probabilities.

Example shape (vocabulary arrays abbreviated here and therefore NOT a loadable fixture):
{"fak_tokenizer_profile":"llama-spm-v1","normalization":"identity","tokens":["<unk>","<s>","</s>"],"scores":[0,0,0],"types":[2,3,3],"options":{"bos_id":1,"eos_id":2,"add_bos":true,"add_eos":false,"add_space_prefix":true}}
The source tests construct complete 256-byte fixtures and exercise the real LoadJSON and explicit serve file/directory helper.

Normal use is /v1/completions with one plain user prompt through WithRawPrompt. Selected SPM chat/templates, multiple/history messages, tools/tool fields, vision and structured carriers refuse. Generic Encode does not add BOS/EOS; the selected raw prompt path calls EncodeSPM(text,true,false) exactly once, adds configured special tokens, and preserves ordinary output/context/cancellation admission. USER_DEFINED pieces still match under the donor's parseSpecial=false policy; UNKNOWN/CONTROL spellings are not parsed there. EncodeSegments retains its stronger trusted-interval contract. Raw streaming seeds reasoning from the same raw text without tokenizing/preparing twice. Piece decoding preserves donor dummy-space bytes rather than silently stripping them.

Legacy BPE parser/encoding and embedded GGUF fallback are unchanged. The two previously proposed GGUF reader files and embedded dispatch are explicitly excluded from this delivery; their foreign ownership remains untouched.

Qualification pending: compiler/tests/model execution, tokenizer parity against real artifacts, quality, throughput, FULL and main integration are UNRUN/unqualified. Proposed focused commands, only on an admitted isolated qualified build:
go test ./internal/tokenizer -run 'Test(SPM|NativeSPM)' -count=1
go test ./internal/agent -run 'TestSPM' -count=1
go test ./cmd/fak -run '^TestExplicitServeSPMTokenizerFileAndDirectory$' -count=1
Return exact source/build identity, model/tokenizer digests and normalization policy, argv/stdout/stderr/exit, ordinary raw prompt token IDs including BOS/EOS, one-shot/stream decoded bytes and refusal outcomes. These CPU source fixtures are not a hardware request or parity result.

Malformed raw UTF-8 and unpaired UTF-16 escapes refuse; valid Unicode escape pairs and literal escaped backslashes remain accepted. Automatic model-directory tokenizer loading does not activate this profile: use explicit --tokenizer. Valid or invalid recognized native profiles on that automatic route fail closed rather than selecting MockPlanner; unrelated legacy BPE/embedded fallback remains unchanged.
