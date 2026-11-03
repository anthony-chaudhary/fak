# Model-bound native text chat templates

This is one optional, bounded Go adaptation of the **llama.cpp Jinja chat-template
subsystem**, not a standalone parser or a Python interpreter. Its normal consumer
is `InKernelPlanner.preparePrompt`: template AST execution emits provenance-marked
segments, the existing `Tokenizer.EncodeSegments` encodes them, and the same
prepared tokens feed `EncodePrompt`, `Complete` and buffered/per-token
`CompleteStream`.

## Provenance and scope

Upstream: `ggml-org/llama.cpp` at
`cd8427d018c07eefab6c110f7f115c31ec4c6157` (MIT).

The port adapts the tag/whitespace lexer, expression and statement AST,
request-local execution scopes/macros/namespaces, and input-marked string
composition from `common/jinja/{lexer,parser,runtime,string,value}.cpp` and
`string.h`. The complete license is retained in `JINJA-LICENSE.txt`.
It does not import a tokenizer, model, Python installation, upstream executable,
or network behavior.

FAK adaptations are a closed finite language profile, independent source/model
binding, fixed resource limits, redacted refusal, and conservative provenance.
The normal native model/tokenizer, sampler, context admission, output projection,
tool policy and permission system remain in charge. This is not universal Jinja,
all llama.cpp templates, all model output protocols, or a model-quality claim.

## Explicit selection

Programmatic callers compile once with:

```go
selected, err := agent.NewNativeChatTemplate(source, trustedSHA256, exactModelID)
// Handle err; never substitute a fallback on an explicit refusal.
cfg.ChatTemplate = selected
```

The ordinary `fak serve` native local-model route accepts this **complete,
explicit CLI** trio (placeholder values below are not an executed launch):

```text
--native-chat-template-file /explicit/operator-selected/template.jinja
--native-chat-template-sha256 <independently-selected-lowercase-SHA-256>
--native-chat-template-model <exact-model-ID-matching---model>
```

Supply all three flags with nonempty values, or none. There is **no new
environment control**, implicit model-directory discovery, URL fetch, credential
lookup or automatic download. The selected route requires a local GGUF model,
no mock/proxy/replicas/delegated runtime, and the exact `--model` identity.
Flag-only inspection exits do not open the file. Actual startup compiles the
explicit regular file once, before model loading/listener binding; final
symlinks, changed identity/size, oversized content and hash mismatch refuse
startup without fallback. The existing `buildGateway` path binds this immutable
object to `InKernelPlannerConfig.ChatTemplate` and refuses a missing native
model/tokenizer. No model loading or serving is performed by this source change
itself.

Programmatic file callers may use `agent.LoadNativeChatTemplateFile(path, sha,
modelID)` and must handle errors before constructing a planner. The compiled
program retains no pathname and is never reloaded during requests. This is an
operator-owned file contract, not a hostile-same-user filesystem guarantee.

A source hash pins bytes; it is not producer authenticity. The exact selected
model ID must match the planner. If the model already carries an authenticated
embedded chat-template digest, that digest must agree as well. Empty/unknown
embedded metadata is not promoted to an upstream-version assertion.

Nil typed selection and absent CLI flags use the existing renderer unchanged.
Explicit bad file/CLI configuration returns typed `ErrNativeChatTemplate` at
startup; incompatible selected requests refuse before tokenization/decode. `RuntimeConfig().ChatTemplate` returns an immutable
object, and renderer identity contains its source digest.

## Admitted input and language

The selected path accepts bounded plain-text messages, function-tool definitions
and function-call arguments. Ordered JSON tool data is parsed as data, with
duplicate object keys, malformed JSON, excessive depth, integer overflow and
nonfinite numbers refused. Input strings never become expression source.
Roles, content, tool names, schemas, arguments and reasoning-content strings are
untrusted segments. Source literals and model-declared BOS/EOS spellings are
trusted. `safe` is a string conversion, **not a trust upgrade**.

The supported language includes:

- Text, expression and comment tags; explicit `-` whitespace control; donor
  chat defaults for newline normalization, one trailing LF, block indentation
  and post-block newline trimming.
- `if/elif/else`, `for` with an optional predicate and `else`, loop metadata,
  tuple unpacking, `break/continue`, assignments, namespace mutation, capture
  assignments, macros with defaults, and `generation` blocks.
- Typed scalar/list/dictionary literals, indexing and slices, attributes,
  boolean/comparison/arithmetic operators, and conditional expressions.
- Closed bounded builtins including `range`, `namespace`, `dict`,
  `raise_exception`, `default`, string/case/trim/replace/split/join,
  length/list/reverse/first/last, object items/keys/values/get/dictsort,
  startswith/endswith, select/reject/map-attribute, tests, and ordered `tojson`.

Only the implemented argument forms are admitted. For example, split requires an
explicit separator; replace accepts its all-occurrences form; JSON sort_keys=true
and chained comparisons are refused. Mixed integer/float comparisons whose integer
cannot be represented exactly in binary64 are refused rather than rounded. Non-scalar identity tests are unsupported.
Includes/imports/extends/call blocks/filter blocks/recursive-for, arbitrary
functions, filesystem/process/network access, clocks/randomness and reflection
are not available. Unsupported constructs fail closed rather than falling back
to legacy ChatML. This is a named finite profile, not quiet partial execution.

Raw-completion requests, selected SPM tokenizers, multimodal/structured message
carriers, thinking-signature/redacted-thinking carriers, legacy function-call
objects, incompatible pinned DeepSeek prompt profiles, DFlash2 renderer
contracts, and per-request chat_template_kwargs are refused on this selected
path. The separately implemented SPM raw-completion policy remains independent;
this port does not enable SPM chat. Existing request response-format/forced-tool
instructions remain request data prepended through the existing helper contract.

## Normal consumer behavior

Admission checks original input before prompt shrinking can discard an
unsupported representation. The existing shrink and context/output limits remain
effective. The selected renderer emits marked segments directly to the existing
BPE encoder, without flattening untrusted text into trusted added tokens. Empty
encoded prompts are refused.

Per-token streaming seeds the existing reasoning projector from the **actual
prepared prompt**, via a request-local internal observer. It does not render and
shrink the request a second time. Retry preparation must retain that same
rendered identity. Structured-output streaming keeps its existing dedicated
path. This is no change to the model's output/tool protocol.

Incremental-context rendering explicitly performs a full marked rebuild and
reports all layers rendered; it does not reuse the legacy string memo. The
legacy shared-system-prefix optimization is disabled for an explicit template
because that old renderer is not proof of this template's prefix. Ordinary
token-prefix KV caching still operates on the actual encoded token sequence.
No cache-sharing or latency improvement is asserted.

## Finite limits

- Source: 64 KiB; total lexical tokens: 32,768; AST nodes: 16,384.
- Parse/block/evaluation depth: 64; macro call depth: 32.
- Request input: 2 MiB of admitted scalar/JSON bytes; 4,096 messages,
  1,024 tools, at most 1,024 calls per message and 16,384 combined input items.
- Render work: 2,000,000 charged operations; output: 1 MiB.
- Cumulative charged string/value content: 16 MiB; charged items: 65,536.

These are logical source/input/output/interpreter budgets, **not** total process
RSS, exact Go allocator accounting, physical disk I/O, or a hard wall-time
guarantee. Cancellation is checked cooperatively during request-local work.
The contextless incremental-render API has finite work bounds but no caller
deadline. No partial result is returned after a refusal. Scopes and namespaces
are per render; the compiled program is immutable and shareable.

## Evidence and remaining qualification

The source fixtures cover a finite language oracle, marked-token provenance,
budgets/cancellation, independent concurrent renders, source/model/explicit-file-and-CLI
binding, default-unset byte identity, ordinary prompt tokenization, complete and
both stream entry paths, sink cancellation, ordered tool JSON and refusal before
sampling. Their runtime cost markers are declarations, not measurements.

Suggested focused receiver commands, **UNRUN**:

```sh
go test ./internal/agent -run 'TestNativeJinja|TestNativeChatTemplate' -count=1
go test ./cmd/fak -run '^TestServeNativeChatTemplate' -count=1
go test -race ./internal/agent -run 'TestNativeJinjaImmutableParallelRender' -count=1
```

Formatter/compiler/tests, installed serving behavior, full donor-template parity,
real-model output behavior, performance, FULL and main integration remain
unqualified until separately evidenced. One coherent renderer port is proposed;
individual parser/filter/test files are not separate subsystem credits.
