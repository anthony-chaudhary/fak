# Bounded native GBNF character grammars

This is one source-level subsystem: parser and compile admission → pushdown
matching → partial-UTF-8 token admission → the normal native structured logit
mask. It is not a parser-only port or an external post-generation validator.

## Pinned donor and adaptations

Adapted from **ggml-org/llama.cpp**, immutable commit
`cd8427d018c07eefab6c110f7f115c31ec4c6157`:

- `src/llama-grammar.cpp`, Git blob
  `4e096a3684c63786fba76ed90b9b54a937677ee0`: character-rule parsing,
  synthesized right-recursive repetition rules, alternative stack expansion,
  Unicode character/range matching and partial-codepoint viability.
- `src/llama-grammar.h`, Git blob
  `2885f770597f0102b330a7db658f50aa3c4ca2d0`: rule and stack contract.
- `grammars/README.md`, Git blob
  `f005fc2522b84265172098e465c6dde1e4c328b1`: GBNF notation.
- `LICENSE`, Git blob `e7dca554bcb802f98408383a864404e3aa4eacca`.
  The complete MIT text is retained in `LLAMA-GBNF-LICENSE.txt`.

FAK adaptations are the Go representation, history-derived synchronized cursor
cache, strict Unicode scalar decoding, independent finite budgets, redacted
typed errors, and existing native request/mask/EOS integration. These are not
claims that the donor supplied FAK's resource or concurrency guarantees.

## Actual consumer and activation

The existing native `InKernelPlanner.newNativeRequestConstraint` accepts an
explicit string in `guided_grammar` or `structured_outputs.grammar`. The
request reaches the same ordinary buffered/per-token streaming and native
sampling path as the existing JSON/regex/choice masks. There is no new CLI
service, network endpoint, permission, persistent cache, environment variable,
or provider credential. Without a grammar carrier, no GBNF object is created.

Only one constraint kind is accepted. The two grammar aliases can coexist only
when their source strings are identical. A text response format may accompany
GBNF; other response formats and JSON/regex/choice combinations are refused.
For `structured_outputs.grammar`, true
`disable_any_whitespace`/`disable_additional_properties` is refused rather
than pretending the arbitrary grammar enforces those options. False/null
options do not change the grammar. Other existing carrier contracts remain.

The existing `nativeRegexTokenDecoder` hides special tokens. EOS uses its ID,
not its decoded spelling. The grammar covers the whole response, with EOS only
when an empty stack is reachable and no codepoint is incomplete. Returned token
bytes are treated as data, never instructions or tool authority.

## Supported character-rule syntax

- Required `root ::= ...`; named rules use ASCII letters, digits and hyphens.
- UTF-8 string literals, empty literals, references, groups and alternatives.
- Character classes, inclusive ranges, class negation and dot (Unicode scalar).
- `?`, `*`, `+`, `{m}`, `{m,n}`, `{m,}`.
- `#` comments; newlines separate top-level rules, and are permitted within
  groups or after an alternative separator.
- Escapes `\\t`, `\\r`, `\\n`, `\\\\`, `\\"`, `\\[`, `\\]`,
  `\\-`, `\\xNN`, `\\uNNNN`, `\\UNNNNNNNN`.

Surrogate scalar escapes, malformed UTF-8 and raw NUL source bytes are refused.
Escaped U+0000 is an ordinary scalar. Ranges exclude Unicode surrogates.
Duplicate/undefined or unproductive rules are rejected, including unreachable
invalid rules. Nullable-prefix cycles and left recursion are refused; ordinary
right recursion and balanced terminal-led recursion are supported within the
runtime stack bound. Repetition of an empty atom and unbounded nullable
repetition are refused.

Direct special-token forms (`<[id]>`, `<text>` and their negations), lazy
triggers, external grammar dialects, regex syntax inside GBNF, automatic
schema conversion and a selectable backend are not supported. This is a
bounded character-rule adaptation, not universal llama.cpp grammar equivalence.

## Finite work and ownership

Compile admission caps source at 64 KiB, rules at 2,048, stored symbols and
normalized ranges at 32,768 each, alternatives per rule at 256, parser nesting
at 64, counted repeat bounds at 256, and validation work at 8,388,608 units.
Character classes have at most 256 input ranges.

Matching independently caps stack depth at 256, active stacks at 256 and
closure states at 1,024. Each checked mask call has 8,388,608 charged work
units, including vocabulary initialization/history/candidate processing.
Vocabulary is at most 262,144 IDs; each token piece at most 4 KiB; copied
decoded-token storage at most 16 MiB. A call can hit its work limit before
those individual maxima. Generated history is at most 32,768 token IDs and
256 KiB of decoded bytes. These are logical byte/count/work limits, not a
measured memory peak or a strict wall-time guarantee. The tokenizer itself
must be immutable and obey its existing in-process decoder contract.

Compile state is immutable. A mutex protects copied decoded pieces and one
exact-history cursor cache. Appends resume from that cursor; rollback or a
different branch replays from the root. Native forks copy their own history,
clear scratch logits and share only this synchronized mask. No cursor is
implicitly advanced by speculative candidate probing. The existing current-main
`nativeStructuredConstraint.fork` contract is preserved.

Ambiguity/stack/work exhaustion returns an error instead of dropping
alternatives. `MaskLogitsChecked` masks all logits on error; the real native
consumer observes that error and stops before selection with
`ErrNativeStructuredNoToken`. The generic `model.LogitMask` adapter also
masks everything on failure, but an arbitrary consumer that samples an all-
minus-infinity vector is not thereby qualified. Existing negative-infinity
scores stay masked; grammar admission never increases scores.

## Witnesses and qualification debt

Authored synthetic witnesses cover finite-language byte-prefix oracle masks,
balanced recursion/repetition, Unicode splits and invalid continuations, EOS
by ID, special-token exclusion, malformed/left-recursive grammar refusals,
ambiguity/history/token budgets, shared concurrent histories, branch rollback,
and ordinary Complete/CompleteStream carrier wiring. The two historical valid-
grammar refusal cases evolve into real positive consumer witnesses; malformed
grammar and incompatible combinations remain pre-decode refusals. Existing
JSON/regex/choice and unsupported-dialect fixtures remain in place.

Suggested focused commands (not executed by this source delivery):

```sh
go test ./internal/grammar -run '^TestGBNF' -count=1
go test ./internal/agent -run '^(TestNativeGBNF|TestNativeGuidedCarrier|TestNativeStructuredOutputs|TestNativeGuidedJSON|TestNativeGuidedChoice)' -count=1
go test -race ./internal/grammar -run '^TestGBNFSharedMaskRollbackAndConcurrentHistories$' -count=1
```

Declared fixture runtime markers are estimates, not measurements. Compiler,
tests/race, installed controller, real model/tokenizer, hardware, latency,
throughput and memory measurements remain **UNRUN/UNMEASURED**. This unit adds no
new acceptance count until independently reviewed and natively delivered; it
does not turn source publication into main integration or FULL qualification.
