// Package breath checks fak's "in one breath" prose contract — the named
// summary block every page under contract opens with, specified in
// docs/ONE-BREATH-CONTRACT.md.
//
// It gates the COUNTABLE half of that contract and refuses the judgement half. That
// split is the whole design, so it is stated here before the API.
//
// # What it gates
//
// Simplicity and brevity, which reduce to counting: the block is present, it sits in
// the lede before the first section heading, it is 2 to 4 sentences, each sentence is
// 15 words or fewer, it carries no em-dash and no parentheses, every ALLCAPS acronym
// is spelled out inside the block, and a `**One line:**` paragraph follows it. Each of
// those is a closed-vocabulary Kind (see Kinds), and each is a count a machine can
// take without an opinion.
//
// # What it refuses to gate, and why a future contributor must argue past this
//
// It says NOTHING about whether the block is TRUE, complete, or faithful to the page
// beneath it. That is not modesty and it is not a TODO. It is the measured scope
// decision, and reaching for the other half inverts the tool's own signal:
//
//   - The TREC PLABA track (arXiv 2507.14096, 2025) ran two years of expert manual
//     evaluation over plain-language adaptations, scoring four axes: accuracy,
//     completeness, simplicity, brevity. Top systems "rivaled human levels of factual
//     accuracy and completeness, but not simplicity or brevity". The axes a machine
//     already gets right are the ones this package cannot mechanically judge, and the
//     two it fails are exactly the two that reduce to counting words and sentences.
//
//   - The tempting extension is to compare the block against the page body and prove
//     the simple version follows from the precise one. PlainQAFact (arXiv 2503.08890,
//     2025) names why that fails: the "elaborative explanation phenomenon". Good
//     plain-language writing ADDS content absent from the source — a definition,
//     background, an example — to make it comprehensible, so an entailment- or
//     QA-style consistency check scores the BEST-written blocks as the LEAST faithful
//     ones. Their own fix classifies each sentence as source-simplified or elaborative
//     before scoring it, which needs a trained model and a judgement this package has
//     no business making. A checker that punished elaboration would push every block
//     toward a lossless summary of the page, which is the opposite of what the block
//     is for.
//
// So: a green run means short, well-placed, and free of the constructions that smuggle
// nuance past a reader who has none. Whether it is true is review's job. Two tests pin
// that boundary in code — TestNoJudgementHalfKind fails if a faithfulness/entailment/
// accuracy Kind is ever added to the closed vocabulary, and TestElaborativeBlockStaysClean
// is a block that deliberately says more than its page body and must stay green. Adding
// an entailment check means deleting both and arguing past this paragraph.
//
// Every report this package renders carries ScopeNotice verbatim, so a consumer reading
// only the tool's output still learns which half was not judged. A gate that silently
// implies it measured faithfulness is the failure mode this package exists to avoid.
//
// # Advisory with a counted ratchet
//
// fak's doc corpus predates the contract, so this is not a refusal. Findings are keyed
// by KIND<TAB>path — stable under editing — and a Baseline stores a COUNT per key, so
// fixing one of two findings tightens the floor on regeneration and adding a third is
// still caught. A baseline row that does not parse is a hard error naming the line
// number, never a skipped line: a lenient parser converts this package's bug into
// someone else's denial. The only claim a green ratchet makes is "this class is not
// growing" — never "the corpus is clean".
//
// The one unconditional refusal is BreathScanFloor. This package reports an ABSENCE, so
// a scan that examined zero pages prints "clean" and is byte-identical to a tree whose
// every block is perfect. A run below Contract.Floor therefore reports the floor breach
// against the gate itself and is never treated as a clean census.
//
// # Consumers
//
// The block is what a budget-aware loader serves when it cannot afford the page: epic
// #3229 (context/token budget) and #3535 (cut the ~10k-token turn-1 AGENTS.md forced
// read via a sectioned loader). The enforced ceiling is what makes a loader's budget
// arithmetic sound rather than hopeful.
//
// # Layering
//
// Pure promptlint subpackage, stdlib only, no filesystem walk of its own beyond the explicit Collect
// helper: the caller supplies documents as bytes so every rule is table-testable with
// no tree and no git. Wired at cmd/fak/breath.go (`fak breath`).
package breath
