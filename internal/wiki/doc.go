// Package wiki is the fak-native, witness-verified repo-wiki core (epic #4277,
// mined from AsyncFuncAI/deepwiki-open @ 16f35a0 — inspire, clean-room Go).
//
// It holds the two deterministic halves the field structurally lacks:
//
//   - Structure (L1, #4278): a section→page tree projected straight from fak's
//     own self-index (internal/devindex) — leaves, lanes, docs — NOT re-inferred
//     by an LLM+embedding pass the way DeepWiki's <wiki_structure> call does. The
//     structure is ground truth fak already owns, so the wiki seeds from it.
//
//   - VerifyCitations (L3, #4280): every `Sources: [path:line]` code citation in
//     a generated page is resolved against the working tree — the file exists and
//     the line range is in bounds — or it is reported as a dangler. This is the
//     anti-hallucination guarantee DeepWiki cannot make: its prompt asks the LLM
//     to cite path:line but nothing ever resolves the cites against the tree.
//
// The package is a pure VIEW: Structure reads a *devindex.Catalog, VerifyCitations
// reads a markdown byte slice and stats files under a root. No LLM, no clock, no
// network — the two steps that make the wiki deterministic and verifiable.
//
// Tier: wiki is a tier-1 composer over devindex(1); see internal/architest.
package wiki
