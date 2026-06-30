// Package trendreport is the generic, consumer-agnostic substrate the fak
// trend-reports share: the durable-JSONL ledger plumbing (parse / latest-prior /
// append-line), the per-tick direction word, the embeddable control-pane Envelope,
// and the advisory gate whose only failing finding is the caller's *_unmeasured
// token.
//
// Three reports — internal/cadencereport (scores/maturity/work/releases),
// internal/milestonereport (the maturity CLIMB + epic ROADMAP), and the dojo board
// — independently re-declared the same machinery: a ParseLedger that tolerates
// blank/bad/empty-Date lines, a latestBefore that orders prior rows by
// (date, generated_at) and excludes a same-generated_at idempotent re-append, an
// AppendLedgerLine JSON marshaller, a directionWord sign-to-word helper, an
// envelope head of schema/ok/verdict/finding/reason/next_action + the ambient
// (workspace, commit, generated_at, date) stamp + two gate fields, and a CheckGate
// that fails ONLY when a dimension could not be measured. This package lifts that
// common shape into generic, parameterized helpers so a fourth report needs no
// copy-paste.
//
// It is a foundation-tier leaf: stdlib + generics only, importing nothing internal.
// Generic over the row type via the Row interface (Key() (date, generatedAt)), so
// each consumer keeps its own flat LedgerRow projection and supplies a one-line
// Key method.
//
// Migrating the existing consumers onto this substrate is the documented FOLLOW-ON
// of #1437; this package only provides the spine + proves it with tests. The live
// reports stay byte-identical until a later behavior-preserving switch-over wave.
package trendreport
