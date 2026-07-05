// Package trendreport is the generic, consumer-agnostic ENVELOPE substrate the
// fak trend-reports share: the embeddable control-pane Envelope, the advisory
// gate whose only failing finding is the caller's *_unmeasured token, the
// per-tick direction word, and the JSONL append-line marshaller.
//
// Three reports — internal/cadencereport (scores/maturity/work/releases),
// internal/milestonereport (the maturity CLIMB + epic ROADMAP), and the dojo
// board — independently re-declared the same machinery: an envelope head of
// schema/ok/verdict/finding/reason/next_action + the ambient (workspace, commit,
// generated_at, date) stamp + two gate fields, a CheckGate that fails ONLY when
// a dimension could not be measured, a WithGate reconciler, an AppendLedgerLine
// JSON marshaller, and a directionWord sign-to-word helper. This package lifts
// that common shape into generic, parameterized helpers so a fourth report needs
// no copy-paste.
//
// The durable-ledger READ plumbing (the tolerant JSONL parse and the
// latest-prior-row scan) is deliberately NOT here: that seam lives in
// internal/jsonlledger (#2526), which the report packages already delegate to.
// trendreport owns the envelope tier above it; together the two substrates cover
// the whole trend-report spine (#1437).
//
// It is a foundation-tier leaf: stdlib + generics only, importing nothing
// internal.
package trendreport
