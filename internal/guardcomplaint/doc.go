// Package guardcomplaint is the agent's APPEAL channel against the kernel — the
// subjective complement to the objective guard RSI loop (internal/guardrsi +
// internal/guardroute).
//
// THE GAP THIS CLOSES
// -------------------
// fak is "the disinterested referee": the capability floor refuses dangerous tool
// calls by structure, and every verdict lands in a hash-chained decision journal.
// internal/guardrsi folds that journal and catches the kernel's OWN honesty-holes
// (a blank reason on a DENY, an out-of-vocabulary verdict, a denial reason that
// recurs); internal/guardroute routes those to deduped issues. That loop is
// objective — it reads only what the journal can prove.
//
// But a referee with no appeal is unaccountable, and the journal has a blind spot
// it cannot fix on its own: a FALSE-POSITIVE refusal — a legitimate tool call the
// floor wrongly blocked — is byte-identical to a CORRECT refusal in the journal
// (both are a DENY with a valid closed-vocabulary reason). Only the agent that
// proposed the call knows it was legitimate. guardrsi can never surface that; the
// judgment is not in the bytes.
//
// This package is that missing channel. `fak complain` lets a governed agent file a
// structured appeal in its own words — "this DENY was a false positive, here is why
// the call was legitimate" — and attaches the WITNESSED verdict pulled from the
// journal (a self-report is not a witness; the journal row is). Appeals dedupe by a
// stable key (kind + appealed reason + refused tool + summary slug) into ONE issue
// that escalates an occurrence count, so a recurring false positive reads as the
// stronger signal it is. It reuses the gh issue-create/update plumbing in
// internal/dogfoodissues and the journal discovery in internal/guardrsi, adding only
// the appeal-specific decision layer.
//
// Safe by default: building the plan touches nothing external; only `--live` fetches
// existing issues and shells out to gh. The same idempotent-sink discipline
// dogfoodissues and guardroute already hold.
package guardcomplaint
