// Package commitsubject reports witness-gradeable commit subject coverage across git history.
//
// Invariant: commit subject folding is fail-closed and verb-checked.
// Non-conforming subjects lacking an imperative verb or conventional structure are rejected into abstain status.
//
// Guard: empty windows and exempt subjects yield zero totals with nil coverage without triggering panics.
package commitsubject
