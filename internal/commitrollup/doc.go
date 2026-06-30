// Package commitrollup plans compatible commit-intent batches without touching git.
//
// The live drain path is expected to supply already-validated commit intents. This
// package only answers the pure compatibility question: which intents can share
// one witnessed commit, which ones must bounce, and whether the final committed
// path set still equals the planned union.
package commitrollup
