// Package edittx applies a batch of full-file edits as one working-tree
// transaction: every target is snapshotted first, checks run against the applied
// set, and any failure restores the touched files before returning.
//
// Invariant: edit transactions are fail-closed and rollback-atomic. If any preflight
// path validation, snapshotting, write operation, or verification check command fails,
// all modified and newly created files are restored to their exact pre-transaction state.
//
// Guard: target paths must not escape the workspace root via parent directory traversals
// or symlinks.
//
// Contract:
//   - Apply performs full-tree snapshots of target paths before applying modifications.
//   - Check commands run against the applied set; non-zero exit codes abort and rollback.
//   - Rollbacks prune newly created empty parent directories under the transaction root.
//   - DryRun simulates path validation and returns the normalized edit plan without disk I/O.
package edittx
