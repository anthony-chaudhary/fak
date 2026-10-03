// Package fakroot is the ONE canonical fak-repo-root discovery ladder, extracted
// from internal/selfupdate/cmd/compat.go so a second surface can answer "where is
// the public fak checkout?" without growing a second copy of the ladder.
//
// The ladder is five ordered rungs, unchanged from the original:
//  1. the enclosing git top level;
//  2. $FAK_ROOT;
//  3. the `use` entries of $cwd/go.work and <gitRoot>/go.work;
//  4. a child directory named "fak" under cwd or the git root;
//  5. the sibling ../fak.
//
// EVERY candidate on EVERY rung is proved by IsRepoRoot — a directory is a fak repo
// root only when `cmd/fak/main.go` exists on disk. That proof is the whole security
// story of this package: `fak-private` has no `cmd/fak` directory, so the ladder can
// never resolve to the private companion checkout. The widening is ASYMMETRIC on
// purpose — a private working tree can discover the public one, never the reverse.
//
// Purity: the ladder itself spawns nothing. The `git rev-parse --show-toplevel`
// subprocess is the CALLER's to run and pass in as Ladder.GitRoot ("" = not probed),
// because a rung-1 probe is a process launch and this leaf is reachable from the
// live request path. selfupdate/cmd keeps its exact previous behavior by supplying
// its exec-backed probe; a caller that must not spawn (the MCP read engine) skips
// rung 1 and loses nothing, because the git root of a fak checkout is that checkout.
//
// Tier: 1 (primitive). stdlib only, no internal imports, off the hot path.
package fakroot
