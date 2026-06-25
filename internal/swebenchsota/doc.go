// Package swebenchsota emits a dated SWE-bench SOTA reference snapshot extracted
// from the official leaderboard at https://www.swebench.com/.
//
// The public leaderboard page embeds the full leaderboard as a JSON document in a
// <script type="application/json" id="leaderboard-data"> tag. This package splits
// the work into a PURE extraction half and a NETWORK fetch half so the fold is
// hermetically testable from a fixture HTML string:
//
//   - ExtractLeaderboard parses the script tag (regex + HTML-unescape) into the
//     list of leaderboard groups;
//   - GroupByName selects a named group ("Verified", "bash-only", ...);
//   - SummarizeGroup sorts a group's rows by resolved-percent (then date) and folds
//     the SOTA top row, a top-N window, and the focal rows matching a pattern;
//   - BuildSnapshot composes those into the versioned snapshot document
//     (schema "fak.swebench-sota-snapshot.v1").
//
// The network fetch (Fetch) and the local-file read (--html) are the only impure
// parts; both feed the same pure pipeline. This mirrors the structure of the Python
// tool it replaces (tools/swebench_sota_snapshot.py), whose test exercised exactly
// the pure path.
//
// Tier: mechanism (2) — a tool-shaped leaf that reads input (HTML source), folds it
// into a SOTA snapshot, and emits one envelope. Imports nothing internal; the stdlib
// net/http fetch is off the hot path.
package swebenchsota
