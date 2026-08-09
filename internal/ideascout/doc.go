// Package ideascout searches configured research, repository, and community
// feeds — arXiv (papers), GitHub (repos), Hacker News and Reddit (real-time
// trending discussion, newest-first via each platform's public API) — dedupes
// candidates against a seen-cache and the existing issue backlog, scores what
// survives, and emits triage-ready idea-scout issue plans. Every source is a
// pure parse over wire bytes (ParseArxivAtom / ParseGitHubRepos /
// ParseHackerNewsJSON / ParseRedditJSON); only the LiveFetcher touches the
// network, and only --live files issues.
//
// The hard part of an UNATTENDED issue filer is not fetching — it is NOT
// spamming. Four dedup rungs gate every candidate before it can become an issue:
//
//  1. seen-cache   .idea-scout/seen.json — a node-local {source_id: record} of
//     every candidate this machine FILED. A pure fast path: the
//     cache is git-ignored, so it can be lost, and losing it must
//     not cost the guarantee. Rung 2 is what makes the promise.
//  2. filed-stamp  the candidate's source_id read back out of the
//     `<!-- idea-scout-source: … -->` stamp on EVERY issue the
//     scout has ever filed. THE DURABLE RUNG: a source filed once
//     is never filed again, even years later. Its index comes from
//     a query TARGETED at the idea-scout label
//     (FetchScoutIssues), not from a fixed-size window of recent
//     issues — so its completeness is a function of how many
//     issues the SCOUT has filed (capped at MaxIssues/day) and
//     never of how fast the tracker as a whole grows. GitHub is
//     the replicated store; nothing local is trusted.
//  3. issue-body   the candidate's source URL appears verbatim in some existing
//     issue body ⇒ a human already opened it by hand.
//     Best-effort: scanned over a recent-issue window
//     (IssueScanLimit), so it is a bonus catch, never the promise.
//  4. title-near   token-overlap (Jaccard) with any existing issue title ⇒ a
//     near-dup a human opened by hand. Same window as rung 3.
//
// Rung 2 is MANDATORY: if its index cannot be built — gh failed, or the label
// scan came back saturated at ScoutScanLimit and may therefore be truncated — Run
// REFUSES instead of filing blind. A dedup index that silently degrades as the
// tracker grows is exactly how already-triaged sources get re-filed (#5543,
// #5544: #528/#1266/#1267 aged out of an 800-issue window and came back as
// #5298/#5308/#5309), so growth has to trip a loud refusal, not a quiet re-file.
//
// GitHub is walked on two lanes from the same topic query: FetchGitHub (all-time
// stars, floored at MinStars) and FetchGitHubFresh (sorted most-recently-updated,
// floored at the lower FreshMinStars) so newly-created / trending / recently-pushed
// repos enter the pool instead of only established incumbents. Scoring then rewards
// a repo's star-velocity (trending) and how recently it was pushed.
package ideascout
