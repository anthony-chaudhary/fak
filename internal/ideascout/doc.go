// Package ideascout searches configured research, repository, and community
// feeds — arXiv (papers), GitHub (repos), Hacker News and Reddit (real-time
// trending discussion, newest-first via each platform's public API) — dedupes
// candidates against a seen-cache and the existing issue backlog, scores what
// survives, and emits triage-ready idea-scout issue plans. Every source is a
// pure parse over wire bytes (ParseArxivAtom / ParseGitHubRepos /
// ParseHackerNewsJSON / ParseRedditJSON); only the LiveFetcher touches the
// network, and only --live files issues.
package ideascout
