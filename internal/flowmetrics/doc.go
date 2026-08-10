// Package flowmetrics measures how work actually flows through this repo, so
// that "reduce WIP" becomes a number instead of a feeling.
//
// The model rests on one observation: this repo already emits every signal a
// flow metric needs, and nobody reads it. Ship commits carry an issue
// reference (`#5420`) and a lane stamp (`(fak agent)`), so git alone says when
// an issue's work STARTED and which lanes it touched. The tracker says when an
// issue was created and closed. Joining the two yields the three durations
// that matter, without asking a single agent to file a status update:
//
//	queue  = created      -> first commit   (how long it sat)
//	cycle  = first commit -> closed         (how long it took once begun)
//	lead   = created      -> closed         (queue + cycle)
//
// The load-bearing distinction is backlog versus in flight, and [Span.Started]
// is the whole of it. An open issue with no commit is BACKLOG; it costs nothing
// to hold. An open issue with a commit is IN FLIGHT; it holds context, blocks a
// lane, and rots. Counting the two together is what makes a 1300-issue backlog
// look like 1300 units of work in progress and makes the real WIP number —
// typically two orders of magnitude smaller — invisible. [WIPCurve] therefore
// counts only started spans as in flight.
//
// Three deliberate refusals, because each is a way this measurement could lie:
//
//   - Percent-complete is reported only when the issue body declares a
//     checklist. [Checklist.PercentComplete] returns known=false otherwise.
//     Commit count is NOT a progress proxy: a rising commit count on one
//     issue is as consistent with thrash as with progress, so inferring
//     completion from it would make the metric read best exactly when work is
//     going worst.
//
//   - Git-observed start is a lower bound on WIP, never an upper one. Work
//     that has begun but not yet committed is invisible to git, which is why
//     [TreeWIP] measures the uncommitted tree as its own class of WIP. In a
//     shared checkout that dark WIP is where the real cost hides.
//
//   - An issue's last commit does not end its life. Only the tracker's
//     closedAt does. [AgingWIP] therefore treats a started-but-unclosed issue
//     as in flight forever, which is the honest reading: quiet work is not
//     finished work.
package flowmetrics
