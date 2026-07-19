// The ORPHANED-FOLLOWON reason token (issue #4957): a loop tick emitted downstream
// work — a durable [ArtifactIssue] baton pointer ("#1234", baton.go), or the issue an
// a2achan.WorkerStatus names — and NOBODY advanced or closed it within a cadence
// window. Progress at the loop's own grain, zero progress at the fleet grain.
//
// The token lives here because the emission being judged is relay vocabulary (the
// baton's durable Artifact pointers), and it joins the Reason* discipline
// (ReasonNoProgress, ReasonIdleParked, ReasonGoalDone, …) so a supervisor reads a
// checkable cause from the closed vocabulary, never free text. Verified, never
// claimed; fail-closed: the verdict that emits it (superloop.ClassifyFollowon) is
// assembled only from durable issue/artifact state, and an UNREADABLE emission never
// earns this token — an orphan is never fabricated from an absence, the same
// asymmetry ReadVerifiedProgress keeps for a missing ledger (ProgressUnknown).
package relay

// ReasonOrphanedFollowon is the closed relay reason token emitted when a member
// loop's emitted follow-on work (an ArtifactIssue / WorkerStatus issue ref) is OPEN
// with no advance within the cadence window: chase or close the emitted work through
// the member's own front door. Like RELAY_NO_PROGRESS it is operator-facing — the
// witness only surfaces the orphan; it never re-files or re-dispatches (#4958 owns
// the live binding and any redirect).
const ReasonOrphanedFollowon = "RELAY_ORPHANED_FOLLOWON"
