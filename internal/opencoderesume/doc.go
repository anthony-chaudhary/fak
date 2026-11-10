// Package opencoderesume verifies OpenCode session identities before
// exact-resume launch, mirroring the fail-closed preflight pattern of
// internal/codexresume.
//
// # Why verification is mandatory
//
// `opencode run --session <id>` CREATES the session when the id does not
// exist. Launching against an unverified id therefore does not fail: it
// silently fabricates a fresh empty session and calls it recovery. Identity
// verification is the only gate between "resume this session" and "invent a
// new one", which is why the recovery cohort refuses every opencode row until
// a verdict of VerdictResumable is witnessed.
//
// # Project-scoped listing semantics
//
// The identity oracle is `opencode session list --format json -n <max>`
// executed with its working directory set to the candidate's project
// directory. The listing is scoped to the project that directory resolves
// to, so a hit proves BOTH that the id exists AND that the session belongs
// to this project/cwd. An id owned by another project is indistinguishable
// from a never-existed id and is refused the same way (VerdictNotFound).
//
// # Fail-closed verdicts
//
// Every verdict except VerdictResumable blocks launch. A missing binary, a
// nonzero exit, malformed output, or an ill-shaped id all refuse; this
// package never guesses. Session ids are `ses_`-prefixed mixed-case
// alphanumerics, not UUIDs, so the shape check is deliberately separate from
// codexresume's canonical UUID pattern and doubles as the injection defense
// for the id that flows into an exec argv.
package opencoderesume
