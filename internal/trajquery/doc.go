// Package trajquery lets an agent query its OWN trajectory corpus with a small SQL
// SELECT — and confines that query to an operator-defined scope by REWRITING it as a
// view, with a validator that proves the rewrite cannot leak rows outside the scope.
//
// # Query your own trajectory
//
// The corpus is the trajectory JSONL — one [Row] per turn/step/query. [Parse] accepts a
// deliberately small SQL subset (SELECT cols FROM rel WHERE f <op> lit AND ... LIMIT n;
// AND-only conjunction, no OR/joins/subqueries) and [Query.Execute] runs it over the
// rows. Small on purpose: a scope predicate ANDed into an AND-only WHERE can only narrow
// the result, never widen it, which is what makes the scope guarantee below decidable.
//
// # Scope by view rewrite
//
// An operator publishes a [View]: a named, scoped window onto the base relation — a scope
// predicate (e.g. session_id = 'S' AND redacted = 'true') plus a column allowlist. Agents
// query the VIEW, never the base relation. [View.Rewrite] inlines the view: it prepends
// the (non-removable) scope predicates to the user's WHERE and retargets FROM at the base
// — the textbook row-level-security rewrite of `SELECT … FROM v WHERE u` into
// `SELECT … FROM (SELECT * FROM base WHERE scope) WHERE u`.
//
// # The validator
//
// [View.Validate] is the load-bearing part: a rewrite you cannot trust is not security.
// It checks, statically, that the user query targets the view (not the base — a direct
// base query is the primary escape and is refused), that every scope predicate survives
// into the rewrite, and that neither the projection nor the WHERE references a column
// outside the allowlist (so a hidden column cannot be filtered-on to infer it). Then,
// dynamically over the supplied corpus, it confirms every row the rewrite returns
// satisfies the scope — a belt-and-suspenders check that the static argument held.
//
// Borrowed from the devindex (#1287) scoping model: an index you query is only safe if
// the scope is enforced by rewrite, not by asking the caller nicely.
package trajquery
