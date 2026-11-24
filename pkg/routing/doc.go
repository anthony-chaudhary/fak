// Package routing is the multi-upstream routing fabric: the pure, deterministic
// seam that answers "where does this request or unit of work go?" separately
// from the control plane that answers "may it proceed, and did it?".
//
// Routing is a pure function of its inputs: no I/O, no learned policy, no
// randomness. The first eligible engine wins, ties break on Priority descending
// then declared order, and every skipped engine carries a closed reason, so a
// decision can be replayed. It never grants a lease, prices usage, or writes
// durable state; prices, fleet inventory, and admission come from the caller.
//
// It imports only the standard library and other pkg/* packages, so a
// standalone router can embed it without any companion module.
package routing
