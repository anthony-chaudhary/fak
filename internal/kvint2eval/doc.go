// Package kvint2eval adjudicates bounded output-aware INT2 KV-cache rotation evidence.
//
// Invariant: INT2 KV rotation evaluations are fail-closed and tamper-evident.
// Guard: Modeled projection records must be dispatched to execution before being accepted as empirical proof.
// Precondition: Provenance pins and witness digests must match ground truth hardware observations exactly.
package kvint2eval
