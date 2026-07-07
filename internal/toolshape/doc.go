// Package toolshape fingerprints the SHAPE of one tool call's input and output —
// the redaction-safe structural record the session-analytics rollup chain consumes
// (epic #2822: this is C1, the keystone leaf C2/C4/C5 build on).
//
// Tier: composer (3) — see internal/architest. It imports only
// internal/trajectory (tier 3) and stdlib; an upward import fails the architest
// gate. It is off the hot path and pure: Fingerprint is total and deterministic
// (no clock, no RNG, never panics).
//
// WHY SHAPE, NOT CONTENT. trajectory.Turn already carries ArgsDigest/ResultDigest —
// opaque content-identity hashes, exactly right for "was this the same payload"
// and exactly wrong for "what KIND of call was this". toolshape derives the
// STRUCTURE instead: which top-level arg names were present (never their values),
// a closed-vocabulary input class, log-scale output-size buckets, and error/
// truncation flags. No raw arg or result content is stored, hashed per-value, or
// logged — only the derived shape, so a ToolShape row is safe to export where a
// payload is not.
//
// PRIOR ART. The arg-name decomposition follows BFCL's AST argument analysis
// (ShishirPatil/gorilla — score a call by its argument structure, not its raw
// string); ArgKeySig follows the genson / Avro schema-fingerprint discipline: a
// stable digest of the sorted key-set + type signature, so two calls with the
// same arg schema collide and a schema drift (a key added) separates.
//
// HOW IT BINDS. It reads only trajectory.Turn's existing fields (Tool, Verdict,
// TokenEstimate, Bytes, ResultDigest, Labels) — no ABI change, no new Turn field.
// Producers that want input shape stamp the OPEN Labels channel with the
// well-known keys declared here (LabelArgKeys, LabelArgTypes, LabelTruncated,
// LabelError); a Turn without them degrades cleanly to ArgClass=unknown, empty
// ArgKeys, and cost-derived output shape — never an error, never a panic.
package toolshape
