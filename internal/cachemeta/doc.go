// Package cachemeta defines the metadata contract for first-class cache entries.
//
// It intentionally stores no payloads and owns no cache. The package only names
// reusable objects, their validity/security/residency metadata, and typed lookup
// verdicts that callers can fold without collapsing every non-hit into "false".
//
// Tier: foundation (1) - see internal/architest. This package may import only
// packages whose tier is <= 1. Higher cache planes adapt their local objects down
// to these records instead of making this package depend on them.
package cachemeta
