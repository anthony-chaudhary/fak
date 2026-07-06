package devindex

// verbStyleBaseline grandfathers the verb-catalog style defects that were present
// when the style gate (#2249) landed, keyed by "<verb>\t<kind>" (see
// VerbStyleViolation.key). It is FROZEN and may only SHRINK: a defect that gets
// fixed must have its entry removed (TestVerbStyleBaselineShrinksOnly refuses a
// stale entry), and a NEW verb or a NEW defect is never added here — it is fixed at
// the source. This is the pythongate-style ratchet the issue asks for: the gate
// reds on anything NOT in this set, so debt can only fall.
//
// Every grandfathered entry was an UNCATALOGED verb — a live cmd/fak dispatch case
// with no curated verbManifest entry, so Verbs() handed it the "not yet cataloged"
// fallback synopsis. The ratchet has now fully drained: every one of those verbs was
// curated in verbManifest (verbs.go), so the baseline is empty and the gate holds
// the whole catalog to the style bar with no grandfathered gaps. A new defect is
// fixed at the source, never re-added here.
var verbStyleBaseline = map[string]bool{}
