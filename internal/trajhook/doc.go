// Package trajhook is the PLUGGABLE TRAJECTORY-SCORER SEAM — the rung that lets a
// trivial application-layer skill garden trajectories (flag bad queries, find
// near-duplicate work, prune dead memory) WITHOUT a core edit to fak.
//
// Tier: composer (3) — see internal/architest. It scores internal/trajectory.Turn
// rows and uses internal/simhash for the near-duplicate scorer; it imports only
// those two leaves (both tier <= 3) and abi-free stdlib. It is entirely off the hot
// path: nothing here runs during adjudication.
//
// THE SEAM. internal/trajectory gives the DATA (per-turn rows, optionally embedded).
// trajhook gives the EXTENSION POINT over that data: a Scorer is a pure function
// Turn -> Finding (or, for cross-turn signals, a CorpusScorer over the whole slice).
// Application code registers a named Scorer into a Registry and runs it over an
// exported corpus — the same "register a driver, don't edit the core" discipline the
// kernel uses for abi.Emitter, but lifted to the analysis layer where no ABI is
// involved at all. A one-off script, a /loop skill, or a hosted control-plane job all
// attach the same way.
//
// WHY THIS IS THE RIGHT ALTITUDE. fak deliberately does NOT ship a learned
// "bad-trajectory classifier" — that is a semantic, application-specific judgment fak
// has no business hard-coding. What fak ships is the substrate that makes such a
// classifier a few lines to write: the typed data, the reference similarity
// primitive, and this registry. The three reference scorers below (duplicate-query,
// cost-outlier, deny-rate) are EXAMPLES, not policy — they prove the seam end to end
// and give a gardening skill something to call on day one, and a real deployment
// replaces or augments them with its own.
package trajhook
