// Package benchauthority is the typed, in-binary source of truth for the PRIMARY
// benchmark NUMBERS fak claims — the "what" half of the benchmark discipline, the
// twin of internal/benchcatalog (which registers the benchmarks that PRODUCE the
// numbers, not the numbers themselves).
//
// Why this exists. BENCHMARK-AUTHORITY.md used to carry every headline number in a
// single hand-typed markdown table where each CELL was a wall of run-on prose —
// the number, its baseline, its provenance SHA, retraction history, and a dozen
// honesty fences all crammed into one table cell. That is unreadable for a human
// operator and unparseable for an agent, and it drifts: `_this commit_` placeholders
// that were never filled, artifact paths that lost a directory prefix, the same
// number maintained by hand in three denormalized places.
//
// The rest of the kernel already solved this: the support-maturity matrix, the
// industry scorecard, and benchcatalog are all GENERATED-not-typed and gated by a
// freshness test so the doc cannot silently drift from the structured source. This
// package brings the benchmark ledger onto that same footing:
//
//   - Each claim is one typed Claim literal in registry.go — the scannable data
//     (headline number, model, baseline, status, commit, artifact, reproduce) is
//     SEPARATED from the narrative (the Fences slice, rendered as bullets).
//   - Block() renders the deterministic ledger: a tight one-line-per-claim table a
//     reader can scan, followed by one compact dossier CARD per claim whose fences
//     are bullets, not a paragraph jammed in a cell.
//   - Splice()/Extract() write and read that block between BEGIN/END markers inside
//     BENCHMARK-AUTHORITY.md, so the surrounding hand-authored prose is untouched.
//   - Validate() enforces the CHECKABLE-MEASUREMENT invariant: a claim that asserts a
//     measured number must carry a structured {Metric, numeric Value, Unit, Baseline}
//     record — "specificity is the deliverable" (#4101) — so a perf number is a
//     resolvable record, never prose crammed into a headline. It is pure over the claim
//     data (no disk, no process state), so it is unit-testable in isolation, and it is
//     the record the commit-audit perf-claim rung is designed to point at. The
//     disk-and-catalog cross-checks are layered by capability-holding callers:
//     ValidateArtifacts(root) does the on-disk half (each Artifact path must exist,
//     catching the dropped-prefix drift the old doc admitted to) and delegates typed
//     benchmark-envelope validation to benchcli before enforcing authority promotion;
//     the "a named Bench is a registered benchcatalog surface" half stays with a
//     caller that imports that registry leaf.
//
// Adding or updating a number means editing one Claim literal and running
// `go run ./cmd/authorityledger -write` — the same additive-leaf discipline the
// benchcatalog registry uses. A stale committed doc reds the freshness test.
package benchauthority

import (
	"fmt"
	"strings"
)

// Status is the governance provenance word for a number — the coarse axis a reader
// needs first: is this a real measurement, an independently reproduced one, a
// model/geometry projection, a number deliberately withheld pending a witness, or a
// retired claim kept as a tombstone. It is the BENCHMARK-GOVERNANCE.md
// THEORETICAL|MEASURED|VERIFIED triad plus the two operational states the ledger
// must be able to show honestly.
type Status string

const (
	// Measured: real data plus real execution produced the number.
	Measured Status = "MEASURED"
	// Verified: MEASURED plus an independent reproduction / external grader.
	Verified Status = "VERIFIED"
	// Theoretical: a formula, deterministic geometry, simulation, or projection —
	// a floor or a plan, never a measured headline.
	Theoretical Status = "THEORETICAL"
	// Gated: no number is claimed yet; the claim records the boundary and the
	// missing witness that would unblock it.
	Gated Status = "GATED"
	// Pending: the result packet shape is committed but the run has not produced a
	// number (e.g. blocked on datacenter GPU access).
	Pending Status = "PENDING"
	// Retracted: a superseded / tombstoned claim, kept visible (never silently
	// deleted) with its replacement.
	Retracted Status = "RETRACTED"
)

// Claim is one primary benchmark number with its full provenance. The first block
// of fields is the SCANNABLE data that renders as one ledger row; Fences is the
// narrative that renders as card bullets.
type Claim struct {
	// ID is a stable kebab-case slug — the card anchor and the ledger's cross-link
	// target. Never renumber; it is a durable handle other docs may deep-link.
	ID string
	// Title is the short human name of the claim (the ledger's "Claim" cell).
	Title string
	// Headline is the ONE number, terse enough for a table cell (the anti-inflation
	// "one primary number per result" rule, enforced by shape). Put the exhaustive
	// decomposition in Fences, not here. Headline is the human one-liner ("60.3x TTFT
	// reduction"); the machine-checkable twin of that number is the {Metric, Value,
	// Unit} record below, which Validate() actually resolves.
	Headline string
	// Metric names WHAT was measured — the axis the number lives on ("pass@1", "TTFT",
	// "tok/s", "cache-hit-rate"). Required for a claim that asserts a measured number;
	// a Value with no Metric is an unlabelled figure. See Validate().
	Metric string
	// Value is the numeric magnitude, authored AS TEXT so a non-number that slipped in
	// ("≈60x", "a lot faster", "") is caught as prose rather than silently coerced.
	// Validate() parses it with strconv.ParseFloat and refuses a non-numeric, NaN, or
	// Inf value. This is the field that closes the "headline is prose" gap (#4101).
	Value string
	// Unit is the unit Value is expressed in ("x", "%", "ms", "tok/s", "s"). Required
	// alongside a Value; "60.3" with no unit is ambiguous. See Validate().
	Unit string
	// Status is the governance provenance word (see Status).
	Status Status
	// Provenance is the finer word when it adds signal (WITNESSED / OBSERVED /
	// MODELED / SIMULATED); "" when Status already says enough.
	Provenance string
	// Model names the model/geometry the number is about ("none" for model-free).
	Model string
	// Baseline is what the number compares against — never omit; a ratio without a
	// baseline is meaningless.
	Baseline string
	// Competitive marks an outward-facing product/runtime comparison: a win,
	// parity result, or honest loss against another implementation. Simulation may
	// rank candidates inside one calibrated envelope, but it can never populate one
	// of these rows; ValidateArtifacts enforces that boundary from the artifact's
	// typed simulation-evidence block rather than trusting headline prose.
	Competitive bool
	// Commit is the provenance SHA, or "artifact" when the SHA is not a public
	// reproduce handle (the reproduce handle is then Artifact + Reproduce).
	Commit string
	// Artifact is the repo-relative path to the committed evidence. Validate()
	// asserts it exists on disk.
	Artifact string
	// Reproduce is the copy-paste command that regenerates the number ("" if the
	// only handle is the artifact itself).
	Reproduce string
	// Bench is the benchcatalog key of the surface that produced this number ("" if
	// none). Validate() asserts a non-empty Bench is registered.
	Bench string
	// Section is an in-doc anchor (a "## Heading" slug) to the deep dossier for this
	// claim, when one exists below the ledger ("" if the card is the full record).
	Section string
	// Fences carry the honesty caveats — one bullet each. This is the content that
	// used to be a run-on paragraph in the table cell.
	Fences []string
	// Replacement, on a Retracted claim, is the ID (or short note) of the claim that
	// supersedes it.
	Replacement string
}

// Marker delimiters for the generated block inside BENCHMARK-AUTHORITY.md. Splice
// replaces everything between them; the surrounding hand-authored prose is left
// exactly as written.
const (
	Begin = "<!-- BEGIN GENERATED benchmark-ledger — edit internal/benchauthority/registry.go then `go run ./cmd/authorityledger -write` -->"
	End   = "<!-- END GENERATED benchmark-ledger -->"
)

// Ledger returns the registered claims in stable ledger order.
func Ledger() []Claim {
	out := make([]Claim, len(registry))
	copy(out, registry)
	return out
}

// Block renders the deterministic ledger body (WITHOUT the markers): a scannable
// one-row-per-claim table followed by one compact dossier card per claim. Two calls
// at one commit are byte-identical (no maps in the render path).
func Block() string {
	var b strings.Builder
	b.WriteString("_Generated from `internal/benchauthority` — the typed claim registry. ")
	b.WriteString("Each row's committed artifact is the exhaustive record; this ledger is the readable index. ")
	b.WriteString("Do not hand-edit between the markers; edit a `Claim` and run `go run ./cmd/authorityledger -write`._\n\n")

	// The scannable ledger — data only, no prose. This is the actual quick reference.
	b.WriteString("| # | Claim | Headline | Status | Model | Baseline |\n")
	b.WriteString("|---|---|---|---|---|---|\n")
	for i, c := range registry {
		fmt.Fprintf(&b, "| %d | [%s](#ledger-%s) | %s | %s | %s | %s |\n",
			i+1, mdCell(c.Title), c.ID, mdCell(c.Headline), statusBadge(c), mdCell(c.Model), mdCell(c.Baseline))
	}
	b.WriteString("\n---\n\n")

	// The dossier cards — the narrative, out of the table, one scannable block each.
	b.WriteString("### Claim dossiers\n\n")
	for i, c := range registry {
		renderCard(&b, i+1, c)
	}
	return strings.TrimRight(b.String(), "\n") + "\n"
}

func renderCard(b *strings.Builder, n int, c Claim) {
	fmt.Fprintf(b, "#### %d. %s <a id=\"ledger-%s\"></a>\n\n", n, c.Title, c.ID)
	fmt.Fprintf(b, "**%s** — %s\n\n", c.Headline, statusBadge(c))

	// One provenance line: model · baseline · commit · bench.
	meta := []string{
		"Model " + orNone(c.Model),
		"vs " + orNone(c.Baseline),
		"Commit `" + orNone(c.Commit) + "`",
	}
	if c.Bench != "" {
		meta = append(meta, "Bench `"+c.Bench+"`")
	}
	fmt.Fprintf(b, "%s\n\n", strings.Join(meta, " · "))

	// Artifact + reproduce — the verifiable anchor.
	fmt.Fprintf(b, "- Artifact: [`%s`](%s)\n", c.Artifact, c.Artifact)
	if c.Reproduce != "" {
		fmt.Fprintf(b, "- Reproduce: `%s`\n", c.Reproduce)
	}
	if c.Section != "" {
		fmt.Fprintf(b, "- Full dossier: [%s](#%s)\n", c.Section, c.Section)
	}
	if c.Status == Retracted && c.Replacement != "" {
		fmt.Fprintf(b, "- ⚠️ Retracted — superseded by: %s\n", c.Replacement)
	}
	b.WriteString("\n")

	if len(c.Fences) > 0 {
		b.WriteString("Fences:\n")
		for _, f := range c.Fences {
			fmt.Fprintf(b, "- %s\n", f)
		}
		b.WriteString("\n")
	}
}

// statusBadge renders the status (and finer provenance when present) as one token.
func statusBadge(c Claim) string {
	if c.Provenance != "" {
		return fmt.Sprintf("`%s` (%s)", c.Status, c.Provenance)
	}
	return fmt.Sprintf("`%s`", c.Status)
}

func orNone(s string) string {
	if strings.TrimSpace(s) == "" {
		return "none"
	}
	return s
}

// mdCell keeps a value safe inside a markdown table cell: no raw pipes or newlines.
func mdCell(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "|", "\\|")
	return strings.TrimSpace(s)
}

// Splice replaces the content between the BEGIN/END markers in doc with a freshly
// rendered Block. It errors if either marker is missing (never guesses where the
// block goes). The returned string equals doc when the block was already fresh.
func Splice(doc string) (string, error) {
	bi := strings.Index(doc, Begin)
	if bi < 0 {
		return "", fmt.Errorf("benchauthority: BEGIN marker not found in doc")
	}
	ei := strings.Index(doc, End)
	if ei < 0 {
		return "", fmt.Errorf("benchauthority: END marker not found in doc")
	}
	if ei < bi {
		return "", fmt.Errorf("benchauthority: END marker precedes BEGIN marker")
	}
	next := doc[:bi] + Begin + "\n\n" + Block() + "\n" + End + doc[ei+len(End):]
	return next, nil
}

// Extract returns the current block content (between the markers, trimmed to match
// Block's shape) and whether both markers were found. The freshness gate compares
// Extract(doc) against Block().
func Extract(doc string) (string, bool) {
	bi := strings.Index(doc, Begin)
	if bi < 0 {
		return "", false
	}
	rest := doc[bi+len(Begin):]
	ei := strings.Index(rest, End)
	if ei < 0 {
		return "", false
	}
	inner := rest[:ei]
	inner = strings.TrimPrefix(inner, "\n")
	inner = strings.TrimPrefix(inner, "\n")
	return strings.TrimRight(inner, "\n") + "\n", true
}
