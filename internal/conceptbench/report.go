// report.go — the conceptbench leaderboard artifact (#2739, epic #2721). It
// turns graded rows (grade.go Verdicts, lifted into ReportRow) into
// fak.conceptbench.report.v1: a per-(model x concept) pass@1 fidelity matrix,
// a per-model rollup with fak-native columns, and — the load-bearing part — an
// honesty gate that computes result_claim_allowed rather than trusting a
// caller to set it. The gate mirrors #868: a headline claim is allowed ONLY
// when every reported row is a non-replay run backed by a real referee
// witness_source. A replay row is labeled and excluded from every headline
// claim (the winner highlight and the gate), so a scaffold can never
// masquerade as a measured result.
package conceptbench

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ReportSchema is the versioned schema id every report carries.
const ReportSchema = "fak.conceptbench.report.v1"

// WitnessReplay is the sentinel witness_source of a replay row — a re-emitted
// scaffold, never a referee reading. It is excluded from every headline claim.
const WitnessReplay = "replay"

// nonWitness is the closed set of witness_source strings that are NOT a real
// referee verdict. A headline row carrying any of these (or an empty source)
// leaves result_claim_allowed false — the exact anti-masquerade rule of #868.
var nonWitness = map[string]bool{
	"":            true,
	WitnessReplay: true,
	"none":        true,
	"scaffold":    true,
	"-":           true,
	"placeholder": true,
	"self_report": true,
}

// isRealWitness reports whether ws names a real referee reading (not a replay /
// scaffold sentinel and not empty).
func isRealWitness(ws string) bool {
	return !nonWitness[strings.ToLower(strings.TrimSpace(ws))]
}

// ReportRow is one graded episode lifted into the report: a (model x concept)
// verdict with its fidelity and the referee that produced it. Replay marks a
// re-emitted scaffold run — labeled and excluded from every headline claim.
// The trailing fak-native fields are optional per-episode telemetry folded into
// the per-model rollup; they are omitempty so a minimal row stays minimal.
type ReportRow struct {
	Model         string  `json:"model"`
	Concept       Concept `json:"concept"`
	Pass          bool    `json:"pass"`
	WitnessSource string  `json:"witness_source"`
	FidelityRate  float64 `json:"fidelity_rate"`
	Evidence      string  `json:"evidence"`
	Replay        bool    `json:"replay,omitempty"`

	// Per-run {model, arm, source} provenance resolved by the model-driver
	// registry (#2731, modelarm.go). SignalClass marks a run that hit a
	// usage/entitlement wall (the sessionsignals-derived class): such a row is
	// CLASSIFIED — excluded from headline scoring — never counted as a concept
	// failure. Distinct from NoCommitReason, which existing rows use for scored
	// guard refusals (e.g. OFF_TRUNK) that ARE genuine failures.
	Arm         string `json:"arm,omitempty"`
	ModelSource string `json:"model_source,omitempty"`
	SignalClass string `json:"signal_class,omitempty"`

	// fak-native per-episode signals (optional), rolled up per model.
	GuardRefused   bool    `json:"guard_refused,omitempty"`
	NoCommitReason string  `json:"no_commit_reason,omitempty"`
	TokensPerTurn  float64 `json:"tokens_per_turn,omitempty"`
	WallClockSec   float64 `json:"wall_clock_sec,omitempty"`
}

// headline reports whether this row can back a headline claim: a non-replay,
// non-walled run with a real referee witness_source. A SignalClass row is a
// classified wall (usage cap / auth / model_unknown), not a measured result.
func (r ReportRow) headline() bool {
	return !r.Replay && r.SignalClass == "" && isRealWitness(r.WitnessSource)
}

// Cell is one (model x concept) aggregate: pass@1 fidelity over the headline
// rows for that pair. Replay is true when the pair has only replay/unwitnessed
// rows (so no measured fidelity exists) — such a cell is never a winner. Winner
// marks the best measured cell for its concept.
type Cell struct {
	Model    string  `json:"model"`
	Concept  Concept `json:"concept"`
	Pass     int     `json:"pass"`
	Total    int     `json:"total"`
	Fidelity float64 `json:"fidelity"`
	Replay   bool    `json:"replay"`
	Winner   bool    `json:"winner"`
}

// ModelRollup is a per-model rollup: measured fidelity over headline rows plus
// the fak-native columns (guard-refusal rate, no-commit reason-class mix,
// tokens/turn, wall-clock) averaged over all of that model's rows.
type ModelRollup struct {
	Model            string         `json:"model"`
	Pass             int            `json:"pass"`
	Total            int            `json:"total"`
	Fidelity         float64        `json:"fidelity"`
	GuardRefusalRate float64        `json:"guard_refusal_rate"`
	NoCommitReasons  map[string]int `json:"no_commit_reasons,omitempty"`
	TokensPerTurn    float64        `json:"tokens_per_turn"`
	WallClockSec     float64        `json:"wall_clock_sec"`
}

// HonestyGate is the computed reasoning behind result_claim_allowed. It counts
// the headline rows, the labeled replay rows, and the headline rows that lack a
// real referee verdict, and states the reason a claim is (or is not) allowed.
type HonestyGate struct {
	HeadlineRows    int    `json:"headline_rows"`
	ReplayRows      int    `json:"replay_rows"`
	UnwitnessedRows int    `json:"unwitnessed_rows"`
	ClassifiedRows  int    `json:"classified_rows,omitempty"`
	Reason          string `json:"reason"`
}

// Report is fak.conceptbench.report.v1: the schema id, the caller-supplied
// generated timestamp (never time.Now, so a render is reproducible), the model
// and concept axes, the graded rows, the (model x concept) matrix cells, the
// per-model rollup, and the computed honesty gate. ResultClaimAllowed is
// derived from HonestyGate — a caller can never hand-set it.
type Report struct {
	Schema             string        `json:"schema"`
	Generated          string        `json:"generated"`
	Models             []string      `json:"models"`
	Concepts           []Concept     `json:"concepts"`
	Rows               []ReportRow   `json:"rows"`
	Cells              []Cell        `json:"cells"`
	Rollup             []ModelRollup `json:"rollup"`
	ResultClaimAllowed bool          `json:"result_claim_allowed"`
	HonestyGate        HonestyGate   `json:"honesty_gate"`
}

// BuildReport folds graded rows into a fak.conceptbench.report.v1 with a
// computed honesty gate. generated is the reproducible timestamp the caller
// supplies (RFC3339); rows are the graded episodes in any order. The returned
// report's ResultClaimAllowed is derived, never trusted from input.
func BuildReport(generated string, rows []ReportRow) Report {
	rep := Report{
		Schema:    ReportSchema,
		Generated: generated,
		Rows:      append([]ReportRow(nil), rows...),
	}
	rep.Models = uniqueModels(rows)
	rep.Concepts = orderConcepts(rows)
	rep.Cells = buildCells(rep.Models, rep.Concepts, rows)
	rep.Rollup = buildRollup(rep.Models, rows)
	rep.HonestyGate = gate(rows)
	rep.ResultClaimAllowed = rep.HonestyGate.HeadlineRows > 0 && rep.HonestyGate.UnwitnessedRows == 0
	return rep
}

// gate computes the honesty gate. A headline row is a non-replay run with a
// real referee witness_source; an unwitnessed row is a non-replay run that
// nonetheless lacks a real witness_source (the failure #868 warns against). A
// claim is allowed only when there is at least one headline row and no
// unwitnessed row.
func gate(rows []ReportRow) HonestyGate {
	g := HonestyGate{}
	for _, r := range rows {
		switch {
		case r.Replay:
			g.ReplayRows++
		case r.SignalClass != "":
			// A classified wall (usage cap / auth / model_unknown) is recorded,
			// never scored — it neither backs a claim nor refuses one.
			g.ClassifiedRows++
		case isRealWitness(r.WitnessSource):
			g.HeadlineRows++
		default:
			g.UnwitnessedRows++
		}
	}
	switch {
	case g.HeadlineRows == 0:
		g.Reason = fmt.Sprintf("no headline row (%d replay, %d unwitnessed) — result claim refused", g.ReplayRows, g.UnwitnessedRows)
	case g.UnwitnessedRows > 0:
		g.Reason = fmt.Sprintf("%d headline row(s) but %d lack a referee verdict — result claim refused", g.HeadlineRows, g.UnwitnessedRows)
	default:
		g.Reason = fmt.Sprintf("%d headline row(s), all referee-witnessed; %d replay row(s) labeled and excluded — result claim allowed", g.HeadlineRows, g.ReplayRows)
	}
	if g.ClassifiedRows > 0 {
		g.Reason += fmt.Sprintf(" (%d walled row(s) classified, not scored)", g.ClassifiedRows)
	}
	return g
}

func uniqueModels(rows []ReportRow) []string {
	seen := map[string]bool{}
	var out []string
	for _, r := range rows {
		if !seen[r.Model] {
			seen[r.Model] = true
			out = append(out, r.Model)
		}
	}
	sort.Strings(out)
	return out
}

// orderConcepts lists the concepts present, canonical graded set first (in the
// #2732 dispatch order), then any remaining concepts sorted — a deterministic,
// stable axis for the matrix.
func orderConcepts(rows []ReportRow) []Concept {
	present := map[Concept]bool{}
	for _, r := range rows {
		present[r.Concept] = true
	}
	var out []Concept
	for _, c := range Concepts() {
		if present[c] {
			out = append(out, c)
			delete(present, c)
		}
	}
	var rest []Concept
	for c := range present {
		rest = append(rest, c)
	}
	sort.Slice(rest, func(i, j int) bool { return rest[i] < rest[j] })
	return append(out, rest...)
}

// buildCells aggregates headline rows into per-(model x concept) fidelity and
// marks the best measured cell per concept as the winner. A pair with no
// headline rows (replay/unwitnessed only) is flagged Replay and never wins.
func buildCells(models []string, concepts []Concept, rows []ReportRow) []Cell {
	type key struct {
		m string
		c Concept
	}
	agg := map[key]*Cell{}
	has := map[key]bool{}
	for _, r := range rows {
		k := key{r.Model, r.Concept}
		has[k] = true
		c := agg[k]
		if c == nil {
			c = &Cell{Model: r.Model, Concept: r.Concept}
			agg[k] = c
		}
		if !r.headline() {
			continue
		}
		c.Total++
		if r.Pass {
			c.Pass++
		}
		c.Fidelity += r.FidelityRate
	}
	var cells []Cell
	for _, m := range models {
		for _, con := range concepts {
			k := key{m, con}
			if !has[k] {
				continue
			}
			c := agg[k]
			if c.Total > 0 {
				c.Fidelity /= float64(c.Total)
			} else {
				c.Replay = true
			}
			cells = append(cells, *c)
		}
	}
	markWinners(cells, concepts)
	return cells
}

// markWinners flags, per concept, the single best measured cell (highest
// fidelity; ties broken by the models' sorted order, which cells already
// follow). Replay-only cells are ignored.
func markWinners(cells []Cell, concepts []Concept) {
	for _, con := range concepts {
		best := -1
		for i := range cells {
			if cells[i].Concept != con || cells[i].Replay || cells[i].Total == 0 {
				continue
			}
			if best == -1 || cells[i].Fidelity > cells[best].Fidelity {
				best = i
			}
		}
		if best >= 0 {
			cells[best].Winner = true
		}
	}
}

func buildRollup(models []string, rows []ReportRow) []ModelRollup {
	byModel := map[string][]ReportRow{}
	for _, r := range rows {
		byModel[r.Model] = append(byModel[r.Model], r)
	}
	var out []ModelRollup
	for _, m := range models {
		rs := byModel[m]
		roll := ModelRollup{Model: m}
		var fidSum, tokSum, wallSum float64
		var guard int
		reasons := map[string]int{}
		for _, r := range rs {
			tokSum += r.TokensPerTurn
			wallSum += r.WallClockSec
			if r.GuardRefused {
				guard++
			}
			if strings.TrimSpace(r.NoCommitReason) != "" {
				reasons[r.NoCommitReason]++
			}
			if r.headline() {
				roll.Total++
				fidSum += r.FidelityRate
				if r.Pass {
					roll.Pass++
				}
			}
		}
		if roll.Total > 0 {
			roll.Fidelity = fidSum / float64(roll.Total)
		}
		if n := len(rs); n > 0 {
			roll.GuardRefusalRate = float64(guard) / float64(n)
			roll.TokensPerTurn = tokSum / float64(n)
			roll.WallClockSec = wallSum / float64(n)
		}
		if len(reasons) > 0 {
			roll.NoCommitReasons = reasons
		}
		out = append(out, roll)
	}
	return out
}

// JSON renders the report as indented JSON — the machine-readable artifact.
func (r Report) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// Markdown renders the report as a leaderboard: a model x concept fidelity
// matrix with the per-concept winner bolded and a per-model rollup, headed by
// the computed honesty gate. A replay-only cell renders "replay"; a
// model x concept pair with no episode renders "—".
func (r Report) Markdown() string {
	var b strings.Builder
	claim := "false"
	if r.ResultClaimAllowed {
		claim = "true"
	}
	fmt.Fprintf(&b, "# conceptbench leaderboard\n\n")
	fmt.Fprintf(&b, "- schema: %s\n", r.Schema)
	fmt.Fprintf(&b, "- generated: %s\n", r.Generated)
	fmt.Fprintf(&b, "- result_claim_allowed: %s\n", claim)
	fmt.Fprintf(&b, "- honesty_gate: %s\n\n", r.HonestyGate.Reason)

	cell := map[string]Cell{}
	for _, c := range r.Cells {
		cell[c.Model+"|"+string(c.Concept)] = c
	}
	rollup := map[string]ModelRollup{}
	for _, ro := range r.Rollup {
		rollup[ro.Model] = ro
	}

	// model x concept fidelity matrix.
	fmt.Fprintf(&b, "## model × concept fidelity (pass@1)\n\n")
	b.WriteString("| model |")
	for _, c := range r.Concepts {
		fmt.Fprintf(&b, " %s |", c)
	}
	b.WriteString(" rollup |\n")
	b.WriteString("| --- |")
	for range r.Concepts {
		b.WriteString(" --- |")
	}
	b.WriteString(" --- |\n")
	for _, m := range r.Models {
		fmt.Fprintf(&b, "| %s |", m)
		for _, c := range r.Concepts {
			b.WriteString(" " + fmtCell(cell[m+"|"+string(c)], hasCell(cell, m, c)) + " |")
		}
		fmt.Fprintf(&b, " %s |\n", fmtFidelity(rollup[m].Fidelity))
	}
	b.WriteString("\n")

	// per-concept winner line.
	b.WriteString("winner per concept: ")
	var wins []string
	for _, c := range r.Concepts {
		if w, ok := winnerFor(r.Cells, c); ok {
			wins = append(wins, fmt.Sprintf("%s → %s", c, w))
		} else {
			wins = append(wins, fmt.Sprintf("%s → (no measured result)", c))
		}
	}
	b.WriteString(strings.Join(wins, " · "))
	b.WriteString("\n\n")

	// per-model rollup with fak-native columns.
	fmt.Fprintf(&b, "## per-model rollup\n\n")
	b.WriteString("| model | fidelity | pass/total | guard_refusal_rate | no_commit_reasons | tokens/turn | wall_clock_s |\n")
	b.WriteString("| --- | --- | --- | --- | --- | --- | --- |\n")
	for _, m := range r.Models {
		ro := rollup[m]
		fmt.Fprintf(&b, "| %s | %s | %d/%d | %s | %s | %s | %s |\n",
			m, fmtFidelity(ro.Fidelity), ro.Pass, ro.Total,
			fmtFidelity(ro.GuardRefusalRate), fmtReasons(ro.NoCommitReasons),
			fmtNum(ro.TokensPerTurn), fmtNum(ro.WallClockSec))
	}
	return b.String()
}

func hasCell(cells map[string]Cell, m string, c Concept) bool {
	_, ok := cells[m+"|"+string(c)]
	return ok
}

func fmtCell(c Cell, present bool) string {
	if !present {
		return "—"
	}
	if c.Replay {
		return "replay"
	}
	s := fmtFidelity(c.Fidelity)
	if c.Winner {
		return "**" + s + "**"
	}
	return s
}

func winnerFor(cells []Cell, c Concept) (string, bool) {
	for _, cell := range cells {
		if cell.Concept == c && cell.Winner {
			return cell.Model, true
		}
	}
	return "", false
}

func fmtFidelity(f float64) string { return fmt.Sprintf("%.2f", f) }

func fmtNum(f float64) string {
	if f == 0 {
		return "—"
	}
	return fmt.Sprintf("%.1f", f)
}

func fmtReasons(m map[string]int) string {
	if len(m) == 0 {
		return "—"
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s×%d", k, m[k]))
	}
	return strings.Join(parts, ", ")
}
