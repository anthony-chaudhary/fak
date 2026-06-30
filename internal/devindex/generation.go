package devindex

import (
	"sort"
	"strings"
)

// Generation is one horizon row in the generation-aware development contract.
// The stream/label/milestone/meaning fields are parsed from docs/generation.md
// when available; the operational query fields below are the stable issue-intake
// contract agents need when they ask the self-index instead of re-reading the
// whole epic.
type Generation struct {
	Stream                 string   `json:"stream"`
	Label                  string   `json:"label"`
	Milestone              string   `json:"milestone"`
	Meaning                string   `json:"meaning"`
	IssueBodySignals       []string `json:"issue_body_signals,omitempty"`
	PromotionEvidence      string   `json:"promotion_evidence,omitempty"`
	DemotionEvidence       string   `json:"demotion_evidence,omitempty"`
	InvalidatingAssumption string   `json:"invalidating_assumption,omitempty"`
}

var generationDefaults = []Generation{
	{
		Stream:            "now",
		Label:             "gen/now",
		Milestone:         "Generation G0 - Now / Immediate",
		Meaning:           "Improves the current product, operator loop, or trunk hygiene with a clear witness and no dependency on a future architecture bet.",
		PromotionEvidence: "Already in the current-product path; keep it now by shipping a focused witness and avoiding stale side work.",
		DemotionEvidence:  "Demote if the current-product witness disappears, a dependency becomes speculative, or a feature gate proves the work is not ready for default exposure.",
	},
	{
		Stream:            "next",
		Label:             "gen/next",
		Milestone:         "Generation G1 - Next Gen",
		Meaning:           "Near-term foundation that should be runnable by agents soon, but still needs a gate, dogfood run, schema, or default-exposure proof.",
		PromotionEvidence: "Promote toward now when a gate, dogfood run, schema contract, or default-exposure proof lands with a focused witness.",
		DemotionEvidence:  "Demote or park when the dogfood path is absent, the witness is stale, or the foundation no longer has a near-term caller.",
	},
	{
		Stream:            "second-next",
		Label:             "gen/second-next",
		Milestone:         "Generation G2 - Second Next Gen",
		Meaning:           "Architectural option that needs simulation, compatibility policy, or cross-generation dependency management before it can become active product work.",
		PromotionEvidence: "Promote when simulation, compatibility policy, or dependency edges prove the option can become next-gen foundation.",
		DemotionEvidence:  "Demote, retire, or park when compatibility fails, dependency cost exceeds value, or the option duplicates a stronger shipped path.",
	},
	{
		Stream:            "future",
		Label:             "gen/future",
		Milestone:         "Generation G3 - Future",
		Meaning:           "Research, market narrative, standards analogue, or long-horizon option that should stay visible without pretending it is on the current release train.",
		PromotionEvidence: "Promote when research yields a decision, scorecard, prototype, or option value that can be shaped into second-next or next work.",
		DemotionEvidence:  "Retire when the assumption fails, the option is superseded, or the carrying cost exceeds the expected value.",
	},
}

var generationIssueBodySignals = []string{
	"Generation stream",
	"matching gen/* label",
	"matching Generation G* milestone",
	"promotion evidence",
	"demotion/retirement evidence",
	"invalidating assumption",
}

const generationInvalidatingAssumption = "The stream label, milestone, project Generation field, and issue body remain cheap and consistent for agents to read; if they drift, classify first and dispatch second."

// defaultGenerations returns a fully decorated copy of the built-in generation
// index. It is used both as the installed-binary fallback and as decoration for
// rows parsed from docs/generation.md.
func defaultGenerations() []Generation {
	out := make([]Generation, len(generationDefaults))
	for i, g := range generationDefaults {
		out[i] = decorateGeneration(g)
	}
	return out
}

func decorateGeneration(g Generation) Generation {
	base, ok := defaultGenerationByStream(g.Stream)
	if ok {
		if g.Label == "" {
			g.Label = base.Label
		}
		if g.Milestone == "" {
			g.Milestone = base.Milestone
		}
		if g.Meaning == "" {
			g.Meaning = base.Meaning
		}
		if g.PromotionEvidence == "" {
			g.PromotionEvidence = base.PromotionEvidence
		}
		if g.DemotionEvidence == "" {
			g.DemotionEvidence = base.DemotionEvidence
		}
	}
	g.IssueBodySignals = append([]string(nil), generationIssueBodySignals...)
	if g.InvalidatingAssumption == "" {
		g.InvalidatingAssumption = generationInvalidatingAssumption
	}
	return g
}

func defaultGenerationByStream(stream string) (Generation, bool) {
	stream = strings.ToLower(strings.TrimSpace(stream))
	for _, g := range generationDefaults {
		if g.Stream == stream {
			return g, true
		}
	}
	return Generation{}, false
}

// parseGenerations reads the Streams table from docs/generation.md. The table is
// the human-maintained source for label/milestone/meaning; if the doc is absent
// or unparsable, the self-index still answers from the built-in contract.
func parseGenerations(text string) []Generation {
	var out []Generation
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if !strings.HasPrefix(line, "|") || strings.Contains(line, "---") {
			continue
		}
		cells := markdownTableCells(line)
		if len(cells) < 4 {
			continue
		}
		stream := strings.ToLower(strings.TrimSpace(cells[0]))
		if _, ok := defaultGenerationByStream(stream); !ok {
			continue
		}
		out = append(out, decorateGeneration(Generation{
			Stream:    stream,
			Label:     cleanMarkdownCell(cells[1]),
			Milestone: cleanMarkdownCell(cells[2]),
			Meaning:   cleanMarkdownCell(cells[3]),
		}))
	}
	if len(out) == 0 {
		return defaultGenerations()
	}
	sort.SliceStable(out, func(i, j int) bool {
		return generationRank(out[i].Stream) < generationRank(out[j].Stream)
	})
	return out
}

func markdownTableCells(line string) []string {
	line = strings.Trim(line, "|")
	raw := strings.Split(line, "|")
	out := make([]string, 0, len(raw))
	for _, cell := range raw {
		out = append(out, strings.TrimSpace(cell))
	}
	return out
}

func cleanMarkdownCell(s string) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), "`", "")
	return strings.Join(strings.Fields(s), " ")
}

func generationRank(stream string) int {
	switch stream {
	case "now":
		return 0
	case "next":
		return 1
	case "second-next":
		return 2
	case "future":
		return 3
	default:
		return 9
	}
}

// GenerationByStream returns the generation row matching a stream name or gen/*
// label, case-insensitive.
func (c *Catalog) GenerationByStream(stream string) (Generation, bool) {
	q := strings.ToLower(strings.TrimSpace(stream))
	q = strings.TrimPrefix(q, "gen/")
	for _, g := range c.Generations {
		if g.Stream == q || strings.TrimPrefix(strings.ToLower(g.Label), "gen/") == q {
			return g, true
		}
	}
	return Generation{}, false
}

// SearchGenerations returns generation rows matching the query. Empty query lists
// all horizons in now -> future order so `fak index generation` is a compact
// self-index overview.
func (c *Catalog) SearchGenerations(query string) []Generation {
	toks := tokens(query)
	if len(toks) == 0 {
		out := make([]Generation, len(c.Generations))
		copy(out, c.Generations)
		return out
	}
	type scored struct {
		g Generation
		s int
	}
	var hits []scored
	for _, g := range c.Generations {
		hay := strings.ToLower(strings.Join([]string{
			g.Stream,
			g.Label,
			g.Milestone,
			g.Meaning,
			strings.Join(g.IssueBodySignals, " "),
			g.PromotionEvidence,
			g.DemotionEvidence,
			g.InvalidatingAssumption,
		}, " "))
		score := 0
		for _, tk := range toks {
			label := strings.TrimPrefix(strings.ToLower(g.Label), "gen/")
			switch {
			case tk == g.Stream || tk == g.Label || tk == label:
				score += 10
			case isGenerationSelector(tk):
				score = -1
			case strings.Contains(hay, tk):
				score++
			default:
				score = -1
			}
			if score < 0 {
				break
			}
		}
		if score >= 0 {
			hits = append(hits, scored{g: g, s: score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		return generationRank(hits[i].g.Stream) < generationRank(hits[j].g.Stream)
	})
	out := make([]Generation, len(hits))
	for i, h := range hits {
		out[i] = h.g
	}
	return out
}

func isGenerationSelector(tok string) bool {
	tok = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(tok)), "gen/")
	switch tok {
	case "now", "next", "second-next", "future":
		return true
	default:
		return false
	}
}
