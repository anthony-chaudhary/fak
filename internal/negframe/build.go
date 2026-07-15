package negframe

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// DefaultTargetFiles are the singleton agent-steer prose surfaces the card gardens by default:
// the two top-level instruction files an agent reads every session. Add a surface here when a
// new always-loaded instruction file lands.
var DefaultTargetFiles = []string{"AGENTS.md", "CLAUDE.md"}

// DefaultTargetGlobDirs are directories whose *.md files are ALL agent-steer prose (the skills,
// each of which is injected when its verb is invoked). Walked for markdown so a new skill is
// gardened without editing this list.
var DefaultTargetGlobDirs = []string{".claude/skills"}

// broadcastTierForPath assigns corpus defaults: instruction files are paid once
// per session; skill/docs are cold; explicit guard-runtime source surfaces are
// paid per turn. Unknown paths conservatively use the cold tier.
func broadcastTierForPath(path string) BroadcastTier {
	p := filepath.ToSlash(path)
	switch {
	case strings.HasPrefix(p, "cmd/fak/guard_"), strings.Contains(p, "refusal_notes"):
		return TierPerTurn
	case p == "AGENTS.md", p == "CLAUDE.md":
		return TierPerSession
	default:
		return TierCold
	}
}

// ResolveTargets returns the default corpus paths (repo-relative, slash form) that exist under
// root: the singleton instruction files plus every *.md under the skill dirs. Missing files are
// skipped so a partial checkout degrades to what is present, never a crash.
func ResolveTargets(root string) []string {
	var out []string
	seen := map[string]bool{}
	add := func(rel string) {
		rel = filepath.ToSlash(rel)
		if !seen[rel] {
			seen[rel] = true
			out = append(out, rel)
		}
	}
	for _, rel := range DefaultTargetFiles {
		if fileExists(filepath.Join(root, filepath.FromSlash(rel))) {
			add(rel)
		}
	}
	for _, dir := range DefaultTargetGlobDirs {
		full := filepath.Join(root, filepath.FromSlash(dir))
		_ = filepath.WalkDir(full, func(path string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() || !strings.EqualFold(filepath.Ext(path), ".md") {
				return nil
			}
			if rel, err := filepath.Rel(root, path); err == nil {
				add(rel)
			}
			return nil
		})
	}
	sort.Strings(out)
	return out
}

// Build reads each target document, classifies its negative framing, and folds the corpus into
// the control-pane payload. paths are repo-relative; when empty, ResolveTargets(root) supplies
// the default corpus. Each of the four categories becomes one KPI whose HARD defects are the
// mechanical (confidently reframable) findings and whose soft signals are the judgement-tier
// ones. The headline negframe_debt is the total mechanical count -- the reframes a maintainer
// can apply right now with the suggestion attached.
func Build(root string, paths []string) scorecard.Payload {
	if len(paths) == 0 {
		paths = ResolveTargets(root)
	}
	var docs []DocResult
	totalSentences := 0
	for _, rel := range paths {
		text := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(rel)))
		d := ScoreDoc(rel, text)
		docs = append(docs, d)
		totalSentences += d.Sentences
	}

	// Bucket findings by category across the whole corpus so each category folds into one KPI.
	byCat := map[Category][]Finding{}
	for _, d := range docs {
		for _, f := range d.Findings {
			byCat[f.Category] = append(byCat[f.Category], f)
		}
	}

	denom := totalSentences
	if denom < 1 {
		denom = 1
	}
	kpis := make([]scorecard.KPI, 0, len(Categories))
	mechTotal := 0
	for _, cat := range Categories {
		fs := byCat[cat]
		var hard, soft []string
		for _, f := range fs {
			if f.Mechanical() {
				hard = append(hard, fmt.Sprintf("%s:%d %s -> reframe: %q  [%s]",
					f.Path, f.Line, scorecard.Clip(f.Text, 80), f.Suggest, cat))
			} else {
				soft = append(soft, fmt.Sprintf("%s:%d %s  [%s: %s]",
					f.Path, f.Line, scorecard.Clip(f.Text, 80), cat, f.Hint))
			}
		}
		mechTotal += len(hard)
		// Value trends with negative DENSITY across both tiers so reframing (of either tier)
		// lifts the continuous signal; the HARD gate stays mechanical-only via Defects.
		score := 100.0 * (1 - clampRatio(len(fs), denom))
		kpis = append(kpis, scorecard.KPI{
			Key:     "positive_" + string(cat),
			Group:   "framing",
			Score:   score,
			Detail:  fmt.Sprintf("%d %s finding(s) (%d mechanical, %d judgement)", len(fs), cat, len(hard), len(soft)),
			Defects: hard,
			Soft:    soft,
		})
	}

	finding := "steer prose leads with the affordance -- no mechanical negative reframes outstanding"
	next := "hold -- re-run after new steer prose lands"
	if mechTotal > 0 {
		finding = fmt.Sprintf("%d mechanical negative(s) carry an unambiguous positive reframe", mechTotal)
		next = "apply the suggested reframes worst-category-first (`fak score negframe --suggest`)"
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStrict,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			"documents":       len(docs),
			"weighted_debt":   weightedMechanicalDebt(docs),
			"sentences":       totalSentences,
			"mechanical_debt": mechTotal,
			"judgement_soft":  countTier(docs, false),
		},
	})
	p.Workspace = root
	return p
}

// AllFindings flattens every finding across the corpus (default or explicit paths) in document
// then line order -- the raw list the CLI's per-doc and --suggest views render from.
func AllFindings(root string, paths []string) []Finding {
	if len(paths) == 0 {
		paths = ResolveTargets(root)
	}
	var out []Finding
	for _, rel := range paths {
		text := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(rel)))
		out = append(out, Classify(rel, text)...)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Weight != out[j].Weight {
			return out[i].Weight > out[j].Weight
		}
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].Line < out[j].Line
	})
	return out
}

// clampRatio returns n/denom clamped to [0,1].
func clampRatio(n, denom int) float64 {
	if denom <= 0 {
		return 0
	}
	r := float64(n) / float64(denom)
	if r > 1 {
		return 1
	}
	if r < 0 {
		return 0
	}
	return r
}

// countTier totals the judgement (mechanical=false) or mechanical (true) findings across docs.
func countTier(docs []DocResult, mechanical bool) int {
	n := 0
	for _, d := range docs {
		if mechanical {
			n += d.Mechanical
		} else {
			n += d.Judgement
		}
	}
	return n
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func weightedMechanicalDebt(docs []DocResult) int {
	total := 0
	for _, d := range docs {
		for _, f := range d.Findings {
			if f.Mechanical() {
				total += f.Weight
			}
		}
	}
	return total
}
