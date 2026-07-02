package popularizationtickets

import (
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

//go:embed tickets.json
var ticketFS embed.FS

type Ticket struct {
	Dim         string   `json:"dim"`
	Title       string   `json:"title"`
	Concepts    []string `json:"concepts"`
	Deliverable string   `json:"deliverable"`
	Why         string   `json:"why"`
	Acceptance  string   `json:"acceptance"`
	Lane        string   `json:"lane"`
}

var dims = map[string]string{
	"A": "Explainer content",
	"B": "Visual & diagram assets",
	"C": "Interactive & runnable demos",
	"D": "Positioning & comparison",
	"E": "Social proof & community",
	"F": "Developer experience & onramp",
	"G": "Integration recipes",
	"H": "Benchmark-as-story",
	"I": "Memorable framing & naming",
	"J": "Distribution & channels",
	"K": "Adoption measurement",
}

var concepts = map[string]string{
	"syscall": "Treat the tool call like a syscall",
	"dos":     "Verify, don't trust (DOS)",
	"kvcache": "Addressable bit-exact KV cache",
	"capgate": "Default-deny capability gate + quarantine",
	"binary":  "One static Go binary, drop-in",
}

func Load() ([]Ticket, error) {
	raw, err := ticketFS.ReadFile("tickets.json")
	if err != nil {
		return nil, err
	}
	var tickets []Ticket
	if err := json.Unmarshal(raw, &tickets); err != nil {
		return nil, err
	}
	if len(tickets) != 50 {
		return nil, fmt.Errorf("expected 50 tickets, have %d", len(tickets))
	}
	return tickets, nil
}

func JSON(tickets []Ticket) ([]byte, error) {
	return json.MarshalIndent(tickets, "", "  ")
}

func List(tickets []Ticket) string {
	var b strings.Builder
	byDim := map[string]int{}
	for i, t := range tickets {
		byDim[t.Dim]++
		fmt.Fprintf(&b, "%2d  [%s] %s\n", i+1, t.Dim, t.Title)
	}
	keys := make([]string, 0, len(byDim))
	for k := range byDim {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	b.WriteString("\nper-dimension: map[")
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%s:%d", k, byDim[k])
	}
	b.WriteString("]")
	return b.String()
}

func LanesTSV(tickets []Ticket) string {
	var b strings.Builder
	for _, t := range tickets {
		title := strings.ReplaceAll(t.Title, "\t", " ")
		fmt.Fprintf(&b, "%s\t%s\n", title, t.Lane)
	}
	return b.String()
}

func EmitFiles(dir, epicRef string, tickets []Ticket) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for i, t := range tickets {
		base := fmt.Sprintf("ticket-%02d", i+1)
		if err := os.WriteFile(filepath.Join(dir, base+".md"), []byte(RenderBody(t, epicRef)), 0o644); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, base+".title"), []byte(t.Title), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func RenderBody(t Ticket, epicRef string) string {
	dimName := dims[t.Dim]
	conceptNames := make([]string, 0, len(t.Concepts))
	for _, c := range t.Concepts {
		conceptNames = append(conceptNames, concepts[c])
	}
	conceptsText := strings.Join(conceptNames, ", ")
	files := likelyFiles(t)
	current := strings.TrimSpace(strings.SplitN(t.Deliverable, ":", 2)[0])
	return fmt.Sprintf(`**Dimension %s - %s** - part of the concept-popularization epic (%s).

**Concepts served:** %s

## Parent context
Concept-popularization epic - `+"`%s`"+`. This is one of 50 self-contained tickets under the `+"`popularization`"+` label making the fak/DOS concepts broadly known and attractive.

## Current state
The concept exists in the code/docs but its human-facing popularization artifact for this dimension does not: %s is not yet present as a standalone, shareable unit. Today a reader has no single artifact for this angle.

## Why now
The AEO/SEO surface (machine-facing discovery) is maintained, but the human-facing half - the artifacts that make a person want and remember the concept - is the current gap. This dimension has no dedicated owner-artifact yet.

## Working spine
%s

## In scope
The single deliverable named above, plus its wiring (index link / front-matter / cross-links) so it is reachable and honest.

## Out of scope
Any other popularization ticket's deliverable; kernel/engineering changes; market-adoption claims; any benchmark not already run; renaming or restructuring existing docs beyond what this artifact needs.

## Why it popularizes the concept
%s

## Done condition
%s

## Witness
The acceptance artifact exists and is checkable: the named file/command is present and correct; `+"`python tools/seo_aeo_scorecard.py`"+` does not regress for any new `+"`docs/*.md`"+`; the ship commit passes `+"`dos commit-audit`"+`.

## Acceptance gate
%s

## Work unit
One doc/example/tool artifact a single worker owns end to end in one sitting; no dependency on another popularization ticket landing first.

## Expected steps
3

## Assumptions
- The five core concepts and honest-scope fences in the epic doc are authoritative.
- Witnessed numbers only (the tuned ~4.1x, not the naive 60x); simulated is labeled simulated.

## Confusion risks
- Do not overclaim market adoption or a novelty the 0/29 prior-art audit refutes.
- Keep this lane disjoint from sibling popularization tickets - touch only the files below.

## Coordination
- One worker per lane; lane `+"`%s`"+` is disjoint from the other 49 tickets.
- Verify lane disjointness via `+"`dos_arbitrate`"+` before writing if the trunk is busy.

## Trigger
Filed as part of the 2026-07-02 concept-popularization epic; dispatched via the account-switching headless resolver.

## Batch policy
One issue per popularization dimension-slot; deduped by title; update the existing issue rather than re-filing. Capped at the 50-ticket epic set.

## Likely files
%s

## Lane
`+"`%s`"+` (disjoint from other popularization tickets - one worker can own it end to end).

## Closure binding
Closed by the ship commit that creates the accepted artifact, stamped `+"`(fak <leaf>)`"+` and referencing this issue number; the commit's `+"`dos commit-audit`"+` verdict is the binding witness.

## Ship discipline
- Trunk only; commit by explicit path; Conventional-Commits subject + a `+"`(fak <leaf>)`"+` stamp.
- New `+"`docs/*.md`"+` need SEO front-matter (`+"`title:`"+`/`+"`description:`"+`) and an `+"`INDEX.md`"+` line.
- Honest-scope fence: no market-adoption claim, no unrun benchmark, no novelty the 0/29 prior-art audit refutes; quote witnessed numbers, label simulated as simulated.

_This is one self-contained unit of work. It does not depend on any other popularization ticket landing first (it may cross-link to them)._
`, t.Dim, dimName, epicRef, conceptsText, epicRef, current, t.Deliverable, t.Why, t.Acceptance, t.Acceptance, t.Lane, files, t.Lane)
}

var codeSpanRE = regexp.MustCompile("`([^`]+)`")

func likelyFiles(t Ticket) string {
	var paths []string
	for _, match := range codeSpanRE.FindAllStringSubmatch(t.Deliverable, -1) {
		s := match[1]
		if strings.Contains(s, " ") {
			continue
		}
		if strings.Contains(s, "/") || strings.HasSuffix(s, ".py") || strings.HasSuffix(s, ".md") {
			paths = append(paths, "`"+s+"`")
		}
	}
	if len(paths) == 0 {
		lane := t.Lane
		switch {
		case strings.HasPrefix(lane, "examples-"):
			paths = []string{"`examples/" + strings.TrimPrefix(lane, "examples-") + "/`"}
		case strings.HasPrefix(lane, "cmd-"):
			parts := strings.Split(lane, "-")
			paths = []string{"`cmd/fak/" + parts[len(parts)-1] + ".go`"}
		case strings.HasPrefix(lane, "tools-"):
			name := strings.ReplaceAll(strings.TrimPrefix(lane, "tools-"), "-", "_")
			paths = []string{"`tools/" + name + ".py`"}
		default:
			paths = []string{"`docs/`"}
		}
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(paths)+1)
	for _, p := range paths {
		if !seen[p] {
			seen[p] = true
			out = append(out, p)
		}
	}
	addIndex := false
	for _, p := range out {
		if strings.Contains(p, "docs/") {
			addIndex = true
			break
		}
	}
	if addIndex && !seen["`INDEX.md`"] {
		out = append(out, "`INDEX.md`")
	}
	return strings.Join(out, ", ")
}
