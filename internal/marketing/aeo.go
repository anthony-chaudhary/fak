package marketing

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// aeo.go — the AEO/AgentEO data producers. AEO is dual: Answer-Engine Optimization (be cited
// correctly by ChatGPT/Claude/Perplexity/Google AI Overviews) AND Agent-Engine Optimization
// (be the path of least resistance for a coding agent to adopt fak). Both want the same thing
// a completion loop can give: a fresh, machine-ingestible, sha-cited record of what shipped.
//
// The division of labor (per the design review): GO produces the witnessed DATA here; the
// existing Python generator (tools/gen_structured_data.py) owns the in-place injection into
// hand-authored docs (llms.txt) via its marker machinery. Go never rewrites llms.txt — it
// emits docs/marketing/updates.json, and the Python side reads that and injects a bounded,
// sentinel-fenced "What's new" block. This keeps one marker engine, in one language, and makes
// it impossible for the loop to clobber hand-written prose.
//
// Every feed item is witnessed: it carries its commit sha. An unwitnessed item cannot be
// produced — the input is []Ship, and a Ship only exists for a trailer|direct stamp.

// repoCommitURL is the base for a commit permalink the feed cites. Kept here (not a tracked
// secret) — it is the public repo, the same URL llms.txt already links.
const repoCommitURL = "https://github.com/anthony-chaudhary/fak/commit/"

// defaultFeedCap bounds the updates feed / What's-new block so a long history doesn't bloat
// the answer-engine surface; the newest N ships are what "recent" means.
const defaultFeedCap = 25

// updatesFeed is the schema.org ItemList an answer engine ingests for recency. Each element is
// a SoftwareSourceCode item anchored to its commit — the witness an engine can cite.
type updatesFeed struct {
	Context  string        `json:"@context"`
	Type     string        `json:"@type"`
	Name     string        `json:"name"`
	Items    []updatesItem `json:"itemListElement"`
	Modified string        `json:"dateModified,omitempty"`
}

type updatesItem struct {
	Type     string         `json:"@type"`
	Position int            `json:"position"`
	Item     updatesSrcCode `json:"item"`
}

type updatesSrcCode struct {
	Type           string `json:"@type"`
	Name           string `json:"name"`
	CodeRepository string `json:"codeRepository"`
	DateModified   string `json:"dateModified,omitempty"`
	Keywords       string `json:"keywords,omitempty"`
}

// UpdatesFeed renders the witnessed ships as a schema.org ItemList (JSON, 2-space indent),
// newest-first and capped. when stamps the feed's dateModified. The result is the
// docs/marketing/updates.json the Python injector reads — and a valid JSON-LD document an
// answer engine can ingest directly.
func UpdatesFeed(ships []Ship, when time.Time) ([]byte, error) {
	ships = sortedShipsDesc(ships)
	if len(ships) > defaultFeedCap {
		ships = ships[:defaultFeedCap]
	}
	feed := updatesFeed{
		Context: "https://schema.org",
		Type:    "ItemList",
		Name:    "fak — what shipped",
	}
	if !when.IsZero() {
		feed.Modified = when.UTC().Format(time.RFC3339)
	}
	for i, s := range ships {
		feed.Items = append(feed.Items, updatesItem{
			Type:     "ListItem",
			Position: i + 1,
			Item: updatesSrcCode{
				Type:           "SoftwareSourceCode",
				Name:           claimText(s),
				CodeRepository: repoCommitURL + s.SHA,
				DateModified:   shipDate(s),
				Keywords:       s.Leaf,
			},
		})
	}
	return json.MarshalIndent(feed, "", "  ")
}

// WhatsNewMarkdown renders the bounded, dated, witnessed "Recent ships" block that the Python
// injector fences into llms.txt (and llms-updates.txt). House style: one `- **date** — claim
// ([sha](commit-url))` line per ship, newest-first, capped. Pure and stable, so re-rendering
// the same ships is a no-op diff (the idempotence the marker injection relies on).
func WhatsNewMarkdown(ships []Ship) string {
	ships = sortedShipsDesc(ships)
	if len(ships) > defaultFeedCap {
		ships = ships[:defaultFeedCap]
	}
	if len(ships) == 0 {
		return "_No witnessed ships recorded yet._"
	}
	var b strings.Builder
	for _, s := range ships {
		date := shipDate(s)
		if date == "" {
			date = "recent"
		}
		fmt.Fprintf(&b, "- **%s** — %s ([`%s`](%s%s))\n", date, claimText(s), s.SHA, repoCommitURL, s.SHA)
	}
	return strings.TrimRight(b.String(), "\n")
}

// LlmsUpdatesText renders the sibling llms-updates.txt corpus — a plain, capped, newest-first
// recency feed an answer engine or agent polls, in the same house style as llms.txt. It leads
// with a one-line self-describing header so a crawler that lands on it alone understands it.
func LlmsUpdatesText(ships []Ship, when time.Time) string {
	var b strings.Builder
	b.WriteString("# fak — what shipped (recent, witnessed)\n\n")
	b.WriteString("> A machine-readable feed of fak's most recent shipped changes. Every line cites the\n")
	b.WriteString("> commit that witnesses it. Regenerated on each completion by `fak marketing aeo`.\n")
	if !when.IsZero() {
		fmt.Fprintf(&b, ">\n> Updated: %s\n", when.UTC().Format(time.RFC3339))
	}
	b.WriteString("\n")
	b.WriteString(WhatsNewMarkdown(ships))
	b.WriteString("\n")
	return b.String()
}

// shipDate renders a ship's date as YYYY-MM-DD, or "" if unparsed.
func shipDate(s Ship) string {
	if s.Date.IsZero() {
		return ""
	}
	return s.Date.UTC().Format("2006-01-02")
}

// sortedShipsDesc returns a newest-first copy (does not mutate the input).
func sortedShipsDesc(ships []Ship) []Ship {
	out := append([]Ship(nil), ships...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].Date.After(out[j-1].Date); j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}
