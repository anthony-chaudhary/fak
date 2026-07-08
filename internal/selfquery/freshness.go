package selfquery

import (
	"path"
	"regexp"
	"strings"
)

// freshness.go is the first checkable step of #3163 (parent epic #1494, the
// query-quality axis): a query-result card proves WHICH bytes it carries
// (FeatureCard.Witness) but says nothing about its CURRENCY, so score() ranks a
// note dated 2026-06-25 identically to one dated 2026-07-06 on the same topic.
// This computes a deterministic, advisory supersession rung from dated-note
// timestamps already indexed in the card's DetailRef path — no front-matter
// read, no LLM judgment, no ranking change.
//
// It is the read-time analogue of dos_recall's re-verification, but for the
// query surface: within a set of same-topic dated notes, the newest date is
// FRESH and every strictly-older one is SUPERSEDED_BY the newest. Precision
// first — the fences from the issue hold here:
//   - Advisory ranking metadata, NOT a trust verdict.
//   - Never auto-deletes or auto-hides: a superseded card still returns,
//     stamped with its rung (a silent drop reads as "nothing there").
//   - Same topic means byte-equal normalized topic slug — conservative, so the
//     signal never fires on a merely-similar title.
//
// Complements (does not replace) fak_index_freshness, which answers a different
// axis: index-vs-tree drift, not semantic supersession.
//
// Not yet computed here (documented follow-ons under #3163): the STALE rung for
// a card whose cited code token no longer exists (bind to dos_recall), and an
// explicit `supersedes:` front-matter edge. noteInfo/freshnessByKey are shaped
// so either can layer on without disturbing the dated-note path.

// Freshness rung values carried by FeatureCard.Freshness.
const (
	// FreshnessFresh marks the newest card among a set of same-topic dated
	// notes (only set when at least one strictly-older sibling exists).
	FreshnessFresh = "FRESH"
	// FreshnessSupersededPrefix is followed by the Name of the superseding
	// (newer, same-topic) card, e.g. "SUPERSEDED_BY:doc:<title>". The suffix is
	// a real card Name, resolvable via the same match findCard uses.
	FreshnessSupersededPrefix = "SUPERSEDED_BY:"
)

// isoDateRE matches an ISO date token (YYYY-MM-DD) embedded anywhere in a note
// filename. Both note-naming styles in docs/notes carry one: the date-prefix
// form (2026-07-07-topic-slug) and the date-suffix form (TOPIC-SLUG-2026-07-07).
// Lexical order over the matched string equals chronological order, so no time
// parsing is needed to compare two notes.
var isoDateRE = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}\b`)

// noteRef is the two facts a supersession decision needs, both derived purely
// from a note's path: when it is dated and what topic it is about.
type noteRef struct {
	date  string // ISO YYYY-MM-DD; lexical compare == chronological compare
	topic string // date-stripped, normalized filename slug
}

// noteInfo extracts (date, topic) from a dated note reference, or ok=false when
// ref is not a dated note. Precision-first admission: ref must be a `.md` file
// under a `docs/notes/` path AND carry an ISO date token. A plain doc
// (docs/gateway.md), a non-note path, or an undated note is never marked.
func noteInfo(ref string) (noteRef, bool) {
	ref = strings.ToLower(strings.TrimSpace(ref))
	if i := strings.IndexAny(ref, "#?"); i >= 0 { // drop an anchor/query suffix
		ref = ref[:i]
	}
	if !strings.Contains(ref, "docs/notes/") {
		return noteRef{}, false
	}
	slug := path.Base(ref)
	if !strings.HasSuffix(slug, ".md") {
		return noteRef{}, false
	}
	slug = strings.TrimSuffix(slug, ".md")
	date := isoDateRE.FindString(slug)
	if date == "" {
		return noteRef{}, false
	}
	// Strip every date token (a filename may repeat one) and normalize the
	// remainder to a stable topic key.
	topic := normalizeTopic(isoDateRE.ReplaceAllString(slug, " "))
	if topic == "" {
		return noteRef{}, false
	}
	return noteRef{date: date, topic: topic}, true
}

// normalizeTopic collapses a date-stripped filename to a stable topic key:
// lowercase alphanumeric runs joined by single dashes, no leading or trailing
// dash. Two notes share a topic iff their keys are byte-equal, so the date's
// position in the filename (prefix vs suffix) does not affect the match.
func normalizeTopic(s string) string {
	var b strings.Builder
	prevDash := true // suppress a leading dash
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			prevDash = false
			continue
		}
		if !prevDash {
			b.WriteByte('-')
			prevDash = true
		}
	}
	return strings.TrimRight(b.String(), "-")
}

// cardKey identifies a card across the two independent slices freshness spans
// (the full candidate set it is computed over, and the ranked result it is
// stamped onto). Kind+Name+DetailRef is unique for the dated-note cards this
// touches; the NUL separators keep the parts unambiguous.
func cardKey(c FeatureCard) string {
	return c.Kind + "\x00" + c.Name + "\x00" + c.DetailRef
}

// freshnessByKey computes the supersession rung for every dated-note card in
// cards, keyed by cardKey. It is computed over the FULL candidate set (not the
// post-limit result) so an older note still in the result is correctly marked
// even when its superseding sibling was ranked out of the top-N. Cards that are
// not dated notes, and lone notes with no same-topic sibling, get no entry.
func freshnessByKey(cards []FeatureCard) map[string]string {
	// Bucket dated-note card indices by topic; a plain slice per topic keeps the
	// input order (already stably sorted upstream) so tie-breaks are stable.
	dates := make([]string, len(cards))
	groups := map[string][]int{}
	for i := range cards {
		nr, ok := noteInfo(cards[i].DetailRef)
		if !ok {
			continue
		}
		dates[i] = nr.date
		groups[nr.topic] = append(groups[nr.topic], i)
	}
	out := map[string]string{}
	for _, idxs := range groups {
		if len(idxs) < 2 {
			continue // a lone note has nothing to supersede or be superseded by
		}
		maxDate := ""
		for _, i := range idxs {
			if dates[i] > maxDate {
				maxDate = dates[i]
			}
		}
		// Require a strictly-older sibling; if every note shares the newest date
		// the order is ambiguous, so claim no supersession (precision-first).
		olderExists := false
		winner := ""
		for _, i := range idxs {
			if dates[i] < maxDate {
				olderExists = true
			}
			if dates[i] == maxDate && winner == "" {
				winner = cards[i].Name // stable: first newest in sorted order
			}
		}
		if !olderExists {
			continue
		}
		for _, i := range idxs {
			if dates[i] == maxDate {
				out[cardKey(cards[i])] = FreshnessFresh
			} else {
				out[cardKey(cards[i])] = FreshnessSupersededPrefix + winner
			}
		}
	}
	return out
}

// applyFreshness stamps the precomputed rungs onto the cards that carry them.
// It only writes the advisory field; it never re-orders or drops a card.
func applyFreshness(cards []FeatureCard, rungs map[string]string) {
	if len(rungs) == 0 {
		return
	}
	for i := range cards {
		if r, ok := rungs[cardKey(cards[i])]; ok {
			cards[i].Freshness = r
		}
	}
}
