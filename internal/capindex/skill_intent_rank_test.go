package capindex

// skill_intent_rank_test.go — the selection-quality evidence for #5560 (epic #3229).
//
// #5560 asks whether the resident skill card may carry a SHORT intent line instead
// of the full frontmatter `description`. There are two distinct copies of that prose
// in a CapCard and they answer to opposite pressures:
//
//   - CardBytes — the SERIALIZED at-rest card. Pure resident cost; nothing ranks on it.
//   - Trigger   — the in-process RANKING key scoreCard() matches intent terms against.
//     Every byte here is recall.
//
// The ticket's blocking design question is "is Trigger allowed to shrink?". These
// tests answer it with a measurement over the REAL .claude/skills tree rather than an
// assumption, and pin the answer so a later trim cannot quietly regress selection.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// realSkillCards indexes the repository's own .claude/skills tree, or skips when the
// test is run somewhere without one (a clean-room export, a vendored consumer).
func realSkillCards(t *testing.T) []CapCard {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Skipf("cannot resolve working directory: %v", err)
	}
	for i := 0; i < 8; i++ {
		cand := filepath.Join(dir, ".claude", "skills")
		if st, err := os.Stat(cand); err == nil && st.IsDir() {
			cards := NewSkillResolver(cand).Index()
			if len(cards) == 0 {
				t.Skipf("skills tree %s indexed no cards", cand)
			}
			return cards
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Skip("no .claude/skills tree above the working directory")
	return nil
}

// rankWith ranks cards through the SHIPPED ranker (Catalog.RankCards → scoreCard), so
// the measurement can never drift from the selection path the loader actually runs.
func rankWith(cards []CapCard, intent string) []CapCard {
	ix := NewIndex()
	ix.RegisterAll(cards)
	cat := NewCatalog()
	cat.index = ix
	return cat.RankCards(intent)
}

// withTrigger returns a copy of cards whose Trigger is replaced by fn(card) — the
// A/B lever for "what if the ranking key were shorter".
func withTrigger(cards []CapCard, fn func(CapCard) string) []CapCard {
	out := make([]CapCard, len(cards))
	copy(out, cards)
	for i := range out {
		out[i].Trigger = fn(out[i])
	}
	return out
}

// rankStopwords are terms too common in English prose to be a distinctive trigger for
// any one skill; they are excluded from the probe intents so the corpus measures
// vocabulary that actually discriminates between skills.
var rankStopwords = map[string]bool{
	"a": true, "an": true, "and": true, "any": true, "are": true, "as": true, "at": true,
	"be": true, "but": true, "by": true, "can": true, "do": true, "does": true, "for": true,
	"from": true, "has": true, "have": true, "in": true, "into": true, "is": true, "it": true,
	"its": true, "not": true, "of": true, "on": true, "one": true, "or": true, "so": true,
	"that": true, "the": true, "then": true, "this": true, "to": true, "up": true, "use": true,
	"used": true, "when": true, "which": true, "who": true, "why": true, "with": true,
	"you": true, "your": true,
}

// distinctiveTailTerms returns the terms that live ONLY in the tail of a card's
// description — absent from its first sentence, its name and its tags — and that are
// rare across the whole corpus (document frequency <= maxDF). These are exactly the
// terms a first-sentence ranking key would stop matching, so they are the honest probe
// for what shrinking Trigger costs. Labels are not invented: the correct answer for a
// probe built from skill X's own prose is skill X.
func distinctiveTailTerms(cards []CapCard, card CapCard, df map[string]int, maxDF int) []string {
	covered := map[string]bool{}
	for _, t := range tokenize(FirstSentence(card.Trigger)) {
		covered[t] = true
	}
	for _, t := range tokenize(card.Ref.Name) {
		covered[t] = true
	}
	for _, t := range tokenize(strings.Join(card.Tags, " ")) {
		covered[t] = true
	}

	seen := map[string]bool{}
	var out []string
	for _, t := range tokenize(card.Trigger) {
		if covered[t] || seen[t] || rankStopwords[t] || len(t) < 4 || df[t] > maxDF {
			continue
		}
		seen[t] = true
		out = append(out, t)
	}
	sort.Strings(out)
	return out
}

// docFreq counts, per term, how many cards' full descriptions contain it.
func docFreq(cards []CapCard) map[string]int {
	df := map[string]int{}
	for _, c := range cards {
		seen := map[string]bool{}
		for _, t := range tokenize(c.Trigger) {
			if seen[t] {
				continue
			}
			seen[t] = true
			df[t]++
		}
	}
	return df
}

// TestShrinkingTheRankingKeyCostsSelectionQuality is #5560's blocking measurement:
// what top-1 selection accuracy costs if the ranking key shrinks from the full
// description to a one-line intent.
//
// The probe corpus is built from the tree itself, never hand-written: for each skill,
// up to four rare terms that appear ONLY in the tail of its own description. The label
// is the skill the phrase was taken from. A ranking key that still holds the tail finds
// it; a first-sentence key cannot.
func TestShrinkingTheRankingKeyCostsSelectionQuality(t *testing.T) {
	cards := realSkillCards(t)
	df := docFreq(cards)

	type probe struct {
		want   string
		intent string
	}
	var probes []probe
	for _, c := range cards {
		terms := distinctiveTailTerms(cards, c, df, 2)
		if len(terms) < 2 {
			continue // no distinctive tail vocabulary: nothing to lose, nothing to measure
		}
		if len(terms) > 4 {
			terms = terms[:4]
		}
		probes = append(probes, probe{want: c.Ref.Name, intent: strings.Join(terms, " ")})
	}
	if len(probes) < 20 {
		t.Fatalf("probe corpus too small to be evidence: %d probes over %d skills", len(probes), len(cards))
	}

	shrunk := withTrigger(cards, func(c CapCard) string { return FirstSentence(c.Trigger) })

	var fullHits, shrunkHits, shrunkMisses int
	var lost []string
	for _, p := range probes {
		fullRank := rankWith(cards, p.intent)
		if len(fullRank) > 0 && fullRank[0].Ref.Name == p.want {
			fullHits++
		}
		shrunkRank := rankWith(shrunk, p.intent)
		switch {
		case len(shrunkRank) == 0:
			shrunkMisses++ // nothing scored at all: the skill became unreachable by intent
			lost = append(lost, p.want)
		case shrunkRank[0].Ref.Name == p.want:
			shrunkHits++
		default:
			lost = append(lost, p.want)
		}
	}

	fullPct := 100 * fullHits / len(probes)
	shrunkPct := 100 * shrunkHits / len(probes)
	t.Logf("selection quality over %d tail-derived probes across %d skills:", len(probes), len(cards))
	t.Logf("  full-description ranking key : top-1 %d/%d (%d%%)", fullHits, len(probes), fullPct)
	t.Logf("  first-sentence ranking key   : top-1 %d/%d (%d%%), %d probes scored NOTHING",
		shrunkHits, len(probes), shrunkPct, shrunkMisses)
	sort.Strings(lost)
	t.Logf("  skills that stop ranking first when the key shrinks (%d): %s", len(lost), strings.Join(lost, " "))

	// The pinned conclusion: shrinking the RANKING key is a real regression. If this
	// ever stops holding, the trade in #5560 changes and the card split may take the
	// Trigger with it — but that must be a measured decision, not a silent one.
	if shrunkHits >= fullHits {
		t.Fatalf("first-sentence ranking key did NOT lose selection quality (full %d, shrunk %d): "+
			"re-open #5560's design question, the residency split may now shrink Trigger too",
			fullHits, shrunkHits)
	}
}

// TestEverySkillStaysInvocableByName is the safety property the whole trim rests on:
// an intent that names a skill selects that skill, with the full description as the
// ranking key AND with a one-line intent — because scoreCard weights a name match
// above a trigger match. Trimming resident prose can therefore never cost name
// addressability.
func TestEverySkillStaysInvocableByName(t *testing.T) {
	cards := realSkillCards(t)
	shrunk := withTrigger(cards, func(c CapCard) string { return FirstSentence(c.Trigger) })
	nameOnly := withTrigger(cards, func(CapCard) string { return "" })

	for _, variant := range []struct {
		label string
		cards []CapCard
	}{
		{"full description", cards},
		{"one-line intent", shrunk},
		{"no trigger at all", nameOnly},
	} {
		var miss []string
		for _, c := range cards {
			ranked := rankWith(variant.cards, c.Ref.Name)
			if len(ranked) == 0 || ranked[0].Ref.Name != c.Ref.Name {
				miss = append(miss, c.Ref.Name)
			}
		}
		if len(miss) != 0 {
			t.Errorf("[%s] %d skill(s) not selected by their own name: %s",
				variant.label, len(miss), strings.Join(miss, " "))
		}
	}
}

// TestResidentCardCarriesIntentNotFullDescription pins the residency split itself: the
// serialized at-rest card holds the SHORT intent line, the full prose is not in it, and
// the full prose is still reachable — through Trigger for ranking and through Fault for
// the body. This is a residency split, not a deletion of prose.
func TestResidentCardCarriesIntentNotFullDescription(t *testing.T) {
	cards := realSkillCards(t)

	var cardTotal, intentTotal, descTotal int
	var fat []string
	for _, c := range cards {
		cardTotal += len(c.CardBytes)
		intentTotal += len(c.Intent)
		descTotal += len(c.Trigger)

		var got struct {
			Name        string `json:"name"`
			Intent      string `json:"intent"`
			Description string `json:"description"`
		}
		if err := json.Unmarshal(c.CardBytes, &got); err != nil {
			t.Fatalf("%s: card bytes are not JSON: %v", c.Ref.Name, err)
		}
		if got.Name != c.Ref.Name {
			t.Errorf("%s: at-rest card lost the name (got %q)", c.Ref.Name, got.Name)
		}
		if got.Description != "" {
			t.Errorf("%s: at-rest card still carries a full `description` field", c.Ref.Name)
		}
		if got.Intent != c.Intent {
			t.Errorf("%s: at-rest card intent %q != card.Intent %q", c.Ref.Name, got.Intent, c.Intent)
		}
		if c.Intent == "" {
			t.Errorf("%s: no resident intent line — the card cannot say when to load it", c.Ref.Name)
		}
		if len(c.Intent) > SkillIntentMaxBytes {
			fat = append(fat, c.Ref.Name)
		}
		if c.Trigger == "" {
			t.Errorf("%s: ranking key is empty — the skill is un-queryable by intent", c.Ref.Name)
		}
	}
	if len(fat) != 0 {
		t.Errorf("%d intent line(s) over the %d-byte cap: %s", len(fat), SkillIntentMaxBytes, strings.Join(fat, " "))
	}

	t.Logf("resident at-rest card floor: %d bytes across %d skills (intent slice %d B, in-process ranking key %d B)",
		cardTotal, len(cards), intentTotal, descTotal)
	if intentTotal >= descTotal {
		t.Fatalf("intent slice (%d B) is not smaller than the description it replaces (%d B)", intentTotal, descTotal)
	}
}

// TestResidentIntentInventory reports the derived intent line for the real tree: the
// length distribution, and by name the skills whose leading sentence does not fit
// SkillIntentMaxBytes and therefore elides. Those are exactly the skills that would
// benefit from an explicit frontmatter `intent:`, so the follow-on migration is a
// list rather than a hunt. It asserts only that the elision stays a minority — if
// most skills need an override, the derivation is the wrong default and the cap (or
// the rule) should be revisited rather than 58 files edited.
func TestResidentIntentInventory(t *testing.T) {
	cards := realSkillCards(t)

	lens := make([]int, 0, len(cards))
	var elided []string
	longest := 0
	for _, c := range cards {
		lens = append(lens, len(c.Intent))
		if strings.HasSuffix(c.Intent, "…") {
			elided = append(elided, c.Ref.Name)
		}
		if n := len(FirstSentence(c.Trigger)); n > longest {
			longest = n
		}
	}
	sort.Ints(lens)
	sort.Strings(elided)
	t.Logf("resident intent lines over %d skills: min %d B, median %d B, max %d B (cap %d); longest leading sentence %d B",
		len(lens), lens[0], lens[len(lens)/2], lens[len(lens)-1], SkillIntentMaxBytes, longest)
	t.Logf("  %d skill(s) elide and want an explicit frontmatter `intent:`: %s", len(elided), strings.Join(elided, " "))

	if len(elided)*2 > len(cards) {
		t.Errorf("%d of %d intent lines elide — deriving from the leading sentence is the wrong default at a %d-byte cap",
			len(elided), len(cards), SkillIntentMaxBytes)
	}
}

// TestExplicitFrontmatterIntentWins pins the escape hatch end to end, through the
// real parser and the real card build: a SKILL.md that declares `intent:` gets that
// line resident, verbatim, instead of a derived one — and its full description is
// still the ranking key. This is the worked example for the frontmatter field #5560
// asks be proposed rather than swept across all 58 skills.
func TestExplicitFrontmatterIntentWins(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "worked-example")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	const long = "A sprawling leading sentence that runs on and on about every trigger phrase " +
		"the author could think of, naming arXiv and tarball and monorepo and crates so the " +
		"ranker can match them, which is exactly why it makes a poor resident one-liner. Tail."
	md := "---\nname: worked-example\nversion: 1.0.0\n" +
		"intent: Study an external repo into a witnessed, filed backlog.\n" +
		"description: " + long + "\ntags: [research]\n---\n\n# Body\n\nfull prose here\n"
	if err := os.WriteFile(filepath.Join(dir, "SKILL.md"), []byte(md), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}

	cards := NewSkillResolver(root).Index()
	if len(cards) != 1 {
		t.Fatalf("indexed %d cards, want 1", len(cards))
	}
	c := cards[0]
	if want := "Study an external repo into a witnessed, filed backlog."; c.Intent != want {
		t.Errorf("Intent = %q, want the explicit frontmatter line %q", c.Intent, want)
	}
	if c.Trigger != long {
		t.Errorf("ranking key lost the full description (%d B, want %d B)", len(c.Trigger), len(long))
	}
	if strings.Contains(string(c.CardBytes), "arXiv") {
		t.Errorf("at-rest card still carries the full description: %s", c.CardBytes)
	}
	// Still selected by a term that lives ONLY in the tail of the description.
	ranked := rankWith(cards, "tarball monorepo")
	if len(ranked) == 0 || ranked[0].Ref.Name != "worked-example" {
		t.Errorf("an explicit intent broke intent-ranking on tail vocabulary: %v", ranked)
	}
}

// TestFaultStillPagesTheWholeSkillBody proves the prose the card no longer carries is
// still there when the skill is actually paged in — the other half of a residency
// split. A trim that made the body unreachable would be a deletion.
func TestFaultStillPagesTheWholeSkillBody(t *testing.T) {
	cards := realSkillCards(t)
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	var skills string
	for i := 0; i < 8; i++ {
		cand := filepath.Join(dir, ".claude", "skills")
		if st, err := os.Stat(cand); err == nil && st.IsDir() {
			skills = cand
			break
		}
		dir = filepath.Dir(dir)
	}
	r := NewSkillResolver(skills)

	for _, c := range cards {
		cp, err := r.Fault(c.Ref)
		if err != nil {
			t.Fatalf("%s: Fault: %v", c.Ref.Name, err)
		}
		body := cp.Materialize()
		if len(body) == 0 {
			t.Fatalf("%s: faulted body is empty", c.Ref.Name)
		}
		// The full description prose is in the faulted body, byte for byte.
		if !strings.Contains(strings.Join(strings.Fields(string(body)), " "),
			strings.Join(strings.Fields(c.Trigger), " ")) {
			t.Errorf("%s: full description is not recoverable from the faulted body", c.Ref.Name)
		}
	}
}

// TestFirstSentenceHandlesRealFrontmatter pins the derivation on shapes that actually
// occur in the corpus, including the ones where a naive "split on the first period"
// would cut mid-abbreviation or mid-path.
func TestFirstSentenceHandlesRealFrontmatter(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"", ""},
		{"One line, no terminator", "One line, no terminator"},
		{"First. Second.", "First."},
		{"Score the repo (e.g. a fork) end to end. Then file.", "Score the repo (e.g. a fork) end to end."},
		{"Reads .claude/skills/x.md and reports. More prose here.", "Reads .claude/skills/x.md and reports."},
		{"Use when asked about i.e. hooks. Not otherwise.", "Use when asked about i.e. hooks."},
		{"Ends with a question? Then more.", "Ends with a question?"},
		{"Trailing space before the cut.  Next.", "Trailing space before the cut."},
	} {
		if got := FirstSentence(tc.in); got != tc.want {
			t.Errorf("FirstSentence(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// TestIntentLineIsCappedAndWordSafe pins the cap: an intent line never exceeds
// SkillIntentMaxBytes, never cuts a word in half, and marks the elision so a reader
// knows prose was left behind rather than the author having written a fragment.
func TestIntentLineIsCappedAndWordSafe(t *testing.T) {
	long := strings.Repeat("alpha bravo charlie delta ", 40) // ~1000 B, no sentence break
	got := intentLine(long, "")
	if len(got) > SkillIntentMaxBytes {
		t.Fatalf("intent line %d bytes, over the %d cap", len(got), SkillIntentMaxBytes)
	}
	if !strings.HasSuffix(got, "…") {
		t.Errorf("elided intent line does not mark the elision: %q", got)
	}
	if strings.Contains(got, "alph…") || strings.Contains(got, "brav…") {
		t.Errorf("intent line cut a word in half: %q", got)
	}

	// An explicit frontmatter intent wins over the derived one, and is capped too.
	if got := intentLine("Derived from this sentence. And more.", "Explicit override."); got != "Explicit override." {
		t.Errorf("explicit intent not honoured: %q", got)
	}
	if got := intentLine("x.", long); len(got) > SkillIntentMaxBytes {
		t.Errorf("explicit intent escaped the cap: %d bytes", len(got))
	}
}
