package issuededup

import (
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/simhash"
)

// TestProbeScores is a throwaway calibration probe (deleted before commit): it
// prints title/body cosine and normalized-body length for the fixture twins and
// for real concept-popularization epic siblings, to find the discriminator that
// separates a genuine body-twin from a short-boilerplate template family.
func TestProbeScores(t *testing.T) {
	report := func(a, b BacklogIssue) {
		na := normalizeBody(a.Body)
		nb := normalizeBody(b.Body)
		ta := simhash.Embed(a.Title)
		tb := simhash.Embed(b.Title)
		ca := simhash.Embed(a.Title + "\n" + na)
		cb := simhash.Embed(b.Title + "\n" + nb)
		t.Logf("#%d/#%d title=%.3f body=%.3f | normBodyTokens %d/%d normBodyRunes %d/%d",
			a.Number, b.Number,
			simhash.Cosine(ta, tb), simhash.Cosine(ca, cb),
			len(strings.Fields(na)), len(strings.Fields(nb)),
			len([]rune(na)), len([]rune(nb)))
	}

	// Fixture twins (the intended positive).
	byNum := map[int]BacklogIssue{}
	for _, is := range censusBacklog {
		byNum[is.Number] = is
	}
	report(byNum[3001], byNum[3002])

	// Real backlog epic siblings (the intended negatives that currently avalanche).
	b, err := os.ReadFile(os.Getenv("BACKLOG_JSON"))
	if err != nil {
		t.Skipf("no BACKLOG_JSON: %v", err)
	}
	issues, err := ParseBacklog(b)
	if err != nil {
		t.Fatal(err)
	}
	real := map[int]BacklogIssue{}
	for _, is := range issues {
		real[is.Number] = is
	}
	for _, pr := range [][2]int{{2311, 2324}, {2311, 2315}, {2319, 2321}, {2324, 2325}, {2267, 2505}} {
		if real[pr[0]].Number != 0 && real[pr[1]].Number != 0 {
			report(real[pr[0]], real[pr[1]])
		}
	}
}
