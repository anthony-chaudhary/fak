package ideascout

// Scoring: term hits, recency, stars, points, push freshness, and star velocity
// folded into one number plus the human-readable reasons behind it.

import (
	"fmt"
	"strings"
	"time"
)

func ScoreCandidate(c Candidate, topic Topic, cfg Config, now time.Time) (int, []string) {
	if now.IsZero() {
		now = time.Now().UTC()
	}
	title := strings.ToLower(c.Title)
	body := strings.ToLower(c.Summary)
	score := 0
	var reasons []string
	var hitTerms []string
	for _, term := range topic.Terms {
		t := strings.ToLower(term)
		switch {
		case strings.Contains(title, t):
			score += WTitleHit
			hitTerms = append(hitTerms, term+"(title)")
		case strings.Contains(body, t):
			score += WBodyHit
			hitTerms = append(hitTerms, term)
		}
	}
	if len(hitTerms) > 0 {
		reasons = append(reasons, "terms: "+strings.Join(hitTerms, ", "))
	}

	if published, ok := parseISO(c.Published); ok {
		age := int(now.Sub(published).Hours() / 24)
		if age >= 0 && age <= cfg.RecentDays {
			score += WRecent180
			reasons = append(reasons, fmt.Sprintf("recent (%dd)", age))
			if age <= 30 {
				score += WRecent30
				reasons = append(reasons, "very fresh (<=30d)")
			}
		}
	}
	stars := intFromExtra(c.Extra, "stars")
	if stars > 0 {
		bonus := stars / StarDivisor
		if bonus > StarCap {
			bonus = StarCap
		}
		if bonus > 0 {
			score += bonus
			reasons = append(reasons, fmt.Sprintf("%d stars (+%d)", stars, bonus))
		}
	}
	points := intFromExtra(c.Extra, "points")
	if points > 0 {
		bonus := points / HNPointDiv
		if bonus > HNPointCap {
			bonus = HNPointCap
		}
		if bonus > 0 {
			score += bonus
			reasons = append(reasons, fmt.Sprintf("%d points (+%d)", points, bonus))
		}
	}
	if pushed, ok := parseISO(stringFromExtra(c.Extra, "pushed_at")); ok {
		days := int(now.Sub(pushed).Hours() / 24)
		window := cfg.FreshWindowDays
		if window <= 0 {
			window = 45
		}
		switch {
		case days >= 0 && days <= window:
			score += WFreshPush
			reasons = append(reasons, fmt.Sprintf("pushed <=%dd (actively updated)", window))
		case days <= 90:
			score += WRecentPush
			reasons = append(reasons, "pushed <=90d")
		}
	}
	// Trending: a young repo already gathering stars (high stars/day) is on the
	// rise; an old repo with the same stars accrued them slowly and scores ~0.
	if stars > 0 {
		if created, ok := parseISO(c.Published); ok {
			ageDays := int(now.Sub(created).Hours() / 24)
			if ageDays < 1 {
				ageDays = 1
			}
			rawVel := stars / ageDays
			if rawVel > 0 {
				bonus := rawVel
				if bonus > TrendingCap {
					bonus = TrendingCap
				}
				score += bonus
				reasons = append(reasons, fmt.Sprintf("trending (%d*/day, +%d)", rawVel, bonus))
			}
		}
	}
	return score, reasons
}
