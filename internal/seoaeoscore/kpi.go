// Per-page and site KPI scoring.
// Split verbatim from seoaeoscore.go along concern seams to hold the god-file ceiling (#3022).
package seoaeoscore

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

// ---------------------------------------------------------------------------
// Per-page KPIs.
// ---------------------------------------------------------------------------

func kpiTitle(fm map[string]string) kpi {
	title := strings.TrimSpace(fm["title"])
	if title == "" {
		return kpi{"title", 0, "missing title", []string{"no front-matter title: (no <title> tag / no blue-link text)"}, nil}
	}
	if degenerate(title) {
		return kpi{"title", 0, "degenerate title", []string{"degenerate title (filler/repeat — no usable <title> text)"}, nil}
	}
	n := utf8.RuneCountInString(title)
	score := 100
	var soft []string
	if n < titleMin {
		soft = append(soft, fmt.Sprintf("title is thin (%d chars; aim %d-%d)", n, titleMin, titleMax))
		score -= 20
	} else if n > titleMax {
		soft = append(soft, fmt.Sprintf("title is long (%d chars; search truncates past ~%d)", n, titleMax))
		score -= 15
	}
	return kpi{"title", clampScore(float64(score)), fmt.Sprintf("title present (%d chars)", n), nil, soft}
}

func kpiDescription(fm map[string]string) kpi {
	desc := strings.TrimSpace(fm["description"])
	if desc == "" {
		return kpi{"description", 0, "missing description", []string{"no front-matter description: (no meta description / SERP snippet)"}, nil}
	}
	if degenerate(desc) {
		return kpi{"description", 0, "degenerate description", []string{"degenerate description (filler/repeat — no usable SERP snippet)"}, nil}
	}
	n := utf8.RuneCountInString(desc)
	score := 100
	var soft []string
	if n < descMin {
		soft = append(soft, fmt.Sprintf("description is thin (%d chars; aim %d-%d)", n, descMin, descMax))
		score -= 20
	} else if n > descMax {
		soft = append(soft, fmt.Sprintf("description is long (%d chars; search truncates past ~%d)", n, descMax))
		score -= 15
	}
	return kpi{"description", clampScore(float64(score)), fmt.Sprintf("description present (%d chars)", n), nil, soft}
}

func kpiHeadings(text string) kpi {
	body := reFence.ReplaceAllString(stripFrontMatter(text), " ")
	var levels []int
	for _, m := range reH.FindAllStringSubmatch(body, -1) {
		levels = append(levels, len(m[1]))
	}
	h1 := 0
	for _, l := range levels {
		if l == 1 {
			h1++
		}
	}
	score := 100
	var defects, soft []string
	if h1 == 0 {
		defects = append(defects, "no H1 heading (a '# Title' line)")
		score -= 30
	} else if h1 > 1 {
		soft = append(soft, fmt.Sprintf("%d H1 headings (expected exactly one per page)", h1))
		score -= 10
	}
	prev := 0
	for _, l := range levels {
		if prev != 0 && l > prev+1 {
			soft = append(soft, fmt.Sprintf("heading level jumps H%d->H%d (skips H%d)", prev, l, prev+1))
			score -= 8
			break
		}
		prev = l
	}
	nLines := strings.Count(body, "\n") + 1
	if nLines > 40 && len(levels) <= 1 {
		defects = append(defects, "long page with no section headings (## …)")
		score -= 20
	}
	return kpi{"headings", clampScore(float64(score)), fmt.Sprintf("%d H1 / %d headings", h1, len(levels)), defects, soft}
}

func kpiLinks(text, rootAbs, docRel string) kpi {
	baseAbs := filepath.Dir(rjoin(rootAbs, docRel))
	text = reFence.ReplaceAllString(text, " ")
	var dead []string
	seen := map[string]bool{}
	total := 0
	for _, m := range reLink.FindAllStringSubmatch(text, -1) {
		target := strings.TrimSpace(m[2])
		if hasSchemePrefix(target) {
			continue
		}
		pathPart := strings.TrimSpace(splitOnce(splitOnce(target, "#"), "?"))
		if pathPart == "" || seen[pathPart] {
			continue
		}
		seen[pathPart] = true
		total++
		if !existsPath(resolveLink(rootAbs, baseAbs, pathPart)) {
			dead = append(dead, pathPart)
		}
	}
	sort.Strings(dead)
	var defects []string
	for _, d := range dead {
		defects = append(defects, "dead link: "+d)
	}
	score := 100 - 20*len(dead)
	detail := fmt.Sprintf("all %d local link(s) resolve", total)
	if len(dead) > 0 {
		detail = fmt.Sprintf("%d/%d local link(s) dead", len(dead), total)
	}
	return kpi{"links", clampScore(float64(score)), detail, defects, nil}
}

func kpiLinksCrawlable(text, rootAbs, docRel string) kpi {
	baseAbs := filepath.Dir(rjoin(rootAbs, docRel))
	text = reFence.ReplaceAllString(text, " ")
	var crawl404, dirlinks []string
	seen := map[string]bool{}
	for _, m := range reLink.FindAllStringSubmatch(text, -1) {
		target := strings.TrimSpace(m[2])
		if hasSchemePrefix(target) {
			continue
		}
		pp := strings.TrimSpace(splitOnce(splitOnce(target, "#"), "?"))
		if pp == "" || seen[pp] {
			continue
		}
		seen[pp] = true
		resolved := resolveLink(rootAbs, baseAbs, pp)
		if !existsPath(resolved) {
			continue
		}
		tgt, ok := relPosix(rootAbs, resolved)
		if !ok {
			continue
		}
		if isDirPath(resolved) {
			dirlinks = append(dirlinks, pp)
			continue
		}
		if strings.HasSuffix(pp, ".md") && strings.HasPrefix(tgt, "docs/") && !published(rootAbs, tgt) {
			crawl404 = append(crawl404, fmt.Sprintf("%s (target %s is excluded from publishing — 404 on the live site)", pp, tgt))
		}
	}
	sort.Strings(crawl404)
	sort.Strings(dirlinks)
	var defects []string
	for _, c := range crawl404 {
		defects = append(defects, "crawl-404: "+c)
	}
	var soft []string
	for _, d := range dirlinks {
		soft = append(soft, fmt.Sprintf("directory link (no canonical published page): %s/", d))
	}
	score := 100 - 25*len(crawl404)
	detail := "every local link is crawlable on the published site"
	if len(crawl404) > 0 {
		detail = fmt.Sprintf("%d link(s) resolve on disk but 404 on the live site", len(crawl404))
	}
	return kpi{"links_crawlable", clampScore(float64(score)), detail, defects, soft}
}

func kpiAnswerability(text string) kpi {
	score := 100
	var soft []string
	if !hasProseOpener(text) {
		soft = append(soft, "no plain-language opening sentence before the first heading/code/table")
		score -= 25
	}
	body := stripFrontMatter(text)
	head := strings.Join(firstNStr(splitLines(body), 25), "\n")
	if !reIsAre.MatchString(head) {
		soft = append(soft, "first screen states no definition ('X is a …') for an answer engine to quote")
		score -= 10
	}
	detail := "first screen is thin for answer-engine quoting"
	if score >= 90 {
		detail = "first screen is quotable prose"
	}
	return kpi{"answerability", clampScore(float64(score)), detail, nil, soft}
}

func kpiAltText(text string) kpi {
	body := reInlineCode.ReplaceAllString(reFence.ReplaceAllString(text, " "), " ")
	var defects, soft []string
	nImg := 0
	check := func(alt, label, where string) {
		if strings.TrimSpace(alt) == "" {
			defects = append(defects, fmt.Sprintf("%s has no alt text: %s", label, truncRunes(where, 60)))
		} else if degenerateAlt(alt) {
			soft = append(soft, fmt.Sprintf("%s alt is generic filler ('%s'): %s", label, strings.TrimSpace(alt), truncRunes(where, 50)))
		}
	}
	for _, m := range reMDImg.FindAllStringSubmatch(body, -1) {
		nImg++
		check(m[1], "image", strings.TrimSpace(m[2]))
	}
	for _, m := range reMDRefImg.FindAllStringSubmatch(body, -1) {
		nImg++
		check(m[1], "image", strings.TrimSpace(m[0]))
	}
	for _, m := range reHTMLImg.FindAllString(body, -1) {
		nImg++
		alt := ""
		if a := reHTMLAlt.FindStringSubmatch(m); a != nil {
			alt = a[1]
		}
		check(alt, "<img>", m)
	}
	score := 100 - 25*len(defects) - 8*len(soft)
	detail := "no images"
	if nImg > 0 {
		if len(defects) == 0 && len(soft) == 0 {
			detail = fmt.Sprintf("%d image(s), all captioned", nImg)
		} else {
			detail = fmt.Sprintf("%d missing + %d weak alt of %d image(s)", len(defects), len(soft), nImg)
		}
	}
	return kpi{"alt_text", clampScore(float64(score)), detail, defects, soft}
}

// resolveLink mirrors `(root / pp.lstrip("/")) if pp.startswith("/") else (base / pp)`.
func resolveLink(rootAbs, baseAbs, pp string) string {
	if strings.HasPrefix(pp, "/") {
		return rjoin(rootAbs, strings.TrimLeft(pp, "/"))
	}
	return filepath.Join(baseAbs, filepath.FromSlash(pp))
}

// ---------------------------------------------------------------------------
// Excellence credit.
// ---------------------------------------------------------------------------

func crawlHubCount(text, rootAbs, docRel string) int {
	baseAbs := filepath.Dir(rjoin(rootAbs, docRel))
	text = reFence.ReplaceAllString(text, " ")
	selfTgt, ok := relPosix(rootAbs, rjoin(rootAbs, docRel))
	if !ok {
		selfTgt = docRel
	}
	seen := map[string]bool{}
	hubs := map[string]bool{}
	for _, m := range reLink.FindAllStringSubmatch(text, -1) {
		target := strings.TrimSpace(m[2])
		if hasSchemePrefix(target) {
			continue
		}
		pp := strings.TrimSpace(splitOnce(splitOnce(target, "#"), "?"))
		if pp == "" || seen[pp] {
			continue
		}
		seen[pp] = true
		if !strings.HasSuffix(pp, ".md") {
			continue
		}
		resolved := resolveLink(rootAbs, baseAbs, pp)
		if !existsPath(resolved) || isDirPath(resolved) {
			continue
		}
		tgt, ok := relPosix(rootAbs, resolved)
		if !ok {
			continue
		}
		if tgt != selfTgt && published(rootAbs, tgt) && discovery(rootAbs, tgt) {
			hubs[tgt] = true
		}
	}
	return len(hubs)
}

func pageCredit(text, docRel, rootAbs string, fm map[string]string) (int, map[string]any) {
	credit := 0
	detail := map[string]any{}
	shared := intersectSets(tokens(fm["title"]), tokens(firstH1Text(text)), ledeTokens(text), tokens(fm["description"]))
	if len(shared) > 0 {
		credit += 2
		sl := sortedStrings(shared)
		detail["keyword_focus"] = map[string]any{"points": 2, "shared": firstNStr(sl, 5)}
	}
	hub := crawlHubCount(text, rootAbs, docRel)
	if hub >= 3 {
		credit += 1
		detail["crawl_hub"] = map[string]any{"points": 1, "links": hub}
	}
	return credit, detail
}

func intersectSets(sets ...map[string]bool) map[string]bool {
	if len(sets) == 0 {
		return map[string]bool{}
	}
	out := map[string]bool{}
	for k := range sets[0] {
		in := true
		for _, s := range sets[1:] {
			if !s[k] {
				in = false
				break
			}
		}
		if in {
			out[k] = true
		}
	}
	return out
}

func siteCredit(jsonldValues []any, robots string, faqQuestions int, faqSyncOK bool) (int, map[string]int) {
	credit := 0
	detail := map[string]int{}
	for _, o := range jsonldObjectsWithType(jsonldValues, "Organization") {
		name, _ := o["name"].(string)
		if strings.TrimSpace(name) == "" {
			continue
		}
		hasURL := jsonldURL(o) != ""
		sameAs, ok := o["sameAs"].([]any)
		if hasURL || (ok && len(sameAs) > 0) {
			credit += 2
			detail["schema_richness"] = 2
			break
		}
	}
	groups := robotsGroups(robots)
	allNamed := true
	for ua := range aiCrawlerUAs {
		if _, ok := groups[ua]; !ok {
			allNamed = false
			break
		}
	}
	if allNamed {
		credit += 2
		detail["crawler_breadth"] = 2
	}
	if faqQuestions >= 2*minFAQQuestions && faqSyncOK {
		credit += 1
		detail["faq_depth"] = 1
	}
	return credit, detail
}
