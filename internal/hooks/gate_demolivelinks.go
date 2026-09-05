package hooks

import (
	"bytes"
	"fmt"
	"html"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// gate_demolivelinks.go — the DEMO_LIVE_LINKS gate, a byte-faithful port of
// tools/demo_live_links.py static audit.
//
// Where DEMO_COMMAND catches broken local command references and BROWSER_CONTRACT
// catches localhost default drift, this gate validates the public hosted demo links
// and metadata in docs/demos.html:
// - catches stale hosted demo paths (e.g. /guarddemo/, /turntax/, /ctxdemo/, /unsee/);
// - enforces uniqueness and role (card vs non-card link) for all hosted URLs;
// - matches the hosted demo URLs against the shared demo registry;
// - verifies the plain HTTP disclosure note;
// - verifies canonical and OpenGraph/Twitter social-preview metadata and asset.
//
// Parity anchor: tools/demo_live_links.py collect() / static_audit() / page_metadata_audit().

const (
	demoLiveLinksHost         = "136.111.250.205"
	demoLiveLinksHubURL       = "http://136.111.250.205/"
	demoLiveLinksDefaultDoc   = "docs/demos.html"
	demoLiveLinksPublishedURL = "https://anthony-chaudhary.github.io/fak/demos.html"
	demoLiveLinksPreviewPath  = "visuals/social-preview.png"
	demoLiveLinksPreviewURL   = "https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/social-preview.png"
	demoLiveLinksPNGMagic     = "\x89PNG\r\n\x1a\n"
)

var demoLiveLinksStalePrefixes = []struct {
	prefix string
	demo   string
}{
	{"http://" + demoLiveLinksHost + "/guarddemo/", "guarddemo"},
	{"http://" + demoLiveLinksHost + ":8151/", "guarddemo"},
	{"http://" + demoLiveLinksHost + ":8151/api/", "guarddemo"},
	{"http://" + demoLiveLinksHost + ":8156/", "unseedemo"},
	{"http://" + demoLiveLinksHost + ":8156/api/", "unseedemo"},
	{"http://" + demoLiveLinksHost + "/turntax/", "turntaxdemo"},
	{"http://" + demoLiveLinksHost + "/ctxdemo/", "ctxdemo"},
	{"http://" + demoLiveLinksHost + "/unsee/", "unseedemo"},
}

// knownStaleDemoMatch returns the matching prefix and demo name if href starts with a known stale prefix.
func knownStaleDemoMatch(href string) (string, string, bool) {
	for _, entry := range demoLiveLinksStalePrefixes {
		if strings.HasPrefix(href, entry.prefix) {
			return entry.prefix, entry.demo, true
		}
	}
	return "", "", false
}

// hostedURLForDemo computes the canonical hosted URL for a demo registry entry, mirroring
// demo_registry.py hosted_url().
func hostedURLForDemo(host string, demo demoReg) string {
	if demo.hostedPath == "" {
		return ""
	}
	port := ""
	if demo.hostedPort != 0 {
		port = fmt.Sprintf(":%d", demo.hostedPort)
	}
	p := "/" + strings.Trim(demo.hostedPath, "/")
	normalized := p
	if p == "/" {
		normalized = ""
	}
	path := "/"
	if normalized != "" {
		path = normalized + "/"
	}
	return fmt.Sprintf("http://%s%s%s", host, port, path)
}

// expectedHostedDemoLinks returns the map of expected hosted URLs to isCard (bool), and the
// expected card count, matching EXPECTED_HOSTED_LINKS in demo_live_links.py.
func expectedHostedDemoLinks(host string) (map[string]bool, int) {
	expected := make(map[string]bool)
	cardCount := 0
	for _, demo := range demoRegistry {
		if u := hostedURLForDemo(host, demo); u != "" {
			expected[u] = true
			cardCount++
		}
	}
	expected["http://"+host+"/"] = false
	return expected, cardCount
}

type demoHTMLLink struct {
	href    string
	text    string
	classes []string
	isCard  bool
}

var (
	reHTMLComment = regexp.MustCompile(`(?s)<!--.*?-->`)
	reAnchorTag   = regexp.MustCompile(`(?is)<a\b([^>]*)>(.*?)</a>`)
	reHTMLTags    = regexp.MustCompile(`<[^>]*>`)
	reHTMLAttr    = regexp.MustCompile(`(?i)([a-zA-Z0-9_:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?`)
)

func parseHTMLTagAttrs(attrText string) map[string]string {
	attrs := make(map[string]string)
	matches := reHTMLAttr.FindAllStringSubmatch(attrText, -1)
	for _, m := range matches {
		k := strings.ToLower(m[1])
		if k == "/" {
			continue
		}
		v := ""
		if m[2] != "" {
			v = m[2]
		} else if m[3] != "" {
			v = m[3]
		} else if m[4] != "" {
			v = m[4]
		}
		attrs[k] = html.UnescapeString(v)
	}
	return attrs
}

func extractDemoHTMLLinks(htmlStr string) []demoHTMLLink {
	clean := reHTMLComment.ReplaceAllString(htmlStr, "")
	var links []demoHTMLLink
	matches := reAnchorTag.FindAllStringSubmatch(clean, -1)
	for _, m := range matches {
		attrs := parseHTMLTagAttrs(m[1])
		href := attrs["href"]
		var classes []string
		if classAttr, ok := attrs["class"]; ok {
			classes = strings.Fields(classAttr)
		}
		isCard := false
		for _, c := range classes {
			if c == "card" {
				isCard = true
				break
			}
		}
		plain := reHTMLTags.ReplaceAllString(m[2], " ")
		text := strings.Join(strings.Fields(plain), " ")
		text = html.UnescapeString(text)
		links = append(links, demoHTMLLink{
			href:    href,
			text:    text,
			classes: classes,
			isCard:  isCard,
		})
	}
	return links
}

type demoHTMLMetadata struct {
	canonical []string
	meta      map[string][]string
}

var reMetaOrLink = regexp.MustCompile(`(?i)<(link|meta)\b([^>]*)>`)

func extractDemoHTMLMetadata(htmlStr string) demoHTMLMetadata {
	clean := reHTMLComment.ReplaceAllString(htmlStr, "")
	res := demoHTMLMetadata{
		meta: make(map[string][]string),
	}
	matches := reMetaOrLink.FindAllStringSubmatch(clean, -1)
	for _, m := range matches {
		tagName := strings.ToLower(m[1])
		attrs := parseHTMLTagAttrs(m[2])
		if tagName == "link" {
			rels := strings.Fields(strings.ToLower(attrs["rel"]))
			hasCanonical := false
			for _, r := range rels {
				if r == "canonical" {
					hasCanonical = true
					break
				}
			}
			if hasCanonical && attrs["href"] != "" {
				res.canonical = append(res.canonical, attrs["href"])
			}
		} else if tagName == "meta" {
			key := attrs["property"]
			if key == "" {
				key = attrs["name"]
			}
			if key != "" {
				if content, ok := attrs["content"]; ok {
					res.meta[key] = append(res.meta[key], content)
				}
			}
		}
	}
	return res
}

func isHostedDemoLink(href, host string) bool {
	u, err := url.Parse(href)
	if err != nil {
		return false
	}
	return (u.Scheme == "http" || u.Scheme == "https") && u.Hostname() == host
}

// staticDemoLinksAudit ports demo_live_links.py static_audit().
func staticDemoLinksAudit(htmlStr string, host string) []string {
	links := extractDemoHTMLLinks(htmlStr)
	var hosted []demoHTMLLink
	for _, link := range links {
		if isHostedDemoLink(link.href, host) {
			hosted = append(hosted, link)
		}
	}

	var hostedCards []demoHTMLLink
	for _, link := range hosted {
		if link.isCard {
			hostedCards = append(hostedCards, link)
		}
	}

	var defects []string

	for _, link := range hosted {
		if _, demo, ok := knownStaleDemoMatch(link.href); ok {
			defects = append(defects, fmt.Sprintf("stale hosted demo link for %s: %s", demo, link.href))
		}
	}

	expectedLinks, expectedCardCount := expectedHostedDemoLinks(host)

	if host == demoLiveLinksHost {
		seen := make(map[string]int)
		for _, link := range hosted {
			seen[link.href]++
		}
		if len(hosted) != len(seen) {
			defects = append(defects, "duplicate hosted demo link found; keep each hosted URL unique")
		}

		var missingExpected []string
		for href := range expectedLinks {
			if _, ok := seen[href]; !ok {
				missingExpected = append(missingExpected, href)
			}
		}
		sort.Strings(missingExpected)
		for _, href := range missingExpected {
			defects = append(defects, fmt.Sprintf("expected hosted demo link missing: %s", href))
		}

		var unexpectedHosted []string
		for href := range seen {
			if _, ok := expectedLinks[href]; !ok {
				unexpectedHosted = append(unexpectedHosted, href)
			}
		}
		sort.Strings(unexpectedHosted)
		for _, href := range unexpectedHosted {
			defects = append(defects, fmt.Sprintf("unexpected hosted demo link: %s", href))
		}

		// Check role consistency in deterministic demoRegistry order
		var orderedExpectedHrefs []string
		for _, demo := range demoRegistry {
			if u := hostedURLForDemo(host, demo); u != "" {
				orderedExpectedHrefs = append(orderedExpectedHrefs, u)
			}
		}
		orderedExpectedHrefs = append(orderedExpectedHrefs, "http://"+host+"/")

		for _, href := range orderedExpectedHrefs {
			wantCard, isExpected := expectedLinks[href]
			if !isExpected {
				continue
			}
			var roles []bool
			for _, link := range hosted {
				if link.href == href {
					roles = append(roles, link.isCard)
				}
			}
			if len(roles) > 0 {
				hasWantedRole := false
				for _, r := range roles {
					if r == wantCard {
						hasWantedRole = true
						break
					}
				}
				if !hasWantedRole {
					want := "card"
					if !wantCard {
						want = "non-card link"
					}
					defects = append(defects, fmt.Sprintf("hosted demo link role changed: %s should be a %s", href, want))
				}
			}
			if !isExpected {
				defects = append(defects, fmt.Sprintf("hosted demo link lacks live witness spec: %s", href))
			}
		}
	}

	if len(hostedCards) != expectedCardCount {
		defects = append(defects, fmt.Sprintf("hosted card count changed: found %d, want %d; update docs and this audit together",
			len(hostedCards), expectedCardCount))
	}

	hasHTTP := false
	var httpsLinks []string
	for _, link := range hosted {
		u, err := url.Parse(link.href)
		if err == nil {
			if u.Scheme == "http" {
				hasHTTP = true
			} else if u.Scheme == "https" {
				httpsLinks = append(httpsLinks, link.href)
			}
		}
	}

	if hasHTTP && !strings.Contains(htmlStr, "plain HTTP") {
		defects = append(defects, "hosted links use http:// but docs/demos.html does not disclose plain HTTP")
	}

	if len(httpsLinks) > 0 {
		defects = append(defects, fmt.Sprintf("hosted demo link uses https:// for the IP host; verify TLS first: %s", httpsLinks[0]))
	}

	return defects
}

func expectDemoMetadataValue(defects *[]string, values []string, label, expected string) {
	uniqueMap := make(map[string]struct{})
	for _, v := range values {
		uniqueMap[v] = struct{}{}
	}
	var unique []string
	for k := range uniqueMap {
		unique = append(unique, k)
	}
	sort.Strings(unique)

	if len(unique) == 0 {
		*defects = append(*defects, fmt.Sprintf("demos page metadata missing %s: %s", label, expected))
		return
	}
	if len(unique) != 1 || unique[0] != expected {
		var parts []string
		for _, u := range unique {
			parts = append(parts, fmt.Sprintf("'%s'", u))
		}
		*defects = append(*defects, fmt.Sprintf("demos page metadata %s=[%s], want '%s'", label, strings.Join(parts, ", "), expected))
	}
}

// demoPageMetadataAudit ports demo_live_links.py page_metadata_audit().
func demoPageMetadataAudit(workspaceRoot, htmlStr string) []string {
	meta := extractDemoHTMLMetadata(htmlStr)
	var defects []string

	expectDemoMetadataValue(&defects, meta.canonical, "canonical", demoLiveLinksPublishedURL)
	expectDemoMetadataValue(&defects, meta.meta["og:url"], "og:url", demoLiveLinksPublishedURL)
	expectDemoMetadataValue(&defects, meta.meta["og:image"], "og:image", demoLiveLinksPreviewURL)
	expectDemoMetadataValue(&defects, meta.meta["twitter:image"], "twitter:image", demoLiveLinksPreviewURL)
	expectDemoMetadataValue(&defects, meta.meta["twitter:card"], "twitter:card", "summary_large_image")

	assetPath := filepath.Join(workspaceRoot, filepath.FromSlash(demoLiveLinksPreviewPath))
	data, err := os.ReadFile(assetPath)
	if err != nil {
		defects = append(defects, fmt.Sprintf("demos page social image asset missing: %s", demoLiveLinksPreviewPath))
	} else {
		magic := []byte(demoLiveLinksPNGMagic)
		if len(data) < len(magic) || !bytes.HasPrefix(data, magic) {
			defects = append(defects, fmt.Sprintf("demos page social image asset is not a PNG: %s", demoLiveLinksPreviewPath))
		} else if len(data) < 1024 {
			defects = append(defects, fmt.Sprintf("demos page social image asset is unexpectedly small: %s", demoLiveLinksPreviewPath))
		}
	}

	return defects
}

// demoLiveLinksDocDefects runs the static link audit and page metadata audit for doc in workspace root.
func demoLiveLinksDocDefects(workspaceRoot, doc string) []string {
	path := filepath.Join(workspaceRoot, filepath.FromSlash(doc))
	data, err := os.ReadFile(path)
	if err != nil {
		return []string{fmt.Sprintf("read %s: %s", doc, err)}
	}
	htmlStr := string(data)
	staticDefects := staticDemoLinksAudit(htmlStr, demoLiveLinksHost)
	metadataDefects := demoPageMetadataAudit(workspaceRoot, htmlStr)
	return append(staticDefects, metadataDefects...)
}

// demoLiveLinksDefects returns defects in the default doc docs/demos.html.
func demoLiveLinksDefects(workspaceRoot string) []string {
	return demoLiveLinksDocDefects(workspaceRoot, demoLiveLinksDefaultDoc)
}

// gateDemoLiveLinksTree is the DEMO_LIVE_LINKS tree hygiene gate.
func gateDemoLiveLinksTree(t *TrackedTree) ([]Finding, error) {
	var findings []Finding
	for _, d := range demoLiveLinksDefects(t.Root) {
		findings = append(findings, Finding{
			Gate:   "DEMO_LIVE_LINKS",
			File:   demoLiveLinksDefaultDoc,
			Detail: d,
		})
	}
	return findings, nil
}
