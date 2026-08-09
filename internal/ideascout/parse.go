package ideascout

// Source-lane parsing: one pure fold per source over its wire bytes. No network
// and no wall clock lives here, so every parser is replayable from a fixture.

import (
	"encoding/json"
	"encoding/xml"
	"regexp"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

func ParseArxivAtom(xmlText, topicKey string) []Candidate {
	type author struct {
		Name string `xml:"name"`
	}
	type entry struct {
		ID        string   `xml:"id"`
		Title     string   `xml:"title"`
		Summary   string   `xml:"summary"`
		Published string   `xml:"published"`
		Authors   []author `xml:"author"`
	}
	type feed struct {
		Entries []entry `xml:"entry"`
	}
	var f feed
	if err := xml.Unmarshal([]byte(xmlText), &f); err != nil {
		return nil
	}
	var out []Candidate
	for _, e := range f.Entries {
		rawID := strings.TrimSpace(e.ID)
		if rawID == "" {
			continue
		}
		absID := rawID
		if idx := strings.LastIndex(absID, "/"); idx >= 0 {
			absID = absID[idx+1:]
		}
		absID = regexp.MustCompile(`v\d+$`).ReplaceAllString(absID, "")
		var authors []string
		for _, a := range e.Authors {
			if name := strings.TrimSpace(a.Name); name != "" {
				authors = append(authors, name)
			}
			if len(authors) == 6 {
				break
			}
		}
		out = append(out, Candidate{
			Source:    "arxiv",
			SourceID:  "arxiv:" + absID,
			URL:       "https://arxiv.org/abs/" + absID,
			Title:     squashSpace(e.Title),
			Summary:   squashSpace(e.Summary),
			Published: strings.TrimSpace(e.Published),
			Topic:     topicKey,
			Extra:     map[string]any{"authors": authors},
		})
	}
	return out
}

func ParseGitHubRepos(items []GitHubRepo, topicKey string) []Candidate {
	var out []Candidate
	for _, it := range items {
		full := strmatch.FirstNonBlank(it.FullName, it.Name)
		if full == "" {
			continue
		}
		u := it.URL
		if u == "" {
			u = "https://github.com/" + full
		}
		pushed := strmatch.FirstNonBlank(it.PushedAt, it.UpdatedAt)
		out = append(out, Candidate{
			Source:    "github",
			SourceID:  "github:" + strings.ToLower(full),
			URL:       u,
			Title:     full,
			Summary:   it.Description,
			Published: it.CreatedAt,
			Topic:     topicKey,
			Extra: map[string]any{
				"stars":     it.StargazersCount,
				"pushed_at": pushed,
				"language":  it.Language,
				"size":      it.Size,
			},
		})
	}
	return out
}

// ParseHackerNewsJSON turns an Algolia HN search response into candidates.
// It is a pure fold over the wire JSON: no network, no clock. Link stories keep
// their outbound URL; text/self posts fall back to the HN item permalink so the
// candidate always resolves to something a triager can open.
func ParseHackerNewsJSON(jsonText, topicKey string) []Candidate {
	var doc struct {
		Hits []struct {
			ObjectID    string `json:"objectID"`
			Title       string `json:"title"`
			StoryTitle  string `json:"story_title"`
			URL         string `json:"url"`
			StoryURL    string `json:"story_url"`
			Author      string `json:"author"`
			Points      int    `json:"points"`
			NumComments int    `json:"num_comments"`
			CreatedAt   string `json:"created_at"`
			StoryText   string `json:"story_text"`
		} `json:"hits"`
	}
	if err := json.Unmarshal([]byte(jsonText), &doc); err != nil {
		return nil
	}
	var out []Candidate
	for _, h := range doc.Hits {
		id := strings.TrimSpace(h.ObjectID)
		if id == "" {
			continue
		}
		title := squashSpace(strmatch.FirstNonBlank(h.Title, h.StoryTitle))
		if title == "" {
			continue
		}
		permalink := "https://news.ycombinator.com/item?id=" + id
		u := strmatch.FirstNonBlank(h.URL, h.StoryURL, permalink)
		out = append(out, Candidate{
			Source:    "hackernews",
			SourceID:  "hn:" + id,
			URL:       u,
			Title:     title,
			Summary:   squashSpace(stripTags(h.StoryText)),
			Published: strings.TrimSpace(h.CreatedAt),
			Topic:     topicKey,
			Extra: map[string]any{
				"points":       h.Points,
				"num_comments": h.NumComments,
				"discussion":   permalink,
				"author":       h.Author,
			},
		})
	}
	return out
}

// ParseRedditJSON turns a Reddit listing/search response into candidates. Like
// the other sources it is a pure fold over the wire JSON. Reddit stamps posts
// with a Unix `created_utc` float rather than an ISO string, so it is converted
// to RFC3339 here (a deterministic transform, no wall clock) to match the shared
// freshness path. Self/text posts carry the permalink in `url`; link posts carry
// the outbound target, and the permalink is always kept as the discussion link.
func ParseRedditJSON(jsonText, topicKey string) []Candidate {
	var doc struct {
		Data struct {
			Children []struct {
				Data struct {
					ID          string  `json:"id"`
					Title       string  `json:"title"`
					URL         string  `json:"url"`
					Permalink   string  `json:"permalink"`
					Score       int     `json:"score"`
					NumComments int     `json:"num_comments"`
					CreatedUTC  float64 `json:"created_utc"`
					Selftext    string  `json:"selftext"`
					Subreddit   string  `json:"subreddit"`
				} `json:"data"`
			} `json:"children"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(jsonText), &doc); err != nil {
		return nil
	}
	var out []Candidate
	for _, ch := range doc.Data.Children {
		h := ch.Data
		id := strings.TrimSpace(h.ID)
		if id == "" {
			continue
		}
		title := squashSpace(h.Title)
		if title == "" {
			continue
		}
		permalink := ""
		if h.Permalink != "" {
			permalink = "https://www.reddit.com" + h.Permalink
		}
		u := strmatch.FirstNonBlank(h.URL, permalink)
		if u == "" {
			u = "https://www.reddit.com/comments/" + id
		}
		published := ""
		if h.CreatedUTC > 0 {
			published = time.Unix(int64(h.CreatedUTC), 0).UTC().Format(time.RFC3339)
		}
		out = append(out, Candidate{
			Source:    "reddit",
			SourceID:  "reddit:" + id,
			URL:       u,
			Title:     title,
			Summary:   squashSpace(stripTags(h.Selftext)),
			Published: published,
			Topic:     topicKey,
			Extra: map[string]any{
				"points":       h.Score,
				"num_comments": h.NumComments,
				"discussion":   strmatch.FirstNonBlank(permalink, u),
				"subreddit":    h.Subreddit,
			},
		})
	}
	return out
}
