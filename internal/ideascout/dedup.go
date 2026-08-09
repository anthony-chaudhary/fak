package ideascout

// The dedup rungs and the indexes they read: the node-local seen-cache (rung 1),
// the durable idea-scout-source stamp index (rung 2), and the windowed
// issue-body / title-near corpus (rungs 3 and 4).

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var tokenRE = regexp.MustCompile(`[a-z0-9]+`)

func Tokenize(text string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, t := range tokenRE.FindAllString(strings.ToLower(text), -1) {
		if len(t) >= 3 {
			out[t] = struct{}{}
		}
	}
	return out
}

func Jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	inter := 0
	for k := range a {
		if _, ok := b[k]; ok {
			inter++
		}
	}
	return float64(inter) / float64(len(a)+len(b)-inter)
}

var stampRE = regexp.MustCompile(`idea-scout-source:\s*([^\s>]+)`)

// StampIndex is rung 2's whole payload: every `idea-scout-source:` stamp carried
// by issues, lower-cased. Case is folded on BOTH sides (here and in IsDuplicate)
// because GitHub repo names are case-insensitive while the search API hands back
// whichever casing it feels like — an un-folded compare lets `Acme/Repo` slip
// past a stamp that reads `acme/repo`.
func StampIndex(issues []ExistingIssue) map[string]struct{} {
	out := map[string]struct{}{}
	for _, iss := range issues {
		for _, m := range stampRE.FindAllStringSubmatch(iss.Body, -1) {
			if len(m) > 1 {
				out[strings.ToLower(strings.TrimSpace(m[1]))] = struct{}{}
			}
		}
	}
	return out
}

func ExistingIssueIndex(issues []ExistingIssue) (map[string]struct{}, []map[string]struct{}, string) {
	titleSets := make([]map[string]struct{}, 0, len(issues))
	bodies := make([]string, 0, len(issues))
	for _, iss := range issues {
		bodies = append(bodies, strings.ToLower(iss.Body))
		titleSets = append(titleSets, Tokenize(iss.Title))
	}
	return StampIndex(issues), titleSets, strings.Join(bodies, "\n")
}

// IsDuplicate names the rung that fires ("seen-cache" / "filed-stamp" /
// "issue-body" / "title-near"), or "" if the candidate is genuinely new.
//
// "filed-stamp" and "issue-body" are reported separately on purpose: the first is
// the durable, complete filing history (rung 2) and the second is a best-effort
// URL sighting inside a recency window (rung 3). Collapsing them would make a
// windowed guess indistinguishable from the guarantee in the run report.
func IsDuplicate(c Candidate, seen map[string]SeenRecord, stamped map[string]struct{}, titleSets []map[string]struct{}, bodiesJoined string, dupJaccard float64) string {
	sidLower := strings.ToLower(c.SourceID)
	if _, ok := seen[c.SourceID]; ok {
		return "seen-cache"
	}
	if _, ok := seen[sidLower]; ok {
		return "seen-cache"
	}
	if _, ok := stamped[sidLower]; ok {
		return "filed-stamp"
	}
	if u := strings.ToLower(c.URL); u != "" && strings.Contains(bodiesJoined, u) {
		return "issue-body"
	}
	ctoks := Tokenize(c.Title)
	for _, tset := range titleSets {
		if Jaccard(ctoks, tset) >= dupJaccard {
			return "title-near"
		}
	}
	return ""
}

func CachePath(workspace string) string {
	return filepath.Join(workspace, CacheDirname, CacheFilename)
}

func LoadSeen(workspace string) (map[string]SeenRecord, error) {
	p := CachePath(workspace)
	raw, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]SeenRecord{}, nil
	}
	if err != nil {
		return map[string]SeenRecord{}, err
	}
	var wrapped struct {
		Schema string                `json:"schema"`
		Seen   map[string]SeenRecord `json:"seen"`
	}
	if err := json.Unmarshal(raw, &wrapped); err == nil && wrapped.Seen != nil {
		return wrapped.Seen, nil
	}
	var flat map[string]SeenRecord
	if err := json.Unmarshal(raw, &flat); err != nil {
		return map[string]SeenRecord{}, nil
	}
	return flat, nil
}

func SaveSeen(workspace string, seen map[string]SeenRecord) error {
	p := CachePath(workspace)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(struct {
		Schema string                `json:"schema"`
		Seen   map[string]SeenRecord `json:"seen"`
	}{Schema: Schema, Seen: seen}, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return os.WriteFile(p, raw, 0o644)
}
