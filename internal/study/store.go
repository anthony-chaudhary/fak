// Package study persists immutable source-to-decision receipts.
package study

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const Schema = "fak-study/1"

type Source struct {
	URL      string `json:"url"`
	Revision string `json:"revision"`
	License  string `json:"license,omitempty"`
}
type Candidate struct {
	Effect      string   `json:"effect"`
	Status      string   `json:"status"`
	Disposition string   `json:"disposition"`
	Evidence    []string `json:"evidence"`
	Issue       string   `json:"issue,omitempty"`
	Outcome     string   `json:"outcome,omitempty"`
}
type Record struct {
	Schema       string      `json:"schema"`
	Title        string      `json:"title"`
	Observed     string      `json:"observed"`
	Sources      []Source    `json:"sources"`
	Observations []string    `json:"observations"`
	Candidates   []Candidate `json:"candidates"`
	Supersedes   string      `json:"supersedes,omitempty"`
}
type Receipt struct {
	ID      string `json:"id"`
	Created bool   `json:"created"`
	Store   string `json:"store"`
}
type Match struct {
	ID     string `json:"id"`
	Record Record `json:"record"`
}

func canonical(r Record) ([]byte, error) {
	if r.Schema != Schema {
		return nil, fmt.Errorf("schema must be %q", Schema)
	}
	if strings.TrimSpace(r.Title) == "" || strings.TrimSpace(r.Observed) == "" {
		return nil, errors.New("title and observed date are required")
	}
	if len(r.Sources) == 0 || len(r.Candidates) == 0 {
		return nil, errors.New("at least one source and candidate are required")
	}
	for _, s := range r.Sources {
		if strings.TrimSpace(s.URL) == "" || strings.TrimSpace(s.Revision) == "" {
			return nil, errors.New("every source requires url and revision")
		}
	}
	for _, c := range r.Candidates {
		if strings.TrimSpace(c.Effect) == "" || len(c.Evidence) == 0 {
			return nil, errors.New("every candidate requires effect and evidence")
		}
		if c.Status != "PRESENT" && c.Status != "PARTIAL" && c.Status != "ABSENT" {
			return nil, fmt.Errorf("invalid status %q", c.Status)
		}
		switch c.Disposition {
		case "DEFAULT", "OPTIONAL-MODULE", "RECIPE", "WATCH", "EXCLUDE":
		default:
			return nil, fmt.Errorf("invalid disposition %q", c.Disposition)
		}
	}
	return json.Marshal(r)
}

func Add(store string, r Record) (Receipt, error) {
	b, err := canonical(r)
	if err != nil {
		return Receipt{}, err
	}
	sum := sha256.Sum256(b)
	id := "study_" + hex.EncodeToString(sum[:])
	if err := os.MkdirAll(store, 0700); err != nil {
		return Receipt{}, fmt.Errorf("storage unavailable: %w", err)
	}
	path := filepath.Join(store, id+".json")
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if errors.Is(err, os.ErrExist) {
		return Receipt{ID: id, Created: false, Store: store}, nil
	}
	if err != nil {
		return Receipt{}, fmt.Errorf("storage unavailable: %w", err)
	}
	ok := false
	defer func() {
		f.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	if _, err = f.Write(append(b, '\n')); err != nil {
		return Receipt{}, fmt.Errorf("storage unavailable: %w", err)
	}
	if err = f.Sync(); err != nil {
		return Receipt{}, fmt.Errorf("storage unavailable: %w", err)
	}
	ok = true
	return Receipt{ID: id, Created: true, Store: store}, nil
}

func Get(store, id string) (Record, error) {
	if !strings.HasPrefix(id, "study_") || strings.ContainsAny(id, "/\\") {
		return Record{}, errors.New("invalid study ID")
	}
	b, err := os.ReadFile(filepath.Join(store, id+".json"))
	if err != nil {
		return Record{}, err
	}
	var r Record
	if err = json.Unmarshal(b, &r); err != nil {
		return Record{}, err
	}
	if _, err = canonical(r); err != nil {
		return Record{}, err
	}
	return r, nil
}

func Search(store, query string, limit int) ([]Match, error) {
	if limit <= 0 || limit > 100 {
		return nil, errors.New("limit must be between 1 and 100")
	}
	entries, err := os.ReadDir(store)
	if errors.Is(err, os.ErrNotExist) {
		return []Match{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("storage unavailable: %w", err)
	}
	q := strings.ToLower(strings.TrimSpace(query))
	out := []Match{}
	for _, e := range entries {
		if e.IsDir() || !strings.HasPrefix(e.Name(), "study_") || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		id := strings.TrimSuffix(e.Name(), ".json")
		r, err := Get(store, id)
		if err != nil {
			return nil, err
		}
		b, _ := json.Marshal(r)
		if q == "" || strings.Contains(strings.ToLower(string(b)), q) {
			out = append(out, Match{ID: id, Record: r})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}
