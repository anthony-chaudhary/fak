package papers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
)

const EnvIndex = "FAK_PAPERS_INDEX"

// DefaultIndexPath resolves flag > FAK_PAPERS_INDEX > <repo-root>/.fak/papers/index.jsonl.
func DefaultIndexPath(flagValue string, getenv func(string) string, cwd string) string {
	if flagValue != "" {
		return flagValue
	}
	if getenv != nil {
		if v := getenv(EnvIndex); v != "" {
			return v
		}
	}
	return filepath.Join(repoRoot(cwd), ".fak", "papers", "index.jsonl")
}

func repoRoot(dir string) string {
	d := dir
	for {
		if _, err := os.Stat(filepath.Join(d, ".git")); err == nil {
			return d
		}
		parent := filepath.Dir(d)
		if parent == d {
			return dir
		}
		d = parent
	}
}

func LoadIndex(path string) ([]Paper, error) {
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []Paper
	sc := bufio.NewScanner(bytes.NewReader(b))
	sc.Buffer(make([]byte, 0, 64*1024), 16<<20)
	line := 0
	for sc.Scan() {
		line++
		raw := bytes.TrimSpace(sc.Bytes())
		if len(raw) == 0 {
			continue
		}
		var p Paper
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		out = append(out, p)
	}
	return out, sc.Err()
}

// SaveIndex writes the index sorted by ID via temp file + rename.
func SaveIndex(path string, ps []Paper) error {
	sorted := append([]Paper(nil), ps...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	for _, p := range sorted {
		if err := enc.Encode(p); err != nil {
			return err
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".index-*.jsonl.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(buf.Bytes()); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		os.Remove(tmpName)
		return err
	}
	return nil
}

type UpsertStats struct {
	New     int `json:"new"`
	Updated int `json:"updated"`
	Total   int `json:"total"`
}

func Upsert(existing, incoming []Paper) ([]Paper, UpsertStats) {
	byID := make(map[string]int, len(existing))
	out := append([]Paper(nil), existing...)
	for i, p := range out {
		byID[p.ID] = i
	}
	var st UpsertStats
	touched := map[string]bool{}
	for _, p := range incoming {
		if p.ID == "" {
			continue
		}
		i, ok := byID[p.ID]
		if !ok {
			byID[p.ID] = len(out)
			out = append(out, p)
			st.New++
			touched[p.ID] = true
			continue
		}
		merged := Merge(out[i], p)
		if !sameContent(out[i], merged) && !touched[p.ID] {
			st.Updated++
			touched[p.ID] = true
		}
		out[i] = merged
	}
	st.Total = len(out)
	return out, st
}

func sameContent(a, b Paper) bool {
	a.FetchedAt, b.FetchedAt = "", ""
	return reflect.DeepEqual(a, b)
}
