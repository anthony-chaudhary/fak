package dispatchcache

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const BacklogSchema = "fak.dispatch-backlog.v1"

type BacklogIssue struct {
	Number int             `json:"number"`
	Data   json.RawMessage `json:"data"`
}

type BacklogSnapshot struct {
	Schema    string         `json:"schema"`
	Key       string         `json:"key"`
	Watermark time.Time      `json:"watermark"`
	Issues    []BacklogIssue `json:"issues"`
}

func MergeBacklog(base []BacklogIssue, changed []BacklogIssue, closed []int) []BacklogIssue {
	byNumber := make(map[int]json.RawMessage, len(base)+len(changed))
	for _, row := range base {
		if row.Number > 0 {
			byNumber[row.Number] = append(json.RawMessage(nil), row.Data...)
		}
	}
	for _, number := range closed {
		delete(byNumber, number)
	}
	for _, row := range changed {
		if row.Number > 0 {
			byNumber[row.Number] = append(json.RawMessage(nil), row.Data...)
		}
	}
	out := make([]BacklogIssue, 0, len(byNumber))
	for number, data := range byNumber {
		out = append(out, BacklogIssue{Number: number, Data: data})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Number < out[j].Number })
	return out
}

func WriteBacklog(path, key string, watermark time.Time, issues []BacklogIssue) error {
	if path == "" || key == "" {
		return errors.New("dispatchcache: backlog path and key are required")
	}
	b, err := json.Marshal(BacklogSnapshot{Schema: BacklogSchema, Key: key, Watermark: watermark.UTC(), Issues: issues})
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".backlog-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(append(b, '\n')); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return err
	}
	return os.Rename(name, path)
}

func ReadBacklog(path, key string) (BacklogSnapshot, bool) {
	var s BacklogSnapshot
	b, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(b, &s) != nil || s.Schema != BacklogSchema || s.Key != key || s.Watermark.IsZero() {
		return BacklogSnapshot{}, false
	}
	return s, true
}
