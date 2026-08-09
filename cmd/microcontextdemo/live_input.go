package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"
)

const (
	maxLiveInputBytes   = 1 << 20
	maxLiveInputRecords = 100
	maxLiveTitleRunes   = 160
)

// liveWorkUnit is the deliberately narrow public read-back of repository work.
// Bodies, paths, credentials, and arbitrary label metadata never enter output.
type liveWorkUnit struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
}

type liveIssueRecord struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
}

func loadLiveWorkUnits(path string) ([]liveWorkUnit, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open live issues: %w", err)
	}
	defer f.Close()
	var records []liveIssueRecord
	dec := json.NewDecoder(io.LimitReader(f, maxLiveInputBytes+1))
	if err := dec.Decode(&records); err != nil {
		return nil, fmt.Errorf("decode live issues: %w", err)
	}
	if len(records) == 0 {
		return nil, fmt.Errorf("live issues snapshot is empty")
	}
	if len(records) > maxLiveInputRecords {
		return nil, fmt.Errorf("live issues snapshot has %d records; limit is %d", len(records), maxLiveInputRecords)
	}
	seen := make(map[int]struct{}, len(records))
	units := make([]liveWorkUnit, 0, len(records))
	for _, record := range records {
		if record.Number < 1 {
			return nil, fmt.Errorf("live issue number must be positive")
		}
		if _, ok := seen[record.Number]; ok {
			return nil, fmt.Errorf("duplicate live issue #%d", record.Number)
		}
		seen[record.Number] = struct{}{}
		title := sanitizeLiveTitle(record.Title)
		if title == "" {
			return nil, fmt.Errorf("live issue #%d has an empty sanitized title", record.Number)
		}
		units = append(units, liveWorkUnit{Number: record.Number, Title: title})
	}
	return units, nil
}

func sanitizeLiveTitle(title string) string {
	title = strings.Map(func(r rune) rune {
		if unicode.IsControl(r) {
			return ' '
		}
		return r
	}, title)
	title = strings.Join(strings.Fields(title), " ")
	runes := []rune(title)
	if len(runes) > maxLiveTitleRunes {
		runes = runes[:maxLiveTitleRunes]
	}
	return string(runes)
}

func (u liveWorkUnit) workID() string { return fmt.Sprintf("issue-%d: %s", u.Number, u.Title) }
