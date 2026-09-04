package issueorchestrator

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// LoadIssues reads issues from a JSON file, stdin ("-"), or default workspace files.
func LoadIssues(path string, workspace string) ([]Issue, error) {
	var data []byte
	var err error

	switch {
	case path == "-":
		data, err = io.ReadAll(os.Stdin)
		if err != nil {
			return nil, fmt.Errorf("read stdin: %w", err)
		}
	case path != "":
		data, err = os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
	default:
		// Try default locations
		candidates := []string{
			filepath.Join(workspace, ".fak", "issues.json"),
			filepath.Join(workspace, "issues.json"),
		}
		found := false
		for _, c := range candidates {
			if b, readErr := os.ReadFile(c); readErr == nil && len(b) > 0 {
				data = b
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("no issues file specified and none found at .fak/issues.json or issues.json; use --from-issues <path>")
		}
	}

	return DecodeIssues(data)
}

// DecodeIssues parses raw JSON bytes into a slice of Issues.
func DecodeIssues(b []byte) ([]Issue, error) {
	trimmed := strings.TrimSpace(string(b))
	if trimmed == "" {
		return nil, fmt.Errorf("empty issue payload")
	}

	// 1. Try decoding as []Issue directly
	var directIssues []Issue
	if err := json.Unmarshal(b, &directIssues); err == nil && len(directIssues) > 0 && directIssues[0].Title != "" {
		return directIssues, nil
	}

	// 2. Try decoding as []issuepolicy.IssueDraft
	var drafts []issuepolicy.IssueDraft
	if err := json.Unmarshal(b, &drafts); err == nil && len(drafts) > 0 {
		var issues []Issue
		for _, d := range drafts {
			cand := issuepolicy.CandidateFromIssueDraft(d)
			rev := issuepolicy.ReviewIssueDraft(d, issuepolicy.Options{})
			var lbls []string
			for _, l := range d.Labels {
				if l.Name != "" {
					lbls = append(lbls, l.Name)
				}
			}
			issues = append(issues, Issue{
				Number:          d.Number,
				Key:             cand.Key,
				Title:           d.Title,
				Lane:            rev.Lane,
				Paths:           append([]string(nil), rev.Paths...),
				ExpectedSteps:   rev.ExpectedSteps,
				Labels:          lbls,
				URL:             d.URL,
				Centrality:      string(cand.ProblemFrame.Centrality),
				ProblemFrame:    cand.ProblemFrame,
				Dispatchability: rev.Dispatchability,
			})
		}
		return issues, nil
	}

	// 3. Try decoding as []issuepolicy.Candidate
	var candidates []issuepolicy.Candidate
	if err := json.Unmarshal(b, &candidates); err == nil && len(candidates) > 0 {
		var issues []Issue
		for _, c := range candidates {
			rev := issuepolicy.ReviewCandidate(c, issuepolicy.Options{})
			num := rev.IssueNumber
			if num == 0 {
				num = parseNumberFromKey(c.Key)
			}
			issues = append(issues, Issue{
				Number:          num,
				Key:             c.Key,
				Title:           c.Title,
				Lane:            rev.Lane,
				Paths:           append([]string(nil), rev.Paths...),
				ExpectedSteps:   rev.ExpectedSteps,
				Labels:          append([]string(nil), c.Labels...),
				Centrality:      string(c.ProblemFrame.Centrality),
				ProblemFrame:    c.ProblemFrame,
				Dispatchability: rev.Dispatchability,
			})
		}
		return issues, nil
	}

	// 4. Try decoding object with "issues", "items", or "candidates"
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(b, &obj); err == nil {
		if raw, ok := obj["issues"]; ok {
			return DecodeIssues(raw)
		}
		if raw, ok := obj["candidates"]; ok {
			return DecodeIssues(raw)
		}
		if raw, ok := obj["items"]; ok {
			return DecodeIssues(raw)
		}
	}

	// 5. Try single issue object
	var singleDraft issuepolicy.IssueDraft
	if err := json.Unmarshal(b, &singleDraft); err == nil && singleDraft.Title != "" {
		arrBytes, _ := json.Marshal([]issuepolicy.IssueDraft{singleDraft})
		return DecodeIssues(arrBytes)
	}

	// 6. Try line-by-line JSON Lines (JSONL)
	lines := strings.Split(trimmed, "\n")
	if len(lines) > 1 {
		var jsonlIssues []Issue
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			if itemIssues, err := DecodeIssues([]byte(line)); err == nil && len(itemIssues) > 0 {
				jsonlIssues = append(jsonlIssues, itemIssues...)
			}
		}
		if len(jsonlIssues) > 0 {
			return jsonlIssues, nil
		}
	}

	// 7. Try generic maps with id/number and paths
	var genericMaps []map[string]any
	if err := json.Unmarshal(b, &genericMaps); err == nil && len(genericMaps) > 0 {
		var genericIssues []Issue
		for _, m := range genericMaps {
			var num int
			if n, ok := m["number"].(float64); ok {
				num = int(n)
			} else if idStr, ok := m["id"].(string); ok {
				if n, err := strconv.Atoi(idStr); err == nil {
					num = n
				}
			}
			title, _ := m["title"].(string)
			if title == "" {
				title = fmt.Sprintf("Issue #%d", num)
			}
			key, _ := m["key"].(string)
			if key == "" {
				key = fmt.Sprintf("issue-%d", num)
			}
			lane, _ := m["lane"].(string)
			var paths []string
			if pRaw, ok := m["paths"].([]any); ok {
				for _, p := range pRaw {
					if ps, ok := p.(string); ok {
						paths = append(paths, ps)
					}
				}
			}
			var steps int
			if s, ok := m["expected_steps"].(float64); ok {
				steps = int(s)
			} else if s, ok := m["steps"].(float64); ok {
				steps = int(s)
			}
			if steps == 0 {
				steps = 3
			}
			genericIssues = append(genericIssues, Issue{
				Number:          num,
				Key:             key,
				Title:           title,
				Lane:            lane,
				Paths:           paths,
				ExpectedSteps:   steps,
				Dispatchability: "dispatchable",
			})
		}
		if len(genericIssues) > 0 {
			return genericIssues, nil
		}
	}

	var singleGeneric map[string]any
	if err := json.Unmarshal(b, &singleGeneric); err == nil && len(singleGeneric) > 0 {
		arrBytes, _ := json.Marshal([]map[string]any{singleGeneric})
		return DecodeIssues(arrBytes)
	}

	return nil, fmt.Errorf("unable to parse issue payload into known issue, draft, or candidate schemas")
}
