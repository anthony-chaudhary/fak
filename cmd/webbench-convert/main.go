// webbench-convert converts WebVoyager dataset to webbench format.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// WebVoyagerTask represents a task from the WebVoyager dataset.
type WebVoyagerTask struct {
	WebName string `json:"web_name"`
	ID      string `json:"id"`
	Ques    string `json:"ques"`
	Web     string `json:"web"`
}

// WebbenchInstance represents a task in webbench format.
type WebbenchInstance struct {
	TaskID          string   `json:"task_id"`
	Benchmark       string   `json:"benchmark"`
	SourceURL       string   `json:"source_url"`
	Description     string   `json:"description"`
	Instructions    string   `json:"instructions"`
	Actions         []Action `json:"actions,omitempty"`
	SuccessCriteria string   `json:"success_criteria"`
	ExpectedOutput  string   `json:"expected_output"`
	Difficulty      string   `json:"difficulty,omitempty"`
	Category        string   `json:"category,omitempty"`
	Domain          string   `json:"domain,omitempty"`
}

// Action represents an expected action in the task.
type Action struct {
	Type   string `json:"type"`
	Target string `json:"target,omitempty"`
	Value  string `json:"value,omitempty"`
	Tokens int    `json:"tokens,omitempty"`
}

// difficultyFromWeb derives a difficulty rating from the website name.
func difficultyFromWeb(webName string) string {
	// Map known websites to difficulty based on task complexity.
	// These are heuristic estimates since WebVoyager doesn't provide difficulty.
	switch webName {
	case "Allrecipes":
		return "medium"
	case "Amazon":
		return "hard"
	case "Apple":
		return "hard"
	case "BBC":
		return "medium"
	case "Coursera":
		return "hard"
	case "ESPN":
		return "easy"
	case "Globotours":
		return "hard"
	case "Google":
		return "medium"
	case "Google Flights":
		return "medium"
	case "Google Maps":
		return "medium"
	case "Google Search":
		return "easy"
	case "Instagram":
		return "hard"
	case "NIKE":
		return "medium"
	case "Reddit":
		return "medium"
	case "Wikipedia":
		return "easy"
	case "Youtube":
		return "easy"
	default:
		return "medium"
	}
}

// categoryFromWeb derives a task category from the website name.
func categoryFromWeb(webName string) string {
	switch webName {
	case "Allrecipes", "Amazon", "NIKE":
		return "shopping"
	case "Apple", "BBC", "Coursera", "Reddit", "Wikipedia":
		return "information"
	case "ESPN", "Youtube":
		return "media"
	case "Google Flights", "Globotours":
		return "travel"
	case "Google Maps":
		return "navigation"
	case "Google Search":
		return "search"
	case "Instagram":
		return "social"
	default:
		return "general"
	}
}

// extractDomain extracts the domain from a URL.
func extractDomain(url string) string {
	parts := strings.Split(url, "://")
	if len(parts) < 2 {
		return url
	}
	domain := parts[1]
	if idx := strings.Index(domain, "/"); idx != -1 {
		domain = domain[:idx]
	}
	return domain
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: webbench-convert <input.jsonl> <output.jsonl>\n")
		os.Exit(1)
	}

	inputPath := os.Args[1]
	outputPath := os.Args[2]

	// Read input file.
	data, err := os.ReadFile(inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input file: %v\n", err)
		os.Exit(1)
	}

	// Open output file.
	outFile, err := os.Create(outputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output file: %v\n", err)
		os.Exit(1)
	}
	defer outFile.Close()

	lines := strings.Split(string(data), "\n")
	converted := 0
	errors := 0

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var wvTask WebVoyagerTask
		if err := json.Unmarshal([]byte(line), &wvTask); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing line: %v\n", err)
			errors++
			continue
		}

		// Convert to webbench format.
		wbTask := WebbenchInstance{
			TaskID:          wvTask.ID,
			Benchmark:       "WebVoyager",
			SourceURL:       wvTask.Web,
			Description:     wvTask.Ques,
			Instructions:    wvTask.Ques, // Use question as instructions.
			SuccessCriteria: "Task completed successfully per the question.",
			ExpectedOutput:  "Task completed",
			Difficulty:      difficultyFromWeb(wvTask.WebName),
			Category:        categoryFromWeb(wvTask.WebName),
			Domain:          extractDomain(wvTask.Web),
		}

		// Marshal to JSON. lineage:exempt — this emits a JSONL stream of converted
		// WebVoyager task fixtures (benchmark INPUT, one task object per line), not a
		// single benchmark RESULT artifact; a per-line lineage block would corrupt the
		// task schema. The lineage lives on the result emitters that consume these (#9).
		jsonBytes, err := json.Marshal(wbTask)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error marshaling task: %v\n", err)
			errors++
			continue
		}

		// Write to output file.
		if _, err := outFile.Write(append(jsonBytes, '\n')); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output: %v\n", err)
			os.Exit(1)
		}
		converted++
	}

	fmt.Fprintf(os.Stderr, "Converted %d tasks, %d errors\n", converted, errors)
}
