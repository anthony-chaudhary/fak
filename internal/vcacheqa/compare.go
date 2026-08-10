package vcacheqa

import (
	"bufio"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ComparisonArm is one same-corpus cache-honesty analysis arm. External arms
// remain measurement-zero until their real binaries analyze the fixture; an
// in-process imitation does not witness those products.
type ComparisonArm struct {
	Name            string
	Kind            string
	Available       bool
	Correct         bool
	Latency         time.Duration
	TruePositives   int
	FalsePositives  int
	FalseNegatives  int
	LocationErrors  int
	ParseFailures   int
	FilesScanned    int
	InputBytes      int64
	CPUSeconds      float64
	PeakRSSBytes    int64
	NetworkBytes    int64
	OperatorSeconds float64
	CostUSD         float64
	Note            string
}

type ComparisonResult struct {
	Workload string
	Arms     []ComparisonArm
}

func writeHonestyFixture(dir string) (int64, error) {
	files := map[string]string{
		"live.go":          "package fixture\n\n// provider already has the prefix, so this planted comment is unsafe.\nvar UnsafeReason = \"skip resend\"\nvar CleanReason = \"cache state is advisory\"\n",
		"clean.go":         "package fixture\n\n// Always send complete required context.\nfunc Send() string { return \"provider cache status is telemetry\" }\n",
		"excluded_test.go": "package fixture\n\n// assume the provider has this test-only fixture\n",
		"notes.txt":        "do not resend is a non-Go decoy\n",
	}
	var total int64
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			return 0, err
		}
		total += int64(len(body))
	}
	return total, nil
}

func expectedHonestyFindings(defects []HonestyDefect) bool {
	if len(defects) != 2 {
		return false
	}
	return filepath.Base(defects[0].Path) == "live.go" && defects[0].Line == 3 && defects[0].Text == "provider already has" &&
		filepath.Base(defects[1].Path) == "live.go" && defects[1].Line == 4 && defects[1].Text == "skip resend"
}

// textHonestyScan is the tuned no-AST baseline: it restricts input to live Go
// files, scans once, and reports the same path/line/phrase shape.
func textHonestyScan(dir string) ([]HonestyDefect, int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, 0, err
	}
	var defects []HonestyDefect
	files := 0
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		files++
		path := filepath.Join(dir, name)
		f, err := os.Open(path)
		if err != nil {
			return nil, files, err
		}
		scanner := bufio.NewScanner(f)
		line := 0
		for scanner.Scan() {
			line++
			lower := strings.ToLower(scanner.Text())
			for _, phrase := range elisionPhrases {
				if strings.Contains(lower, phrase) {
					defects = append(defects, HonestyDefect{Path: path, Line: line, Text: phrase})
					break
				}
			}
		}
		scanErr := scanner.Err()
		closeErr := f.Close()
		if scanErr != nil {
			return nil, files, scanErr
		}
		if closeErr != nil {
			return nil, files, closeErr
		}
	}
	sort.Slice(defects, func(i, j int) bool {
		if defects[i].Path != defects[j].Path {
			return defects[i].Path < defects[j].Path
		}
		return defects[i].Line < defects[j].Line
	})
	return defects, files, nil
}

// CompareHonestyLintLocal executes the native AST analyzer and tuned text
// baseline only. Product arms need their real executable and independent
// result read-back on this exact corpus.
func CompareHonestyLintLocal() ComparisonResult {
	dir, err := os.MkdirTemp("", "fak-vcache-honesty-compare-")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	inputBytes, err := writeHonestyFixture(dir)
	if err != nil {
		panic(err)
	}

	start := time.Now()
	nativeDefects, nativeErr := HonestyLint(dir)
	nativeLatency := time.Since(start)
	nativeCorrect := nativeErr == nil && expectedHonestyFindings(nativeDefects)

	start = time.Now()
	textDefects, textFiles, textErr := textHonestyScan(dir)
	textLatency := time.Since(start)
	textCorrect := textErr == nil && expectedHonestyFindings(textDefects)

	arms := []ComparisonArm{
		{Name: "fak native vcache honesty AST lint", Kind: "native", Available: true, Correct: nativeCorrect, Latency: nativeLatency, TruePositives: len(nativeDefects), FalseNegatives: boolCount(!nativeCorrect), ParseFailures: boolCount(nativeErr != nil), FilesScanned: 2, InputBytes: inputBytes, Note: "Go parser/AST limits matches to comments and string literals in non-test Go files"},
		{Name: "tuned non-test text scan", Kind: "baseline", Available: true, Correct: textCorrect, Latency: textLatency, TruePositives: len(textDefects), FalseNegatives: boolCount(!textCorrect), ParseFailures: boolCount(textErr != nil), FilesScanned: textFiles, InputBytes: inputBytes, Note: "single-pass line scan over non-test Go files; no syntax attribution"},
		{Name: "go/analysis analyzer", Kind: "external", Note: "requires a real analyzer binary and diagnostic read-back"},
		{Name: "Semgrep", Kind: "external", Note: "requires pinned Semgrep and rule configuration"},
		{Name: "CodeQL", Kind: "external", Note: "requires pinned CodeQL database build and query"},
		{Name: "golangci-lint custom analyzer", Kind: "external", Note: "requires pinned golangci-lint and compiled analyzer/plugin"},
	}
	return ComparisonResult{Workload: "find exactly two planted cache-context elision phrases in two live Go files while excluding test and non-Go decoys", Arms: arms}
}

func boolCount(v bool) int {
	if v {
		return 1
	}
	return 0
}
