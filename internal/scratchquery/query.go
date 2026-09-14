// Package scratchquery is the read-only CONTENT half of the harness session
// scratchpad: `fak scratch-janitor` owns the lifecycle (age + resume reference),
// and this leaf owns "what did session <id> leave in here?".
//
// It is deliberately two deterministic, stdlib-only queries over a scratchpad
// directory:
//
//   - Query: a bounded literal content match, the read half of the scratchpad
//     concept (docs/notes/CONCEPT-HARNESS-SESSION-SCRATCHPAD-2026-07-02.md).
//   - Promotable: the promotion-candidate surface (see promote.go) that flags
//     parked artifacts looking like unfiled session flags before the reap window.
//
// Invariant: inspection is fail-closed and deterministic. A Query over a missing
// or empty scratchpad yields a clean empty result, never an error; a file above
// MaxFileBytes is skipped (its size, not its content, is reported); a binary file
// is skipped. Results are sorted by (file, line) so two runs over the same tree
// are byte-identical.
package scratchquery

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	// MaxFileBytes bounds the content of any ONE scratchpad file a query reads.
	// A blob larger than this is reported as skipped, never streamed into a hit
	// list: the scratchpad is the busiest artifact surface on the fleet and an
	// unbounded read is how a query floods an agent's context.
	MaxFileBytes = 1 << 20 // 1 MiB

	// binarySniffBytes is how much of a file's head is inspected to decide it is
	// binary. A NUL byte in this prefix (the same conservative test `grep -I`
	// uses) marks the file binary and out of scope.
	binarySniffBytes = 8 * 1024
)

// Hit is one matching line in one scratchpad file. Line is 1-based; Match is the
// trimmed line text, so a caller reads what matched without re-opening the file.
type Hit struct {
	File  string `json:"file"`
	Line  int    `json:"line"`
	Match string `json:"match"`
	Bytes int    `json:"bytes,omitempty"`
}

// Skipped is a file a query declined to read, with the closed reason why. It is
// surfaced (never silently dropped) so "no hits" is distinguishable from "not
// searched".
type Skipped struct {
	File   string `json:"file"`
	Reason string `json:"reason"`
	Bytes  int    `json:"bytes"`
}

const (
	// SkipTooLarge marks a file above MaxFileBytes.
	SkipTooLarge = "too_large"
	// SkipBinary marks a file whose head contains a NUL byte.
	SkipBinary = "binary"
)

// QueryResult is the deterministic answer to one content query over one
// scratchpad directory.
type QueryResult struct {
	Root      string    `json:"root"`
	Pattern   string    `json:"pattern"`
	Hits      []Hit     `json:"hits"`
	Skipped   []Skipped `json:"skipped"`
	Files     int       `json:"files"`
	Matched   int       `json:"matched"`
	Truncated bool      `json:"truncated,omitempty"`
}

// QueryOptions is the bounded input to Query. Pattern is a literal (or, with
// Regex, a RE2) substring; Limit caps returned hits (0 = unlimited).
type QueryOptions struct {
	Pattern string
	Regex   bool
	Limit   int
}

// Query runs a bounded, read-only content match over every regular file under
// root. A missing root or an empty directory returns a clean empty result (no
// error): the caller's question "what did this session leave here?" has the
// answer "nothing", which is not a failure.
func Query(root string, opts QueryOptions) (QueryResult, error) {
	result := QueryResult{Root: root, Pattern: opts.Pattern, Hits: []Hit{}, Skipped: []Skipped{}}
	if strings.TrimSpace(root) == "" {
		return result, fmt.Errorf("root is required")
	}
	pattern := opts.Pattern

	info, err := os.Stat(root)
	if err != nil {
		if os.IsNotExist(err) {
			return result, nil // an absent scratchpad is an empty one
		}
		return result, fmt.Errorf("stat root %q: %w", root, err)
	}
	if !info.IsDir() {
		return result, fmt.Errorf("root %q is not a directory", root)
	}

	files, err := scratchFiles(root)
	if err != nil {
		return result, err
	}
	var re *regexp.Regexp
	if opts.Regex && pattern != "" {
		compiled, err := regexp.Compile(pattern)
		if err != nil {
			return result, fmt.Errorf("bad --regex pattern %q: %w", pattern, err)
		}
		re = compiled
	}
	for _, rel := range files {
		full := filepath.Join(root, filepath.FromSlash(rel))
		fi, err := os.Stat(full)
		if err != nil {
			continue // a file that vanished mid-scan simply drops out
		}
		if fi.Size() > MaxFileBytes {
			result.Skipped = append(result.Skipped, Skipped{File: rel, Reason: SkipTooLarge, Bytes: int(fi.Size())})
			continue
		}
		data, binary, err := readBounded(full)
		if err != nil {
			continue // unreadable file is skipped, never fatal
		}
		if binary {
			result.Skipped = append(result.Skipped, Skipped{File: rel, Reason: SkipBinary, Bytes: len(data)})
			continue
		}
		result.Files++
		fileHits := matchLines(rel, data, pattern, re)
		if len(fileHits) > 0 {
			result.Matched++
			result.Hits = append(result.Hits, fileHits...)
		}
	}

	sortHits(result.Hits)
	sortSkipped(result.Skipped)
	if opts.Limit > 0 && len(result.Hits) > opts.Limit {
		result.Hits = result.Hits[:opts.Limit]
		result.Truncated = true
	}
	return result, nil
}

// File returns a single scratchpad file's bounded content lines, so a caller can
// read one artifact without walking the whole directory. It is the read half the
// hit list points back into; a missing file is an empty result, never an error.
func File(root, rel string) ([]string, error) {
	if strings.TrimSpace(rel) == "" {
		return nil, fmt.Errorf("file is required")
	}
	clean := filepath.Clean(rel)
	if filepath.IsAbs(clean) || strings.HasPrefix(clean, "..") {
		return nil, fmt.Errorf("file %q escapes the scratchpad root", rel)
	}
	full := filepath.Join(root, clean)
	fi, err := os.Stat(full)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("stat %q: %w", full, err)
	}
	if fi.Size() > MaxFileBytes {
		return nil, fmt.Errorf("file %q is %d bytes, above the %d-byte cap", rel, fi.Size(), MaxFileBytes)
	}
	data, binary, err := readBounded(full)
	if err != nil {
		return nil, err
	}
	if binary {
		return nil, fmt.Errorf("file %q is binary", rel)
	}
	return splitLines(data), nil
}

// scratchFiles walks root and returns every regular file as a slash-separated
// path relative to root, sorted so the scan order is deterministic.
func scratchFiles(root string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil // skip an unreadable subtree rather than abort the scan
		}
		if d.IsDir() {
			return nil
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return nil
		}
		out = append(out, filepath.ToSlash(rel))
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk %q: %w", root, err)
	}
	sort.Strings(out)
	return out, nil
}

// readBounded reads at most MaxFileBytes of a file and reports whether its head
// looks binary (a NUL byte in the first binarySniffBytes).
func readBounded(path string) ([]byte, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, MaxFileBytes))
	if err != nil {
		return nil, false, err
	}
	return data, looksBinary(data), nil
}

// looksBinary reports whether the head of data contains a NUL byte, the same
// conservative test `grep -I` applies.
func looksBinary(data []byte) bool {
	head := data
	if len(head) > binarySniffBytes {
		head = head[:binarySniffBytes]
	}
	for _, b := range head {
		if b == 0 {
			return true
		}
	}
	return false
}

// matchLines returns the hits for one file's content. An empty pattern matches
// nothing (a query with no pattern is a listing, not a flood); the match text is
// the trimmed full line, so a hit is readable without re-opening the artifact.
func matchLines(rel string, data []byte, pattern string, re *regexp.Regexp) []Hit {
	if pattern == "" {
		return nil
	}
	lines := splitLines(data)
	var hits []Hit
	for i, raw := range lines {
		line := strings.TrimRight(raw, "\r")
		var matched bool
		if re != nil {
			matched = re.MatchString(line)
		} else {
			matched = strings.Contains(line, pattern)
		}
		if !matched {
			continue
		}
		hits = append(hits, Hit{File: rel, Line: i + 1, Match: strings.TrimSpace(line), Bytes: len(data)})
	}
	return hits
}

// splitLines splits on \n, dropping a trailing empty element so "a\n" is one line.
func splitLines(data []byte) []string {
	if len(data) == 0 {
		return nil
	}
	text := strings.TrimSuffix(string(data), "\n")
	if text == "" {
		return nil
	}
	return strings.Split(text, "\n")
}

// sortHits orders hits by (file, line), the deterministic order two runs agree on.
func sortHits(hits []Hit) {
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].File != hits[j].File {
			return hits[i].File < hits[j].File
		}
		return hits[i].Line < hits[j].Line
	})
}

// sortSkipped orders skipped files by (file, reason).
func sortSkipped(skipped []Skipped) {
	sort.Slice(skipped, func(i, j int) bool {
		if skipped[i].File != skipped[j].File {
			return skipped[i].File < skipped[j].File
		}
		return skipped[i].Reason < skipped[j].Reason
	})
}
