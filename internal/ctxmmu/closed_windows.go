package ctxmmu

// Reverse history traversal adapted from SWE-agent's ClosedWindowHistoryProcessor,
// sweagent/agent/history_processors.py@3ea751c087f32b16e039a2233dd6eefecef325d5.
// Native renderer validation and byte-identical/tool-scoped keys are Fak-specific.
//
// MIT License
// Copyright (c) 2024 John Yang, Carlos E. Jimenez, Alexander Wettig, Shunyu Yao,
// Karthik Narasimhan, Ofir Press
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"regexp"
	"slices"
	"strconv"
)

var fileWindowHeader = regexp.MustCompile(`^=== \[(.+)\] \(lines ([0-9]+)-([0-9]+) of ([0-9]+)\) ===$`)

type closedFileWindow struct {
	duplicate bool
	tombstone Tombstone
	content   []byte
}

// fileWindows recognizes only complete, standalone toolbound.WindowView output.
// Tool names are supplied by the trusted caller, never inferred from output text.
// The payload digest keeps changed contents and different line ranges distinct;
// it is an equality key, not evidence of a filesystem revision.
func (c *Compactor) fileWindows(pages []TokenPage) map[int]closedFileWindow {
	if !c.config.DeduplicateFileWindows || len(c.config.FileWindowTools) == 0 {
		return nil
	}
	type key struct {
		tool   string
		digest [32]byte
	}
	seen := make(map[key]int)
	windows := make(map[int]closedFileWindow)
	for i := len(pages) - 1; i >= 0; i-- {
		p := &pages[i]
		if p.Kind != PageKindToolResult || p.Role != "tool" || p.ToolName == "" || p.TurnIndex < 1 ||
			p.Tombstone.Active || p.IsContinuation ||
			(i+1 < len(pages) && pages[i+1].IsContinuation) ||
			!slices.Contains(c.config.FileWindowTools, p.ToolName) {
			continue
		}
		path, start, end, ok := parseFileWindow(p.Content)
		if !ok {
			continue
		}
		digest := sha256.Sum256(p.Content)
		k := key{p.ToolName, digest}
		newer, duplicate := seen[k]
		duplicate = duplicate && bytes.Equal(p.Content, pages[newer].Content)
		seen[k] = i
		w := closedFileWindow{duplicate: duplicate}
		if duplicate {
			tokens := p.Tokens
			if tokens <= 0 {
				tokens = EstimateTokens(p.Content)
			}
			w.tombstone = Tombstone{
				Active: true, Digest: digest, OriginalBytes: len(p.Content),
				OriginalTokens: tokens, Tool: p.ToolName,
				Summary: fmt.Sprintf("File window closed: %s lines %d-%d", path, start, end),
			}
			w.content = FormatTombstone(w.tombstone, nil)
			// A short view or supplied token estimate may cost less than the marker.
			if len(w.content) >= len(p.Content) || EstimateTokens(w.content) >= tokens {
				w.duplicate = false
			}
		}
		windows[i] = w
	}
	return windows
}

func parseFileWindow(body []byte) (path string, start, end int, ok bool) {
	header, rest, found := bytes.Cut(body, []byte{'\n'})
	if !found {
		return
	}
	m := fileWindowHeader.FindSubmatch(header)
	if m == nil {
		return
	}
	path = string(m[1])
	start, err := strconv.Atoi(string(m[2]))
	if err != nil || start < 1 {
		return path, 0, 0, false
	}
	end, err = strconv.Atoi(string(m[3]))
	if err != nil || end < start {
		return path, 0, 0, false
	}
	total, err := strconv.Atoi(string(m[4]))
	if err != nil || total < end {
		return path, 0, 0, false
	}
	for line := start; ; line++ {
		row, remaining, found := bytes.Cut(rest, []byte{'\n'})
		if !found || !bytes.HasPrefix(row, []byte(strconv.Itoa(line)+": ")) {
			return path, 0, 0, false
		}
		rest = remaining
		if line == end {
			break
		}
	}
	status := "End of Window"
	if end == total {
		status = "[EOF]"
	}
	footer := "\n=== Navigation: ScrollDown(n), ScrollUp(n), Goto(line) | " + status + " ==="
	return path, start, end, bytes.Equal(rest, []byte(footer))
}
