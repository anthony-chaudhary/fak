// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// DIRECT-PORT of the incremental SSE event reader from ai-dynamo/aiperf
// (AsyncSSEStreamReader.__aiter__), pinned at commit
// ab610add4e8876fca223197596c728e17365f381 (aiperf v0.14.0):
//
//	https://github.com/ai-dynamo/aiperf/blob/ab610add4e8876fca223197596c728e17365f381/src/aiperf/transports/sse_utils.py
//
// The upstream algorithm is preserved byte-for-byte in behavior: a consumed
// offset instead of front-deletion (no O(remaining) memmove per message), a
// remembered search offset that backs up by longest-delimiter-minus-one (3
// bytes) so a delimiter straddling two chunks is re-scanned, \n\n checked
// first with \r\n\r\n only as fallback, padding stripped from message ends,
// and buffer compaction once per chunk instead of per message.

package main

import (
	"bytes"
	"strings"
	"time"
)

// sseMessagePadding is the byte-level padding set stripped from complete SSE
// message bodies. Bytes-level so non-ASCII separators survive into the parser.
const sseMessagePadding = " \t\r\n"

// The only event delimiters checked; see feed for why \n\n leads and \r\n\r\n
// is fallback-only.
var (
	sseDelimLF   = []byte("\n\n")
	sseDelimCRLF = []byte("\r\n\r\n")
)

// sseStreamParser incrementally parses an SSE byte stream into complete event
// bodies. It tracks a consumed offset instead of deleting from the front on
// every message, which avoids an O(remaining) copy per message; the buffer is
// compacted once per chunk instead.
type sseStreamParser struct {
	buf          []byte
	consumed     int // bytes already yielded as part of complete messages
	searchOffset int // where the next delimiter search starts (avoids re-scanning checked bytes)
}

// sseMessage is one complete SSE event body plus the arrival time of the
// network chunk that completed it. Stamping at chunk arrival (not at line or
// parse time) is what makes TTFT and inter-chunk measurements accurate.
type sseMessage struct {
	body string
	at   time.Time
}

func newSSEStreamParser() *sseStreamParser { return &sseStreamParser{} }

// feed appends one network chunk to the parser and returns every complete SSE
// message it contains, each stamped with now (the chunk's arrival time). A
// delimiter can span two chunks: all but its last byte may already be in the
// buffer from the previous chunk, so the next search backs up by the length of
// the longest delimiter (\r\n\r\n) minus 1, i.e. 3 bytes, to re-scan that
// overlap region.
//
// Only \n\n and \r\n\r\n are checked. The spec also allows \r\r and mixed
// combos like \n\r\n, but no real LLM server produces those. \n\n is checked
// first (more common with LLM servers) and \r\n\r\n is only used when absent:
// a server uses one delimiter style for the entire stream, so searching for
// both and picking the earliest would double the find cost on every iteration
// for no practical correctness gain.
func (p *sseStreamParser) feed(chunk []byte, now time.Time) []sseMessage {
	p.buf = append(p.buf, chunk...)
	var out []sseMessage
	for {
		// Invariant: consumed <= len(buf) and searchOffset >= consumed, so
		// scanFrom below is always a valid slice index.
		scanFrom := p.consumed
		if back := p.searchOffset - 3; back > scanFrom {
			scanFrom = back
		}
		delimIndex := bytes.Index(p.buf[scanFrom:], sseDelimLF)
		delimLength := 2
		if delimIndex < 0 {
			delimIndex = bytes.Index(p.buf[scanFrom:], sseDelimCRLF)
			delimLength = 4
		}
		if delimIndex < 0 {
			p.searchOffset = len(p.buf)
			break
		}
		delimIndex += scanFrom
		rawMessage := strings.Trim(string(p.buf[p.consumed:delimIndex]), sseMessagePadding)
		p.consumed = delimIndex + delimLength
		p.searchOffset = p.consumed
		if rawMessage == "" {
			continue // empty SSE message: skip
		}
		out = append(out, sseMessage{body: rawMessage, at: now})
	}
	// Compact: discard consumed bytes once per chunk instead of per message.
	if p.consumed > 0 {
		rest := copy(p.buf, p.buf[p.consumed:])
		p.buf = p.buf[:rest]
		p.searchOffset -= p.consumed
		p.consumed = 0
	}
	return out
}

// finish flushes any remaining data in the buffer after the stream ends. Some
// servers do not send the final delimiter; the trailing partial message is
// returned stamped with now (the end-of-stream time). After finish the parser
// is empty and further calls yield nothing.
func (p *sseStreamParser) finish(now time.Time) []sseMessage {
	remaining := strings.Trim(string(p.buf[p.consumed:]), sseMessagePadding)
	p.buf = nil
	p.consumed = 0
	p.searchOffset = 0
	if remaining == "" {
		return nil
	}
	return []sseMessage{{body: remaining, at: now}}
}
