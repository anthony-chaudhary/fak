package headroom

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// NativeName is the in-process, dependency-free structural compressor: the plugin
// that proves the seam end-to-end with ZERO external service. It applies only
// safe, model-readable transforms — JSON minification (lossless), consecutive
// duplicate-line collapse with an explicit elision count, blank-run collapse, and
// trailing-whitespace trim. It is NOT Headroom's ML compression (no SmartCrusher,
// no Kompress model); it is the honest local baseline that always works offline.
// The original bytes are preserved in the shared CAS by the gate, so even the
// lossy line-collapse stays reversible (the CCR promise).
const NativeName = "native"

// dupRunMin: a run of at least this many identical consecutive lines collapses to
// one copy plus an elision marker.
const dupRunMin = 3

type nativeCompressor struct{}

func (nativeCompressor) Name() string { return NativeName }

func (nativeCompressor) Compress(_ context.Context, in Input) (Output, error) {
	orig := in.Bytes
	if len(orig) == 0 {
		return passthrough(in), nil
	}
	kind := in.Kind
	if kind == KindUnknown {
		kind = Detect(orig)
	}

	var out []byte
	var codecs []string
	if kind == KindJSON {
		if mini, ok := minifyJSON(orig); ok {
			out, codecs = mini, append(codecs, "json-min")
		}
	}
	if out == nil {
		out, codecs = normalizeLines(orig)
	}

	if len(codecs) == 0 || len(out) >= len(orig) {
		return passthrough(in), nil // nothing helped -> admit the original as-is
	}
	return Output{
		Bytes:      out,
		Compressed: true,
		Codec:      strings.Join(codecs, "+"),
		OrigLen:    len(orig),
		NewLen:     len(out),
	}, nil
}

// minifyJSON re-renders valid JSON with all insignificant whitespace removed. It
// is lossless: json.Compact preserves the document's semantics exactly.
func minifyJSON(b []byte) ([]byte, bool) {
	if !json.Valid(b) {
		return nil, false
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, b); err != nil {
		return nil, false
	}
	return buf.Bytes(), true
}

// normalizeLines collapses consecutive duplicate lines into one copy plus a count
// marker, collapses blank runs to a single blank line, and trims trailing
// whitespace. It returns the rewritten bytes and the names of the transforms that
// actually fired (empty if the text was already minimal).
func normalizeLines(b []byte) ([]byte, []string) {
	lines := strings.Split(string(b), "\n")
	out := make([]string, 0, len(lines))
	var didDedup, didBlank, didTrim bool
	blank := 0
	i := 0
	for i < len(lines) {
		ln := strings.TrimRight(lines[i], " \t\r")
		if ln != lines[i] {
			didTrim = true
		}
		if ln == "" {
			blank++
			if blank <= 1 {
				out = append(out, ln)
			} else {
				didBlank = true
			}
			i++
			continue
		}
		blank = 0
		// measure the run of identical (post-trim) lines starting at i.
		j := i + 1
		for j < len(lines) && strings.TrimRight(lines[j], " \t\r") == ln {
			j++
		}
		run := j - i
		out = append(out, ln)
		if run >= dupRunMin {
			out = append(out, fmt.Sprintf("… (×%d identical lines elided) …", run-1))
			didDedup = true
		} else {
			for k := 1; k < run; k++ { // emit the rest of a short run verbatim
				out = append(out, ln)
			}
		}
		i = j
	}
	var codecs []string
	if didDedup {
		codecs = append(codecs, "line-dedup")
	}
	if didBlank {
		codecs = append(codecs, "blank-collapse")
	}
	if didTrim {
		codecs = append(codecs, "trim")
	}
	return []byte(strings.Join(out, "\n")), codecs
}

func passthrough(in Input) Output {
	return Output{
		Bytes:      in.Bytes,
		Compressed: false,
		Codec:      "identity",
		OrigLen:    len(in.Bytes),
		NewLen:     len(in.Bytes),
	}
}

func init() { Register(nativeCompressor{}) }
