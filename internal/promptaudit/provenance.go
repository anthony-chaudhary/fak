package promptaudit

// This file adds the PROVENANCE + DIGEST layer on top of the #1691 stego
// scanner (Scan / Finding in promptaudit.go). The scanner answers "is there a
// hidden marker in this text?"; this layer answers the operator's two follow-up
// questions:
//
//  1. WHO produced the bytes that crossed the model boundary? Each context
//     field is attributed to a Source — user config, fak policy, an integration
//     adapter, a provider shim, the local environment, fetched content, or
//     unknown. The documented threat (a client mutating the boring `currentDate`
//     field from local env + a base-URL classifier) is exactly a SourceLocalEnv
//     or SourceProviderShim segment that no audit surface otherwise attributes.
//
//  2. Did a TINY marker change the request? A stable digest over the exact raw
//     segment bytes, concatenated in order, gives a fingerprint that flips when
//     two otherwise-identical prompts differ only by U+2019 vs ASCII apostrophe
//     or '/' vs '-' in a date token — so cache/debug reports can PROVE a marker
//     mutation happened.
//
// Nothing here mutates the input. The scanner is reused, never duplicated.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Source attributes a rendered context field to the producer that wrote it. The
// members are exactly the producers the ticket names; SourceUnknown is the
// fail-open default so an unattributed field is visible rather than silently
// trusted.
type Source string

const (
	// SourceUserConfig is operator/user configuration (config files, flags, a
	// user-authored system prompt).
	SourceUserConfig Source = "user-config"
	// SourceFakPolicy is text fak itself injects as policy (guard rules,
	// safety preamble, fak-authored instructions).
	SourceFakPolicy Source = "fak-policy"
	// SourceIntegration is text contributed by an integration/tool adapter
	// (an MCP server, a skill, a connected tool's instructions).
	SourceIntegration Source = "integration-adapter"
	// SourceProviderShim is text a provider-specific shim/adapter injects when
	// shaping the request for a given upstream (the base-URL classifier path the
	// article warns about lives here).
	SourceProviderShim Source = "provider-shim"
	// SourceLocalEnv is text derived from the local machine environment —
	// hostname, timezone, the current date. This is the field class the
	// steganography article mutates without an audit surface.
	SourceLocalEnv Source = "local-env"
	// SourceFetched is content fetched from a remote/untrusted source and
	// spliced into the context (a web page, a fetched document).
	SourceFetched Source = "fetched-content"
	// SourceUnknown is the fail-open default: a field whose producer is not (yet)
	// attributed. It is surfaced, never hidden.
	SourceUnknown Source = "unknown"
)

// knownSources is the closed set used to validate/normalize a Source.
var knownSources = map[Source]struct{}{
	SourceUserConfig:   {},
	SourceFakPolicy:    {},
	SourceIntegration:  {},
	SourceProviderShim: {},
	SourceLocalEnv:     {},
	SourceFetched:      {},
	SourceUnknown:      {},
}

// Valid reports whether s is a member of the closed Source vocabulary.
func (s Source) Valid() bool {
	_, ok := knownSources[s]
	return ok
}

// normalize returns s if known, else SourceUnknown — fail open, never drop a
// segment because its Source label was unrecognized.
func (s Source) normalize() Source {
	if s.Valid() {
		return s
	}
	return SourceUnknown
}

// Segment is one rendered context field as it crossed (or will cross) the model
// boundary: the EXACT bytes, who produced them, and the field's logical name.
// The normalized display form is derived (NormalizedForm) rather than stored, so
// it can never drift from Raw.
type Segment struct {
	// Field is the logical name of the context field (e.g. "currentDate",
	// "systemPrompt", "userInstructions"). Used for display and attribution.
	Field string
	// Source is the producer that wrote Raw.
	Source Source
	// Raw is the exact rendered bytes of this field that cross the model
	// boundary. This is what the digest hashes and what the scanner reads.
	Raw string
}

// NormalizedForm returns a display form of the segment's raw bytes with any
// detected markers folded to their benign default (ASCII apostrophe, '-' date
// separator) and invisible runes shown as their codepoint. It is built FROM the
// scanner's findings so the raw and normalized views can never disagree about
// what a marker was. Raw is never mutated.
func (s Segment) NormalizedForm() string {
	return normalizeWithFindings(s.Raw, Scan(s.Raw))
}

// normalizeWithFindings rewrites text by replacing each finding's Raw run with
// its Normalized rendering. Findings are applied right-to-left so earlier byte
// offsets stay valid as we splice. Overlapping findings are skipped after the
// first to keep the rewrite well-defined.
func normalizeWithFindings(text string, fs []Finding) string {
	if len(fs) == 0 {
		return text
	}
	ordered := make([]Finding, len(fs))
	copy(ordered, fs)
	sort.SliceStable(ordered, func(i, j int) bool {
		return ordered[i].ByteOffset < ordered[j].ByteOffset
	})
	out := text
	// Walk from the last finding to the first so splice offsets remain valid.
	prevStart := len(text) + 1
	for i := len(ordered) - 1; i >= 0; i-- {
		f := ordered[i]
		start := f.ByteOffset
		end := start + len(f.Raw)
		if start < 0 || end > len(out) || end > prevStart {
			// Out of range or overlaps a run we already rewrote — skip.
			continue
		}
		if out[start:end] != f.Raw {
			// The bytes at the recorded offset no longer match (e.g. a prior
			// splice shifted things); skip rather than corrupt the string.
			continue
		}
		out = out[:start] + f.Normalized + out[end:]
		prevStart = start
	}
	return out
}

// Report is the assembled audit over a sequence of segments. It holds the
// segments in boundary order plus the scanner findings attributed back to the
// segment (and Source) that produced them.
type Report struct {
	// Segments are the rendered context fields, in the order they cross the
	// model boundary. Order is load-bearing: the digest hashes Raw in this order.
	Segments []Segment
	// Findings are the per-segment scanner findings, attributed to their
	// producing segment and Source.
	Findings []SegmentFinding
}

// SegmentFinding pairs a scanner Finding with the segment it was found in, so a
// confusable/format-sensitive detection is attributed to its producer rather
// than floating free of provenance. Offsets in Finding are RELATIVE to the
// segment's Raw bytes.
type SegmentFinding struct {
	// SegmentIndex is the index into Report.Segments of the producing segment.
	SegmentIndex int
	// Field is the producing segment's logical field name (copied for display).
	Field string
	// Source is the producing segment's Source (copied for display/routing).
	Source Source
	// Finding is the underlying scanner detection, with offsets relative to the
	// segment's Raw bytes.
	Finding Finding
}

// String renders one attributed finding as an operator line.
func (sf SegmentFinding) String() string {
	return fmt.Sprintf("field=%q source=%s seg=%d  %s",
		sf.Field, sf.Source, sf.SegmentIndex, sf.Finding.String())
}

// Audit assembles a Report from the given segments. For each segment it records
// the exact rendered bytes (kept verbatim in the segment), normalizes the
// segment's Source to the closed vocabulary (fail open to SourceUnknown), runs
// the existing #1691 Scan over the segment's raw bytes, and attributes each
// finding back to that segment and its Source. The input slice is not mutated.
func Audit(segments []Segment) Report {
	r := Report{
		Segments: make([]Segment, len(segments)),
	}
	for i, seg := range segments {
		seg.Source = seg.Source.normalize()
		r.Segments[i] = seg
		for _, f := range Scan(seg.Raw) {
			r.Findings = append(r.Findings, SegmentFinding{
				SegmentIndex: i,
				Field:        seg.Field,
				Source:       seg.Source,
				Finding:      f,
			})
		}
	}
	return r
}

// Digest is a stable SHA-256 over the EXACT raw segment bytes, concatenated in
// boundary order with a length-prefixed framing so two different segmentations
// can never collide. Because it hashes Raw verbatim, the digest CHANGES when a
// single marker rune changes (U+2019 vs ASCII ', or '/' vs '-' in a date token)
// and is IDENTICAL for byte-identical prompts. It is the fingerprint a
// cache/debug report uses to prove a marker mutated the request.
func (r Report) Digest() string {
	h := sha256.New()
	for _, seg := range r.Segments {
		// Length-prefix the raw bytes so concatenation is unambiguous; two
		// segmentations that flatten to the same byte stream still differ.
		fmt.Fprintf(h, "%d:", len(seg.Raw))
		h.Write([]byte(seg.Raw))
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// HasFindings reports whether any segment produced a scanner finding.
func (r Report) HasFindings() bool {
	return len(r.Findings) > 0
}

// String renders a human-readable, operator-inspectable audit. Per segment it
// shows the field name, the attributed Source, the raw bytes, the normalized
// display form SEPARATELY (so an operator sees why the field looked harmless),
// and — for any segment with findings — each finding's raw marker codepoints and
// byte offsets. The whole-report Digest is printed last.
func (r Report) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "prompt-audit report: %d segment(s), %d finding(s)\n", len(r.Segments), len(r.Findings))
	fmt.Fprintf(&b, "digest: %s\n", r.Digest())
	for i, seg := range r.Segments {
		norm := seg.NormalizedForm()
		fmt.Fprintf(&b, "  [%d] field=%q source=%s\n", i, seg.Field, seg.Source)
		fmt.Fprintf(&b, "      raw=%q\n", seg.Raw)
		if norm != seg.Raw {
			fmt.Fprintf(&b, "      normalized=%q  (differs — a marker was folded for display)\n", norm)
		} else {
			fmt.Fprintf(&b, "      normalized=%q\n", norm)
		}
		for _, sf := range r.Findings {
			if sf.SegmentIndex != i {
				continue
			}
			f := sf.Finding
			fmt.Fprintf(&b, "      ! %s codepoint=%s byteOffset=%d channel=%s — raw=%q normalized=%q\n",
				f.Kind, strings.Join(f.Codepoints, " "), f.ByteOffset, f.Channel, f.Raw, f.Normalized)
		}
	}
	return b.String()
}

// reportJSON is the wire shape of a Report, kept separate from the in-memory
// struct so the JSON is stable regardless of how the Go fields evolve.
type reportJSON struct {
	Digest   string        `json:"digest"`
	Segments []segmentJSON `json:"segments"`
	Findings []findingJSON `json:"findings"`
}

type segmentJSON struct {
	Index      int    `json:"index"`
	Field      string `json:"field"`
	Source     Source `json:"source"`
	Raw        string `json:"raw"`
	Normalized string `json:"normalized"`
}

type findingJSON struct {
	SegmentIndex int      `json:"segment_index"`
	Field        string   `json:"field"`
	Source       Source   `json:"source"`
	Kind         Kind     `json:"kind"`
	Channel      Channel  `json:"channel"`
	Codepoints   []string `json:"codepoints"`
	ByteOffset   int      `json:"byte_offset"`
	RuneOffset   int      `json:"rune_offset"`
	Raw          string   `json:"raw"`
	Normalized   string   `json:"normalized"`
	Detail       string   `json:"detail"`
}

// JSON renders the report as indented JSON for machine consumption: the digest,
// each segment with its raw + normalized form + source, and each attributed
// finding with codepoints and byte offsets. Marshaling cannot fail for this
// shape, but any error is surfaced as a JSON object rather than panicking.
func (r Report) JSON() string {
	rj := reportJSON{
		Digest:   r.Digest(),
		Segments: make([]segmentJSON, len(r.Segments)),
	}
	for i, seg := range r.Segments {
		rj.Segments[i] = segmentJSON{
			Index:      i,
			Field:      seg.Field,
			Source:     seg.Source,
			Raw:        seg.Raw,
			Normalized: seg.NormalizedForm(),
		}
	}
	for _, sf := range r.Findings {
		f := sf.Finding
		rj.Findings = append(rj.Findings, findingJSON{
			SegmentIndex: sf.SegmentIndex,
			Field:        sf.Field,
			Source:       sf.Source,
			Kind:         f.Kind,
			Channel:      f.Channel,
			Codepoints:   f.Codepoints,
			ByteOffset:   f.ByteOffset,
			RuneOffset:   f.RuneOffset,
			Raw:          f.Raw,
			Normalized:   f.Normalized,
			Detail:       f.Detail,
		})
	}
	out, err := json.MarshalIndent(rj, "", "  ")
	if err != nil {
		return `{"error":` + strconv.Quote(err.Error()) + `}`
	}
	return string(out)
}
