package promptaudit

import (
	"encoding/json"
	"strings"
	"testing"
)

// dateSegment builds a "currentDate" context segment whose value is a
// "Today's date is ..." carrier sentence with a chosen apostrophe rune and date
// separator, attributed to the local-environment producer — the exact field
// class the steganography article mutates without an audit surface.
func dateSegment(apostrophe, dateSep string) Segment {
	date := "2026" + dateSep + "06" + dateSep + "30"
	return Segment{
		Field:  "currentDate",
		Source: SourceLocalEnv,
		Raw:    "Today" + apostrophe + "s date is " + date + ".",
	}
}

// TestReportShowsRawMarkerAndSource is acceptance criterion #1: a report with a
// date-marker segment surfaces the raw marker codepoint AND attributes it to the
// segment's source field.
func TestReportShowsRawMarkerAndSource(t *testing.T) {
	r := Audit([]Segment{
		{Field: "systemPrompt", Source: SourceFakPolicy, Raw: "You are a helpful assistant."},
		dateSegment("’", "-"), // U+2019 right single quote marker, benign date sep
	})

	if len(r.Findings) == 0 {
		t.Fatalf("expected at least one finding for the marker segment, got none")
	}

	var apos *SegmentFinding
	for i := range r.Findings {
		if r.Findings[i].Finding.Kind == KindLookalikeApostrophe {
			apos = &r.Findings[i]
			break
		}
	}
	if apos == nil {
		t.Fatalf("expected a lookalike-apostrophe finding, got %v", r.Findings)
	}

	// Raw marker codepoint is surfaced.
	if len(apos.Finding.Codepoints) != 1 || apos.Finding.Codepoints[0] != "U+2019" {
		t.Errorf("codepoints = %v, want [U+2019]", apos.Finding.Codepoints)
	}
	// Attributed to the producing field + source.
	if apos.Field != "currentDate" {
		t.Errorf("finding.Field = %q, want currentDate", apos.Field)
	}
	if apos.Source != SourceLocalEnv {
		t.Errorf("finding.Source = %q, want %q", apos.Source, SourceLocalEnv)
	}
	if apos.SegmentIndex != 1 {
		t.Errorf("finding.SegmentIndex = %d, want 1", apos.SegmentIndex)
	}

	// The human render must mention both the codepoint and the source field.
	s := r.String()
	if !strings.Contains(s, "U+2019") {
		t.Errorf("String() should show the raw marker codepoint U+2019:\n%s", s)
	}
	if !strings.Contains(s, "currentDate") {
		t.Errorf("String() should show the source field currentDate:\n%s", s)
	}
	if !strings.Contains(s, string(SourceLocalEnv)) {
		t.Errorf("String() should show the source %q:\n%s", SourceLocalEnv, s)
	}
}

// TestNormalizedShownSeparately is acceptance criterion #2: the normalized form
// is shown SEPARATELY from the raw, so an operator sees why the field looked
// harmless. The raw bytes are never mutated.
func TestNormalizedShownSeparately(t *testing.T) {
	seg := dateSegment("’", "/") // both channels active
	r := Audit([]Segment{seg})

	norm := r.Segments[0].NormalizedForm()
	if norm == seg.Raw {
		t.Fatalf("normalized form should differ from raw when a marker is present: raw=%q norm=%q", seg.Raw, norm)
	}
	// Normalization folds the markers to their benign defaults.
	if strings.ContainsRune(norm, '’') {
		t.Errorf("normalized form still contains the U+2019 marker: %q", norm)
	}
	if !strings.Contains(norm, "2026-06-30") {
		t.Errorf("normalized form should carry the dash-normalized date: %q", norm)
	}
	if !strings.Contains(norm, "Today's date is") {
		t.Errorf("normalized form should fold the apostrophe to ASCII: %q", norm)
	}
	// Raw must be preserved verbatim — evidence not destroyed.
	if r.Segments[0].Raw != seg.Raw {
		t.Errorf("raw bytes were mutated: got %q want %q", r.Segments[0].Raw, seg.Raw)
	}

	// String() shows both forms, on separate lines.
	s := r.String()
	if !strings.Contains(s, "raw=") || !strings.Contains(s, "normalized=") {
		t.Errorf("String() must show raw and normalized separately:\n%s", s)
	}
}

// TestDigestChangesOnApostropheMarker is acceptance criterion #3a: two prompts
// differing ONLY by the apostrophe variant (U+2019 vs ASCII ') produce different
// digests.
func TestDigestChangesOnApostropheMarker(t *testing.T) {
	ascii := Audit([]Segment{dateSegment("'", "-")})
	curly := Audit([]Segment{dateSegment("’", "-")})

	if ascii.Digest() == curly.Digest() {
		t.Fatalf("digest did not change when only the apostrophe variant changed: %s", ascii.Digest())
	}
}

// TestDigestChangesOnDateSeparator is acceptance criterion #3b: two prompts
// differing ONLY by the date separator ('/' vs '-') produce different digests.
func TestDigestChangesOnDateSeparator(t *testing.T) {
	dash := Audit([]Segment{dateSegment("'", "-")})
	slash := Audit([]Segment{dateSegment("'", "/")})

	if dash.Digest() == slash.Digest() {
		t.Fatalf("digest did not change when only the date separator changed: %s", dash.Digest())
	}
}

// TestDigestStableForIdenticalBytes is the other half of criterion #3: byte-
// identical prompts produce identical digests, and the digest is deterministic
// across calls.
func TestDigestStableForIdenticalBytes(t *testing.T) {
	a := Audit([]Segment{
		{Field: "systemPrompt", Source: SourceFakPolicy, Raw: "Be concise."},
		dateSegment("’", "/"),
	})
	b := Audit([]Segment{
		{Field: "systemPrompt", Source: SourceFakPolicy, Raw: "Be concise."},
		dateSegment("’", "/"),
	})
	if a.Digest() != b.Digest() {
		t.Errorf("identical byte segments produced different digests: %s vs %s", a.Digest(), b.Digest())
	}
	// Deterministic across repeated calls on the same report.
	if a.Digest() != a.Digest() {
		t.Errorf("digest is not stable across calls")
	}
	if !strings.HasPrefix(a.Digest(), "sha256:") {
		t.Errorf("digest should be sha256-prefixed: %s", a.Digest())
	}
}

// TestDigestDependsOnSegmentBoundaries confirms the length-prefix framing: two
// reports whose raw bytes flatten to the SAME stream but split at different
// boundaries get DIFFERENT digests, so a segmentation change is also visible.
func TestDigestDependsOnSegmentBoundaries(t *testing.T) {
	one := Audit([]Segment{{Field: "all", Source: SourceUnknown, Raw: "abcdef"}})
	two := Audit([]Segment{
		{Field: "a", Source: SourceUnknown, Raw: "abc"},
		{Field: "b", Source: SourceUnknown, Raw: "def"},
	})
	if one.Digest() == two.Digest() {
		t.Errorf("different segment boundaries collided in the digest: %s", one.Digest())
	}
}

// TestSourceNormalizationFailsOpen confirms an unrecognized Source is surfaced
// as SourceUnknown rather than dropped, and that the known sources validate.
func TestSourceNormalizationFailsOpen(t *testing.T) {
	r := Audit([]Segment{{Field: "x", Source: Source("not-a-real-source"), Raw: "hello"}})
	if r.Segments[0].Source != SourceUnknown {
		t.Errorf("unknown source = %q, want %q (fail open)", r.Segments[0].Source, SourceUnknown)
	}
	for _, s := range []Source{
		SourceUserConfig, SourceFakPolicy, SourceIntegration,
		SourceProviderShim, SourceLocalEnv, SourceFetched, SourceUnknown,
	} {
		if !s.Valid() {
			t.Errorf("source %q should be valid", s)
		}
	}
	if Source("bogus").Valid() {
		t.Errorf("bogus source should not be valid")
	}
}

// TestAttributionAcrossSources confirms findings in different segments are
// attributed to their respective producers.
func TestAttributionAcrossSources(t *testing.T) {
	r := Audit([]Segment{
		{Field: "fetchedDoc", Source: SourceFetched, Raw: "remote text with a zero-width​space"},
		dateSegment("ʼ", "-"), // local-env apostrophe marker
	})
	bySource := map[Source]Kind{}
	for _, sf := range r.Findings {
		bySource[sf.Source] = sf.Finding.Kind
	}
	if bySource[SourceFetched] != KindInvisibleRune {
		t.Errorf("fetched segment should carry the invisible-rune finding, got %v", bySource)
	}
	if bySource[SourceLocalEnv] != KindLookalikeApostrophe {
		t.Errorf("local-env segment should carry the apostrophe finding, got %v", bySource)
	}
}

// TestJSONRoundTrips confirms JSON() emits valid JSON carrying the digest, the
// raw + normalized per segment, and the attributed findings with codepoints.
func TestJSONRoundTrips(t *testing.T) {
	r := Audit([]Segment{dateSegment("’", "/")})
	js := r.JSON()

	var decoded struct {
		Digest   string `json:"digest"`
		Segments []struct {
			Field      string `json:"field"`
			Source     string `json:"source"`
			Raw        string `json:"raw"`
			Normalized string `json:"normalized"`
		} `json:"segments"`
		Findings []struct {
			Source     string   `json:"source"`
			Codepoints []string `json:"codepoints"`
			ByteOffset int      `json:"byte_offset"`
		} `json:"findings"`
	}
	if err := json.Unmarshal([]byte(js), &decoded); err != nil {
		t.Fatalf("JSON() did not produce valid JSON: %v\n%s", err, js)
	}
	if !strings.HasPrefix(decoded.Digest, "sha256:") {
		t.Errorf("json digest = %q, want sha256-prefixed", decoded.Digest)
	}
	if len(decoded.Segments) != 1 {
		t.Fatalf("want 1 segment in json, got %d", len(decoded.Segments))
	}
	if decoded.Segments[0].Raw == decoded.Segments[0].Normalized {
		t.Errorf("json raw and normalized should differ for a marker segment")
	}
	if decoded.Segments[0].Source != string(SourceLocalEnv) {
		t.Errorf("json source = %q, want %q", decoded.Segments[0].Source, SourceLocalEnv)
	}
	if len(decoded.Findings) == 0 {
		t.Fatalf("expected findings in json, got none")
	}
	if decoded.Findings[0].Source != string(SourceLocalEnv) {
		t.Errorf("json finding source = %q, want %q", decoded.Findings[0].Source, SourceLocalEnv)
	}
}

// TestEmptyReport confirms the zero-segment report is well-defined: a stable
// digest of the empty stream and no findings.
func TestEmptyReport(t *testing.T) {
	r := Audit(nil)
	if r.HasFindings() {
		t.Errorf("empty report should have no findings")
	}
	if !strings.HasPrefix(r.Digest(), "sha256:") {
		t.Errorf("empty report digest = %q, want sha256-prefixed", r.Digest())
	}
	// Two empty reports agree.
	if Audit(nil).Digest() != Audit([]Segment{}).Digest() {
		t.Errorf("nil and empty-slice reports should have the same digest")
	}
}
