package ctxplan

import (
	"testing"
)

// media_test.go — witnesses for the first-class media span descriptor (#5167): the
// closed vocabularies fail closed, an annotated media span becomes reachable through the
// EXISTING relevance and durability machinery (no new access path), and the descriptor
// survives the persisted-index round-trip.

func TestNormMediaTypeFailsClosed(t *testing.T) {
	cases := map[string]string{
		"image":       MediaImage,
		"IMAGE":       MediaImage,
		"image/png":   MediaImage,
		"audio/wav":   MediaAudio,
		"video/mp4":   MediaVideo,
		"binary":      MediaBinary,
		"":            MediaBinary,
		"application": MediaBinary, // unknown kind fails closed to opaque bytes
		"parquet":     MediaBinary,
	}
	for in, want := range cases {
		if got := NormMediaType(in); got != want {
			t.Errorf("NormMediaType(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestMediaSizeClassBoundaries(t *testing.T) {
	cases := []struct {
		bytes int64
		want  string
	}{
		{0, MediaSizeSmall},
		{-1, MediaSizeSmall},
		{64 << 10, MediaSizeSmall},
		{64<<10 + 1, MediaSizeMedium},
		{1 << 20, MediaSizeMedium},
		{1<<20 + 1, MediaSizeLarge},
	}
	for _, c := range cases {
		if got := MediaSizeClassOf(c.bytes); got != c.want {
			t.Errorf("MediaSizeClassOf(%d) = %q, want %q", c.bytes, got, c.want)
		}
	}
}

// TestAnnotateMediaDurabilityDefault is the fail-closed durability witness: a digest-
// bearing media span (recoverable from the durable store) defaults to bounded, a
// digest-less one (unrecoverable once elided) stays turn-scoped, and an explicitly set
// durability is never overridden.
func TestAnnotateMediaDurabilityDefault(t *testing.T) {
	withDigest := Span{ID: "m1", Bytes: 2 << 20, Digest: "sha256:abc"}
	AnnotateMedia(&withDigest, "image/png", "")
	if withDigest.Durability != DurabilityBounded {
		t.Errorf("digest-bearing media durability = %q, want %q", withDigest.Durability, DurabilityBounded)
	}

	noDigest := Span{ID: "m2", Bytes: 2 << 20}
	AnnotateMedia(&noDigest, "image/png", "")
	if noDigest.Durability != DurabilityTurn {
		t.Errorf("digest-less media durability = %q, want %q", noDigest.Durability, DurabilityTurn)
	}

	explicit := Span{ID: "m3", Bytes: 10, Durability: DurabilityDurable}
	AnnotateMedia(&explicit, "image/png", "")
	if explicit.Durability != DurabilityDurable {
		t.Errorf("explicit durability overridden to %q", explicit.Durability)
	}
}

// TestAnnotateMediaPreservesExistingDescriptor: annotation prepends the media words, it
// never clobbers descriptor text a producer already supplied.
func TestAnnotateMediaPreservesExistingDescriptor(t *testing.T) {
	s := Span{ID: "m1", Bytes: 100, Descriptor: "gateway latency chart"}
	AnnotateMedia(&s, "image", "p99 over time")
	for _, tok := range []string{"media", "image", "small", "p99", "gateway latency chart"} {
		if !containsSub(s.Descriptor, tok) {
			t.Errorf("descriptor %q missing %q", s.Descriptor, tok)
		}
	}
}

func containsSub(hay, needle string) bool {
	return len(hay) >= len(needle) && (hay == needle || indexOf(hay, needle) >= 0)
}

func indexOf(hay, needle string) int {
	for i := 0; i+len(needle) <= len(hay); i++ {
		if hay[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}

// TestMediaSpanReachableByRelevance is the headline #5167 witness: an annotated image
// span buried under noise is probed back by a forecast whose intent overlaps its ALT
// TEXT — through the ordinary relevance access path, with zero planner changes. The
// same span un-annotated (no descriptor tokens) is invisible to the same forecast.
func TestMediaSpanReachableByRelevance(t *testing.T) {
	build := func(annotate bool) *Index {
		spans := make([]Span, 0, 40)
		img := Span{ID: "img", Step: 0, Role: "Read", Bytes: 3 << 20, Digest: "sha256:d1"}
		if annotate {
			AnnotateMedia(&img, "image/png", "architecture diagram of the gateway relay seam")
		}
		spans = append(spans, img)
		for i := 1; i <= 30; i++ {
			spans = append(spans, Span{
				ID: "noise" + itoaTest(i), Step: i, Role: "Bash",
				Descriptor: "build log line " + itoaTest(i) + " compiled quietly",
				Bytes:      200, Durability: DurabilityTurn,
			})
		}
		return BuildIndex(spans)
	}

	f := Forecast{Intents: []string{"the architecture diagram of the relay seam"}, Horizon: 1}
	opts := ProbeOptions{RecencyWindow: 4, MaxCandidates: 8}

	if got := probeIDset(build(true).Probe(f, opts)); !got["img"] {
		t.Fatalf("annotated media span not probed by relevance; got %v", got)
	}
	if got := probeIDset(build(false).Probe(f, opts)); got["img"] {
		t.Fatalf("un-annotated media span unexpectedly probed (should be invisible to relevance)")
	}
}

// TestMediaDescriptorSurvivesImageRoundTrip: the descriptor rides Attrs, which the
// persisted IndexImage already serializes — so a media span is still first-class after
// a save/re-attach, and MediaDescriptorOf re-normalizes on read.
func TestMediaDescriptorSurvivesImageRoundTrip(t *testing.T) {
	s := Span{ID: "img", Bytes: 500 << 10, Digest: "sha256:d2"}
	want := AnnotateMedia(&s, "image/jpeg", "whiteboard photo")
	ix := BuildIndex([]Span{s})

	data, err := MarshalIndexImage(ix)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	rix, err := UnmarshalIndexImage(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	spans := rix.Spans()
	if len(spans) != 1 {
		t.Fatalf("restored %d spans, want 1", len(spans))
	}
	if !IsMediaSpan(spans[0]) {
		t.Fatalf("restored span lost its media descriptor")
	}
	got, ok := MediaDescriptorOf(spans[0])
	if !ok || got != want {
		t.Fatalf("MediaDescriptorOf after round-trip = %+v (ok=%v), want %+v", got, ok, want)
	}
}

// TestMediaDescriptorOfNonMedia: a plain text span reads back as not-media.
func TestMediaDescriptorOfNonMedia(t *testing.T) {
	s := Span{ID: "t1", Descriptor: "a text note"}
	if IsMediaSpan(s) {
		t.Fatal("text span reported as media")
	}
	if _, ok := MediaDescriptorOf(s); ok {
		t.Fatal("MediaDescriptorOf returned ok for a non-media span")
	}
}
