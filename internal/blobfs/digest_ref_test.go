package blobfs

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
)

// One temporary store and one small durable write; runtime estimate is unmeasured.
// fak-test:runtime fast est=20ms lane=default
func TestResolveRejectsMalformedDigest(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s, err := NewWithBudget(t.TempDir(), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []abi.RefKind{abi.RefBlob, abi.RefRegion} {
		for _, digest := range []string{
			"", "abc", strings.Repeat("a", digestHexLen-1), strings.Repeat("a", digestHexLen+1),
			strings.Repeat("g", digestHexLen), "../../escape",
		} {
			got, err := s.Resolve(ctx, abi.Ref{Kind: kind, Digest: digest})
			if err == nil || err.Error() != "blobfs: invalid SHA-256 digest" || got != nil {
				t.Fatalf("Resolve kind=%d digest=%q: bytes=%v err=%v; want validation error and no bytes", kind, digest, got, err)
			}
		}
	}

	// Valid stored references still read and verify the content-addressed bytes.
	want := bytes.Repeat([]byte("v"), InlineMax+1)
	r, err := s.Put(ctx, want)
	if err != nil {
		t.Fatal(err)
	}
	if r.Kind != abi.RefBlob {
		t.Fatalf("Put kind=%d, want durable blob", r.Kind)
	}
	for _, kind := range []abi.RefKind{abi.RefBlob, abi.RefRegion} {
		r.Kind = kind
		got, err := s.Resolve(ctx, r)
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("valid Resolve kind=%d: bytes=%v err=%v", kind, got, err)
		}
	}

	// A well-formed missing digest reaches the existing not-found path.
	missing := abi.Ref{Kind: abi.RefBlob, Digest: blob.Digest([]byte("absent"))}
	if got, err := s.Resolve(ctx, missing); err == nil || !strings.Contains(err.Error(), "unknown digest") || got != nil {
		t.Fatalf("missing Resolve: bytes=%v err=%v", got, err)
	}
	// Inline refs do not need a digest, and still return an owned copy.
	inline := []byte("inline")
	got, err := s.Resolve(ctx, abi.Ref{Kind: abi.RefInline, Inline: inline})
	if err != nil || !bytes.Equal(got, inline) {
		t.Fatalf("inline Resolve: bytes=%v err=%v", got, err)
	}
	got[0] = 'X'
	if string(inline) != "inline" {
		t.Fatal("inline Resolve returned aliased bytes")
	}
}
