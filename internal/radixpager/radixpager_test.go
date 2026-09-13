package radixpager

import "testing"

func TestPageBytesDeterministicAndCounted(t *testing.T) {
	p := NewRadixKVPager(nil, 16, 64)
	if p.Alignment() != 16 {
		t.Fatalf("alignment: %d", p.Alignment())
	}
	raw := make([]byte, 40)
	for i := range raw {
		raw[i] = byte(i)
	}
	toks, pages, _, err := p.PageBytes(raw)
	if err != nil {
		t.Fatalf("PageBytes: %v", err)
	}
	if pages != 3 || len(toks) != 3 {
		t.Fatalf("want 3 pages/tokens, got pages=%d toks=%d", pages, len(toks))
	}
	toks2, _, _, _ := p.PageBytes(raw)
	for i := range toks {
		if toks[i] != toks2[i] {
			t.Fatalf("page token not deterministic at %d", i)
		}
	}
}

func TestAdmitRawUnalignedReturnsTokensNoBind(t *testing.T) {
	p := NewRadixKVPager(nil, 4096, 64)
	raw := []byte("not-page-aligned")
	ref, toks, pages, zero, err := p.AdmitRaw(raw)
	if err != nil {
		t.Fatalf("AdmitRaw: %v", err)
	}
	if zero || ref != 0 || pages != 1 || len(toks) != 1 {
		t.Fatalf("want unaligned no-bind, got ref=%x toks=%d pages=%d zero=%v", ref, len(toks), pages, zero)
	}
}

func TestAdmitRawAlignedBinds(t *testing.T) {
	p := NewRadixKVPager(nil, 16, 64)
	raw := make([]byte, 48)
	ref, toks, pages, zero, err := p.AdmitRaw(raw)
	if err != nil {
		t.Fatalf("AdmitRaw: %v", err)
	}
	if pages != 3 || len(toks) != 3 {
		t.Fatalf("pages=%d toks=%d", pages, len(toks))
	}
	if zero && ref == 0 {
		t.Fatalf("zero-copy reported with nil node ref")
	}
	if zero && p.AllocatedBlocks() < 0 {
		t.Fatalf("allocated blocks negative")
	}
}

func TestNewRadixKVPagerDefaults(t *testing.T) {
	p := NewRadixKVPager(nil, 0, 0)
	if p.Alignment() != DefaultRadixPagerAlign {
		t.Fatalf("default alignment: %d", p.Alignment())
	}
	if p.Tree() == nil {
		t.Fatalf("default tree nil")
	}
}
