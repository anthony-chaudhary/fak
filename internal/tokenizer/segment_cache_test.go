package tokenizer

import (
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestEncodeMemoSharedSegmentCacheAcrossConversations(t *testing.T) {
	tok := memoTestTokenizer(t, 256)
	split := tok.split
	if split == nil {
		split = preTokenizeByteLevel
	}
	actualBytes := 0
	tok.split = func(text string) []string {
		actualBytes += len(text)
		return split(text)
	}
	stable := strings.Repeat("ab ", 128)
	convA := "<|im_start|>system\n" + stable + "<|im_end|>\n<|im_start|>user\nalpha<|im_end|>\n<|im_start|>assistant\n"
	convB := "<|im_start|>system\n" + stable + "<|im_end|>\n<|im_start|>user\nbeta<|im_end|>\n<|im_start|>assistant\n"

	shared := NewSegmentCache(0, 0)
	memoA := &EncodeMemo{Shared: shared}
	memoB := &EncodeMemo{Shared: shared}

	if _, _, err := memoA.Encode(tok, convA); err != nil {
		t.Fatal(err)
	}
	want, err := tok.Encode(convB)
	if err != nil {
		t.Fatal(err)
	}
	actualBytes = 0
	got, work, err := memoB.Encode(tok, convB)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("conversation B: error=%v ids_equal=%t", err, reflect.DeepEqual(got, want))
	}
	if work.OrdinaryBytesReused < len(stable) || actualBytes >= len(stable) {
		t.Fatalf("conversation B did not reuse A's shared system run: work=%+v encoded=%d", work, actualBytes)
	}
	if st := shared.Stats(); st.Hits == 0 || st.Entries == 0 {
		t.Fatalf("shared cache stats=%+v, want hits and resident entries", st)
	}

	// Without Shared the second conversation must pay for the system run itself.
	actualBytes = 0
	if _, work, err := (&EncodeMemo{}).Encode(tok, convB); err != nil || work.OrdinaryBytesReused >= len(stable) {
		t.Fatalf("unshared memo reused=%d err=%v, want less than the system run", work.OrdinaryBytesReused, err)
	}

	// A cache bound to one tokenizer is bypassed by another.
	other := memoTestTokenizer(t, 512)
	if _, work, err := (&EncodeMemo{Shared: shared}).Encode(other, convB); err != nil || work.OrdinaryBytesReused >= len(stable) {
		t.Fatalf("foreign tokenizer reused=%d err=%v, want bypass", work.OrdinaryBytesReused, err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestSegmentCacheEvictsLeastRecentlyUsedWithinBounds(t *testing.T) {
	tok := memoTestTokenizer(t, 256)
	c := NewSegmentCache(1<<20, 2)
	c.put(tok, "one", []int{1})
	c.put(tok, "two", []int{2})
	if _, ok := c.get(tok, "one"); !ok { // touch "one" so "two" is coldest
		t.Fatal("one missing")
	}
	c.put(tok, "three", []int{3})
	if _, ok := c.get(tok, "two"); ok {
		t.Fatal("coldest entry survived eviction")
	}
	if _, ok := c.get(tok, "one"); !ok {
		t.Fatal("recently used entry evicted")
	}
	if st := c.Stats(); st.Entries != 2 || st.Evictions != 1 {
		t.Fatalf("stats=%+v, want 2 entries and 1 eviction", st)
	}
}
