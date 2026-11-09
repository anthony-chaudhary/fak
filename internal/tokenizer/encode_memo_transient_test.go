package tokenizer

import (
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestEncodeTransientReusesPrefixWithoutEvictingRequest(t *testing.T) {
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
	shared := "<|im_start|>system\n" + stable + "<|im_end|>\n"
	full := shared + "<|im_start|>user\nunique registry content<|im_end|>\n<|im_start|>assistant\n"
	var memo EncodeMemo
	wantFull, _, err := memo.Encode(tok, full)
	if err != nil {
		t.Fatal(err)
	}
	coldBytes := actualBytes
	wantPrefix, err := tok.Encode(shared)
	if err != nil {
		t.Fatal(err)
	}
	actualBytes = 0
	prefix, work, err := memo.EncodeTransient(tok, shared)
	if err != nil || !reflect.DeepEqual(prefix, wantPrefix) || actualBytes != 0 || work.OrdinaryBytesEncoded != 0 || work.OrdinaryBytesReused < len(stable) {
		t.Fatalf("prefix parity/work: err=%v actual=%d work=%+v", err, actualBytes, work)
	}
	t.Logf("cold_full_split_bytes=%d shared_prefix_split_bytes=%d work=%+v", coldBytes, actualBytes, work)
	for i := range prefix {
		prefix[i] = -1
	}
	actualBytes = 0
	got, work, err := memo.Encode(tok, full)
	if err != nil || !reflect.DeepEqual(got, wantFull) || actualBytes != 0 || work.OrdinaryBytesEncoded != 0 {
		t.Fatalf("prefix probe evicted/corrupted request: err=%v actual=%d work=%+v", err, actualBytes, work)
	}
	t.Logf("full_request_after_prefix_split_bytes=%d work=%+v", actualBytes, work)
}

// fak-test:runtime fast est=100ms lane=default
func TestEncodeTransientDoesNotPublishMissOrOtherTokenizer(t *testing.T) {
	tok := memoTestTokenizer(t, 256)
	other := memoTestTokenizer(t, 270)
	plain, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"decoder":{"type":"ByteLevel"}}`))
	if err != nil {
		t.Fatal(err)
	}
	full := "<|im_start|>ab original<|im_end|>"
	var memo EncodeMemo
	wantFull, _, err := memo.Encode(tok, full)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		tok  *Tokenizer
		text string
	}{{tok, "<|im_start|>unrelated<|im_end|>"}, {other, full}, {plain, "a"}} {
		want, err := tc.tok.Encode(tc.text)
		if err != nil {
			t.Fatal(err)
		}
		got, _, err := memo.EncodeTransient(tc.tok, tc.text)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("transient encode lost canonical parity: %v", err)
		}
	}
	if _, _, err := memo.EncodeTransient(nil, "a"); err == nil {
		t.Fatal("nil tokenizer should refuse")
	}
	got, work, err := memo.Encode(tok, full)
	if err != nil || !reflect.DeepEqual(got, wantFull) || work.OrdinaryBytesEncoded != 0 {
		t.Fatalf("transient work replaced retained generation: err=%v work=%+v", err, work)
	}
}
