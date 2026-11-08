package tokenizer

import (
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func memoTestTokenizer(t *testing.T, firstSpecialID int) *Tokenizer {
	t.Helper()
	vocab := make(map[string]int)
	for b := 0; b < 256; b++ {
		vocab[byteLevelEncode(string([]byte{byte(b)}))] = b
	}
	vocab["ab"] = firstSpecialID + 44
	added := []map[string]any{}
	for i, text := range []string{"<|im_start|>", "<|im_end|>", "<|im_end|>\n<|im_start|>assistant"} {
		vocab[text] = firstSpecialID + i
		added = append(added, map[string]any{"id": firstSpecialID + i, "content": text, "special": true})
	}
	raw, err := json.Marshal(map[string]any{
		"model":   map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{"a b"}},
		"decoder": map[string]any{"type": "ByteLevel"}, "added_tokens": added,
	})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// fak-test:runtime fast est=100ms lane=default
func TestEncodeMemoAvoidsUnchangedOrdinaryWork(t *testing.T) {
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
	first := "<|im_start|>system\n" + stable + "<|im_end|>\n<|im_start|>user\nfirst<|im_end|>\n<|im_start|>assistant\n"
	second := first + "answer<|im_end|>\n<|im_start|>user\nnext<|im_end|>\n<|im_start|>assistant\n"
	var memo EncodeMemo
	for i, text := range []string{first, second, second} {
		want, err := tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		actualBytes = 0
		got, work, err := memo.Encode(tok, text)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("request %d: error=%v ids_equal=%t", i, err, reflect.DeepEqual(got, want))
		}
		if work.OrdinaryBytesEncoded != actualBytes {
			t.Fatalf("work accounting=%d actual splitter input=%d", work.OrdinaryBytesEncoded, actualBytes)
		}
		if i == 1 && (actualBytes >= len(stable) || work.OrdinaryBytesReused < len(stable)) {
			t.Fatalf("unchanged source was not reused: %+v actual=%d", work, actualBytes)
		}
		if i == 2 && actualBytes != 0 {
			t.Fatalf("identical request re-tokenized %d ordinary bytes", actualBytes)
		}
		t.Logf("request=%d actual_split_bytes=%d work=%+v", i, actualBytes, work)
		for j := range got {
			got[j] = -1 // Returned slices must not corrupt retained ordinary IDs.
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestEncodeMemoKeepsCurrentTokenBoundaries(t *testing.T) {
	tok := memoTestTokenizer(t, 256)
	var memo EncodeMemo
	for _, text := range []string{
		"<|im_start|>a", "<|im_start|>ab", // Ordinary BPE merge changes.
		"<|im_end|>\n", "<|im_end|>\n<|im_start|>assistant", // Longer added match crosses the old end.
		"<|im_start|>héllo 世界\n<|im_end|>", "<|im_start|>héllo 世界\n \n<|im_end|>",
	} {
		want, err := tok.Encode(text)
		if err != nil {
			t.Fatal(err)
		}
		got, _, err := memo.Encode(tok, text)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("boundary case %q: error=%v ids_equal=%t", text, err, reflect.DeepEqual(got, want))
		}
	}
	other := memoTestTokenizer(t, 270)
	text := "<|im_start|>ab<|im_end|>"
	_, _, _ = memo.Encode(tok, text)
	want, _ := other.Encode(text)
	got, _, err := memo.Encode(other, text)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatal("memo reused IDs across tokenizer instances")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestEncodeMemoFailureDoesNotPublish(t *testing.T) {
	tok, err := ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0,"<s>":1,"</s>":2},"merges":[]},"decoder":{"type":"ByteLevel"},"added_tokens":[{"id":1,"content":"<s>","special":true},{"id":2,"content":"</s>","special":true}]}`))
	if err != nil {
		t.Fatal(err)
	}
	var memo EncodeMemo
	text := "<s>a</s><s>aa</s>"
	want, _, err := memo.Encode(tok, text)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := memo.Encode(tok, "<s>a</s><s>z</s>"); err == nil {
		t.Fatal("fixture should reject the missing ordinary symbol")
	}
	got, work, err := memo.Encode(tok, text)
	if err != nil || !reflect.DeepEqual(got, want) || work.OrdinaryBytesEncoded != 0 {
		t.Fatalf("failed encode changed retained state: error=%v work=%+v", err, work)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestEncodeMemoRetentionIsBounded(t *testing.T) {
	run := &encodeRunMemo{next: make(map[string]encodedRun)}
	run.remember(strings.Repeat("x", encodeMemoPayloadLimit+1), nil)
	if len(run.next) != 0 {
		t.Fatal("oversized run was retained")
	}
	for i := 0; i < encodeMemoEntryLimit+10; i++ {
		run.remember(strconv.Itoa(i), []int{i})
	}
	if len(run.next) != encodeMemoEntryLimit || run.work.CachedPayloadBytes > encodeMemoPayloadLimit {
		t.Fatalf("retention exceeded bounds: entries=%d work=%+v", len(run.next), run.work)
	}
}
