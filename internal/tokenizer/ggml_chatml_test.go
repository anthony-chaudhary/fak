package tokenizer

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// TestGGMLChatMLSpecialTokens asserts the embedded tokenizer recognizes the Qwen
// ChatML control tokens as single special ids (not byte-splattered text). This is
// the path the simpledemo's prompt template depends on; the corpus oracle does not
// exercise these tokens, so a regression here would silently make the model
// role-leak. Gated on a Qwen2 GGUF being present.
func TestGGMLChatMLSpecialTokens(t *testing.T) {
	ggufPath := findQwen2GGUF(t)
	if ggufPath == "" {
		t.Skip("no Qwen2 GGUF found (set FAK_GGUF=<path>); skipping ChatML special-token gate")
	}
	f, err := ggufload.Open(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok {
		t.Skip("no embedded BPE tokenizer")
	}
	tok, err := FromGGML(gt.Tokens, gt.Merges, gt.TokenTypes, gt.Pre)
	if err != nil {
		t.Fatal(err)
	}

	// Stable across the Qwen2.5 family.
	want := map[string][]int{
		"<|endoftext|>": {151643},
		"<|im_start|>":  {151644},
		"<|im_end|>":    {151645},
	}
	for s, ids := range want {
		got, err := tok.Encode(s)
		if err != nil || !reflect.DeepEqual(got, ids) {
			t.Errorf("Encode(%q) = %v, %v; want %v (special token must be one id)", s, got, err, ids)
		}
	}

	// A full ChatML turn keeps the control tokens intact and opens the assistant
	// turn (…<|im_start|>assistant\n), i.e. ends with 151644 …(assistant)… 198.
	prompt := "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
	got, err := tok.Encode(prompt)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(got); n < 4 || got[0] != 151644 || got[n-1] != 198 || got[n-3] != 151644 {
		t.Errorf("ChatML prompt not framed by control tokens: %v", got)
	}
	// The control tokens must survive inside the prompt, exactly twice each for
	// the two completed turns plus the open assistant turn (151644 x3, 151645 x2).
	count := func(id int) int {
		c := 0
		for _, x := range got {
			if x == id {
				c++
			}
		}
		return c
	}
	if count(151644) != 3 || count(151645) != 2 {
		t.Errorf("ChatML control-token counts wrong: <|im_start|>=%d (want 3), <|im_end|>=%d (want 2)", count(151644), count(151645))
	}
}
