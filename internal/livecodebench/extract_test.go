package livecodebench

import "testing"

// TestExtractCodeGolden pins the extractor across the shapes #2103 names: fenced
// (with a language tag), unfenced (a tag-less fence), multi-block (last fence
// wins), and the empty/garbage no-code verdicts. starter-merge has its own test.
func TestExtractCodeGolden(t *testing.T) {
	cases := []struct {
		name     string
		raw      string
		starter  string
		wantCode string
		wantLang string
		wantNo   bool
	}{
		{
			name:     "fenced_with_language",
			raw:      "Here is my solution:\n```python\nprint(1)\n```",
			wantCode: "print(1)",
			wantLang: "python",
		},
		{
			name:     "unfenced_tagless_fence",
			raw:      "```\nprint(2)\n```",
			wantCode: "print(2)",
			wantLang: "",
		},
		{
			name:     "multi_block_last_wins",
			raw:      "First attempt:\n```python\nreturn 0\n```\nFixed:\n```python\nreturn 1\n```",
			wantCode: "return 1",
			wantLang: "python",
		},
		{
			name:     "preserves_internal_indentation",
			raw:      "```python\ndef f():\n    if x:\n        return 1\n```",
			wantCode: "def f():\n    if x:\n        return 1",
			wantLang: "python",
		},
		{
			name:     "cpp_language_tag",
			raw:      "```cpp\nint main(){}\n```",
			wantCode: "int main(){}",
			wantLang: "cpp",
		},
		{
			name:   "empty_output_no_code",
			raw:    "",
			wantNo: true,
		},
		{
			name:   "prose_only_no_fence_no_code",
			raw:    "I am not sure how to solve this problem.",
			wantNo: true,
		},
		{
			name:   "empty_fenced_block_no_code",
			raw:    "```python\n\n```",
			wantNo: true,
		},
		{
			name:   "single_unclosed_fence_no_code",
			raw:    "```python\nprint(3)",
			wantNo: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ExtractCode(tc.raw, tc.starter)
			if got.NoCode != tc.wantNo {
				t.Fatalf("NoCode = %v, want %v (code=%q)", got.NoCode, tc.wantNo, got.Code)
			}
			if tc.wantNo {
				return
			}
			if got.Code != tc.wantCode {
				t.Errorf("Code = %q, want %q", got.Code, tc.wantCode)
			}
			if got.Language != tc.wantLang {
				t.Errorf("Language = %q, want %q", got.Language, tc.wantLang)
			}
		})
	}
}

// TestExtractCodeStarterMerge pins the starter-merge case: a bare completion gets
// the starter prepended, a full solution that already carries the signature does
// not, and a code-generation problem (no starter) is untouched.
func TestExtractCodeStarterMerge(t *testing.T) {
	starter := "class Solution:\n    def solve(self, a):"

	// Bare completion: signature absent -> starter prepended.
	bare := ExtractCode("```python\n        return a + 1\n```", starter)
	if bare.NoCode {
		t.Fatal("bare completion reported NoCode")
	}
	want := starter + "\n        return a + 1"
	if bare.Code != want {
		t.Errorf("merged Code = %q, want %q", bare.Code, want)
	}

	// Full solution already carrying the signature -> no duplication.
	full := ExtractCode("```python\nclass Solution:\n    def solve(self, a):\n        return a + 1\n```", starter)
	if full.Code != "class Solution:\n    def solve(self, a):\n        return a + 1" {
		t.Errorf("full solution was rewritten: %q", full.Code)
	}

	// No starter (code-generation problem) -> block returned verbatim.
	none := ExtractCode("```python\nprint(4)\n```", "")
	if none.Code != "print(4)" {
		t.Errorf("no-starter Code = %q, want %q", none.Code, "print(4)")
	}
}
