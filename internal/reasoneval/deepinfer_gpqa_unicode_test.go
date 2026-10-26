package reasoneval

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestDeepInferGPQAUnicodeExtraction(t *testing.T) {
	t.Parallel()
	// Source-derived cases for DeepInfer@553741557a57df44a46fb3ab72b6aafeaf945337
	// eval/gpqa/bench.py:111-134. Python regex word characters include all numeric
	// categories, not only decimal digits; marks and non-underscore Pc do not count.
	cases := []struct {
		name, text, prediction, source string
	}{
		{"answer NBSP", "ANSWER:\u00a0B\nfinal C", "B", "answer_line"},
		{"boxed NBSP", "\\boxed{\u00a0B}\nfinal C", "B", "boxed"},
		{"letter is word", "ANSWER: Bβ\nfinal C", "C", "tail_letter"},
		{"decimal is word", "ANSWER: B٣\nfinal C", "C", "tail_letter"},
		{"letter number is word", "ANSWER: BⅧ\nfinal C", "C", "tail_letter"},
		{"other number is word", "ANSWER: B²\nfinal C", "C", "tail_letter"},
		{"underscore is word", "ANSWER: B_\nfinal C", "C", "tail_letter"},
		{"letter-folding combining mark is nonword", "ANSWER: B\u0345\nfinal C", "B", "answer_line"},
		{"other connector is nonword", "ANSWER: B\u203f\nfinal C", "B", "answer_line"},
		{"Python whitespace at each position", "\u001cANSWER\u0085：\u2028(d)\nfinal C", "D", "answer_line"},
		{"boxed Python whitespace", "\\boxed{\u000b\u001d\u001e\u001f(B)\u2003\u2029\u3000}\nfinal C", "B", "boxed"},
		{"zero width space is not whitespace", "ANSWER:\u200bB\nfinal C", "C", "tail_letter"},
		{"adjacent answer lines keep last priority", "ANSWER: A\nANSWER: (d)\n\\boxed{B}\nfinal C", "D", "answer_line"},
		{"failed multiline prefix retries next line", "ANSWER:\nANSWER: B\nfinal C", "B", "answer_line"},
		{"answer boundary at end", "answer: b", "B", "answer_line"},
		{"boxed case and fallback ASCII boundary unchanged", "\\boxed{b}\nBβ", "B", "tail_letter"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ScoreRule("gpqa", tc.text, tc.prediction)
			if err != nil {
				t.Fatal(err)
			}
			want := RuleResult{Prediction: tc.prediction, Source: tc.source, Correct: true}
			if got != want {
				t.Fatalf("ScoreRule(%q) = %+v, want %+v", tc.text, got, want)
			}
		})
	}
}
