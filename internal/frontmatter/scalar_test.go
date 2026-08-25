package frontmatter

import "testing"

func TestDecodeScalarDecodesQuotedYAML(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "double quoted escapes and backslashes",
			raw:  `"Use a colon: safely and say \"hello\" from C:\\tools.\nNext\tstep."`,
			want: "Use a colon: safely and say \"hello\" from C:\\tools.\nNext\tstep.",
		},
		{
			name: "single quoted doubled apostrophes",
			raw:  `'Bob''s skill says ''hello''.'`,
			want: "Bob's skill says 'hello'.",
		},
		{
			name: "plain scalar",
			raw:  "  unchanged: value  ",
			want: "unchanged: value",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := DecodeScalar(tc.raw)
			if !ok {
				t.Fatalf("DecodeScalar(%q) rejected valid scalar", tc.raw)
			}
			if got != tc.want {
				t.Fatalf("DecodeScalar(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

func TestDecodeScalarRejectsMalformedQuotedYAML(t *testing.T) {
	for _, raw := range []string{
		`"unterminated`,
		`'unterminated`,
		`"unknown \q escape"`,
		`'unescaped ' apostrophe'`,
		`"surrogate \uD800"`,
	} {
		t.Run(raw, func(t *testing.T) {
			got, ok := DecodeScalar(raw)
			if ok {
				t.Fatalf("DecodeScalar(%q) = %q, want malformed", raw, got)
			}
			if got != raw {
				t.Fatalf("DecodeScalar(%q) returned %q, want source preserved", raw, got)
			}
		})
	}
}
