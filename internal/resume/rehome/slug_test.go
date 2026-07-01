package rehome

import "testing"

func TestProjectSlug(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"windows path", `C:\work\fak`, "C--work-fak"},
		{"windows nested", `C:\work\slack-helpers`, "C--work-slack-helpers"},
		{"unix path", "/home/user/repo", "-home-user-repo"},
		{"already slug", "C--work-fak", "C--work-fak"},
		{"digits kept", "run2026", "run2026"},
		{"punctuation each becomes dash", "a@b~c(d)", "a-b-c-d-"},
		{"space becomes dash", "my repo", "my-repo"},
		{"empty", "", ""},
		// A non-ASCII rune maps to a single '-', not one per UTF-8 byte: the
		// two-byte é is one code point, so one dash — matching Python's re.sub
		// over code points, not bytes.
		{"unicode single dash", "café", "caf-"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ProjectSlug(tc.in); got != tc.want {
				t.Fatalf("ProjectSlug(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}
