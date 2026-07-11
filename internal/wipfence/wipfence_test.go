package wipfence

import (
	"strings"
	"testing"
)

func TestSlugFromPath(t *testing.T) {
	cases := []struct {
		name string
		path string
		want string
	}{
		{"dir stripping slash", "internal/Foo/new-thing.go", "new_thing"},
		{"dir stripping backslash", `cmd\fak\Guard_Fleet.go`, "guard_fleet"},
		{"go suffix stripped", "thing.go", "thing"},
		{"lowercasing", "MixedCase.go", "mixedcase"},
		{"non-alnum to underscore", "new-thing.go", "new_thing"},
		{"collapsing runs", "a--b!!.c.go", "a_b_c"},
		{"leading digit prefixed", "123.go", "f_123"},
		{"sanitizes to empty", "--.go", "f_"},
		{"bare .go", ".go", "f_"},
		{"already clean", "clean_name.go", "clean_name"},
		{"leading underscore trimmed", "_x.go", "x"},
		{"no extension", "internal/foo/thing", "thing"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := SlugFromPath(tc.path); got != tc.want {
				t.Errorf("SlugFromPath(%q) = %q, want %q", tc.path, got, tc.want)
			}
		})
	}
}

func TestFence(t *testing.T) {
	const body = "package main\n\nfunc x() {}\n"
	cases := []struct {
		name        string
		content     string
		slug        string
		want        string
		wantChanged bool
		wantErr     bool
	}{
		{
			name:        "unfenced gets fence",
			content:     body,
			slug:        "myfeat",
			want:        "//go:build wip_myfeat\n\n" + body,
			wantChanged: true,
		},
		{
			name:        "slug sanitized like a path base",
			content:     body,
			slug:        "New-Thing",
			want:        "//go:build wip_new_thing\n\n" + body,
			wantChanged: true,
		},
		{
			name:    "idempotent same slug",
			content: "//go:build wip_myfeat\n\n" + body,
			slug:    "myfeat",
			want:    "//go:build wip_myfeat\n\n" + body,
		},
		{
			name:    "clobber guard non-wip constraint",
			content: "//go:build linux\n\n" + body,
			slug:    "myfeat",
			wantErr: true,
		},
		{
			name:    "clobber guard different wip slug",
			content: "//go:build wip_other\n\n" + body,
			slug:    "myfeat",
			wantErr: true,
		},
		{
			name:    "empty slug",
			content: body,
			slug:    "",
			wantErr: true,
		},
		{
			name:    "slug sanitizes to empty",
			content: body,
			slug:    "!!!",
			wantErr: true,
		},
		{
			name:        "empty content",
			content:     "",
			slug:        "myfeat",
			want:        "//go:build wip_myfeat\n\n",
			wantChanged: true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, changed, err := Fence(tc.content, tc.slug)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("Fence(%q, %q) error = nil, want error", tc.content, tc.slug)
				}
				return
			}
			if err != nil {
				t.Fatalf("Fence(%q, %q) unexpected error: %v", tc.content, tc.slug, err)
			}
			if out != tc.want {
				t.Errorf("Fence(%q, %q) out = %q, want %q", tc.content, tc.slug, out, tc.want)
			}
			if changed != tc.wantChanged {
				t.Errorf("Fence(%q, %q) changed = %v, want %v", tc.content, tc.slug, changed, tc.wantChanged)
			}
		})
	}
}

func TestFenceOutputShape(t *testing.T) {
	const body = "package main\n\nfunc x() {}\n"
	out, changed, err := Fence(body, "myfeat")
	if err != nil {
		t.Fatalf("Fence: %v", err)
	}
	if !changed {
		t.Error("Fence changed = false, want true")
	}
	lines := strings.SplitN(out, "\n", 3)
	if len(lines) != 3 {
		t.Fatalf("fenced output has %d lines, want at least 3", len(lines))
	}
	if lines[0] != "//go:build wip_myfeat" {
		t.Errorf("first line = %q, want %q", lines[0], "//go:build wip_myfeat")
	}
	if lines[1] != "" {
		t.Errorf("second line = %q, want blank", lines[1])
	}
	if lines[2] != body {
		t.Errorf("body after fence = %q, want original %q", lines[2], body)
	}
}

func TestFenceWipPrefixNormalization(t *testing.T) {
	const body = "package main\n"
	bare, _, err := Fence(body, "foo")
	if err != nil {
		t.Fatalf("Fence(body, \"foo\"): %v", err)
	}
	tagged, _, err := Fence(body, "wip_foo")
	if err != nil {
		t.Fatalf("Fence(body, \"wip_foo\"): %v", err)
	}
	if bare != tagged {
		t.Errorf("Fence with slug \"foo\" = %q, with \"wip_foo\" = %q; want identical", bare, tagged)
	}
	if !strings.HasPrefix(bare, "//go:build wip_foo\n") {
		t.Errorf("output %q does not start with //go:build wip_foo", bare)
	}
	if strings.Contains(tagged, "wip_wip_") {
		t.Errorf("output %q contains a double wip_ prefix", tagged)
	}
}

func TestUnfence(t *testing.T) {
	const body = "package main\n\nfunc x() {}\n"
	cases := []struct {
		name        string
		content     string
		want        string
		wantChanged bool
	}{
		{
			name:        "removes fence and blank line",
			content:     "//go:build wip_myfeat\n\n" + body,
			want:        body,
			wantChanged: true,
		},
		{
			name:        "removes fence with any wip slug",
			content:     "//go:build wip_other_slug\n\n" + body,
			want:        body,
			wantChanged: true,
		},
		{
			name:        "fence without following blank line",
			content:     "//go:build wip_myfeat\n" + body,
			want:        body,
			wantChanged: true,
		},
		{
			name:        "fence line only",
			content:     "//go:build wip_myfeat",
			want:        "",
			wantChanged: true,
		},
		{
			name:    "idempotent when unfenced",
			content: body,
			want:    body,
		},
		{
			name:    "leaves non-wip constraint untouched",
			content: "//go:build linux\n\n" + body,
			want:    "//go:build linux\n\n" + body,
		},
		{
			name:    "empty content",
			content: "",
			want:    "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, changed, err := Unfence(tc.content)
			if err != nil {
				t.Fatalf("Unfence(%q) unexpected error: %v", tc.content, err)
			}
			if out != tc.want {
				t.Errorf("Unfence(%q) out = %q, want %q", tc.content, out, tc.want)
			}
			if changed != tc.wantChanged {
				t.Errorf("Unfence(%q) changed = %v, want %v", tc.content, changed, tc.wantChanged)
			}
		})
	}
}

func TestFenceUnfenceRoundTrip(t *testing.T) {
	bodies := []struct {
		name string
		body string
	}{
		{"lf body", "package main\n\nfunc x() {}\n"},
		{"crlf body", "package main\r\n\r\nfunc x() {}\r\n"},
		{"no trailing newline", "package main\n\nfunc x() {}"},
		{"leading blank lines", "\n\npackage main\n"},
		{"empty body", ""},
	}
	for _, tc := range bodies {
		t.Run(tc.name, func(t *testing.T) {
			fenced, changed, err := Fence(tc.body, "roundtrip")
			if err != nil {
				t.Fatalf("Fence: %v", err)
			}
			if !changed {
				t.Fatal("Fence changed = false, want true")
			}
			if tag, ok := IsFenced(fenced); !ok || tag != "wip_roundtrip" {
				t.Fatalf("IsFenced(fenced) = (%q, %v), want (%q, true)", tag, ok, "wip_roundtrip")
			}
			back, changed, err := Unfence(fenced)
			if err != nil {
				t.Fatalf("Unfence: %v", err)
			}
			if !changed {
				t.Fatal("Unfence changed = false, want true")
			}
			if back != tc.body {
				t.Errorf("round trip = %q, want byte-identical original %q", back, tc.body)
			}
		})
	}
}

func TestIsFenced(t *testing.T) {
	cases := []struct {
		name    string
		content string
		wantTag string
		wantOK  bool
	}{
		{"fenced", "//go:build wip_foo\n\npackage x\n", "wip_foo", true},
		{"fenced crlf first line", "//go:build wip_foo\r\n\r\npackage x\r\n", "wip_foo", true},
		{"fence line only", "//go:build wip_foo", "wip_foo", true},
		{"plain file", "package x\n", "", false},
		{"non-wip constraint", "//go:build linux\n\npackage x\n", "", false},
		{"fence not on first line", "\n//go:build wip_foo\npackage x\n", "", false},
		{"legacy comment form", "// +build wip_foo\n\npackage x\n", "", false},
		{"invalid ident after wip_", "//go:build wip_123\n\npackage x\n", "", false},
		{"compound constraint", "//go:build wip_foo && linux\n\npackage x\n", "", false},
		{"empty content", "", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tag, ok := IsFenced(tc.content)
			if tag != tc.wantTag || ok != tc.wantOK {
				t.Errorf("IsFenced(%q) = (%q, %v), want (%q, %v)", tc.content, tag, ok, tc.wantTag, tc.wantOK)
			}
		})
	}
}
