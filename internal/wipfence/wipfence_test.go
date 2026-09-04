package wipfence

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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

func TestIsolateAndStrip(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "my_file.go")
	const original = "package main\n\nfunc hello() {}\n"
	if err := os.WriteFile(p, []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}

	// Isolate with explicit session
	if err := Isolate(p, "sess1"); err != nil {
		t.Fatalf("Isolate: %v", err)
	}
	content, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	tag, ok := IsFenced(string(content))
	if !ok || tag != "wip_sess1" {
		t.Fatalf("IsFenced = (%q, %v), want (\"wip_sess1\", true)", tag, ok)
	}

	// Idempotent Isolate
	if err := Isolate(p, "sess1"); err != nil {
		t.Fatalf("idempotent Isolate: %v", err)
	}

	// StripSession mismatch does not strip
	if err := StripSession(p, "other_sess"); err != nil {
		t.Fatalf("StripSession: %v", err)
	}
	content, _ = os.ReadFile(p)
	if _, ok := IsFenced(string(content)); !ok {
		t.Fatal("StripSession stripped mismatched session tag")
	}

	// StripSession matching strips
	if err := StripSession(p, "sess1"); err != nil {
		t.Fatalf("StripSession: %v", err)
	}
	content, _ = os.ReadFile(p)
	if string(content) != original {
		t.Fatalf("content after StripSession = %q, want %q", string(content), original)
	}

	// Isolate with empty session derives from path
	if err := Isolate(p, ""); err != nil {
		t.Fatalf("Isolate with empty session: %v", err)
	}
	content, _ = os.ReadFile(p)
	tag, ok = IsFenced(string(content))
	if !ok || tag != "wip_my_file" {
		t.Fatalf("IsFenced = (%q, %v), want (\"wip_my_file\", true)", tag, ok)
	}

	// Plain Strip
	if err := Strip(p); err != nil {
		t.Fatalf("Strip: %v", err)
	}
	content, _ = os.ReadFile(p)
	if string(content) != original {
		t.Fatalf("content after Strip = %q, want %q", string(content), original)
	}
}

func TestDetectAndIsolateUntracked(t *testing.T) {
	dir := t.TempDir()
	f1 := filepath.Join(dir, "f1.go")
	f2 := filepath.Join(dir, "sub", "f2.go")
	if err := os.MkdirAll(filepath.Dir(f2), 0o755); err != nil {
		t.Fatal(err)
	}
	const src = "package p\n"
	if err := os.WriteFile(f1, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(f2, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	files, err := DetectUntracked(dir)
	if err != nil {
		t.Fatalf("DetectUntracked: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("DetectUntracked found %d files, want 2 (%v)", len(files), files)
	}

	modified, err := IsolateUntracked(dir, "session_test")
	if err != nil {
		t.Fatalf("IsolateUntracked: %v", err)
	}
	if len(modified) != 2 {
		t.Fatalf("IsolateUntracked modified %d files, want 2 (%v)", len(modified), modified)
	}

	// Check files are fenced
	for _, f := range modified {
		full := filepath.Join(dir, filepath.FromSlash(f))
		b, _ := os.ReadFile(full)
		if tag, ok := IsFenced(string(b)); !ok || tag != "wip_session_test" {
			t.Errorf("file %s tag=(%q, %v), want (\"wip_session_test\", true)", f, tag, ok)
		}
	}

	// StripUntracked
	stripped, err := StripUntracked(dir, "session_test")
	if err != nil {
		t.Fatalf("StripUntracked: %v", err)
	}
	if len(stripped) != 2 {
		t.Fatalf("StripUntracked stripped %d files, want 2 (%v)", len(stripped), stripped)
	}
	for _, f := range stripped {
		full := filepath.Join(dir, filepath.FromSlash(f))
		b, _ := os.ReadFile(full)
		if _, ok := IsFenced(string(b)); ok {
			t.Errorf("file %s is still fenced after StripUntracked", f)
		}
	}
}

func TestWIPFenceHidesUntrackedFromPeerBuild(t *testing.T) {
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH; skipping build test")
	}
	dir := t.TempDir()
	write := func(name, body string) string {
		t.Helper()
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	write("go.mod", "module testwipfence\n\ngo 1.21\n")
	write("main.go", "package main\n\nfunc main() {}\n")
	badPath := write("broken.go", "package main\n\nthis is invalid syntax !!!\n")

	nullDev := "/dev/null"
	if runtime.GOOS == "windows" {
		nullDev = "NUL"
	}

	runBuild := func(tags string) (string, error) {
		t.Helper()
		args := []string{"build"}
		if tags != "" {
			args = append(args, "-tags", tags)
		}
		args = append(args, "-o", nullDev, "./...")
		cmd := exec.Command(goBin, args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=")
		out, berr := cmd.CombinedOutput()
		return string(out), berr
	}

	// 1. Without wipfence, the untracked syntax-broken file must fail native go build
	out, err := runBuild("")
	if err == nil {
		t.Fatalf("expected native go build to fail with syntax-broken file, but got exit 0:\n%s", out)
	}

	// 2. Enable wipfence on the untracked broken file using Isolate
	session := "session_11231"
	if err := Isolate(badPath, session); err != nil {
		t.Fatalf("Isolate(%q, %q) failed: %v", badPath, session, err)
	}

	// 3. With wipfence enabled, peer build without the tag must exit 0!
	out, err = runBuild("")
	if err != nil {
		t.Fatalf("expected peer build without tag to exit 0 with wipfence enabled, but failed:\n%s", out)
	}

	// 4. Verifying with the tag compiles the broken file back in (fails as expected)
	out, err = runBuild("wip_" + session)
	if err == nil {
		t.Fatalf("expected build with tag wip_%s to fail, but succeeded:\n%s", session, out)
	}

	// 5. Strip the fence: native build should fail again
	if err := Strip(badPath); err != nil {
		t.Fatalf("Strip(%q) failed: %v", badPath, err)
	}
	out, err = runBuild("")
	if err == nil {
		t.Fatalf("expected native go build to fail after Strip, but succeeded:\n%s", out)
	}
}
