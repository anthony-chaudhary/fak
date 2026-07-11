package commitlane

import "testing"

// TestCommandMentionsRoot pins the lock-holder attribution predicate.
//
// commandMentionsRoot decides whether a running process's command line refers
// to the repo root, which is how Status() attributes an index/commit lock to a
// live owner. A false negative makes a busy lane look free (two writers can then
// collide on .git/index); a false positive blames an unrelated process. The
// load-bearing invariants are therefore: an empty root must never match, and a
// match must survive case folding and mixed path separators, because process
// command lines on this host appear as C:\work\fak, C:/work/fak, and c:/work/fak
// interchangeably (MSYS/git-bash vs native). These are exercised by no direct
// test — the Status() tests use a single fixed path form.
func TestCommandMentionsRoot(t *testing.T) {
	const root = `C:\work\fak`
	cases := []struct {
		name    string
		root    string
		command string
		want    bool
	}{
		{"empty root never matches a real command", "", `git -C C:\work\fak commit`, false},
		{"empty root and empty command", "", "", false},
		{"exact backslash form", root, `git -C C:\work\fak commit`, true},
		{"forward-slash command matches backslash root", root, `git -C C:/work/fak commit`, true},
		{"backslash command matches forward-slash root", `C:/work/fak`, `bash C:\work\fak\run.sh`, true},
		{"match is case-insensitive", `C:\Work\FAK`, `GIT -C c:\work\fak COMMIT`, true},
		{"trailing separator on root is normalized away", `C:/work/fak/`, `git -C C:/work/fak commit`, true},
		{"unrelated repo path does not match", root, `git -C C:\other\repo status`, false},
		{"command with no path does not match", root, `go test ./...`, false},
		// Documents the current substring-contains semantics: a sibling
		// directory that shares the root as a prefix does match. Locked here so
		// any future move to boundary-aware matching is a deliberate change.
		{"sibling path sharing the root prefix matches (contains semantics)", root, `tail C:/work/fak-notes/run.log`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := commandMentionsRoot(tc.command, tc.root); got != tc.want {
				t.Fatalf("commandMentionsRoot(%q, %q) = %v, want %v", tc.command, tc.root, got, tc.want)
			}
		})
	}
}
