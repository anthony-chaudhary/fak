package toolproc

import (
	"os"
	"path/filepath"
	"testing"
)

// The journal's own readers and appenders must not block a concurrent
// session's compaction swap: while either holds the journal open, the
// replaceFileAtomic rename must still supersede it (#3555). On Windows that
// takes both halves of the fix — handles opened with FILE_SHARE_DELETE
// (OpenShareDelete / OpenAppendShareDelete) and a POSIX-semantics rename
// (renameOverOpenHandles); with either half missing the swap fails
// ERROR_ACCESS_DENIED. On POSIX a rename over an open file succeeds
// regardless, so the assertions hold everywhere and only Windows exercises
// the fix.
func TestCompactionSwapSupersedesOpenJournal(t *testing.T) {
	old := []byte(`{"kind":"spawn"}` + "\n")
	replacement := []byte(`{"kind":"exit"}` + "\n")
	cases := []struct {
		name string
		open func(string) (*os.File, error)
	}{
		{"reader", OpenShareDelete},
		{"appender", OpenAppendShareDelete},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			journal := filepath.Join(t.TempDir(), "journal.jsonl")
			if err := os.WriteFile(journal, old, 0o644); err != nil {
				t.Fatal(err)
			}
			f, err := tc.open(journal)
			if err != nil {
				t.Fatalf("open journal: %v", err)
			}
			defer f.Close()
			if err := replaceFileAtomic(journal, replacement); err != nil {
				t.Fatalf("swap while journal held open by %s: %v", tc.name, err)
			}
			got, err := os.ReadFile(journal)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(replacement) {
				t.Fatalf("journal after swap = %q, want %q", got, replacement)
			}
		})
	}
}

// The superseded handle must keep observing the complete pre-swap journal —
// the "pre- or post-compaction file, both complete" promise CompactJournalFile
// documents for concurrent readers.
func TestSupersededReaderSeesPreSwapJournal(t *testing.T) {
	old := []byte(`{"kind":"spawn"}` + "\n")
	journal := filepath.Join(t.TempDir(), "journal.jsonl")
	if err := os.WriteFile(journal, old, 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := OpenShareDelete(journal)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := replaceFileAtomic(journal, []byte(`{"kind":"exit"}`+"\n")); err != nil {
		t.Fatalf("swap while journal held open: %v", err)
	}
	got := make([]byte, len(old)+16)
	n, _ := f.Read(got)
	if string(got[:n]) != string(old) {
		t.Fatalf("superseded handle read %q, want pre-swap %q", got[:n], old)
	}
}

// OpenAppendShareDelete must keep O_APPEND|O_CREATE|O_WRONLY semantics: create
// a missing journal, and append at end-of-file rather than overwrite — the
// Windows implementation reproduces them via CreateFile rather than inheriting
// them from os.OpenFile.
func TestOpenAppendShareDeleteSemantics(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "journal.jsonl")

	f, err := OpenAppendShareDelete(journal) // missing: must create
	if err != nil {
		t.Fatalf("open missing journal: %v", err)
	}
	if _, err := f.Write([]byte("one\n")); err != nil {
		t.Fatalf("first append: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	f, err = OpenAppendShareDelete(journal) // existing: must append, not truncate
	if err != nil {
		t.Fatalf("reopen journal: %v", err)
	}
	if _, err := f.Write([]byte("two\n")); err != nil {
		t.Fatalf("second append: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(journal)
	if err != nil {
		t.Fatal(err)
	}
	if want := "one\ntwo\n"; string(got) != want {
		t.Fatalf("journal content = %q, want %q", got, want)
	}
}

// A missing journal must keep reporting os.IsNotExist through OpenShareDelete
// so ParseTailFile's fresh-workspace fail-open (nil, nil) keeps working.
func TestOpenShareDeleteMissingFile(t *testing.T) {
	_, err := OpenShareDelete(filepath.Join(t.TempDir(), "does-not-exist.jsonl"))
	if !os.IsNotExist(err) {
		t.Fatalf("open missing journal: err = %v, want os.IsNotExist", err)
	}
}
