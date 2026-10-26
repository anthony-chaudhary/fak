package journal

import (
	"bufio"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

type journalCloseInjectedErrorWriter struct {
	err error
}

func (w journalCloseInjectedErrorWriter) Write([]byte) (int, error) {
	return 0, w.err
}

// fak-test:runtime fast est=20ms
func TestJournalCloseFlushErrorStillClosesFile(t *testing.T) {
	flushSentinel := errors.New("injected flush failure")
	for _, tc := range []struct {
		name      string
		preclosed bool
	}{
		{name: "live file"},
		{name: "already closed file", preclosed: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "journal.jsonl")
			j, err := Open(path)
			if err != nil {
				t.Fatalf("Open returned an error: %v", err)
			}
			actualFile := j.f
			t.Cleanup(func() { _ = actualFile.Close() })

			if tc.preclosed {
				if err := actualFile.Close(); err != nil {
					t.Fatalf("preclose returned an error: %v", err)
				}
			}
			j.bw = bufio.NewWriter(journalCloseInjectedErrorWriter{err: flushSentinel})
			if _, err := j.bw.WriteString("pending"); err != nil {
				t.Fatalf("buffer payload: %v", err)
			}

			closeErr := j.Close()
			if !errors.Is(closeErr, flushSentinel) {
				t.Fatalf("Close error does not preserve the flush failure: %v", closeErr)
			}
			if tc.preclosed && !errors.Is(closeErr, os.ErrClosed) {
				t.Fatalf("Close error does not compose the file-close failure: %v", closeErr)
			}
			if _, err := actualFile.Write([]byte("probe")); !errors.Is(err, os.ErrClosed) {
				t.Fatalf("saved file handle remains writable after Close: %v", err)
			}
		})
	}
}
