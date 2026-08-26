package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
)

// The codex loop hook used to hand its open *os.File straight to diagnose, so an
// injected, slow, or panicking diagnose path kept a live handle on the transcript
// after the hook had already allowed the turn. Windows refuses to unlink a file with
// a live handle, so the stranded handle wedges whoever rotates the transcript next.
// sessions_codex_loop.go now snapshots the file under the codexLoopLaunchMaxBytes
// ceiling and closes the handle BEFORE diagnose is entered.
//
// Both assertions are red against the pre-fix source, which passed fh directly: the
// reader asserted as *os.File, and the faithfulness check never got to run.
func TestCodexLoopHookClosesTranscriptBeforeDiagnose(t *testing.T) {
	t.Setenv(guardActiveEnv, "")
	home, sessionID := writeCodexHookSession(t, "openai")

	var (
		sawFile  bool
		sawBytes []byte
		onDisk   []byte
		readErr  error
	)
	var stdout, stderr bytes.Buffer
	code := sessionsCodexLoopHookUnbounded(&stdout, &stderr,
		strings.NewReader(`{"session_id":"`+sessionID+`"}`),
		[]string{"--hardened", "--codex-home", home},
		func(r io.Reader, path string) (codexLoopDiagnosis, error) {
			// Witness 1: the handle is already gone by the time diagnosis starts.
			_, sawFile = r.(*os.File)
			// Witness 2: closing early cost diagnose no content. Without this, the
			// leak could be "fixed" by handing over an empty or exhausted reader.
			sawBytes, readErr = io.ReadAll(r)
			onDisk, _ = os.ReadFile(path)
			return codexLoopDiagnosis{}, nil
		})

	if code != 0 {
		t.Fatalf("exit=%d stderr=%q", code, stderr.String())
	}
	if sawFile {
		t.Fatal("diagnose received the live *os.File — the transcript handle is still open across diagnosis")
	}
	if readErr != nil {
		t.Fatalf("diagnose could not read its reader: %v", readErr)
	}
	if len(onDisk) == 0 {
		t.Fatal("fixture transcript is empty — the byte comparison below would be vacuous")
	}
	if !bytes.Equal(sawBytes, onDisk) {
		t.Fatalf("diagnose saw %d bytes, transcript holds %d — the snapshot is not faithful",
			len(sawBytes), len(onDisk))
	}
}
