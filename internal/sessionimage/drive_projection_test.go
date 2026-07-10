package sessionimage

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// driveState returns a representative spent-down drive: TokensLeft at 20% of a 1000 cap,
// a scheduling Priority, a throttle Pace, a pinned objective (PinID+Digest), and a
// non-original Generation — every axis the projection is meant to carry.
func driveState(id string) session.State {
	return session.State{
		TraceID:  id,
		Run:      session.Throttled,
		Reason:   "operator-offload", // a full-State field the projection must NOT carry
		Budget:   session.Budget{TurnsLeft: 2, TokensLeft: 200, ContextTokensLeft: 4096, SpendMicroCentsLeft: 750},
		Priority: 5,
		Pace:     session.Pace{MaxTokensPerTurn: 512, MinTurnGapMs: 100},
		ObjectivePin: ctxplan.ObjectivePin{
			PinID:  "obj-42",
			Text:   "ship the witnessed checkpoint fold",
			Digest: "d1a9f0c0feed",
			Step:   3,
		},
		Generation: 4,
		Rev:        11,
	}
}

// TestDriveProjection is the #4126 witness: DumpDir on an Input whose Drive carries a
// spent budget/priority/pace/pin writes a drive.json part; LoadDir lists and verifies it
// in Meta.Parts; Image.DriveProjection() round-trips the projected axes byte-for-byte equal
// to the source session.State fields — and the written block is the lean projection, not
// the full session.json transcript. A byte-flip of drive.json fails Load closed.
func TestDriveProjection(t *testing.T) {
	t.Run("round-trips projected axes and integrity-indexes the part", func(t *testing.T) {
		dir := t.TempDir()
		st := driveState("sess-dp")
		if _, err := DumpDir(dir, Input{SessionID: "sess-dp", Drive: st, Now: 1}); err != nil {
			t.Fatalf("DumpDir: %v", err)
		}

		img, err := LoadDir(dir)
		if err != nil {
			t.Fatalf("LoadDir: %v", err)
		}
		dp, ok, err := img.DriveProjection()
		if err != nil {
			t.Fatalf("DriveProjection: %v", err)
		}
		if !ok {
			t.Fatalf("DriveProjection present=false for a drive-carrying image")
		}

		// The four remaining-budget axes round-trip byte-for-byte equal to the source.
		if dp.Budget.TurnsLeft != st.Budget.TurnsLeft ||
			dp.Budget.TokensLeft != st.Budget.TokensLeft ||
			dp.Budget.ContextTokensLeft != st.Budget.ContextTokensLeft ||
			dp.Budget.SpendMicroCentsLeft != st.Budget.SpendMicroCentsLeft {
			t.Fatalf("budget axes changed: got %+v want %+v", dp.Budget, st.Budget)
		}
		if dp.Priority != st.Priority {
			t.Fatalf("priority changed: got %d want %d", dp.Priority, st.Priority)
		}
		if dp.Pace != st.Pace {
			t.Fatalf("pace changed: got %+v want %+v", dp.Pace, st.Pace)
		}
		if dp.ObjectivePin.PinID != st.ObjectivePin.PinID || dp.ObjectivePin.Digest != st.ObjectivePin.Digest {
			t.Fatalf("objective pin changed: got %+v want PinID=%q Digest=%q",
				dp.ObjectivePin, st.ObjectivePin.PinID, st.ObjectivePin.Digest)
		}
		if dp.Generation != st.Generation {
			t.Fatalf("generation changed: got %d want %d", dp.Generation, st.Generation)
		}

		// drive.json is in the sha256 integrity index like every other sibling.
		var listed bool
		for _, p := range img.Meta.Parts {
			if p.Name == DriveFile {
				listed = true
			}
		}
		if !listed {
			t.Fatalf("%s not in the image integrity index: %+v", DriveFile, img.Meta.Parts)
		}

		// The written block is the lean projection, NOT the full-State transcript:
		// session.json carries trace_id/run/reason; drive.json carries none of them.
		db, err := os.ReadFile(filepath.Join(dir, DriveFile))
		if err != nil {
			t.Fatalf("read %s: %v", DriveFile, err)
		}
		body := string(db)
		if !strings.Contains(body, "obj-42") || !strings.Contains(body, "\"tokens_left\": 200") {
			t.Fatalf("%s missing projected axes: %s", DriveFile, body)
		}
		for _, leak := range []string{"trace_id", "\"run\"", "operator-offload"} {
			if strings.Contains(body, leak) {
				t.Fatalf("%s carried full-State field %q (should be a lean projection): %s", DriveFile, leak, body)
			}
		}
	})

	t.Run("a byte-flip of drive.json fails Load closed", func(t *testing.T) {
		dir := t.TempDir()
		if _, err := DumpDir(dir, Input{SessionID: "sess-dp2", Drive: driveState("sess-dp2"), Now: 1}); err != nil {
			t.Fatalf("DumpDir: %v", err)
		}
		path := filepath.Join(dir, DriveFile)
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", DriveFile, err)
		}
		b[len(b)/2] ^= 0xFF // flip a byte in the middle
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatalf("write tampered %s: %v", DriveFile, err)
		}
		if _, err := LoadDir(dir); err == nil {
			t.Fatalf("LoadDir accepted a tampered %s", DriveFile)
		} else if !strings.Contains(err.Error(), DriveFile) {
			t.Fatalf("Load error did not name the tampered part: %v", err)
		}
	})

	t.Run("a drive-only zero projection stays absent", func(t *testing.T) {
		dir := t.TempDir()
		// A default state carries no spent budget/priority/pace/pin — projection is zero.
		st := session.State{TraceID: "sess-dp3"}
		if _, err := DumpDir(dir, Input{SessionID: "sess-dp3", Drive: st, Now: 1}); err != nil {
			t.Fatalf("DumpDir: %v", err)
		}
		img, err := LoadDir(dir)
		if err != nil {
			t.Fatalf("LoadDir: %v", err)
		}
		_, ok, err := img.DriveProjection()
		if err != nil {
			t.Fatalf("DriveProjection: %v", err)
		}
		if ok {
			t.Fatalf("a zero-projection session wrote a %s part", DriveFile)
		}
		for _, p := range img.Meta.Parts {
			if p.Name == DriveFile {
				t.Fatalf("%s listed for a session with nothing to project", DriveFile)
			}
		}
	})
}
