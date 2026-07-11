package sessionimage

import (
	"context"
	"encoding/json"
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

// TestDriveProjectionLeakGate is the #4128 witness: the drive projection is portable ACROSS
// hosts and accounts, so it must carry NO origin identity. A projection built from a State
// whose CacheAffinity is populated (an account-scoped cache route) and whose ParentTrace is
// set, dumped into an image whose Meta.Host/Account are set, produces a drive.json that holds
// the STRUCTURAL axes (budget/priority/pace/pin/generation) but NONE of the host, account,
// affinity, or trace tokens — and a re-home Migration to a new host does not reintroduce them.
// The host/account provenance stays on image.json Meta (the audited record), never the drive.
func TestDriveProjectionLeakGate(t *testing.T) {
	const (
		host      = "boxA"
		toHost    = "boxB"
		account   = "acct-7"
		affinity  = "acct-7-route"
		fromTrace = "trace-from-acct-7"
		toTrace   = "trace-to-acct-7"
		parentTr  = "parent-trace-acct-7"
	)
	// The origin tokens a re-home must never carry across the offload boundary. "acct-7" is
	// listed on its own so a substring leak (affinity/fromTrace all contain it) is caught even
	// if a fuller token somehow survived.
	forbidden := []string{host, toHost, account, affinity, fromTrace, toTrace, parentTr}

	// A spent-down drive whose SOURCE State is packed with the account/host-derived lineage the
	// projection must drop: an account-scoped cache-affinity decision and a parent trace id.
	st := driveState("sess-leak")
	st.ParentTrace = parentTr
	st.CacheAffinity = session.CacheAffinityDecision{
		Action:      session.CacheAffinityPreserve,
		AffinityKey: affinity,
		FromTraceID: fromTrace,
		ToTraceID:   toTrace,
	}

	dir := t.TempDir()
	if _, err := DumpDir(dir, Input{
		SessionID: "sess-leak",
		Drive:     st,
		Host:      host,
		Account:   account,
		Now:       1,
	}); err != nil {
		t.Fatalf("DumpDir: %v", err)
	}

	// assertGated re-reads drive.json and asserts it is origin-free with the structural axes
	// the projection is meant to carry still intact — the same check before and after a re-home.
	assertGated := func(when string) {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(dir, DriveFile))
		if err != nil {
			t.Fatalf("read %s (%s): %v", DriveFile, when, err)
		}
		body := string(b)
		for _, tok := range forbidden {
			if strings.Contains(body, tok) {
				t.Errorf("%s leaked origin token %q (%s):\n%s", DriveFile, tok, when, body)
			}
		}
		var dp DriveProjection
		if err := json.Unmarshal(b, &dp); err != nil {
			t.Fatalf("decode %s (%s): %v", DriveFile, when, err)
		}
		if dp.Budget.TokensLeft != st.Budget.TokensLeft {
			t.Errorf("tokens_left dropped by the scrub (%s): got %d want %d", when, dp.Budget.TokensLeft, st.Budget.TokensLeft)
		}
		if dp.Priority != st.Priority {
			t.Errorf("priority dropped by the scrub (%s): got %d want %d", when, dp.Priority, st.Priority)
		}
		if dp.Pace != st.Pace {
			t.Errorf("pace dropped by the scrub (%s): got %+v want %+v", when, dp.Pace, st.Pace)
		}
		if dp.ObjectivePin.PinID != st.ObjectivePin.PinID || dp.ObjectivePin.Digest != st.ObjectivePin.Digest {
			t.Errorf("objective pin dropped by the scrub (%s): got %+v", when, dp.ObjectivePin)
		}
		if dp.Generation != st.Generation {
			t.Errorf("generation dropped by the scrub (%s): got %d want %d", when, dp.Generation, st.Generation)
		}
	}

	// At dump: the leak-gate held, the structural axes survived.
	assertGated("after DumpDir")

	// A re-home to a new host re-writes the projection through the leak-gate. It stays
	// origin-free, the audited Migration is recorded on Meta, and the integrity index still
	// verifies the freshly written drive.json.
	img, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	res, err := img.Rehydrate(context.Background(), RehydrateOptions{ToHost: toHost, WriteBack: true, Now: 2})
	if err != nil {
		t.Fatalf("Rehydrate re-home: %v", err)
	}
	if !res.Migrated {
		t.Fatalf("re-home to a new host recorded no migration")
	}
	assertGated("after re-home Migration")

	// The re-write did not break the sha256 integrity index (LoadDir fails closed otherwise).
	reloaded, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("image integrity broken after re-home re-write: %v", err)
	}
	// The destination host IS recorded on image.json Meta (the audited provenance record) —
	// proving the token exists in the image, just never in the drive projection.
	if reloaded.Meta.Host != toHost {
		t.Fatalf("re-home did not record the destination host on Meta: got %q want %q", reloaded.Meta.Host, toHost)
	}
}
