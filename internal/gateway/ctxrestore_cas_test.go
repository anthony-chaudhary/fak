package gateway

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ctxrestore_cas_test.go — witnesses for the durable media-restore CAS (#5163): a media (image)
// turn's restore handle survives both the #5164 media-cap eviction and a gateway restart, because
// its verbatim bytes were persisted to the on-disk content-addressed store; text stays RAM-only; a
// tampered entry fails closed to a miss; and an operator suppression purges the durable copy so it
// cannot resurrect a gated span across a restart.

// casTestBlob builds a media-class (>= ctxRestoreMediaThreshold) dropped-turn payload whose content
// varies by seed, so each call has a distinct digest.
func casTestBlob(seed rune) []byte {
	return []byte(`{"role":"user","content":"` + strings.Repeat(string(seed), ctxRestoreMediaThreshold) + `"}`)
}

// wipeStash empties the server's in-RAM restore stash — the observable effect of a gateway process
// restart on the stash (the durable CAS is exactly what must outlive it).
func wipeStash(srv *Server) {
	srv.ctxRestoreMu.Lock()
	srv.ctxRestore = nil
	srv.ctxRestoreMu.Unlock()
}

// TestRestoreDurableCASSurvivesEvictionAndRestart (#5163): a media entry evicted from the RAM stash
// by the media cap — and one lost to a process restart — still restores verbatim by its handle from
// the durable CAS.
func TestRestoreDurableCASSurvivesEvictionAndRestart(t *testing.T) {
	t.Setenv(ctxRestoreCASEnvDir, t.TempDir())
	srv := newTestServer(t)
	const trace = "t-cas"

	first := casTestBlob('a')
	firstID := ctxplan.Digest(first)
	srv.stashRestore(trace, firstID, "an image turn", first)

	// Overflow the media cap so the first media entry is evicted from RAM (#5164).
	for i := 0; i < maxCtxRestoreMediaEntriesPerSession; i++ {
		blob := casTestBlob(rune('b' + i))
		srv.stashRestore(trace, ctxplan.Digest(blob), "an image turn", blob)
	}

	got, err := srv.restoreContext("", ContextRestoreRequest{ID: firstID, TraceID: trace})
	if err != nil {
		t.Fatalf("evicted media entry must restore from the durable CAS: %v", err)
	}
	if got.Bytes != string(first) {
		t.Fatalf("durable restore bytes differ from the dropped turn")
	}
	if got.Provenance != "WITNESSED" {
		t.Fatalf("durable restore provenance = %q, want WITNESSED", got.Provenance)
	}

	// Restart: the RAM stash is gone, the durable copy still answers the same handle.
	wipeStash(srv)
	got, err = srv.restoreContext("", ContextRestoreRequest{ID: firstID, TraceID: trace})
	if err != nil {
		t.Fatalf("post-restart restore from the durable CAS: %v", err)
	}
	if got.Bytes != string(first) {
		t.Fatalf("post-restart durable restore bytes differ from the dropped turn")
	}
}

// TestRestoreDurableCASMediaOnlyAndTamperFailsClosed (#5163): only media-class payloads are
// persisted (text keeps its RAM-only story), and a durable entry whose bytes no longer hash to
// their digest address is refused — the read falls closed to a miss, never serving unproven bytes.
func TestRestoreDurableCASMediaOnlyAndTamperFailsClosed(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(ctxRestoreCASEnvDir, dir)
	srv := newTestServer(t)
	const trace = "t-cas-media"

	text := []byte(`{"role":"user","content":"a small text turn"}`)
	textID := ctxplan.Digest(text)
	srv.stashRestore(trace, textID, "text", text)
	if _, err := os.Stat(filepath.Join(dir, textID)); !os.IsNotExist(err) {
		t.Fatalf("text turn must not be persisted to the durable CAS (stat err = %v)", err)
	}

	blob := casTestBlob('m')
	id := ctxplan.Digest(blob)
	srv.stashRestore(trace, id, "an image turn", blob)
	if _, err := os.Stat(filepath.Join(dir, id)); err != nil {
		t.Fatalf("media turn must be persisted to the durable CAS: %v", err)
	}

	// Tamper with the durable entry, then simulate a restart: the digest re-verify must refuse it.
	if err := os.WriteFile(filepath.Join(dir, id), []byte("tampered"), 0o600); err != nil {
		t.Fatalf("tamper write: %v", err)
	}
	wipeStash(srv)
	if _, err := srv.restoreContext("", ContextRestoreRequest{ID: id, TraceID: trace}); !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("tampered durable entry err = %v, want ErrRestoreMiss (fail closed)", err)
	}
}

// TestRestoreDurableCASSuppressionPurges (#5163): an operator tombstone reaches the durable copy —
// the CAS file is removed at gate time, so after a restart (which forgets the in-RAM gate flags)
// the suppressed span still cannot be paged back in through its handle.
func TestRestoreDurableCASSuppressionPurges(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(ctxRestoreCASEnvDir, dir)
	srv := newTestServer(t)
	const trace = "t-cas-gate"

	blob := casTestBlob('s')
	id := ctxplan.Digest(blob)
	srv.stashRestore(trace, id, "an image turn", blob)
	if _, err := os.Stat(filepath.Join(dir, id)); err != nil {
		t.Fatalf("media turn must be persisted before suppression: %v", err)
	}

	if n := srv.tombstoneRestore(id); n != 1 {
		t.Fatalf("tombstoneRestore suppressed %d handles, want 1", n)
	}
	if _, err := os.Stat(filepath.Join(dir, id)); !os.IsNotExist(err) {
		t.Fatalf("suppression must purge the durable CAS entry (stat err = %v)", err)
	}

	// While the process lives, the stash refuses authoritatively (the gate flag is set).
	if _, err := srv.restoreContext("", ContextRestoreRequest{ID: id, TraceID: trace}); !errors.Is(err, ErrRestoreRefused) {
		t.Fatalf("gated stash entry err = %v, want ErrRestoreRefused", err)
	}
	// After a restart the gate flags are gone — and so is the durable copy: a plain miss, never a
	// resurrection.
	wipeStash(srv)
	if _, err := srv.restoreContext("", ContextRestoreRequest{ID: id, TraceID: trace}); !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("post-restart suppressed span err = %v, want ErrRestoreMiss", err)
	}
}

// TestRestoreDurableCASOffEnv (#5163): the env kill-switch restores the #5164 RAM-only behavior —
// nothing is written, and an evicted or restart-lost media entry is a plain miss again.
func TestRestoreDurableCASOffEnv(t *testing.T) {
	t.Setenv(ctxRestoreCASEnvDir, "off")
	srv := newTestServer(t)
	const trace = "t-cas-off"

	blob := casTestBlob('o')
	id := ctxplan.Digest(blob)
	srv.stashRestore(trace, id, "an image turn", blob)
	wipeStash(srv)
	if _, err := srv.restoreContext("", ContextRestoreRequest{ID: id, TraceID: trace}); !errors.Is(err, ErrRestoreMiss) {
		t.Fatalf("with the CAS off, a restart-lost media entry err = %v, want ErrRestoreMiss", err)
	}
}
