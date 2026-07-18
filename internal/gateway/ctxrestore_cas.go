package gateway

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ctxrestore_cas.go — the DURABLE half of the media restore handle (#5163): a content-addressed
// on-disk byte store backing the RAM-only per-trace stash in ctxrestore.go. The stash is both
// non-durable (a gateway restart empties it, so a resumed session's fak_context_restore(id) handle
// goes to ErrRestoreMiss) and capacity-bounded (#5164's media cap evicts oldest-media-out) — and for
// a media (image) turn the stash held the payload's ONLY copy: text has an out-of-band recovery
// story, a pasted image does not. This file persists media-class restore bytes to a durable CAS
// keyed by the SAME sha256-hex digest scheme as ctxplan.Digest / recall's cas.json, so an image
// handle survives both restart and eviction. ctxplan already persists only a Digest (image.go); the
// durable byte store here is the missing half.
//
// The design axis mirrors recall's core image (recall.Persist / Load): the digest IS the address,
// so the store needs no index — one file per digest, and a read re-verifies the bytes hash to their
// address (a tampered entry fails CLOSED to a miss, exactly recall.Load's corrupt-CAS rule). Every
// write is best-effort and fail-open (the request path must never fault because durable persistence
// did), every read is fail-closed (bytes are served only when they provably match the handle).
//
// Trust gate: durability must not outlive suppression. The in-RAM seal/tombstone flags die with the
// process, so the durable copy is purged the moment an operator gates the digest
// (gateRestoreByDigest → purgeRestoreCAS) — deletion is the only suppression that survives a
// restart. The read fall-through in restoreContext therefore never resurrects a suppressed span:
// a stashed-and-gated entry refuses authoritatively before the CAS is consulted, and a gated digest
// no longer exists on disk.

const (
	// ctxRestoreCASEnvDir overrides the durable media-restore CAS directory; "off" (or "0"/"none")
	// disables durable persistence entirely, leaving the #5164 RAM-only behavior.
	ctxRestoreCASEnvDir = "FAK_CTXRESTORE_CAS_DIR"

	// ctxRestoreCASDirRel is the default workspace-relative durable CAS directory — the same
	// .fak/* neighborhood the toolproc journal writes (mcp_toolproc.go), so a workspace's durable
	// gateway state lives under one gitignored root.
	ctxRestoreCASDirRel = ".fak/ctxrestore/cas"

	// maxCtxRestoreCASEntries bounds the durable directory, oldest-modified-out on overflow —
	// the disk analogue of the stash caps, generous because a durable entry is one media turn.
	maxCtxRestoreCASEntries = 64
)

// ctxRestoreCASDir resolves the durable CAS directory from the environment at call time (the same
// per-call resolution mcpToolprocAppend uses, so tests and operators can flip it without a server
// rebuild). "" means durable persistence is off.
func ctxRestoreCASDir() string {
	dir := strings.TrimSpace(os.Getenv(ctxRestoreCASEnvDir))
	switch strings.ToLower(dir) {
	case "off", "0", "none":
		return ""
	case "":
		return ctxRestoreCASDirRel
	}
	return dir
}

// isRestoreDigest reports whether id is a well-formed content address (64 lowercase sha256 hex
// chars — the ctxplan.Digest shape). The durable store joins the id into a filesystem path, and the
// id arrives from the caller on the read side, so anything that is not exactly a digest is refused
// before it can name a path component.
func isRestoreDigest(id string) bool {
	if len(id) != 64 {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// persistRestoreCAS best-effort-writes a media-class dropped turn's VERBATIM bytes to the durable
// CAS under its digest address. Content-addressing makes the write idempotent — an existing entry
// is the same bytes by construction, so it is left untouched. The write is tmp+rename so a crashed
// writer never leaves a half-written entry at a digest address (a partial file at a tmp name fails
// the digest re-verify anyway, but it also never shadows the real entry). All failures are
// swallowed: durability is an upgrade over #5164's RAM-only stash, never a new fault path.
func persistRestoreCAS(id string, taskBytes []byte) {
	dir := ctxRestoreCASDir()
	if dir == "" || !isRestoreDigest(id) || len(taskBytes) == 0 {
		return
	}
	path := filepath.Join(dir, id)
	if _, err := os.Stat(path); err == nil {
		return // already durable: same digest ⇒ same bytes
	}
	if os.MkdirAll(dir, 0o755) != nil {
		return
	}
	tmp := path + ".tmp" + strconv.Itoa(os.Getpid())
	if os.WriteFile(tmp, taskBytes, 0o600) != nil {
		return
	}
	if os.Rename(tmp, path) != nil {
		_ = os.Remove(tmp)
		return
	}
	pruneRestoreCAS(dir)
}

// loadRestoreCAS reads the durable CAS entry for a digest and re-verifies the bytes hash to their
// address before serving them — recall.Load's corrupt-CAS rule applied per-entry: a tampered or
// truncated swap file fails CLOSED to a miss, never to serving bytes the handle does not name.
func loadRestoreCAS(id string) ([]byte, bool) {
	dir := ctxRestoreCASDir()
	if dir == "" || !isRestoreDigest(id) {
		return nil, false
	}
	b, err := os.ReadFile(filepath.Join(dir, id))
	if err != nil || len(b) == 0 {
		return nil, false
	}
	if ctxplan.Digest(b) != id {
		return nil, false // digest mismatch: the entry does not prove itself, refuse it
	}
	return b, true
}

// purgeRestoreCAS removes a digest's durable entry — the suppression propagation edge (#5163).
// Called unconditionally from gateRestoreByDigest: the in-RAM seal/tombstone flags do not survive a
// restart, so deleting the durable copy is the only way an operator's suppression outlives the
// process. Idempotent and best-effort by design (a digest that was never persisted is a valid
// no-op, matching the gate's fail-open-on-a-miss contract).
func purgeRestoreCAS(id string) {
	dir := ctxRestoreCASDir()
	if dir == "" || !isRestoreDigest(id) {
		return
	}
	_ = os.Remove(filepath.Join(dir, id))
}

// pruneRestoreCAS bounds the durable directory to maxCtxRestoreCASEntries, removing
// oldest-modified entries first — the disk analogue of the stash's oldest-out overflow, so the
// most recent media turn a resuming model is likeliest to ask for is the one kept. Only
// digest-named files are counted or removed; anything else in the directory is left alone.
func pruneRestoreCAS(dir string) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	type casEnt struct {
		name string
		mod  int64
	}
	var cas []casEnt
	for _, e := range ents {
		if e.IsDir() || !isRestoreDigest(e.Name()) {
			continue
		}
		info, ierr := e.Info()
		if ierr != nil {
			continue
		}
		cas = append(cas, casEnt{name: e.Name(), mod: info.ModTime().UnixNano()})
	}
	if len(cas) <= maxCtxRestoreCASEntries {
		return
	}
	sort.Slice(cas, func(i, j int) bool { return cas[i].mod < cas[j].mod })
	for _, e := range cas[:len(cas)-maxCtxRestoreCASEntries] {
		_ = os.Remove(filepath.Join(dir, e.name))
	}
}
