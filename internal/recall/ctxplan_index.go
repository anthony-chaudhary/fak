package recall

import (
	"context"
	"errors"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// ctxplan_index.go — persist a ctxplan candidate Index ALONGSIDE the recall core image, so a
// resumed session RE-ATTACHES its index instead of rebuilding it from the page table every
// time (issue #558, half a). The core image is manifest.json (the page table) + cas.json (the
// swap device); this adds a sibling index.json (the SAFE-metadata candidate index). The three
// files together are the durable session: the manifest is the history, the CAS is its bytes,
// and the index is its access path.
//
// Why the index belongs next to the core image and not inside the manifest: the index is
// DERIVED, regenerable state (BuildIndex over the page table reproduces it bit-for-bit), so it
// is a CACHE of the manifest, not a second source of truth. Keeping it a separate optional file
// means a missing/garbage index.json is never fatal — LoadIndex reports os.ErrNotExist and the
// caller falls back to AttachIndex (a one-time rebuild from the manifest). The manifest stays
// the authority; the index is the fast path.

// IndexFile is the sibling filename the candidate index persists to, in the same core-image
// directory as manifest.json and cas.json.
const IndexFile = "index.json"

// PersistIndex writes a ctxplan candidate index's SAFE-metadata image to index.json in the
// core-image directory dir (the same dir Recorder.Persist / Session.Persist write
// manifest.json + cas.json to). The image carries no bytes — only the same SAFE Span metadata
// the planner already reasons over (a sealed span persists with its Sealed flag and its
// sealed-safe descriptor), so writing it leaks nothing the manifest did not already expose. A
// nil index is refused (there is nothing to persist).
func PersistIndex(dir string, ix *ctxplan.Index) error {
	if ix == nil {
		return errors.New("recall: PersistIndex: nil index")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	b, err := ctxplan.MarshalIndexImage(ix)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, IndexFile), b, 0o644)
}

// LoadIndex reads a persisted candidate index from dir/index.json and re-attaches it
// (rederiving the inverted posting lists + durable set via ctxplan.RestoreIndex, a one-time
// O(spans) rebuild). It returns the underlying os.ErrNotExist when no index was persisted, so
// a caller can branch on errors.Is(err, fs.ErrNotExist) and fall back to AttachIndex. A
// present-but-corrupt or version-skewed image is a non-NotExist error (fail closed: a wrong
// index is worse than a rebuilt one).
func LoadIndex(dir string) (*ctxplan.Index, error) {
	b, err := os.ReadFile(filepath.Join(dir, IndexFile))
	if err != nil {
		return nil, err // os.IsNotExist(err) is true when no index was persisted
	}
	return ctxplan.UnmarshalIndexImage(b)
}

// AttachIndex builds a candidate index over a reloaded session's page table — the one-time
// O(N) rebuild a resumed session pays the FIRST time, before any index.json exists, or the
// fallback when LoadIndex reports os.ErrNotExist. It lowers the session through the same
// CtxStore the planner views (so a sealed page becomes a Sealed span, a tombstoned page a
// Tombstoned span), then BuildIndexes the result. After AttachIndex + PersistIndex, later
// resumes re-attach in one read via LoadIndex instead of re-scanning the manifest.
func AttachIndex(ctx context.Context, s *Session) (*ctxplan.Index, error) {
	spans, err := NewCtxStore(s).Spans(ctx)
	if err != nil {
		return nil, err
	}
	return ctxplan.BuildIndex(spans), nil
}

// LoadOrAttachIndex is the resume convenience: re-attach the persisted index if dir/index.json
// exists, else rebuild it from the session's page table (and the caller may then PersistIndex
// to make the next resume a fast re-attach). A corrupt/version-skewed index is propagated as an
// error rather than silently rebuilt — only a genuinely ABSENT index falls back to a rebuild,
// so a tampered cache never downgrades to a quiet re-scan that hides it.
func LoadOrAttachIndex(ctx context.Context, dir string, s *Session) (*ctxplan.Index, error) {
	ix, err := LoadIndex(dir)
	if err == nil {
		return ix, nil
	}
	if !os.IsNotExist(err) {
		return nil, err
	}
	return AttachIndex(ctx, s)
}
