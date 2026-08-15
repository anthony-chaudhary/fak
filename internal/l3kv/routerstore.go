package l3kv

// routerStore is the storedrv-backed Store the package doc of store.go reserved a
// slot for: the span→content MANIFEST lives here (span residency digest → content
// ref), while the content BYTES route through the EXISTING internal/storedrv
// Router — blobfs as the local durable stand-in, blobhttp as the remote pool —
// instead of a second purpose-built flat-file format. That closes the #1472 gap
// the diskStore stand-in left open: a span staged here is physically relocatable
// OFF-BOX (the remote pool holds it), not merely onto the local disk.
//
// Tiering. With no remote pool named, the router has one durable tier: blobfs
// rooted at <dir>/content. When FAK_BLOB_HTTP_URL names a remote pool (the SAME
// opt-in env internal/blobhttp itself uses), the pool becomes the PRIMARY put
// tier and blobfs its local mirror: a demote's durable write is confirmed by the
// REMOTE store or Put fails — a typed FAULT the capacity executor retains the
// live span on — never a silent local-only OK masquerading as off-box residency.
// Resolve still tries blobfs first (mirror hit: no network round-trip) and falls
// back to the pool, so a restore survives losing EITHER copy.
//
// Manifest durability. A RestoreSpan arrives with the span digest and no payload,
// so the span→content map must itself survive a crash: every Put rewrites
// <dir>/manifest.json via the same temp+fsync+rename commit diskStore uses, and
// open reloads it. Content commits BEFORE the manifest entry, so a crash between
// the two leaves an orphan blob (a clean MISS at restore), never a manifest entry
// pointing at bytes that were never stored. Payloads <= InlineMax ride inline in
// the manifest entry itself (the router stores nothing for them), which the
// durable manifest write makes restart-safe.

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/blobfs"
	"github.com/anthony-chaudhary/fak/internal/blobhttp"
	"github.com/anthony-chaudhary/fak/internal/storedrv"
)

const (
	// manifestName is the durable span→content map inside the store directory.
	manifestName = "manifest.json"
	// contentDirName roots the blobfs local content tier inside the store directory.
	contentDirName = "content"
	// manifestVersion is the manifest format version; an unknown version is refused
	// at open (fail-closed to the default backend), never mis-parsed.
	manifestVersion = 1

	// manifestKindInline / manifestKindBlob are the two content-ref kinds a
	// manifest entry records: payload bytes carried in the entry itself
	// (<= InlineMax; the router stores nothing) vs a content digest resolvable
	// from the router's tiers.
	manifestKindInline = "inline"
	manifestKindBlob   = "blob"
)

// manifestEntry is one span's content reference: the durable projection of the
// abi.Ref the router returned at stage time.
type manifestEntry struct {
	Kind   string `json:"kind"`             // "inline" | "blob"
	Digest string `json:"digest"`           // sha256 content digest of the payload
	Len    int64  `json:"len"`              // payload length in bytes
	Inline []byte `json:"inline,omitempty"` // the payload itself for kind "inline"
}

// ref reconstructs the abi.Ref a stored entry stands for. An unknown kind is a
// typed fault (a manifest written by a future build), refused rather than guessed.
func (e manifestEntry) ref() (abi.Ref, error) {
	switch e.Kind {
	case manifestKindInline:
		return abi.Ref{Kind: abi.RefInline, Digest: e.Digest, Inline: e.Inline, Len: e.Len}, nil
	case manifestKindBlob:
		return abi.Ref{Kind: abi.RefBlob, Digest: e.Digest, Len: e.Len}, nil
	default:
		return abi.Ref{}, fmt.Errorf("unknown manifest content kind %q (this build reads %q|%q)", e.Kind, manifestKindInline, manifestKindBlob)
	}
}

// manifestFile is the on-disk JSON envelope: a version discriminant plus the
// span-digest → content-ref map.
type manifestFile struct {
	Version int                      `json:"version"`
	Spans   map[string]manifestEntry `json:"spans"`
}

// routerStore implements Store over a storedrv.Router plus a durable manifest.
type routerStore struct {
	router *storedrv.Router
	path   string // manifest file path

	mu    sync.Mutex
	spans map[string]manifestEntry // span residency digest -> content ref
}

// fsDriver adapts *blobfs.Store to the storedrv.Driver SPI (it lacks only the
// stable ID); embedding forwards Put/Resolve/Pin/Unpin/PageOut/PageIn.
type fsDriver struct{ *blobfs.Store }

func (fsDriver) ID() string { return "blobfs" }

// newRouterStore opens (creating if absent) the router-backed L3 store rooted at
// dir. remote is the blobhttp pool base URL ("" = local stand-in only) and token
// its bearer secret — init passes FAK_BLOB_HTTP_URL / FAK_BLOB_HTTP_TOKEN; a test
// passes an httptest server URL directly.
func newRouterStore(dir, remote, token string) (*routerStore, error) {
	return newRouterStoreWithLocalMirror(dir, remote, token, true)
}

// NewRemoteStore opens the l3kv span-keyed manifest over a blobhttp primary
// without a local content mirror. The small manifest remains crash-safe locally;
// every snapshot payload byte is PUT to and GET from the remote HTTP store.
func NewRemoteStore(dir, remote, token string) (Store, error) {
	if remote == "" {
		return nil, fmt.Errorf("l3kv: remote snapshot store requires FAK_BLOB_HTTP_URL")
	}
	return newRouterStoreWithLocalMirror(dir, remote, token, false)
}

func newRouterStoreWithLocalMirror(dir, remote, token string, localMirror bool) (*routerStore, error) {
	if dir == "" {
		return nil, fmt.Errorf("l3kv: empty store directory")
	}
	fs, err := blobfs.New(filepath.Join(dir, contentDirName))
	if err != nil {
		return nil, fmt.Errorf("l3kv: open local content tier: %w", err)
	}
	tiers := []storedrv.Tier{{Driver: fsDriver{fs}, Durable: true}}
	mirror := false
	if remote != "" {
		// The remote pool is the PRIMARY put tier: blobfs stops accepting puts
		// (Accept false) so tierFor falls through to the pool, and mirror writes
		// the local stand-in copy through best-effort. A put the pool refuses
		// FAILS (typed FAULT upstream; the live span is retained) — off-box
		// residency is confirmed, never assumed. Resolve order is unchanged
		// (blobfs first): a local mirror hit needs no network round-trip.
		tiers[0].Accept = func(int) bool { return false }
		tiers = append(tiers, storedrv.Tier{Driver: blobhttp.New(remote, blobhttp.WithBearer(token)), Durable: true})
		mirror = localMirror
	}
	router, err := storedrv.New(tiers, mirror)
	if err != nil {
		return nil, fmt.Errorf("l3kv: build content router: %w", err)
	}
	s := &routerStore{
		router: router,
		path:   filepath.Join(dir, manifestName),
		spans:  map[string]manifestEntry{},
	}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

// load reloads the durable manifest at open. A missing file is a fresh store;
// unparseable bytes or an unknown version are a refused open (fail-closed — init
// leaves the in-process default backend live), never a silently empty tier.
func (s *routerStore) load() error {
	b, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // fresh store: nothing staged yet
		}
		return fmt.Errorf("l3kv: read manifest %s: %w", s.path, err)
	}
	var mf manifestFile
	if err := json.Unmarshal(b, &mf); err != nil {
		return fmt.Errorf("l3kv: parse manifest %s: %w", s.path, err)
	}
	if mf.Version != manifestVersion {
		return fmt.Errorf("l3kv: unsupported manifest version %d in %s (this build reads v%d)", mf.Version, s.path, manifestVersion)
	}
	if mf.Spans != nil {
		s.spans = mf.Spans
	}
	return nil
}

// persistLocked rewrites the manifest via the shared temp+fsync+rename commit.
// Caller holds s.mu.
func (s *routerStore) persistLocked() error {
	b, err := json.Marshal(manifestFile{Version: manifestVersion, Spans: s.spans})
	if err != nil {
		return fmt.Errorf("l3kv: encode manifest: %w", err)
	}
	return atomicWrite(s.path, b)
}

// Put routes payload through the storedrv router (content-addressed, durable) and
// then commits the span→content manifest entry. Content lands BEFORE the manifest
// records it, so a crash between the two is a clean MISS at restore, never a
// dangling entry; a failed manifest commit rolls the in-memory entry back so the
// map never claims more than the disk holds.
func (s *routerStore) Put(ctx context.Context, key string, payload []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !validKey(key) {
		return fmt.Errorf("l3kv: invalid span key %q", key)
	}
	ref, err := s.router.Put(ctx, payload)
	if err != nil {
		return fmt.Errorf("l3kv: stage span content: %w", err)
	}
	entry := manifestEntry{Kind: manifestKindBlob, Digest: ref.Digest, Len: ref.Len}
	if ref.Kind == abi.RefInline {
		entry.Kind = manifestKindInline
		entry.Inline = append([]byte(nil), ref.Inline...)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	prev, had := s.spans[key]
	s.spans[key] = entry
	if err := s.persistLocked(); err != nil {
		if had {
			s.spans[key] = prev
		} else {
			delete(s.spans, key)
		}
		return err
	}
	return nil
}

// Get looks the span up in the manifest and resolves its content through the
// router (blobfs first, then the remote pool). No manifest entry is a clean MISS;
// a resolve failure or bytes that no longer hash to the recorded content digest
// are a FAULT — refused, never returned. The digest re-verify is uniform over
// every source (inline bytes included), so a tampered manifest or tier can never
// hand back a wrong hit.
func (s *routerStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if !validKey(key) {
		return nil, false, fmt.Errorf("l3kv: invalid span key %q", key)
	}
	s.mu.Lock()
	entry, ok := s.spans[key]
	s.mu.Unlock()
	if !ok {
		return nil, false, nil // clean MISS: never staged (or the manifest predates it)
	}
	ref, err := entry.ref()
	if err != nil {
		return nil, false, fmt.Errorf("l3kv: manifest entry for %s: %w", key, err)
	}
	payload, err := s.router.Resolve(ctx, ref)
	if err != nil {
		return nil, false, fmt.Errorf("l3kv: resolve span content for %s: %w", key, err)
	}
	if got := blob.Digest(payload); got != entry.Digest {
		return nil, false, fmt.Errorf("l3kv: integrity check failed for %s (content hashes to %s, manifest records %s)", key, got, entry.Digest)
	}
	return payload, true, nil
}
