// Package l3kv is a durable disk-backed L3 KV residency backend: StageSpan /
// RestoreSpan persist a demoted span to a crash-safe on-disk store by digest, so a
// demote to L3 survives off-box instead of being dropped (#1472).
//
// Why this leaf exists. The in-process default abi.KVBackend
// (internal/model/kvbackend.go) has a StageSpan that returns OK while moving ZERO
// bytes — it drops the live span and reports success, so a later RestoreSpan is a
// guaranteed MISS and the "demote to L3" the capacity executor performs is really a
// silent eviction. The off-box residency seam (abi.KVBackend.StageSpan /
// RestoreSpan, added by #638) shipped but had no producer. This leaf is that
// producer: a backend that, on StageSpan, serializes the span's fak-owned bytes and
// commits them to a durable tier keyed by the span digest, and on RestoreSpan pages
// them back — a real demote that survives.
//
// Posture. It wraps the in-process default (delegating the local ops
// Len/Prefill/Evict/ModelID/CanEvict) and overrides only the residency pair, and it
// is registered LAST-WINS behind the opt-in env FAK_L3_KVBACKEND — default builds
// keep the in-process default live, byte-identical to today. It stays on the
// control/admission path: the only integrity work is a fail-closed digest re-verify
// at restore (in the Store), never an inline scan of a hot read.
//
// Honest fence. The bytes StageSpan serializes come from the wrapped backend via
// the SpanStager capability (reachable by type assertion, exactly like the model
// backend's CanEvict). The in-process model backend gains that capability in a
// paired change; until a wrapped backend exposes it, StageSpan returns a typed
// FAULT — honest, and fail-safe (the capacity executor retains the live span on a
// FAULT), never the in-process default's silent OK-that-drops. RestoreSpan completes
// only after the wrapped backend's additive SpanRestorer validates and installs the
// recovered image; absent that capability it reports a typed FAULT.
package l3kv

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// EnvSpec is the opt-in environment variable. Unset/empty leaves the in-process
// default KV backend live (byte-identical to today); set to a directory path to
// enable the durable disk-backed L3 residency tier rooted there.
const EnvSpec = "FAK_L3_KVBACKEND"

const (
	EnvRemoteURL   = "FAK_BLOB_HTTP_URL"
	EnvRemoteToken = "FAK_BLOB_HTTP_TOKEN"
)

var (
	configuredStore Store
	configuredErr   error
)

// ConfiguredRemoteStore returns the process-wide l3kv/blobhttp store opened at
// boot when both L3 and remote HTTP configuration are present. It is the single
// production instance shared by the ABI backend and native prefix snapshots, so
// two independent manifest maps can never race over the same directory.
func ConfiguredRemoteStore() (Store, bool, error) {
	if strings.TrimSpace(os.Getenv(EnvSpec)) == "" || strings.TrimSpace(os.Getenv(EnvRemoteURL)) == "" {
		return nil, false, nil
	}
	if configuredErr != nil {
		return nil, true, configuredErr
	}
	if configuredStore == nil {
		return nil, true, fmt.Errorf("l3kv: remote store configured but unavailable")
	}
	return configuredStore, true, nil
}

// SpanStager is the additive capability a KV backend exposes so its fak-owned span
// rows can be serialized off-box. StageSpanBytes returns the opaque serialization of
// the [from, from+n) span (the pre-RoPE Kraw rows + values + positions the owner
// alone can read) or a typed error for a cache variant it cannot serialize (e.g. a
// recurrent/hybrid cache with no per-token journal). The in-process model backend
// implements it (reachable by type assertion, like its CanEvict); a backend that
// does not is honestly unable to stage.
type SpanStager interface {
	StageSpanBytes(from, n int) ([]byte, error)
}

// SpanRestorer is the additive inverse of SpanStager. It parses and installs an
// opaque span image into the wrapped live cache, returning the number of
// positions actually installed. A durable read is not a restore success until
// this capability commits the bytes.
type SpanRestorer interface {
	RestoreSpanBytes(payload []byte) (positions int, err error)
}

// SpanFileResolver is an optional capability interface exposed by an inner KV backend
// or Store allowing cache span eviction to map token/position ranges to backing file extents.
type SpanFileResolver interface {
	SpanFileRange(from, n int) (file *os.File, offset, length int64, ok bool)
}

// backend wraps an inner abi.KVBackend with the durable L3 residency tier. The four
// local ops delegate to inner; the residency pair moves bytes through store.
type backend struct {
	inner       abi.KVBackend
	stager      SpanStager   // nil when inner exposes no span byte-source
	restorer    SpanRestorer // nil when inner cannot install recovered bytes
	store       Store
	deallocator *AsyncDeallocator
}

// New wraps inner with the durable L3 residency tier at store. If inner also
// implements SpanStager, StageSpan serializes and persists real span bytes;
// otherwise StageSpan is a typed FAULT (honest) and the capacity executor retains
// the live span. Exported so a host or a test can compose it over any backend + Store.
func New(inner abi.KVBackend, store Store) abi.KVBackend {
	b := &backend{inner: inner, store: store}
	if st, ok := inner.(SpanStager); ok {
		b.stager = st
	}
	if restore, ok := inner.(SpanRestorer); ok {
		b.restorer = restore
	}
	return b
}

// WithDeallocator configures b to issue asynchronous deallocate/TRIM commands on span eviction.
func WithDeallocator(b abi.KVBackend, d *AsyncDeallocator) abi.KVBackend {
	if bk, ok := b.(*backend); ok {
		bk.deallocator = d
	}
	return b
}

func (b *backend) Len() int                    { return b.inner.Len() }
func (b *backend) Prefill(ids []int) []float32 { return b.inner.Prefill(ids) }
func (b *backend) Evict(from, n int) int {
	removed := b.inner.Evict(from, n)
	if b.deallocator != nil && removed > 0 {
		if res, ok := b.inner.(SpanFileResolver); ok {
			if f, off, len, ok := res.SpanFileRange(from, n); ok {
				_ = b.deallocator.Submit(f, off, len)
			}
		} else if res, ok := b.store.(SpanFileResolver); ok {
			if f, off, len, ok := res.SpanFileRange(from, n); ok {
				_ = b.deallocator.Submit(f, off, len)
			}
		}
	}
	return removed
}
func (b *backend) ModelID() string { return b.inner.ModelID() }

// CanEvict forwards the wrapped backend's span-eviction verdict so the KV-MMU still
// sees a recurrent cache's typed limitation THROUGH the wrapper. Without this
// passthrough, wrapping would make a recurrent backend look "evictable by absence"
// and a later Evict would panic inside the model. It is the same additive,
// type-asserted capability the in-process default exposes.
func (b *backend) CanEvict() error {
	if ce, ok := b.inner.(interface{ CanEvict() error }); ok {
		return ce.CanEvict()
	}
	return nil
}

// StageSpan serializes the [from, from+n) span via the wrapped backend's byte-source
// and commits it to the durable tier keyed by digest. It returns OK with the real
// bytes moved on a confirmed durable write; a typed FAULT (never a silent OK-drop)
// when there is no byte-source, the span cannot be serialized, or the durable write
// fails — on a FAULT the capacity executor retains the live span (fail-safe).
func (b *backend) StageSpan(ctx context.Context, digest string, from, n int) (abi.KVResidency, error) {
	if b.stager == nil {
		return fault(digest, "l3kv: wrapped backend exposes no span byte-source (SpanStager); span retained"), nil
	}
	payload, err := b.stager.StageSpanBytes(from, n)
	if err != nil {
		return fault(digest, "l3kv: serialize span: "+err.Error()), nil
	}
	if len(payload) == 0 {
		return fault(digest, "l3kv: empty span serialization"), nil
	}
	if err := b.store.Put(ctx, digest, payload); err != nil {
		return fault(digest, "l3kv: stage to durable tier: "+err.Error()), nil
	}
	return abi.KVResidency{
		Outcome:    abi.KVResidencyOK,
		Digest:     digest,
		Positions:  n,
		BytesMoved: int64(len(payload)),
	}, nil
}

// RestoreSpan pages a previously staged span back from the durable tier by digest.
// OK on a hit (bytes recovered, integrity-verified by the store), a typed MISS when
// the tier no longer holds the span (the caller recomputes but is TOLD), a typed
// FAULT on an I/O or integrity failure — never a silent recompute or a wrong hit.
func (b *backend) RestoreSpan(ctx context.Context, digest string) (abi.KVResidency, error) {
	payload, found, err := b.store.Get(ctx, digest)
	if err != nil {
		return fault(digest, "l3kv: restore from durable tier: "+err.Error()), nil
	}
	if !found {
		return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest, Reason: "l3kv: span not resident in L3 tier"}, nil
	}
	if b.restorer == nil {
		return fault(digest, "l3kv: wrapped backend exposes no span installer (SpanRestorer)"), nil
	}
	positions, err := b.restorer.RestoreSpanBytes(payload)
	if err != nil {
		return fault(digest, "l3kv: install restored span: "+err.Error()), nil
	}
	if positions <= 0 {
		return fault(digest, "l3kv: span installer committed no positions"), nil
	}
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest, Positions: positions, BytesMoved: int64(len(payload))}, nil
}

func fault(digest, reason string) abi.KVResidency {
	return abi.KVResidency{Outcome: abi.KVResidencyFault, Digest: digest, Reason: reason}
}

// Factory returns an abi.KVBackendFactory that wraps the in-process default backend
// for a session with the durable L3 tier at store. It fails closed: a session value
// the in-process backend does not own yields ok=false, so the caller never enforces
// against a mis-constructed backend.
func Factory(store Store) abi.KVBackendFactory {
	return func(session any) (abi.KVBackend, bool) {
		inner, ok := model.KVBackendFor(session)
		if !ok {
			return nil, false
		}
		return New(inner, store), true
	}
}

// init registers the durable L3 KV backend LAST-WINS when FAK_L3_KVBACKEND names a
// store directory. It runs after internal/modelengine's init (which registers the
// in-process default) because internal/registrations blank-imports modelengine
// before this leaf, so this registration wins. The store routes span content
// through the storedrv router (blobfs local stand-in, plus the FAK_BLOB_HTTP_URL
// remote pool when set) behind a durable span→content manifest. Unset env, or a
// store that cannot be opened (unreadable manifest included), leaves the
// in-process default live (fail-closed to the default — a broken tier is never
// registered).
func init() {
	dir := os.Getenv(EnvSpec)
	if dir == "" {
		return
	}
	// The L3 tier stages spans to an OFF-BOX content pool when FAK_BLOB_HTTP_URL is
	// set (the SAME opt-in env internal/blobhttp's own registration uses; bearer token
	// via FAK_BLOB_HTTP_TOKEN, read identically). Set, the pool becomes the router's
	// primary put tier (a demote is confirmed off-box or FAULTs) with blobfs mirroring
	// locally; unset, the durable on-box stand-in alone serves.
	remote := os.Getenv(EnvRemoteURL)
	var store Store
	var err error
	if remote != "" {
		// A configured remote L3 is a true remote tier: no local payload mirror can
		// satisfy a later read while being counted as off-host recovery.
		store, err = NewRemoteStore(dir, remote, os.Getenv(EnvRemoteToken))
	} else {
		store, err = newRouterStore(dir, "", "")
	}
	if err != nil {
		configuredErr = err
		return
	}
	configuredStore = store
	abi.RegisterKVBackend(Factory(store))
}
