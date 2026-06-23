package storedrv

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/blobfs"
	"github.com/anthony-chaudhary/fak/internal/blobhttp"
)

// DefaultThreshold is the byte size at or above which a payload prefers a durable
// tier (smaller payloads stay hot in RAM). It mirrors ctxmmu.OversizeBytes: the
// same "small stays in context / large spills" boundary the MMU already uses.
const DefaultThreshold = 4096

// Factory builds an external Driver from its spec argument — the extension seam an
// optional/build-tagged backend (DuckDB columnar, an embedded KV, a vendor SDK)
// registers so it can appear in FAK_STORE without storedrv importing it (which
// would pull a module onto the zero-dependency default build). The built-in
// schemes "mem", "disk", and "http"/"https" are handled directly and need no
// Factory.
type Factory func(arg string) (Driver, error)

// RegisterFactory registers an external-driver Factory under a scheme name (e.g.
// "duckdb", "redis", "s3"). Call it from the driver package's init(); blank-import
// that package (in a build-tagged defconfig) to make the scheme available to
// FAK_STORE. Re-registering a scheme overwrites it (last wins).
func RegisterFactory(scheme string, f Factory) {
	factoryMu.Lock()
	defer factoryMu.Unlock()
	factories[scheme] = f
}

// active is the registered router (nil unless FAK_STORE opted in).
var active *Router

// Active returns the registered storage router, or nil if FAK_STORE was unset at
// boot (the package is inert and the in-memory blob store is the live backend).
func Active() *Router { return active }

// buildTiers parses a FAK_STORE spec into an ordered tier list. The spec is a
// "+"-separated list of tier tokens, hottest first, each "scheme[:arg]":
//
//	mem                      the in-memory blob store (hot, volatile) — blob.Default
//	disk[:/path]             a durable on-disk content-addressed store (blobfs)
//	http://host/bucket       a remote HTTP object store (blobhttp); https:// too
//	<scheme>[:arg]           an external driver registered via RegisterFactory
//
// e.g. "mem+disk:/var/lib/fak/blobs" (hot RAM in front of durable disk) or
// "mem+https://s3.example.com/fak-blobs". A hot (non-durable) tier with a durable
// tier behind it accepts only payloads smaller than the threshold
// (FAK_STORE_THRESHOLD, default DefaultThreshold); larger payloads route to the
// durable tier. FAK_STORE_MIRROR=1 makes every Put write through to durable tiers
// (full persistence + hot cache).
func buildTiers(spec string) (tiers []Tier, mirror bool, err error) {
	threshold := DefaultThreshold
	if v := os.Getenv("FAK_STORE_THRESHOLD"); v != "" {
		if n, e := strconv.Atoi(v); e == nil && n >= 0 {
			threshold = n
		}
	}
	mirror = boolEnv("FAK_STORE_MIRROR")

	tokens := strings.Split(spec, "+")
	drivers := make([]Driver, 0, len(tokens))
	durable := make([]bool, 0, len(tokens))
	for _, tok := range tokens {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		d, dur, e := buildDriver(tok)
		if e != nil {
			return nil, false, e
		}
		drivers = append(drivers, d)
		durable = append(durable, dur)
	}
	if len(drivers) == 0 {
		return nil, false, fmt.Errorf("storedrv: FAK_STORE=%q named no tiers", spec)
	}

	hasDurable := false
	for _, d := range durable {
		if d {
			hasDurable = true
		}
	}
	for i, d := range drivers {
		t := Tier{Driver: d, Durable: durable[i]}
		// A hot tier with a durable tier behind it caps at the threshold so large
		// payloads spill to disk/remote; everything else is a catch-all (accept all).
		if !durable[i] && hasDurable {
			th := threshold
			t.Accept = func(n int) bool { return n < th }
		}
		tiers = append(tiers, t)
	}
	return tiers, mirror, nil
}

// buildDriver constructs one tier's Driver from a "scheme[:arg]" token, returning
// whether it is a durable (persistent) tier.
func buildDriver(tok string) (Driver, bool, error) {
	// http/https: the whole token is the URL.
	if strings.HasPrefix(tok, "http://") || strings.HasPrefix(tok, "https://") {
		return httpDriver{blobhttp.New(tok, blobhttp.WithBearer(os.Getenv("FAK_BLOB_HTTP_TOKEN")))}, true, nil
	}
	scheme, arg := tok, ""
	if i := strings.Index(tok, ":"); i >= 0 {
		scheme, arg = tok[:i], tok[i+1:]
	}
	switch scheme {
	case "mem", "ram", "blob":
		return memDriver{blob.Default}, false, nil
	case "disk", "blobfs", "fs":
		dir := arg
		if dir == "" {
			dir = os.Getenv("FAK_BLOB_DIR")
		}
		if dir == "" {
			return nil, false, fmt.Errorf("storedrv: disk tier needs a path (disk:/path) or FAK_BLOB_DIR")
		}
		s, err := blobfs.New(dir)
		if err != nil {
			return nil, false, fmt.Errorf("storedrv: disk tier: %w", err)
		}
		return diskDriver{s}, true, nil
	default:
		factoryMu.RLock()
		f, ok := factories[scheme]
		factoryMu.RUnlock()
		if !ok {
			known := sortedFactorySchemes()
			return nil, false, fmt.Errorf("storedrv: unknown tier scheme %q (built-in: mem, disk, http(s); registered: %v)", scheme, known)
		}
		d, err := f(arg)
		if err != nil {
			return nil, false, fmt.Errorf("storedrv: %s tier: %w", scheme, err)
		}
		return d, true, nil
	}
}

func boolEnv(key string) bool {
	switch strings.ToLower(os.Getenv(key)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// diskDriver adapts *blobfs.Store to the Driver SPI (it already implements
// abi.Resolver + abi.CASPinner + abi.PageOutBackend; this only adds the ID).
type diskDriver struct{ s *blobfs.Store }

func (diskDriver) ID() string { return "blobfs" }
func (d diskDriver) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	return d.s.Put(ctx, b)
}
func (d diskDriver) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	return d.s.Resolve(ctx, r)
}
func (d diskDriver) Pin(digest string)   { d.s.Pin(digest) }
func (d diskDriver) Unpin(digest string) { d.s.Unpin(digest) }
func (d diskDriver) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	return d.s.PageOut(ctx, r)
}
func (d diskDriver) PageIn(ctx context.Context, h abi.Ref) (abi.Ref, error) {
	return d.s.PageIn(ctx, h)
}

// httpDriver adapts *blobhttp.Store to the Driver SPI (it already implements
// abi.Resolver + abi.PageOutBackend + Deleter; this only re-states the ID).
type httpDriver struct{ s *blobhttp.Store }

func (httpDriver) ID() string { return "blobhttp" }
func (h httpDriver) Put(ctx context.Context, b []byte) (abi.Ref, error) {
	return h.s.Put(ctx, b)
}
func (h httpDriver) Resolve(ctx context.Context, r abi.Ref) ([]byte, error) {
	return h.s.Resolve(ctx, r)
}
func (h httpDriver) PageOut(ctx context.Context, r abi.Ref) (abi.Ref, error) {
	return h.s.PageOut(ctx, r)
}
func (h httpDriver) PageIn(ctx context.Context, hd abi.Ref) (abi.Ref, error) {
	return h.s.PageIn(ctx, hd)
}
func (h httpDriver) Delete(ctx context.Context, digest string) error {
	return h.s.Delete(ctx, digest)
}

func init() {
	spec := os.Getenv("FAK_STORE")
	if spec == "" {
		return // OPT-IN: unset => package inert, blob stays the singleton RegionBackend
	}
	tiers, mirror, err := buildTiers(spec)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak: storage router disabled — %v\n", err)
		return
	}
	r, err := New(tiers, mirror)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak: storage router disabled — %v\n", err)
		return
	}
	active = r
	// The DELIBERATE, reviewed RegionBackend swap (architest regionBackendRole): the
	// router becomes the single live Ref backend, fanning across its tiers. blob's
	// init ran first (storedrv imports blob), so this override is order-deterministic.
	abi.RegisterRegionBackend(r)
	abi.RegisterPageOutBackend("store", r)
	fmt.Fprintf(os.Stderr, "fak: storage router -> %s (id=store)\n", r.Describe())
}
