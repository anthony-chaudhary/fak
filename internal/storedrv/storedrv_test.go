package storedrv

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/blobfs"
	"github.com/anthony-chaudhary/fak/internal/blobhttp"
)

func payload(n int, fill byte) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = fill
	}
	return b
}

const testThreshold = 4096

func memDiskRouter(t *testing.T, mirror bool) (*Router, *blob.Store, *blobfs.Store) {
	t.Helper()
	mem := blob.New()
	disk, err := blobfs.New(t.TempDir())
	if err != nil {
		t.Fatalf("blobfs.New: %v", err)
	}
	th := testThreshold
	tiers := []Tier{
		{Driver: memDriver{mem}, Accept: func(n int) bool { return n < th }, Durable: false},
		{Driver: diskDriver{disk}, Durable: true},
	}
	r, err := New(tiers, mirror)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return r, mem, disk
}

// TestRoutesBySize proves a small payload lands in the hot (mem) tier and a large
// one in the durable (disk) tier — the core "data lives in the place that fits it".
func TestRoutesBySize(t *testing.T) {
	ctx := context.Background()
	r, mem, disk := memDiskRouter(t, false)

	small := payload(1000, 's') // > InlineMax, < threshold -> mem
	if _, err := r.Put(ctx, small); err != nil {
		t.Fatalf("Put small: %v", err)
	}
	if mem.Len() != 1 {
		t.Fatalf("small payload not in mem tier: mem.Len=%d", mem.Len())
	}
	if c, _, _ := disk.Resident(); c != 0 {
		t.Fatalf("small payload leaked to disk tier: disk=%d", c)
	}

	large := payload(5000, 'l') // >= threshold -> disk
	rl, err := r.Put(ctx, large)
	if err != nil {
		t.Fatalf("Put large: %v", err)
	}
	if c, _, _ := disk.Resident(); c != 1 {
		t.Fatalf("large payload not in disk tier: disk=%d", c)
	}
	if mem.Len() != 1 {
		t.Fatalf("large payload leaked to mem tier: mem.Len=%d", mem.Len())
	}
	// Both resolve through the router regardless of which tier holds them.
	got, err := r.Resolve(ctx, rl)
	if err != nil || !bytes.Equal(got, large) {
		t.Fatalf("resolve large through router: err=%v equal=%v", err, bytes.Equal(got, large))
	}
}

// TestResolveAcrossTiers proves Resolve finds a digest in a non-hot tier (the
// content address is global across tiers).
func TestResolveAcrossTiers(t *testing.T) {
	ctx := context.Background()
	r, _, disk := memDiskRouter(t, false)
	large := payload(6000, 'x')
	ref, err := r.Put(ctx, large)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if c, _, _ := disk.Resident(); c != 1 {
		t.Fatalf("expected payload in disk tier")
	}
	got, err := r.Resolve(ctx, ref)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !bytes.Equal(got, large) {
		t.Fatalf("resolved bytes differ")
	}
}

// TestMirrorWriteThrough proves mirror mode write-throughs a hot-routed payload to
// the durable tier, so it is both hot-cached AND persistent.
func TestMirrorWriteThrough(t *testing.T) {
	ctx := context.Background()
	r, mem, disk := memDiskRouter(t, true) // mirror on

	small := payload(1000, 'm') // routes to mem (primary)
	ref, err := r.Put(ctx, small)
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if mem.Len() != 1 {
		t.Fatalf("not in hot tier")
	}
	if c, _, _ := disk.Resident(); c != 1 {
		t.Fatalf("mirror did not write through to durable tier: disk=%d", c)
	}
	// Resolvable even if the hot tier were cleared — prove via the disk tier directly.
	got, err := disk.Resolve(ctx, ref)
	if err != nil || !bytes.Equal(got, small) {
		t.Fatalf("durable mirror copy not resolvable: err=%v", err)
	}
}

// TestPinFanOutProtectsDurableTier proves router.Pin fans to every pin-aware tier,
// so a pinned digest survives the durable tier's byte-budget GC.
func TestPinFanOutProtectsDurableTier(t *testing.T) {
	ctx := context.Background()
	mem := blob.New()
	disk, err := blobfs.NewWithBudget(t.TempDir(), 3000) // holds ~2 of the 1KiB blobs
	if err != nil {
		t.Fatalf("blobfs.NewWithBudget: %v", err)
	}
	tiers := []Tier{
		{Driver: memDriver{mem}, Accept: func(n int) bool { return false }, Durable: false}, // never accept -> everything to disk
		{Driver: diskDriver{disk}, Durable: true},
	}
	r, err := New(tiers, false)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	first, err := r.Put(ctx, payload(1024, 0))
	if err != nil {
		t.Fatalf("Put first: %v", err)
	}
	r.Pin(first.Digest) // fans to disk's CASPinner
	for i := 1; i < 10; i++ {
		if _, err := r.Put(ctx, payload(1024, byte(i))); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	if _, _, ev := disk.Resident(); ev == 0 {
		t.Fatalf("expected GC eviction under the tight budget")
	}
	if _, err := r.Resolve(ctx, first); err != nil {
		t.Fatalf("pinned digest was evicted despite router.Pin: %v", err)
	}
}

// TestDeleteFanOut proves router.Delete erases from every tier that supports
// erasure — the aggregate provable-deletion primitive.
func TestDeleteFanOut(t *testing.T) {
	ctx := context.Background()
	srv := httptest.NewServer(newObjectServer())
	defer srv.Close()
	mem := blob.New()
	tiers := []Tier{
		{Driver: memDriver{mem}, Accept: func(n int) bool { return false }, Durable: false},
		{Driver: httpDriver{blobhttp.New(srv.URL)}, Durable: true},
	}
	r, err := New(tiers, false)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ref, err := r.Put(ctx, payload(2000, 'd'))
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := r.Resolve(ctx, ref); err != nil {
		t.Fatalf("Resolve before delete: %v", err)
	}
	if err := r.Delete(ctx, ref.Digest); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := r.Resolve(ctx, ref); err == nil {
		t.Fatal("expected resolve to fail after Delete fan-out")
	}
}

// TestPageOutLandsDurable proves router page-out persists even a SMALL body to the
// durable tier and returns a bytes-absent handle that pages back in.
func TestPageOutLandsDurable(t *testing.T) {
	ctx := context.Background()
	r, _, disk := memDiskRouter(t, false)

	body := payload(100, 'p') // small — must still page out durably (a quarantined snippet)
	handle, err := r.PageOut(ctx, abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))})
	if err != nil {
		t.Fatalf("PageOut: %v", err)
	}
	if handle.Kind != abi.RefBlob || len(handle.Inline) != 0 {
		t.Fatalf("page-out handle must be a bytes-absent RefBlob, got %+v", handle)
	}
	if c, _, _ := disk.Resident(); c != 1 {
		t.Fatalf("page-out did not land in the durable tier: disk=%d", c)
	}
	back, err := r.PageIn(ctx, handle)
	if err != nil || !bytes.Equal(back.Inline, body) {
		t.Fatalf("PageIn: err=%v", err)
	}
}

// TestPutHintedRoutesSealedToDurable proves a quarantined-taint hint sends a
// small payload to the durable tier (sealed bytes never sit only in volatile RAM),
// overriding the size policy that would keep it hot.
func TestPutHintedRoutesSealedToDurable(t *testing.T) {
	ctx := context.Background()
	r, mem, disk := memDiskRouter(t, false)

	small := payload(1000, 'q') // by size -> mem
	if _, err := r.PutHinted(ctx, small, Hint{Taint: abi.TaintQuarantined}); err != nil {
		t.Fatalf("PutHinted: %v", err)
	}
	if c, _, _ := disk.Resident(); c != 1 {
		t.Fatalf("quarantined hint did not route to durable tier: disk=%d", c)
	}
	if mem.Len() != 0 {
		t.Fatalf("quarantined payload leaked to volatile hot tier: mem.Len=%d", mem.Len())
	}
}

// TestInlineSmallPayload proves a <=InlineMax payload rides inline and touches no
// tier.
func TestInlineSmallPayload(t *testing.T) {
	ctx := context.Background()
	r, mem, disk := memDiskRouter(t, false)
	r.Put(ctx, payload(InlineMax, 'i'))
	if mem.Len() != 0 {
		t.Fatalf("inline payload touched mem")
	}
	if c, _, _ := disk.Resident(); c != 0 {
		t.Fatalf("inline payload touched disk")
	}
}

// TestSelfCheckRoundTrips proves the liveness probe passes on a live router.
func TestSelfCheckRoundTrips(t *testing.T) {
	ctx := context.Background()
	r, _, _ := memDiskRouter(t, false)
	if err := r.SelfCheck(ctx); err != nil {
		t.Fatalf("SelfCheck: %v", err)
	}
}

// TestBuildTiersSpec exercises the FAK_STORE spec parser.
func TestBuildTiersSpec(t *testing.T) {
	// single mem tier
	tiers, mirror, err := buildTiers("mem")
	if err != nil || len(tiers) != 1 || mirror {
		t.Fatalf("mem spec: tiers=%d mirror=%v err=%v", len(tiers), mirror, err)
	}
	if tiers[0].Durable {
		t.Fatalf("mem tier should not be durable")
	}

	// mem + disk
	tiers, _, err = buildTiers("mem+disk:" + t.TempDir())
	if err != nil {
		t.Fatalf("mem+disk spec: %v", err)
	}
	if len(tiers) != 2 || tiers[0].Durable || !tiers[1].Durable {
		t.Fatalf("mem+disk: tiers=%d durable=[%v %v]", len(tiers), tiers[0].Durable, tiers[1].Durable)
	}
	if tiers[0].Accept == nil {
		t.Fatalf("hot tier with a durable tier behind it should have a size cap")
	}

	// unknown scheme errors
	if _, _, err := buildTiers("bogusdriver:x"); err == nil {
		t.Fatal("expected an error for an unknown tier scheme")
	}
}

// TestRegisterFactory proves an external driver scheme resolves through the factory
// registry — the extension seam for build-tagged backends (DuckDB, embedded KV).
func TestRegisterFactory(t *testing.T) {
	RegisterFactory("faketestdrv", func(arg string) (Driver, error) {
		return memDriver{blob.New()}, nil
	})
	d, durable, err := buildDriver("faketestdrv:whatever")
	if err != nil {
		t.Fatalf("buildDriver via factory: %v", err)
	}
	if d == nil || !durable {
		t.Fatalf("factory tier should be a non-nil durable driver")
	}
}

func TestRouterInterfaces(t *testing.T) {
	var _ abi.Resolver = (*Router)(nil)
	var _ abi.RegionBackend = (*Router)(nil)
	var _ abi.PageOutBackend = (*Router)(nil)
	var _ abi.CASPinner = (*Router)(nil)
}

// objectServer is a minimal in-memory path-style object endpoint for the http tier.
type objectServer struct {
	mu      sync.Mutex
	objects map[string][]byte
}

func newObjectServer() *objectServer { return &objectServer{objects: map[string][]byte{}} }

func (o *objectServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Path[1:]
	o.mu.Lock()
	defer o.mu.Unlock()
	switch r.Method {
	case http.MethodHead:
		if _, ok := o.objects[key]; ok {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	case http.MethodPut:
		b, _ := io.ReadAll(r.Body)
		o.objects[key] = b
		w.WriteHeader(http.StatusCreated)
	case http.MethodGet:
		if b, ok := o.objects[key]; ok {
			w.Write(b)
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	case http.MethodDelete:
		delete(o.objects, key)
		w.WriteHeader(http.StatusNoContent)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}
