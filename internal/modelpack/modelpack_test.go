package modelpack

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func signedManifest(t *testing.T, priv ed25519.PrivateKey, id, rev string, payload []byte) Manifest {
	t.Helper()
	sum := sha256.Sum256(payload)
	m := Manifest{Schema: Schema, PackID: id, Revision: rev, Chunks: []Chunk{{Digest: hex.EncodeToString(sum[:]), Size: int64(len(payload))}}, Fixtures: []Fixture{{Name: "smoke", Input: "2+2", Expected: "4"}}}
	if err := Sign(&m, priv); err != nil {
		t.Fatal(err)
	}
	return m
}
func source(payload []byte) Fetch {
	return func(_ string, off int64, dst io.Writer) error { _, e := dst.Write(payload[off:]); return e }
}

func TestLifecycleWitness(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	root := t.TempDir()
	mgr, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	payload1 := []byte("model-one-contents")
	v1 := signedManifest(t, priv, "assistant", "v1", payload1)
	calls := 0
	interrupted := func(_ string, off int64, dst io.Writer) error {
		calls++
		if off != 0 {
			t.Fatalf("first offset=%d", off)
		}
		dst.Write(payload1[:5])
		return errors.New("network reset")
	}
	if _, err = mgr.Install(v1, pub, 100, interrupted, nil); err == nil {
		t.Fatal("interrupted acquisition succeeded")
	}
	if got := mgr.Forecast(v1); got != int64(len(payload1)-5) {
		t.Fatalf("resume forecast=%d", got)
	}
	resumedOffset := int64(-1)
	resume := func(_ string, off int64, dst io.Writer) error {
		resumedOffset = off
		_, e := dst.Write(payload1[off:])
		return e
	}
	r, err := mgr.Install(v1, pub, 100, resume, func(_ string, f []Fixture) error {
		if len(f) != 1 {
			return errors.New("fixture missing")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if resumedOffset != 5 || r.State != "activated" || mgr.Active("assistant") != "v1" {
		t.Fatalf("resume/activation failed offset=%d receipt=%+v active=%q", resumedOffset, r, mgr.Active("assistant"))
	}

	badSig := v1
	badSig.Revision = "tampered"
	if _, err = mgr.Install(badSig, pub, 100, source(payload1), nil); !errors.Is(err, ErrSignature) {
		t.Fatalf("signature error=%v", err)
	}
	vCorrupt := signedManifest(t, priv, "assistant", "corrupt", []byte("expected"))
	if _, err = mgr.Install(vCorrupt, pub, 100, source([]byte("bad-data")), nil); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("corruption error=%v", err)
	}
	v2data := []byte("model-two-contents")
	v2 := signedManifest(t, priv, "assistant", "v2", v2data)
	if _, err = mgr.Install(v2, pub, 100, source(v2data), func(string, []Fixture) error { return errors.New("quality regression") }); err == nil {
		t.Fatal("failing canary activated")
	}
	if mgr.Active("assistant") != "v1" {
		t.Fatal("canary changed active revision")
	}
	if _, err = mgr.Install(v2, pub, 100, source(v2data), nil); err != nil {
		t.Fatal(err)
	}
	if mgr.Active("assistant") != "v2" {
		t.Fatal("v2 not active")
	}
	if _, err = mgr.Rollback("assistant"); err != nil {
		t.Fatal(err)
	}
	if mgr.Active("assistant") != "v1" {
		t.Fatal("rollback did not restore v1")
	}

	v0data := []byte("old-model")
	v0 := signedManifest(t, priv, "assistant", "v0", v0data)
	if _, err = mgr.Install(v0, pub, 100, source(v0data), nil); err != nil {
		t.Fatal(err)
	}
	// v0 is active and v1 is LKG. Rolling back leaves v0 protected as LKG and v1 active;
	// v2 is now the only safely evictable installed revision.
	if _, err = mgr.Rollback("assistant"); err != nil {
		t.Fatal(err)
	}
	freed, err := mgr.Evict(1)
	if err != nil || freed == 0 {
		t.Fatalf("eviction freed=%d err=%v", freed, err)
	}
	if _, err = os.Stat(filepath.Join(root, "packs", "assistant", "v1")); err != nil {
		t.Fatal("active evicted")
	}
	if _, err = os.Stat(filepath.Join(root, "packs", "assistant", "v0")); err != nil {
		t.Fatal("last-known-good evicted")
	}
	if _, err = os.Stat(filepath.Join(root, "packs", "assistant", "v2")); !os.IsNotExist(err) {
		t.Fatal("inactive v2 retained under pressure")
	}
	if _, err = mgr.Revoke("assistant", "v1"); err != nil {
		t.Fatal(err)
	}
	if mgr.Active("assistant") != "v0" {
		t.Fatalf("revocation did not restore LKG: %q", mgr.Active("assistant"))
	}
	if _, err = mgr.Install(v1, pub, 100, source(payload1), nil); !errors.Is(err, ErrRevoked) {
		t.Fatalf("revoked revision reinstall error=%v", err)
	}

	restarted, err := Open(root)
	if err != nil {
		t.Fatal(err)
	}
	if restarted.Active("assistant") != "v0" {
		t.Fatal("active state not durable")
	}
	events := restarted.Events()
	states := map[string]bool{}
	for i, e := range events {
		if e.Sequence != uint64(i+1) {
			t.Fatal("event order is not durable")
		}
		states[e.State] = true
	}
	for _, want := range []string{"interrupted", "reserved", "activated", "refused", "canary_failed", "rolled_back", "evicted", "revoked"} {
		if !states[want] {
			t.Fatalf("missing %s event in %#v", want, states)
		}
	}
}

func TestCapacityRefusalDoesNotFetch(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	data := []byte("large")
	m := signedManifest(t, priv, "p", "r", data)
	mgr, _ := Open(t.TempDir())
	called := false
	_, err := mgr.Install(m, pub, 1, func(string, int64, io.Writer) error { called = true; return nil }, nil)
	if !errors.Is(err, ErrCapacity) || called {
		t.Fatalf("err=%v fetch_called=%v", err, called)
	}
}

func BenchmarkModelPack(b *testing.B) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		b.Fatal(err)
	}
	payload := []byte("benchmark-model-payload-data-for-modelpack-performance")
	sum := sha256.Sum256(payload)
	manifest := Manifest{
		Schema:   Schema,
		PackID:   "bench-model",
		Revision: "v1",
		Chunks: []Chunk{
			{Digest: hex.EncodeToString(sum[:]), Size: int64(len(payload))},
		},
		Fixtures: []Fixture{
			{Name: "smoke", Input: "prompt", Expected: "output"},
		},
	}
	if err := Sign(&manifest, priv); err != nil {
		b.Fatal(err)
	}

	b.Run("SignAndVerify", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			m := manifest
			m.Signature = ""
			if err := Sign(&m, priv); err != nil {
				b.Fatal(err)
			}
			if err := Verify(m, pub); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Forecast", func(b *testing.B) {
		root := b.TempDir()
		mgr, err := Open(root)
		if err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = mgr.Forecast(manifest)
		}
	})

	b.Run("InstallAtomic", func(b *testing.B) {
		fetcher := source(payload)
		for i := 0; i < b.N; i++ {
			root := b.TempDir()
			mgr, err := Open(root)
			if err != nil {
				b.Fatal(err)
			}
			if _, err := mgr.Install(manifest, pub, 1024, fetcher, nil); err != nil {
				b.Fatal(err)
			}
		}
	})
}
