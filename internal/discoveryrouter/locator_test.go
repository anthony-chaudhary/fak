package discoveryrouter

import (
	"context"
	"crypto/ed25519"
	"errors"
	"testing"
	"time"
)

func signedRecord(t *testing.T, key ed25519.PrivateKey, generation uint64, endpoint string) LocatorRecord {
	t.Helper()
	r := LocatorRecord{LogicalID: "logical-7", Generation: generation, Epoch: 1, Endpoints: []string{endpoint}, ExpiresAt: time.Unix(2000, 0)}
	if err := SignLocator(&r, key); err != nil {
		t.Fatal(err)
	}
	return r
}
func requireCode(t *testing.T, err error, code ResolveCode) {
	t.Helper()
	var typed *ResolveError
	if !errors.As(err, &typed) || typed.Code != code {
		t.Fatalf("error=%v, want %s", err, code)
	}
}

func TestResolveSurvivesRelayAndPrimaryLocatorLossWithoutRollback(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(nil)
	primary := &MemoryLocator{BackendName: "primary"}
	independent := &MemoryLocator{BackendName: "independent"}
	ctx := context.Background()
	n := signedRecord(t, priv, 1, "https://relay-a.example/session")
	if err := primary.Put(ctx, n); err != nil {
		t.Fatal(err)
	}
	if err := independent.Put(ctx, n); err != nil {
		t.Fatal(err)
	}
	attached := &Resolver{Backends: []LocatorBackend{primary, independent}, TrustedKeys: map[string]ed25519.PublicKey{"logical-7": pub}, Now: func() time.Time { return time.Unix(1000, 0) }}
	if got, err := attached.Resolve(ctx, "logical-7"); err != nil || got.Generation != 1 {
		t.Fatalf("initial resolve=(%+v,%v)", got, err)
	}
	primary.Err = errors.New("failure domain cut")
	moved := signedRecord(t, priv, 2, "https://relay-b.example/session")
	if err := independent.Put(ctx, moved); err != nil {
		t.Fatal(err)
	}
	fresh := &Resolver{Backends: []LocatorBackend{primary, independent}, TrustedKeys: map[string]ed25519.PublicKey{"logical-7": pub}, Now: attached.Now}
	for name, resolver := range map[string]*Resolver{"fresh": fresh, "attached": attached} {
		got, err := resolver.Resolve(ctx, "logical-7")
		if err != nil || got.Generation != 2 {
			t.Fatalf("%s resolve=(%+v,%v)", name, got, err)
		}
	}
	independent.records["logical-7"] = []LocatorRecord{n}
	_, err := attached.Resolve(ctx, "logical-7")
	requireCode(t, err, TooOld)
}

func TestResolveRejectsConflictsAndUntrustedOrUnsafeRecords(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(nil)
	a := &MemoryLocator{BackendName: "a"}
	b := &MemoryLocator{BackendName: "b"}
	ctx := context.Background()
	left := signedRecord(t, priv, 3, "https://relay-a.example/session")
	right := signedRecord(t, priv, 3, "https://relay-b.example/session")
	a.records = map[string][]LocatorRecord{"logical-7": {left}}
	b.records = map[string][]LocatorRecord{"logical-7": {right}}
	r := &Resolver{Backends: []LocatorBackend{a, b}, TrustedKeys: map[string]ed25519.PublicKey{"logical-7": pub}, Now: func() time.Time { return time.Unix(1000, 0) }}
	_, err := r.Resolve(ctx, "logical-7")
	requireCode(t, err, SplitBrain)
	unsafe := LocatorRecord{LogicalID: "logical-7", Generation: 4, Epoch: 1, Endpoints: []string{"https://127.0.0.1/session"}, ExpiresAt: time.Unix(2000, 0)}
	if err := SignLocator(&unsafe, priv); err == nil {
		t.Fatal("private hostname accepted")
	}
}

func TestResolveReportsRevokedExpiredNotFoundAndUnreachable(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(nil)
	now := time.Unix(1000, 0)
	ctx := context.Background()
	backend := &MemoryLocator{BackendName: "one"}
	r := &Resolver{Backends: []LocatorBackend{backend}, TrustedKeys: map[string]ed25519.PublicKey{"logical-7": pub}, Now: func() time.Time { return now }}
	_, err := r.Resolve(ctx, "logical-7")
	requireCode(t, err, NotFound)
	backend.Err = errors.New("down")
	_, err = r.Resolve(ctx, "logical-7")
	requireCode(t, err, Unreachable)
	backend.Err = nil
	revoked := LocatorRecord{LogicalID: "logical-7", Generation: 1, Epoch: 1, ExpiresAt: time.Unix(2000, 0), Revoked: true}
	if err := SignLocator(&revoked, priv); err != nil {
		t.Fatal(err)
	}
	if err := backend.Put(ctx, revoked); err != nil {
		t.Fatal(err)
	}
	_, err = r.Resolve(ctx, "logical-7")
	requireCode(t, err, Revoked)
	expired := signedRecord(t, priv, 2, "https://relay.example/session")
	expired.ExpiresAt = time.Unix(999, 0)
	if err := SignLocator(&expired, priv); err != nil {
		t.Fatal(err)
	}
	if err := backend.Put(ctx, expired); err != nil {
		t.Fatal(err)
	}
	_, err = r.Resolve(ctx, "logical-7")
	requireCode(t, err, Expired)
}
