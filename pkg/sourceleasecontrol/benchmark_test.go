package sourceleasecontrol

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// BenchmarkParseRequest measures strict JSON request parsing: one valid
// release request and one duplicate-key rejection per iteration.
func BenchmarkParseRequest(b *testing.B) {
	digest := strings.Repeat("a", 64)
	valid := []byte(`{"profile_id":"bench-release","plan_digest":"` + digest + `"}`)
	invalid := []byte(`{"profile_id":"bench-release","profile_id":"other","plan_digest":"` + digest + `"}`)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := ParseRequest(valid); err != nil {
			b.Fatal(err)
		}
		if _, err := ParseRequest(invalid); !errors.Is(err, ErrInvalid) {
			b.Fatalf("duplicate-key request admitted: %v", err)
		}
	}
}

// BenchmarkPlanDigest measures live-scope planning of a canonical release
// definition: filesystem pinning plus canonical-JSON digest hashing.
func BenchmarkPlanDigest(b *testing.B) {
	root := b.TempDir()
	store, lock := filepath.Join(root, "store"), filepath.Join(root, "lock")
	if err := os.WriteFile(store, []byte("row\n"), 0600); err != nil {
		b.Fatal(err)
	}
	footprint := []string{store, lock}
	sort.Strings(footprint)
	d := Definition{ProfileID: "bench-release", Operation: OperationRelease, WorkspaceRoot: root, StorePath: store, LockPath: lock, LockEffect: LockEffectExclusiveMutex, LeaseKey: "bench-key", Generation: 1, RowSHA256: strings.Repeat("a", 64), Footprint: footprint, ExpiresAt: time.Now().UTC().Add(10 * time.Minute)}
	expected, err := PlanDigest(d)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := PlanDigest(d)
		if err != nil || got != expected {
			b.Fatalf("digest drift: %v (%q vs %q)", err, got, expected)
		}
	}
}

type benchAuthority struct {
	d Definition
}

func (a *benchAuthority) Resolve(context.Context, string) (Definition, error) {
	return a.d, nil
}

func (a *benchAuthority) Validate(context.Context, Definition) error {
	return nil
}

// Execute appends to the pinned store each iteration; revalidate survives
// because pins compare file identity (os.SameFile), not content.
func (a *benchAuthority) Execute(_ context.Context, d Definition) (Receipt, error) {
	started := time.Now().UTC()
	f, err := os.OpenFile(d.StorePath, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		return Receipt{}, err
	}
	_, werr := f.WriteString("released\n")
	cerr := f.Close()
	if werr != nil {
		return Receipt{}, werr
	}
	if cerr != nil {
		return Receipt{}, cerr
	}
	digest, err := DefinitionDigest(d)
	if err != nil {
		return Receipt{}, err
	}
	return Receipt{Schema: ReceiptSchema, ProfileID: d.ProfileID, PlanDigest: digest, Generation: d.Generation, RowSHA256: d.RowSHA256, Outcome: OutcomeReleased, Verified: true, StartedAt: started, EndedAt: time.Now().UTC()}, nil
}

// BenchmarkPrepareExecute measures the full prepare-to-effect cycle: resolve,
// pin, digest, validate, provider effect, and receipt verification.
func BenchmarkPrepareExecute(b *testing.B) {
	root := b.TempDir()
	store, lock := filepath.Join(root, "store"), filepath.Join(root, "lock")
	if err := os.WriteFile(store, []byte("row\n"), 0600); err != nil {
		b.Fatal(err)
	}
	footprint := []string{store, lock}
	sort.Strings(footprint)
	a := &benchAuthority{d: Definition{ProfileID: "bench-release", Operation: OperationRelease, WorkspaceRoot: root, StorePath: store, LockPath: lock, LockEffect: LockEffectExclusiveMutex, LeaseKey: "bench-key", Generation: 1, RowSHA256: strings.Repeat("a", 64), Footprint: footprint, ExpiresAt: time.Now().UTC().Add(10 * time.Minute)}}
	digest, err := PlanDigest(a.d)
	if err != nil {
		b.Fatal(err)
	}
	req := Request{ProfileID: a.d.ProfileID, PlanDigest: digest}
	warm, err := Prepare(context.Background(), a, req)
	if err != nil {
		b.Fatal(err)
	}
	if _, err := warm.Execute(context.Background()); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p, err := Prepare(context.Background(), a, req)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := p.Execute(context.Background()); err != nil {
			b.Fatal(err)
		}
	}
}
