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

// fak-test:runtime fast est=1s
func TestSourceLeaseControlRequest(t *testing.T) {
	digest := strings.Repeat("a", 64)
	for _, raw := range []string{
		`{"profile_id":"release","plan_digest":"` + digest + `","actor":"owner"}`,
		`{"profile_id":"release","profile_id":"other","plan_digest":"` + digest + `"}`,
		`{"profile_id":"release","plan_digest":"` + strings.ToUpper(digest) + `"}`,
		`{"profile_id":"release","plan_digest":"` + digest + `"} {}`,
	} {
		if _, err := ParseRequest([]byte(raw)); !errors.Is(err, ErrInvalid) {
			t.Fatalf("invalid request admitted: %v", err)
		}
	}
	if _, err := ParseRequest([]byte(`{"profile_id":"release","plan_digest":"` + digest + `"}`)); err != nil {
		t.Fatal(err)
	}
}

type sourceAuthority struct {
	d             Definition
	deny          bool
	changed       bool
	calls         int
	resolves      int
	validations   int
	onResolve     func()
	onValidate    func()
	afterEffect   func()
	mutateReceipt func(*Receipt)
}

func (a *sourceAuthority) Resolve(context.Context, string) (Definition, error) {
	a.resolves++
	if a.onResolve != nil {
		a.onResolve()
	}
	return a.d, nil
}
func (a *sourceAuthority) Validate(context.Context, Definition) error {
	a.validations++
	if a.onValidate != nil {
		a.onValidate()
	}
	if a.deny {
		return ErrAuthority
	}
	return nil
}
func (a *sourceAuthority) Execute(_ context.Context, d Definition) (Receipt, error) {
	a.calls++
	// The fixture represents a source provider's transaction boundary. A
	// preliminary Validate cannot stand in for this final actor/row check.
	if a.deny || a.changed {
		return Receipt{}, ErrAuthority
	}
	started := time.Now().UTC()
	f, err := os.OpenFile(d.StorePath, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		return Receipt{}, err
	}
	_, err = f.WriteString("released\n")
	closeErr := f.Close()
	if err != nil {
		return Receipt{}, err
	}
	if closeErr != nil {
		return Receipt{}, closeErr
	}
	if a.afterEffect != nil {
		a.afterEffect()
	}
	r := Receipt{Schema: ReceiptSchema, ProfileID: d.ProfileID, PlanDigest: definitionDigest(d), Generation: d.Generation, RowSHA256: d.RowSHA256, Outcome: OutcomeReleased, Verified: true, StartedAt: started, EndedAt: time.Now().UTC()}
	if a.mutateReceipt != nil {
		a.mutateReceipt(&r)
	}
	return r, nil
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlCanceledBeforeEffect(t *testing.T) {
	for _, stage := range []string{"before-prepare", "after-resolve", "after-prepare", "after-validation"} {
		t.Run(stage, func(t *testing.T) {
			a, request := sourceFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if stage == "before-prepare" {
				cancel()
			}
			if stage == "after-resolve" {
				a.onResolve = cancel
			}
			p, err := Prepare(ctx, a, request)
			if stage == "after-prepare" || stage == "after-validation" {
				if err != nil {
					t.Fatal(err)
				}
				if stage == "after-prepare" {
					cancel()
				} else {
					a.onValidate = cancel
				}
				_, err = p.Execute(ctx)
			}
			if !errors.Is(err, context.Canceled) || a.calls != 0 {
				t.Fatalf("canceled operation reached effect: %v", err)
			}
			if stage == "before-prepare" && (a.resolves != 0 || a.validations != 0) {
				t.Fatal("canceled preparation entered provider")
			}
			if stage == "after-resolve" && a.validations != 0 {
				t.Fatal("canceled resolve entered validation")
			}
			if stage == "after-prepare" && a.validations != 1 {
				t.Fatal("canceled execution entered validation")
			}
			data, err := os.ReadFile(a.d.StorePath)
			if err != nil || string(data) != "row\n" {
				t.Fatal("canceled operation changed store")
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlPanicOrCancellationAfterEffect(t *testing.T) {
	for _, kind := range []string{"panic", "cancellation"} {
		t.Run(kind, func(t *testing.T) {
			a, request := sourceFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			a.afterEffect = func() {
				if kind == "panic" {
					panic("untrusted provider failure")
				}
				cancel()
			}
			p, err := Prepare(ctx, a, request)
			if err != nil {
				t.Fatal(err)
			}
			receipt, err := p.Execute(ctx)
			if !errors.Is(err, ErrUnknownEffect) || receipt.Outcome != OutcomeUnknown || receipt.Verified || receipt.PlanDigest != request.PlanDigest || receipt.Generation != a.d.Generation || receipt.RowSHA256 != a.d.RowSHA256 {
				t.Fatalf("uncertain effect promoted: %v", err)
			}
			if _, err := p.Execute(context.Background()); !errors.Is(err, ErrConsumed) || a.calls != 1 {
				t.Fatal("uncertain effect retried")
			}
			data, err := os.ReadFile(a.d.StorePath)
			if err != nil || string(data) != "row\nreleased\n" {
				t.Fatal("post-effect failure fixture did not write exactly once")
			}
		})
	}
}

func sourceFixture(t *testing.T) (*sourceAuthority, Request) {
	t.Helper()
	root := t.TempDir()
	store, lock := filepath.Join(root, "store"), filepath.Join(root, "lock")
	for _, name := range []string{store} {
		if err := os.WriteFile(name, []byte("row\n"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	paths := []string{store, lock}
	sort.Strings(paths)
	a := &sourceAuthority{d: Definition{ProfileID: "release", Operation: OperationRelease, WorkspaceRoot: root, StorePath: store, LockPath: lock, LockEffect: LockEffectExclusiveMutex, LeaseKey: "lease-1", Generation: 1, RowSHA256: strings.Repeat("a", 64), Footprint: paths, ExpiresAt: time.Now().UTC().Add(time.Minute)}}
	digest, err := PlanDigest(a.d)
	if err != nil {
		t.Fatal(err)
	}
	return a, Request{ProfileID: a.d.ProfileID, PlanDigest: digest}
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlPreparedSingleUse(t *testing.T) {
	a, r := sourceFixture(t)
	p, err := Prepare(context.Background(), a, r)
	if err != nil {
		t.Fatal(err)
	}
	copy := p.Definition()
	copy.Footprint[0] = "changed"
	a.d.Footprint[0] = "source-changed"
	if p.Definition().Footprint[0] == "changed" || p.Definition().Footprint[0] == "source-changed" {
		t.Fatal("mutable scope escaped")
	}
	receipt, err := p.Execute(context.Background())
	if err != nil || !receipt.Verified {
		t.Fatalf("effect: %v", err)
	}
	b, err := os.ReadFile(p.Definition().StorePath)
	if err != nil || string(b) != "row\nreleased\n" {
		t.Fatalf("temporary-store readback: %v", err)
	}
	if _, err := p.Execute(context.Background()); !errors.Is(err, ErrConsumed) || a.calls != 1 {
		t.Fatalf("replay: %v calls=%d", err, a.calls)
	}
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlSourceDenialAndChangedPins(t *testing.T) {
	for _, kind := range []string{"nil", "typed-nil", "actor", "digest", "generation", "file-replaced", "provider-row-changed"} {
		t.Run(kind, func(t *testing.T) {
			a, r := sourceFixture(t)
			var provider Authority = a
			switch kind {
			case "nil":
				provider = nil
			case "typed-nil":
				var absent *sourceAuthority
				provider = absent
			case "actor":
				a.deny = true
			case "digest":
				r.PlanDigest = strings.Repeat("b", 64)
			case "generation":
				a.d.Generation++
			}
			p, err := Prepare(context.Background(), provider, r)
			if kind == "file-replaced" || kind == "provider-row-changed" {
				if err != nil {
					t.Fatal(err)
				}
				if kind == "file-replaced" {
					if err := os.Rename(a.d.StorePath, a.d.StorePath+".old"); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(a.d.StorePath, []byte("row\n"), 0600); err != nil {
						t.Fatal(err)
					}
				} else {
					a.changed = true
				}
				_, err = p.Execute(context.Background())
			}
			if err == nil {
				t.Fatal("invalid authority or changed scope admitted")
			}
			if kind != "provider-row-changed" && a.calls != 0 {
				t.Fatal("provider effect entered")
			}
			b, readErr := os.ReadFile(a.d.StorePath)
			if readErr != nil || string(b) != "row\n" {
				t.Fatalf("refused effect altered store: %v", readErr)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlScope(t *testing.T) {
	for _, kind := range []string{"outside", "directory", "missing-lock", "duplicate", "unsorted", "expiry", "zero-generation", "unknown-operation", "uppercase-row"} {
		t.Run(kind, func(t *testing.T) {
			a, _ := sourceFixture(t)
			switch kind {
			case "outside":
				outside := filepath.Join(t.TempDir(), "outside")
				if err := os.WriteFile(outside, nil, 0600); err != nil {
					t.Fatal(err)
				}
				a.d.Footprint = append(a.d.Footprint, outside)
				sort.Strings(a.d.Footprint)
			case "directory":
				a.d.Footprint = append(a.d.Footprint, a.d.WorkspaceRoot)
				sort.Strings(a.d.Footprint)
			case "missing-lock":
				a.d.LockPath = filepath.Join(a.d.WorkspaceRoot, "missing")
			case "duplicate":
				a.d.Footprint = append(a.d.Footprint, a.d.Footprint[0])
				sort.Strings(a.d.Footprint)
			case "unsorted":
				a.d.Footprint[0], a.d.Footprint[1] = a.d.Footprint[1], a.d.Footprint[0]
			case "expiry":
				a.d.ExpiresAt = time.Now().Add(-time.Second)
			case "zero-generation":
				a.d.Generation = 0
			case "unknown-operation":
				a.d.Operation = "execute"
			case "uppercase-row":
				a.d.RowSHA256 = strings.Repeat("A", 64)
			}
			if _, err := PlanDigest(a.d); !errors.Is(err, ErrInvalid) {
				t.Fatalf("invalid scope admitted: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlReceiptMutationUnknown(t *testing.T) {
	for _, kind := range []string{"profile", "digest", "generation", "row", "outcome", "verified", "replayed-time"} {
		t.Run(kind, func(t *testing.T) {
			a, r := sourceFixture(t)
			a.mutateReceipt = func(receipt *Receipt) {
				switch kind {
				case "profile":
					receipt.ProfileID = "peer"
				case "digest":
					receipt.PlanDigest = strings.Repeat("b", 64)
				case "generation":
					receipt.Generation++
				case "row":
					receipt.RowSHA256 = strings.Repeat("b", 64)
				case "outcome":
					receipt.Outcome = "other"
				case "verified":
					receipt.Verified = false
				case "replayed-time":
					receipt.StartedAt = time.Now().Add(-time.Minute)
				}
			}
			p, err := Prepare(context.Background(), a, r)
			if err != nil {
				t.Fatal(err)
			}
			got, err := p.Execute(context.Background())
			if !errors.Is(err, ErrUnknownEffect) || got.Outcome != OutcomeUnknown || got.Verified || got.PlanDigest != r.PlanDigest {
				t.Fatalf("unbound effect promoted: %v", err)
			}
			if _, err := p.Execute(context.Background()); !errors.Is(err, ErrConsumed) || a.calls != 1 {
				t.Fatal("unknown effect retried")
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlExclusiveMutex(t *testing.T) {
	for _, kind := range []string{"absent", "peer-present", "appeared-after-prepare", "parent-replaced", "symlink-parent"} {
		t.Run(kind, func(t *testing.T) {
			a, _ := sourceFixture(t)
			parent := filepath.Join(a.d.WorkspaceRoot, "mutex-parent")
			if err := os.Mkdir(parent, 0700); err != nil {
				t.Fatal(err)
			}
			a.d.LockPath = filepath.Join(parent, "lock")
			a.d.Footprint = []string{a.d.StorePath, a.d.LockPath}
			sort.Strings(a.d.Footprint)
			if kind == "peer-present" {
				if err := os.WriteFile(a.d.LockPath, []byte("peer"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if kind == "symlink-parent" {
				if err := os.Rename(parent, parent+"-real"); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(parent+"-real", parent); err != nil {
					t.Skip("host cannot create directory symlink")
				}
			}
			digest, err := PlanDigest(a.d)
			if kind == "peer-present" || kind == "symlink-parent" {
				if err == nil {
					t.Fatal("peer or redirected mutex admitted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			p, err := Prepare(context.Background(), a, Request{ProfileID: a.d.ProfileID, PlanDigest: digest})
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "appeared-after-prepare":
				if err := os.WriteFile(a.d.LockPath, []byte("peer"), 0600); err != nil {
					t.Fatal(err)
				}
			case "parent-replaced":
				if err := os.Rename(parent, parent+"-old"); err != nil {
					t.Fatal(err)
				}
				if err := os.Mkdir(parent, 0700); err != nil {
					t.Fatal(err)
				}
			}
			_, err = p.Execute(context.Background())
			if kind == "absent" {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || a.calls != 0 {
				t.Fatal("changed mutex identity reached provider")
			}
		})
	}
}
