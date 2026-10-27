package sourceleasecontrol_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/sourceleasecontrol"
)

type externalAuthority struct {
	definition  sourceleasecontrol.Definition
	validations int
	effects     int
	t           *testing.T
}

func (a *externalAuthority) Resolve(context.Context, string) (sourceleasecontrol.Definition, error) {
	return a.definition, nil
}

func (a *externalAuthority) Validate(context.Context, sourceleasecontrol.Definition) error {
	a.validations++
	return nil
}

func (a *externalAuthority) Execute(_ context.Context, d sourceleasecontrol.Definition) (sourceleasecontrol.Receipt, error) {
	a.effects++
	started := time.Now().UTC()
	lock, err := os.OpenFile(d.LockPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return sourceleasecontrol.Receipt{}, err
	}
	defer func() {
		if err := lock.Close(); err != nil {
			a.t.Error(err)
		}
		if err := os.Remove(d.LockPath); err != nil {
			a.t.Error(err)
		}
	}()
	if _, err := sourceleasecontrol.PlanDigest(d); err == nil {
		a.t.Fatal("live-scope planner admitted an occupied mutex")
	}
	digest, err := sourceleasecontrol.DefinitionDigest(d)
	if err != nil {
		return sourceleasecontrol.Receipt{}, err
	}
	store, err := os.OpenFile(d.StorePath, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		return sourceleasecontrol.Receipt{}, err
	}
	_, writeErr := store.WriteString("released\n")
	closeErr := store.Close()
	if writeErr != nil {
		return sourceleasecontrol.Receipt{}, writeErr
	}
	if closeErr != nil {
		return sourceleasecontrol.Receipt{}, closeErr
	}
	return sourceleasecontrol.Receipt{Schema: sourceleasecontrol.ReceiptSchema, ProfileID: d.ProfileID, PlanDigest: digest, Generation: d.Generation, RowSHA256: d.RowSHA256, Outcome: sourceleasecontrol.OutcomeReleased, Verified: true, StartedAt: started, EndedAt: time.Now().UTC()}, nil
}

func externalFixture(t *testing.T) *externalAuthority {
	t.Helper()
	root := t.TempDir()
	store, lock := filepath.Join(root, "store"), filepath.Join(root, "mutex")
	if err := os.WriteFile(store, []byte("row\n"), 0600); err != nil {
		t.Fatal(err)
	}
	footprint := []string{store, lock}
	sort.Strings(footprint)
	return &externalAuthority{t: t, definition: sourceleasecontrol.Definition{ProfileID: "external-release", Operation: sourceleasecontrol.OperationRelease, WorkspaceRoot: root, StorePath: store, LockPath: lock, LockEffect: sourceleasecontrol.LockEffectExclusiveMutex, LeaseKey: "external-key", Generation: 1, RowSHA256: strings.Repeat("a", 64), Footprint: footprint, ExpiresAt: time.Now().UTC().Add(time.Minute)}}
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlExternalHeldMutexReceipt(t *testing.T) {
	a := externalFixture(t)
	digest, err := sourceleasecontrol.PlanDigest(a.definition)
	if err != nil {
		t.Fatal(err)
	}
	p, err := sourceleasecontrol.Prepare(context.Background(), a, sourceleasecontrol.Request{ProfileID: a.definition.ProfileID, PlanDigest: digest})
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := p.Execute(context.Background())
	if err != nil || !receipt.Verified || receipt.PlanDigest != digest || a.effects != 1 {
		t.Fatalf("external provider receipt refused: %v", err)
	}
	data, err := os.ReadFile(a.definition.StorePath)
	if err != nil || string(data) != "row\nreleased\n" {
		t.Fatalf("temporary-store readback: %v", err)
	}
	if _, err := os.Lstat(a.definition.LockPath); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("provider mutex was not released")
	}
}

// fak-test:runtime fast est=1s
func TestSourceLeaseControlExternalHashGrantsNoAuthority(t *testing.T) {
	a := externalFixture(t)
	a.definition.Generation = 0
	digest, err := sourceleasecontrol.DefinitionDigest(a.definition)
	if err != nil || digest == "" {
		t.Fatalf("hash-only helper performed scope validation: %v", err)
	}
	_, err = sourceleasecontrol.Prepare(context.Background(), a, sourceleasecontrol.Request{ProfileID: a.definition.ProfileID, PlanDigest: digest})
	if !errors.Is(err, sourceleasecontrol.ErrInvalid) || a.validations != 0 || a.effects != 0 {
		t.Fatalf("hash granted invalid definition authority: %v", err)
	}
}
