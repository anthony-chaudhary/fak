package remotemaint

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeAuthority struct {
	grant          Grant
	validateCalls  []Grant
	executeCalls   []Grant
	executeErr     error
	invalidReceipt bool
	receiptMutator func(*Receipt)
	failureOutcome string
	waitForContext bool
}

func (f *fakeAuthority) Resolve(context.Context, Request) (Grant, error) { return f.grant, nil }
func (f *fakeAuthority) Validate(_ context.Context, g Grant) error {
	f.validateCalls = append(f.validateCalls, cloneGrant(g))
	return nil
}
func (f *fakeAuthority) Execute(ctx context.Context, g Grant) (Receipt, error) {
	f.executeCalls = append(f.executeCalls, cloneGrant(g))
	if f.waitForContext {
		<-ctx.Done()
		return Receipt{}, ctx.Err()
	}
	if f.executeErr != nil {
		return Receipt{}, f.executeErr
	}
	now := time.Now().UTC()
	receipt := receiptFor(g, now)
	if f.failureOutcome != "" {
		receipt.Outcome = f.failureOutcome
		receipt.Verified = false
		receipt.BackupDisposition = BackupRetained
		if f.receiptMutator != nil {
			f.receiptMutator(&receipt)
		}
		return receipt, errors.New("remote adapter reported failure")
	}
	if f.invalidReceipt {
		receipt.Verified = false
	}
	if f.receiptMutator != nil {
		f.receiptMutator(&receipt)
	}
	return receipt, nil
}

// fak-test:runtime fast est=0.1s
func TestPreparedPropagatesBoundFailureReceiptOnce(t *testing.T) {
	grant := validTestGrant(time.Now().UTC())
	authority := &fakeAuthority{grant: grant, failureOutcome: OutcomeUnknownEffect}
	prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	receipt, err := prepared.Execute(context.Background())
	if !errors.Is(err, ErrUnknownEffect) || receipt.Outcome != OutcomeUnknownEffect || receipt.BackupDisposition != BackupRetained {
		t.Fatalf("Execute() = (%+v, %v), want bound unknown-effect receipt", receipt, err)
	}
	if second, err := prepared.Execute(context.Background()); !errors.Is(err, ErrChanged) || !reflect.DeepEqual(second, Receipt{}) {
		t.Fatalf("retry Execute() = (%+v, %v), want zero/ErrChanged", second, err)
	}
	if len(authority.executeCalls) != 1 {
		t.Fatalf("authority calls = %d, want 1", len(authority.executeCalls))
	}
}

// fak-test:runtime fast est=0.1s
func TestPreparedRedactsMismatchedFailureReceipt(t *testing.T) {
	grant := validTestGrant(time.Now().UTC())
	authority := &fakeAuthority{
		grant: grant, failureOutcome: OutcomeRollbackFailed,
		receiptMutator: func(r *Receipt) { r.BootID = "wrong-boot" },
	}
	prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	receipt, err := prepared.Execute(context.Background())
	if !errors.Is(err, ErrUnknownEffect) || receipt.Outcome != OutcomeUnknownEffect || receipt.BootID != grant.BootID || receipt.BackupDisposition != BackupUnknown {
		t.Fatalf("Execute() = (%+v, %v), want synthesized bound unknown effect", receipt, err)
	}
}

// fak-test:runtime fast est=0.1s
func TestPreparedDeadlineSynthesizesUnknownEffectOnce(t *testing.T) {
	grant := validTestGrant(time.Now().UTC())
	authority := &fakeAuthority{grant: grant, waitForContext: true}
	prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	receipt, err := prepared.Execute(ctx)
	if !errors.Is(err, ErrUnknownEffect) || receipt.GrantDigest != GrantDigest(grant) || receipt.Outcome != OutcomeUnknownEffect || receipt.BackupDisposition != BackupUnknown {
		t.Fatalf("Execute(deadline) = (%+v, %v), want synthesized bound unknown effect", receipt, err)
	}
	if second, err := prepared.Execute(context.Background()); !errors.Is(err, ErrChanged) || !reflect.DeepEqual(second, Receipt{}) {
		t.Fatalf("retry Execute() = (%+v, %v), want zero/ErrChanged", second, err)
	}
}

// fak-test:runtime fast est=0.1s
func TestPreparedConcurrentExecuteCallsAuthorityOnce(t *testing.T) {
	grant := validTestGrant(time.Now().UTC())
	authority := &fakeAuthority{grant: grant}
	prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	var successes atomic.Int32
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := prepared.Execute(context.Background()); err == nil {
				successes.Add(1)
			}
		}()
	}
	wg.Wait()
	if successes.Load() != 1 || len(authority.executeCalls) != 1 {
		t.Fatalf("successes=%d authority calls=%d, want 1/1", successes.Load(), len(authority.executeCalls))
	}
}

// fak-test:runtime fast est=0.1s
func TestPreparedExecuteErrorConsumesGrant(t *testing.T) {
	grant := validTestGrant(time.Now().UTC())
	authority := &fakeAuthority{grant: grant, executeErr: errors.New("uncertain remote outcome")}
	prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	if receipt, err := prepared.Execute(context.Background()); !errors.Is(err, ErrUnknownEffect) || receipt.Outcome != OutcomeUnknownEffect {
		t.Fatalf("Execute() = (%+v, %v), want synthesized unknown effect", receipt, err)
	}
	if _, err := prepared.Execute(context.Background()); !errors.Is(err, ErrChanged) {
		t.Fatalf("retry Execute() error = %v, want ErrChanged", err)
	}
	if len(authority.executeCalls) != 1 {
		t.Fatalf("authority Execute calls = %d, want 1", len(authority.executeCalls))
	}
}

// fak-test:runtime fast est=0.1s
func TestPreparedInvalidReceiptConsumesGrant(t *testing.T) {
	grant := validTestGrant(time.Now().UTC())
	authority := &fakeAuthority{grant: grant, invalidReceipt: true}
	prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	if receipt, err := prepared.Execute(context.Background()); !errors.Is(err, ErrUnknownEffect) || receipt.Outcome != OutcomeUnknownEffect {
		t.Fatalf("Execute() = (%+v, %v), want synthesized unknown effect", receipt, err)
	}
	if _, err := prepared.Execute(context.Background()); !errors.Is(err, ErrChanged) {
		t.Fatalf("retry Execute() error = %v, want ErrChanged", err)
	}
	if len(authority.executeCalls) != 1 {
		t.Fatalf("authority Execute calls = %d, want 1", len(authority.executeCalls))
	}
}

// fak-test:runtime fast est=0.1s
func TestPreparedRejectsMismatchedReceiptEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Receipt)
	}{
		{name: "digest", mutate: func(r *Receipt) { r.GrantDigest = repeated("e") }},
		{name: "boot", mutate: func(r *Receipt) { r.BootID = "different-boot" }},
		{name: "time", mutate: func(r *Receipt) { r.EndedAt = time.Now().UTC().Add(2 * time.Minute) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			grant := validTestGrant(time.Now().UTC())
			authority := &fakeAuthority{grant: grant, receiptMutator: test.mutate}
			prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
			if err != nil {
				t.Fatalf("Prepare() error = %v", err)
			}
			receipt, err := prepared.Execute(context.Background())
			if !errors.Is(err, ErrUnknownEffect) || receipt.GrantDigest != GrantDigest(grant) || receipt.BootID != grant.BootID {
				t.Fatalf("Execute() = (%+v, %v), want synthesized bound unknown effect", receipt, err)
			}
			if len(authority.executeCalls) != 1 {
				t.Fatalf("authority calls = %d, want 1", len(authority.executeCalls))
			}
		})
	}
}

// fak-test:runtime fast est=0.1s
func TestPreparedDeepCloneAndSameGrant(t *testing.T) {
	grant := validTestGrant(time.Now().UTC())
	authority := &fakeAuthority{grant: grant}
	prepared, err := Prepare(context.Background(), authority, Request{ProfileID: grant.ProfileID, OperationDigest: grant.OperationDigest})
	if err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	exposed := prepared.Grant()
	exposed.Paths[0] = "/tampered"
	exposed.LeasePins[0].ID = "tampered"
	receipt, err := prepared.Execute(context.Background())
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if receipt.GrantDigest != GrantDigest(grant) {
		t.Fatalf("receipt digest = %q, want original grant", receipt.GrantDigest)
	}
	if len(authority.executeCalls) != 1 || authority.executeCalls[0].Paths[0] != grant.Paths[0] || authority.executeCalls[0].LeasePins[0].ID != grant.LeasePins[0].ID {
		t.Fatal("authority received caller-mutated grant")
	}
	if _, err := prepared.Execute(context.Background()); !errors.Is(err, ErrChanged) {
		t.Fatalf("second Execute() error = %v, want ErrChanged", err)
	}
}

// fak-test:runtime fast est=0.1s
func TestParseRequestRejectsCallerAuthority(t *testing.T) {
	raw := []byte(`{"profile_id":"reviewed","operation_digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","target":"caller-selected"}`)
	if _, err := ParseRequest(raw); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("ParseRequest() error = %v, want ErrInvalidRequest", err)
	}
}

// fak-test:runtime fast est=0.1s
func TestParseRequestRejectsTrailingJSON(t *testing.T) {
	request := `{"profile_id":"reviewed","operation_digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}`
	for _, raw := range []string{request + ` {}`, request + ` trailing`} {
		if _, err := ParseRequest([]byte(raw)); !errors.Is(err, ErrInvalidRequest) {
			t.Fatalf("ParseRequest(%q) error = %v, want ErrInvalidRequest", raw, err)
		}
	}
}

// fak-test:runtime fast est=0.1s
func TestValidateGrantRejectsExpiredLeasePin(t *testing.T) {
	now := time.Now().UTC()
	grant := validTestGrant(now)
	grant.LeasePins[0].ExpiresAt = now.Add(-time.Second)
	if err := validateGrant(grant, now); !errors.Is(err, ErrExpired) {
		t.Fatalf("validateGrant() error = %v, want ErrExpired", err)
	}
}

// fak-test:runtime fast est=0.1s
func TestValidateGrantRejectsUnsafePathPrefixes(t *testing.T) {
	now := time.Now().UTC()
	for _, prefix := range []PathPrefix{
		{Directory: "/etc", FilenamePrefix: ""},
		{Directory: "/var/tmp", FilenamePrefix: "../stage"},
		{Directory: "/var/tmp", FilenamePrefix: "stage*"},
	} {
		grant := validTestGrant(now)
		grant.PathPrefixes = []PathPrefix{prefix}
		if err := validateGrant(grant, now); !errors.Is(err, ErrScope) {
			t.Fatalf("validateGrant(%+v) error = %v, want ErrScope", prefix, err)
		}
	}
}

func validTestGrant(now time.Time) Grant {
	expires := now.Add(10 * time.Minute).UTC()
	return Grant{
		ProfileID:        "reviewed",
		OperationDigest:  repeated("b"),
		ContextID:        repeated("c"),
		Target:           "maintenance.example.invalid",
		HostKey:          "SHA256:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
		Operation:        OperationSystemdProfileInstallReload,
		PayloadSHA256:    repeated("a"),
		Holder:           "guard",
		BootID:           "boot-1",
		PriorStateSHA256: repeated("d"),
		Paths:            []string{"/etc/systemd/system/fak.service"},
		PathPrefixes:     []PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "fak-stage."}},
		Units:            []string{"fak.service"},
		LeasePins:        []LeasePin{{NodeID: "node-1", ID: "lease-1", Epoch: 1, ExpiresAt: expires.Add(time.Minute)}},
		ExpiresAt:        expires,
	}
}

func receiptFor(g Grant, now time.Time) Receipt {
	return Receipt{
		Schema: ReceiptSchema, GrantDigest: GrantDigest(g), ProfileID: g.ProfileID,
		OperationDigest: g.OperationDigest, ContextID: g.ContextID, Target: g.Target,
		HostKey: g.HostKey, Operation: g.Operation, BootID: g.BootID,
		StartedAt: now.Add(-time.Second), EndedAt: now, Outcome: "applied", Verified: true,
	}
}

func repeated(s string) string {
	out := ""
	for len(out) < 64 {
		out += s
	}
	return out[:64]
}
