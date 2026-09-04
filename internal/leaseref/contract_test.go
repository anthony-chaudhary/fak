package leaseref

import (
	"context"
	"encoding/json"
	"errors"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestContractAcquisitionStateUpdateRenewalReap verifies the entire contract lifecycle:
// Acquire -> UpdateState -> Renew -> Expire -> LiveContracts -> Reap -> Gone.
func TestContractAcquisitionStateUpdateRenewalReap(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(10000, 0)

	// 1. Acquire new contract.
	in := ContractRecord{
		TicketID:    "issue-1324",
		Holder:      "worker-mac-1",
		SessionID:   "sess-guard-1",
		State:       ContractStatePending,
		PaceTier:    PaceTierCommodity,
		TokenBudget: 100000,
		TurnLimit:   25,
		WorktreeDir: ".worktrees/ticket-1324",
		BaseSHA:     "abcdef1234567890",
		VerifyCmd:   "go test ./...",
		TTLSeconds:  300,
	}

	c1, err := s.AcquireContract(ctx(), in, t0)
	if err != nil {
		t.Fatalf("AcquireContract: %v", err)
	}
	if c1.TicketID != "issue-1324" {
		t.Fatalf("TicketID = %q, want issue-1324", c1.TicketID)
	}
	if c1.Generation != 1 {
		t.Fatalf("Generation = %d, want 1", c1.Generation)
	}
	if c1.State != ContractStatePending {
		t.Fatalf("State = %q, want %q", c1.State, ContractStatePending)
	}
	if c1.AcquiredAt != t0.Unix() {
		t.Fatalf("AcquiredAt = %d, want %d", c1.AcquiredAt, t0.Unix())
	}
	if c1.RenewedAt != 0 {
		t.Fatalf("RenewedAt = %d, want 0", c1.RenewedAt)
	}

	// 2. Update state to EXECUTING.
	t1 := t0.Add(10 * time.Second)
	c2, err := s.UpdateContractState(ctx(), "issue-1324", ContractStateExecuting, t1)
	if err != nil {
		t.Fatalf("UpdateContractState(EXECUTING): %v", err)
	}
	if c2.State != ContractStateExecuting {
		t.Fatalf("State = %q, want EXECUTING", c2.State)
	}
	if c2.RenewedAt != t1.Unix() {
		t.Fatalf("RenewedAt = %d, want %d", c2.RenewedAt, t1.Unix())
	}

	// 3. Update state to YIELDED_IO.
	t2 := t0.Add(30 * time.Second)
	c3, err := s.UpdateContractState(ctx(), "issue-1324", ContractStateYieldedIO, t2)
	if err != nil {
		t.Fatalf("UpdateContractState(YIELDED_IO): %v", err)
	}
	if c3.State != ContractStateYieldedIO {
		t.Fatalf("State = %q, want YIELDED_IO", c3.State)
	}

	// 4. Update state to VERIFYING.
	t3 := t0.Add(50 * time.Second)
	c4, err := s.UpdateContractState(ctx(), "issue-1324", ContractStateVerifying, t3)
	if err != nil {
		t.Fatalf("UpdateContractState(VERIFYING): %v", err)
	}
	if c4.State != ContractStateVerifying {
		t.Fatalf("State = %q, want VERIFYING", c4.State)
	}

	// 5. Renew contract.
	t4 := t0.Add(100 * time.Second)
	c5, err := s.RenewContract(ctx(), "issue-1324", "worker-mac-1", 600, t4)
	if err != nil {
		t.Fatalf("RenewContract: %v", err)
	}
	if c5.Generation != 1 {
		t.Fatalf("RenewContract must not bump generation; got %d, want 1", c5.Generation)
	}
	if c5.RenewedAt != t4.Unix() {
		t.Fatalf("RenewedAt = %d, want %d", c5.RenewedAt, t4.Unix())
	}
	if c5.TTLSeconds != 600 {
		t.Fatalf("TTLSeconds = %d, want 600", c5.TTLSeconds)
	}

	// 6. GetContract reads the latest state.
	got, ok, err := s.GetContract(ctx(), "issue-1324")
	if err != nil || !ok {
		t.Fatalf("GetContract ok=%v err=%v", ok, err)
	}
	if got.State != ContractStateVerifying || got.Holder != "worker-mac-1" || got.TTLSeconds != 600 {
		t.Fatalf("GetContract = %+v, unexpected fields", got)
	}

	// 7. Update to SUCCEEDED.
	c6, err := s.UpdateContractState(ctx(), "issue-1324", ContractStateSucceeded, t4.Add(10*time.Second))
	if err != nil {
		t.Fatalf("UpdateContractState(SUCCEEDED): %v", err)
	}
	if c6.State != ContractStateSucceeded {
		t.Fatalf("State = %q, want SUCCEEDED", c6.State)
	}

	// 8. Test live vs expired: renewed at t4 (10100) with TTL 600 expires at 10700.
	tBeforeExp := time.Unix(10650, 0)
	live, expired, err := s.LiveContracts(ctx(), tBeforeExp)
	if err != nil {
		t.Fatalf("LiveContracts: %v", err)
	}
	if len(live) != 1 || live[0].TicketID != "issue-1324" {
		t.Fatalf("LiveContracts live=%+v, want [issue-1324]", live)
	}
	if len(expired) != 0 {
		t.Fatalf("LiveContracts expired=%v, want empty", expired)
	}

	tAfterExp := time.Unix(10800, 0)
	live, expired, err = s.LiveContracts(ctx(), tAfterExp)
	if err != nil {
		t.Fatalf("LiveContracts: %v", err)
	}
	if len(live) != 0 {
		t.Fatalf("LiveContracts live=%+v, want empty", live)
	}
	if len(expired) != 1 || expired[0] != "issue-1324" {
		t.Fatalf("LiveContracts expired=%v, want [issue-1324]", expired)
	}

	// 9. ReapContracts deletes expired contract.
	reaped, err := s.ReapContracts(ctx(), tAfterExp)
	if err != nil {
		t.Fatalf("ReapContracts: %v", err)
	}
	if len(reaped) != 1 || reaped[0] != "issue-1324" {
		t.Fatalf("ReapContracts reaped=%v, want [issue-1324]", reaped)
	}

	// 10. Post-reap check.
	_, ok, err = s.GetContract(ctx(), "issue-1324")
	if err != nil || ok {
		t.Fatalf("post-reap GetContract ok=%v err=%v, want false", ok, err)
	}
	live, expired, err = s.LiveContracts(ctx(), tAfterExp)
	if err != nil || len(live) != 0 || len(expired) != 0 {
		t.Fatalf("post-reap LiveContracts live=%v expired=%v", live, expired)
	}
}

// TestContractGenerationIncrementsAndTakeover tests monotonic generation progression:
// fresh acquire (gen 1) -> blocked rival -> expired takeover (gen 2) -> renew (gen 2) -> rerun (gen 3).
func TestContractGenerationIncrementsAndTakeover(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(5000, 0)

	// Worker A acquires contract at t0 with TTL 60.
	recA := ContractRecord{
		TicketID:   "issue-2000",
		Holder:     "worker-A",
		State:      ContractStateExecuting,
		TTLSeconds: 60,
	}
	cA, err := s.AcquireContract(ctx(), recA, t0)
	if err != nil {
		t.Fatalf("Worker A AcquireContract: %v", err)
	}
	if cA.Generation != 1 {
		t.Fatalf("Generation = %d, want 1", cA.Generation)
	}

	// Rival Worker B tries to acquire same ticket at t0 + 10s (unexpired). Must fail with ErrContractHeld.
	recB := ContractRecord{
		TicketID:   "issue-2000",
		Holder:     "worker-B",
		State:      ContractStatePending,
		TTLSeconds: 60,
	}
	_, err = s.AcquireContract(ctx(), recB, t0.Add(10*time.Second))
	if err == nil || !errors.Is(err, ErrContractHeld) {
		t.Fatalf("Worker B early acquire err=%v, want ErrContractHeld", err)
	}

	// Rival Worker B tries to renew contract. Must fail with ErrContractHeld.
	_, err = s.RenewContract(ctx(), "issue-2000", "worker-B", 60, t0.Add(15*time.Second))
	if err == nil || !errors.Is(err, ErrContractHeld) {
		t.Fatalf("Worker B unauthorized renew err=%v, want ErrContractHeld", err)
	}

	// Worker B acquires at t0 + 100s (past TTL 60). Takeover bumps generation to 2!
	cB, err := s.AcquireContract(ctx(), recB, t0.Add(100*time.Second))
	if err != nil {
		t.Fatalf("Worker B takeover: %v", err)
	}
	if cB.Generation != 2 {
		t.Fatalf("Takeover generation = %d, want 2", cB.Generation)
	}
	if cB.Holder != "worker-B" {
		t.Fatalf("Holder = %q, want worker-B", cB.Holder)
	}

	// Worker B renews: generation stays 2.
	cB2, err := s.RenewContract(ctx(), "issue-2000", "worker-B", 60, t0.Add(120*time.Second))
	if err != nil {
		t.Fatalf("Worker B renew: %v", err)
	}
	if cB2.Generation != 2 {
		t.Fatalf("Renew generation = %d, want 2", cB2.Generation)
	}

	// Worker B succeeds.
	_, err = s.UpdateContractState(ctx(), "issue-2000", ContractStateSucceeded, t0.Add(130*time.Second))
	if err != nil {
		t.Fatalf("Worker B state update: %v", err)
	}

	// Subsequent rerun by Worker C on completed contract increments generation to 3!
	recC := ContractRecord{
		TicketID:   "issue-2000",
		Holder:     "worker-C",
		State:      ContractStatePending,
		TTLSeconds: 60,
	}
	cC, err := s.AcquireContract(ctx(), recC, t0.Add(140*time.Second))
	if err != nil {
		t.Fatalf("Worker C rerun acquire: %v", err)
	}
	if cC.Generation != 3 {
		t.Fatalf("Rerun generation = %d, want 3", cC.Generation)
	}
}

// TestContractCASContention verifies that when an update-ref CAS detects a concurrent modification,
// it returns ErrCASContended.
func TestContractCASContention(t *testing.T) {
	g := newFakeGit()
	t0 := time.Unix(2000, 0)

	raced := false
	racer := func(c context.Context, dir string, args ...string) (string, int, error) {
		if !raced && len(args) > 0 && args[0] == "update-ref" && len(args) >= 4 {
			raced = true
			// Concurrent writer silently updates the ref in Git behind our back.
			g.next++
			racerID := synthID(g.next)
			g.blobs[racerID] = []byte(`{"ticket_id":"issue-cas","holder":"racer","state":"EXECUTING","acquired_unix":2000,"ttl_seconds":3600,"generation":1}`)
			g.refs[args[1]] = racerID
		}
		return g.run(c, dir, args...)
	}

	s := NewWithRunner(racer, "")

	// Acquire contract should hit CAS race if created concurrently.
	_, err := s.AcquireContract(ctx(), ContractRecord{
		TicketID:   "issue-cas",
		Holder:     "worker-orig",
		TTLSeconds: 300,
	}, t0)
	if err == nil || !errors.Is(err, ErrCASContended) {
		t.Fatalf("AcquireContract with raced CAS err=%v, want ErrCASContended", err)
	}

	// Now test CAS race on UpdateContractState:
	g2 := newFakeGit()
	s2 := NewWithRunner(g2.run, "")
	_, err = s2.AcquireContract(ctx(), ContractRecord{
		TicketID:   "issue-cas-state",
		Holder:     "worker-orig",
		TTLSeconds: 300,
	}, t0)
	if err != nil {
		t.Fatalf("setup AcquireContract: %v", err)
	}

	racedState := false
	racerState := func(c context.Context, dir string, args ...string) (string, int, error) {
		if !racedState && len(args) > 0 && args[0] == "update-ref" && len(args) >= 4 {
			racedState = true
			g2.next++
			racerID := synthID(g2.next)
			g2.blobs[racerID] = []byte(`{"ticket_id":"issue-cas-state","holder":"racer","state":"YIELDED_IO","acquired_unix":2000,"ttl_seconds":300,"generation":1}`)
			g2.refs[args[1]] = racerID
		}
		return g2.run(c, dir, args...)
	}

	s2Raced := NewWithRunner(racerState, "")
	_, err = s2Raced.UpdateContractState(ctx(), "issue-cas-state", ContractStateExecuting, t0.Add(time.Second))
	if err == nil || !errors.Is(err, ErrCASContended) {
		t.Fatalf("UpdateContractState with raced CAS err=%v, want ErrCASContended", err)
	}
}

// TestContractNamespaceSeparation is the load-bearing witness for requirement 5:
// a live contract-* ref is NEVER returned by store.Live(), store.List(), or store.LiveLeases(),
// and vice-versa for session and intent views.
func TestContractNamespaceSeparation(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(3000, 0)

	// 1. Acquire a lock lease.
	lockRef, err := s.Acquire(ctx(), Record{
		ID:         "kernel-lane",
		TreeGlobs:  []string{"internal/leaseref/**"},
		Holder:     "lock-holder",
		AcquiredAt: t0.Unix(),
		TTLSeconds: 3600,
	})
	if err != nil {
		t.Fatalf("Acquire lock: %v", err)
	}
	if lockRef != "refs/fak/locks/kernel-lane" {
		t.Fatalf("lockRef = %q", lockRef)
	}

	// 2. Publish a session descriptor.
	sessRef, err := s.PublishSession(ctx(), SessionDescriptor{
		ID:        "guard-sess-1",
		Host:      "host-A",
		PCBState:  "RUNNING",
		UpdatedAt: t0.Unix(),
		TTLSecs:   3600,
	})
	if err != nil {
		t.Fatalf("PublishSession: %v", err)
	}
	if sessRef != "refs/fak/locks/session-guard-sess-1" {
		t.Fatalf("sessRef = %q", sessRef)
	}

	// 3. Claim an intent lease.
	_, intentVerd, err := s.ClaimIntent(ctx(), IntentRecord{
		Target:     "issue-999",
		Holder:     "intent-holder",
		TTLSeconds: 3600,
	}, t0)
	if err != nil || !intentVerd.OK {
		t.Fatalf("ClaimIntent: %v %+v", err, intentVerd)
	}

	// 4. Acquire a contract lease.
	contractRec, err := s.AcquireContract(ctx(), ContractRecord{
		TicketID:   "issue-1324",
		Holder:     "contract-holder",
		State:      ContractStateExecuting,
		TTLSeconds: 3600,
	}, t0)
	if err != nil {
		t.Fatalf("AcquireContract: %v", err)
	}
	if contractRec.Ref() != "refs/fak/locks/contract-issue-1324" {
		t.Fatalf("contractRef = %q", contractRec.Ref())
	}

	// Check Store.List(): MUST return only the lock lease ("kernel-lane").
	locks, err := s.List(ctx())
	if err != nil {
		t.Fatalf("store.List(): %v", err)
	}
	if len(locks) != 1 || locks[0].ID != "kernel-lane" {
		t.Fatalf("store.List() = %+v, want exactly [kernel-lane]", locks)
	}

	// Check Store.Live(): MUST return only the lock lease ("kernel-lane").
	liveLocks, expiredLocks, err := s.Live(ctx(), t0)
	if err != nil {
		t.Fatalf("store.Live(): %v", err)
	}
	if len(liveLocks) != 1 || liveLocks[0].ID != "kernel-lane" || len(expiredLocks) != 0 {
		t.Fatalf("store.Live() = live:%+v expired:%v, want exactly live [kernel-lane]", liveLocks, expiredLocks)
	}

	// Check Store.LiveLeases() (arbiter projection): MUST return only "kernel-lane".
	arbiterLeases, err := s.LiveLeases(ctx(), t0)
	if err != nil {
		t.Fatalf("store.LiveLeases(): %v", err)
	}
	if len(arbiterLeases) != 1 || arbiterLeases[0].Lane != "kernel-lane" {
		t.Fatalf("store.LiveLeases() = %+v, want exactly [kernel-lane]", arbiterLeases)
	}

	// Check Store.ListSessions() / LiveSessions(): MUST return only "guard-sess-1".
	sessions, err := s.ListSessions(ctx())
	if err != nil {
		t.Fatalf("store.ListSessions(): %v", err)
	}
	if len(sessions) != 1 || sessions[0].ID != "guard-sess-1" {
		t.Fatalf("store.ListSessions() = %+v, want exactly [guard-sess-1]", sessions)
	}

	// Check Store.ListIntents() / LiveIntents(): MUST return only "issue-999".
	intents, err := s.ListIntents(ctx())
	if err != nil {
		t.Fatalf("store.ListIntents(): %v", err)
	}
	if len(intents) != 1 || intents[0].Key != "issue-999" {
		t.Fatalf("store.ListIntents() = %+v, want exactly [issue-999]", intents)
	}

	// Check Store.ListContracts() / LiveContracts(): MUST return only "issue-1324".
	contracts, err := s.ListContracts(ctx())
	if err != nil {
		t.Fatalf("store.ListContracts(): %v", err)
	}
	if len(contracts) != 1 || contracts[0].TicketID != "issue-1324" {
		t.Fatalf("store.ListContracts() = %+v, want exactly [issue-1324]", contracts)
	}

	liveContracts, expiredContracts, err := s.LiveContracts(ctx(), t0)
	if err != nil {
		t.Fatalf("store.LiveContracts(): %v", err)
	}
	if len(liveContracts) != 1 || liveContracts[0].TicketID != "issue-1324" || len(expiredContracts) != 0 {
		t.Fatalf("store.LiveContracts() = live:%+v expired:%v, want exactly live [issue-1324]", liveContracts, expiredContracts)
	}
}

// TestContractHelperFunctions tests isContractRef and contractRefName.
func TestContractHelperFunctions(t *testing.T) {
	if !isContractRef("refs/fak/locks/contract-issue-1324") {
		t.Fatalf("isContractRef(refs/fak/locks/contract-issue-1324) must be true")
	}
	if !isContractRef("refs/fak/locks/contract-xyz") {
		t.Fatalf("isContractRef(refs/fak/locks/contract-xyz) must be true")
	}
	if isContractRef("refs/fak/locks/session-guard-1") {
		t.Fatalf("isContractRef(session ref) must be false")
	}
	if isContractRef("refs/fak/locks/intent-issue-1") {
		t.Fatalf("isContractRef(intent ref) must be false")
	}
	if isContractRef("refs/fak/locks/kernel-lane") {
		t.Fatalf("isContractRef(lock lease ref) must be false")
	}
	if isContractRef("refs/heads/main") {
		t.Fatalf("isContractRef(refs/heads/main) must be false")
	}

	cases := []struct {
		input string
		want  string
	}{
		{"issue-1324", "refs/fak/locks/contract-issue-1324"},
		{"contract-issue-1324", "refs/fak/locks/contract-issue-1324"},
		{"refs/fak/locks/contract-issue-1324", "refs/fak/locks/contract-issue-1324"},
		{"gh-500", "refs/fak/locks/contract-gh-500"},
	}
	for _, tc := range cases {
		got := contractRefName(tc.input)
		if got != tc.want {
			t.Errorf("contractRefName(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

// TestContractRecordJSONSerialization pins the JSON serialization and field names.
func TestContractRecordJSONSerialization(t *testing.T) {
	rec := ContractRecord{
		TicketID:    "issue-100",
		Holder:      "h-1",
		SessionID:   "s-1",
		State:       ContractStateYieldedIO,
		PaceTier:    PaceTierFrontier,
		TokenBudget: 50000,
		TokensUsed:  12000,
		TurnLimit:   10,
		WorktreeDir: ".worktrees/t-100",
		BaseSHA:     "11223344",
		VerifyCmd:   "make test",
		Generation:  4,
		AcquiredAt:  12345,
		RenewedAt:   12400,
		TTLSeconds:  300,
	}

	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var decoded ContractRecord
	if err := json.Unmarshal(b, &decoded); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if decoded.TicketID != "issue-100" || decoded.State != ContractStateYieldedIO ||
		decoded.TokenBudget != 50000 || decoded.TokensUsed != 12000 ||
		decoded.TurnLimit != 10 || decoded.Generation != 4 ||
		decoded.AcquiredAt != 12345 || decoded.RenewedAt != 12400 || decoded.TTLSeconds != 300 {
		t.Fatalf("decoded record mismatch: %+v", decoded)
	}

	// Test alternate JSON tags (acquired_at, renewed_at).
	altJSON := `{"ticket_id":"issue-alt","holder":"h2","state":"EXECUTING","acquired_at":9990,"renewed_at":9995,"ttl_seconds":60}`
	var altDecoded ContractRecord
	if err := json.Unmarshal([]byte(altJSON), &altDecoded); err != nil {
		t.Fatalf("Unmarshal alt JSON: %v", err)
	}
	if altDecoded.AcquiredAt != 9990 || altDecoded.RenewedAt != 9995 {
		t.Fatalf("altDecoded timestamps: AcquiredAt=%d, RenewedAt=%d", altDecoded.AcquiredAt, altDecoded.RenewedAt)
	}
}

// TestContractRealGitRoundTripAndZeroRefsLeaked runs against real git in an isolated temp dir,
// verifies operations end-to-end, and proves zero test refs leaked into the real repository.
func TestContractRealGitRoundTripAndZeroRefsLeaked(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}

	// 1. Create temporary isolated git repository.
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "test@example.com"},
		{"config", "user.name", "tester"},
	} {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}

	s := NewInDir(dir)
	t0 := time.Now()

	// 2. Acquire contract.
	rec := ContractRecord{
		TicketID:    "issue-42",
		Holder:      "real-worker",
		SessionID:   "real-sess",
		State:       ContractStateExecuting,
		PaceTier:    PaceTierCommodity,
		TokenBudget: 75000,
		TTLSeconds:  60,
	}

	acquired, err := s.AcquireContract(ctx(), rec, t0)
	if err != nil {
		t.Fatalf("real git AcquireContract: %v", err)
	}
	if acquired.Generation != 1 {
		t.Fatalf("Generation = %d, want 1", acquired.Generation)
	}

	// 3. Verify on-disk ref exists in temp repo under refs/fak/locks/contract-issue-42.
	listing, code, err := gitRunner(ctx(), dir, "for-each-ref", "--format=%(refname)", "refs/fak/locks/")
	if err != nil || code != 0 {
		t.Fatalf("for-each-ref: %v (code %d)", err, code)
	}
	if !strings.Contains(listing, "refs/fak/locks/contract-issue-42") {
		t.Fatalf("ref not found in temp repo: %q", listing)
	}

	// 4. Update state and renew.
	updated, err := s.UpdateContractState(ctx(), "issue-42", ContractStateVerifying, t0.Add(5*time.Second))
	if err != nil {
		t.Fatalf("UpdateContractState: %v", err)
	}
	if updated.State != ContractStateVerifying {
		t.Fatalf("State = %q, want VERIFYING", updated.State)
	}

	renewed, err := s.RenewContract(ctx(), "issue-42", "real-worker", 120, t0.Add(10*time.Second))
	if err != nil {
		t.Fatalf("RenewContract: %v", err)
	}
	if renewed.TTLSeconds != 120 {
		t.Fatalf("TTLSeconds = %d, want 120", renewed.TTLSeconds)
	}

	// 5. Release contract.
	if err := s.ReleaseContract(ctx(), "issue-42"); err != nil {
		t.Fatalf("ReleaseContract: %v", err)
	}

	// Verify temp repo has no contract refs after release.
	listingAfter, _, _ := gitRunner(ctx(), dir, "for-each-ref", "--format=%(refname)", "refs/fak/locks/contract-")
	if strings.TrimSpace(listingAfter) != "" {
		t.Fatalf("contract refs still in temp repo after release: %q", listingAfter)
	}

	// 6. Host repository ref-leak assertion:
	// Verify that the host repository's refs/fak/locks/ contains zero contract-* refs if in a git repo.
	if _, isRepoCode, _ := gitRunner(ctx(), "", "rev-parse", "--is-inside-work-tree"); isRepoCode == 0 {
		hostListing, code, err := gitRunner(ctx(), "", "for-each-ref", "--format=%(refname)", "refs/fak/locks/contract-")
		if err != nil || code != 0 {
			t.Fatalf("host for-each-ref: %v (code %d)", err, code)
		}
		if strings.TrimSpace(hostListing) != "" {
			t.Fatalf("LEAK DETECTED: host repo has contract-* refs: %q", hostListing)
		}
	}
}
