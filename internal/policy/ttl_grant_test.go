package policy

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestTTLGrantLifecycle(t *testing.T) {
	reg := NewTTLGrantRegistry()
	holder := "worker-alpha"
	capName := "package:install"
	target := "curl"

	// Prior to granting, should not be granted.
	if reg.IsGranted(holder, capName, target) {
		t.Fatalf("expected IsGranted to be false before grant")
	}
	if len(reg.ActiveGrants(holder)) != 0 {
		t.Fatalf("expected ActiveGrants to be empty before grant")
	}

	// Grant capability for 1 hour.
	ttl := 1 * time.Hour
	before := time.Now()
	g := reg.Grant(holder, capName, target, ttl)
	after := time.Now()

	if g == nil {
		t.Fatalf("expected non-nil grant")
	}
	if g.HolderID != holder {
		t.Errorf("expected HolderID %q, got %q", holder, g.HolderID)
	}
	if g.Capability != capName {
		t.Errorf("expected Capability %q, got %q", capName, g.Capability)
	}
	if g.Target != target {
		t.Errorf("expected Target %q, got %q", target, g.Target)
	}
	if g.GrantedAt.Before(before) || g.GrantedAt.After(after) {
		t.Errorf("GrantedAt %v outside expected window [%v, %v]", g.GrantedAt, before, after)
	}
	expectedExpires := g.GrantedAt.Add(ttl)
	if !g.ExpiresAt.Equal(expectedExpires) {
		t.Errorf("ExpiresAt %v != expected %v", g.ExpiresAt, expectedExpires)
	}

	// Verify active
	if !reg.IsGranted(holder, capName, target) {
		t.Fatalf("expected IsGranted to be true after grant")
	}

	active := reg.ActiveGrants(holder)
	if len(active) != 1 {
		t.Fatalf("expected 1 active grant, got %d", len(active))
	}
	if active[0].Capability != capName || active[0].Target != target {
		t.Errorf("active grant mismatch: got %+v", active[0])
	}

	// Another holder should not see this grant.
	if reg.IsGranted("worker-beta", capName, target) {
		t.Errorf("different holder should not have grant")
	}
	if len(reg.ActiveGrants("worker-beta")) != 0 {
		t.Errorf("different holder should have 0 active grants")
	}

	// Revoke
	if !reg.Revoke(holder, capName, target) {
		t.Fatalf("expected Revoke to return true for active grant")
	}
	if reg.IsGranted(holder, capName, target) {
		t.Fatalf("expected IsGranted to be false after revocation")
	}
	if len(reg.ActiveGrants(holder)) != 0 {
		t.Fatalf("expected ActiveGrants to be empty after revocation")
	}

	// Second revoke should return false.
	if reg.Revoke(holder, capName, target) {
		t.Fatalf("expected second Revoke to return false")
	}
}

func TestTTLGrantActiveWindowValidity(t *testing.T) {
	reg := NewTTLGrantRegistry()
	baseTime := time.Date(2026, 9, 3, 10, 0, 0, 0, time.UTC)
	currentTime := baseTime

	reg.now = func() time.Time {
		return currentTime
	}

	holder := "operator-1"
	capName := "migration:run"
	target := "db_staging"

	ttl := 10 * time.Minute
	g := reg.Grant(holder, capName, target, ttl)
	if g.ExpiresAt != baseTime.Add(ttl) {
		t.Fatalf("expected ExpiresAt %v, got %v", baseTime.Add(ttl), g.ExpiresAt)
	}

	// Within active window
	currentTime = baseTime.Add(5 * time.Minute)
	if !reg.IsGranted(holder, capName, target) {
		t.Errorf("expected IsGranted to be true at 5 minutes")
	}

	// Just before expiry
	currentTime = baseTime.Add(10*time.Minute - time.Nanosecond)
	if !reg.IsGranted(holder, capName, target) {
		t.Errorf("expected IsGranted to be true at 9m59.999s")
	}

	// Exactly at expiry (time.Now().Before(grant.ExpiresAt) is false when now == ExpiresAt)
	currentTime = baseTime.Add(10 * time.Minute)
	if reg.IsGranted(holder, capName, target) {
		t.Errorf("expected IsGranted to be false exactly at expiry")
	}

	// Past expiry
	currentTime = baseTime.Add(11 * time.Minute)
	if reg.IsGranted(holder, capName, target) {
		t.Errorf("expected IsGranted to be false after expiry")
	}
}

func TestTTLGrantActiveWindowRealClock(t *testing.T) {
	reg := NewTTLGrantRegistry()
	holder := "short-lived-worker"
	capName := "file:write"
	target := "/tmp/staging"

	reg.Grant(holder, capName, target, 25*time.Millisecond)

	if !reg.IsGranted(holder, capName, target) {
		t.Fatalf("expected grant to be active immediately")
	}

	time.Sleep(35 * time.Millisecond)

	if reg.IsGranted(holder, capName, target) {
		t.Fatalf("expected grant to be expired after sleep")
	}
	if len(reg.ActiveGrants(holder)) != 0 {
		t.Fatalf("expected ActiveGrants to exclude expired grant")
	}
}

func TestTTLGrantExpiredDrop(t *testing.T) {
	reg := NewTTLGrantRegistry()
	baseTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	currentTime := baseTime

	reg.now = func() time.Time {
		return currentTime
	}

	holder := "agent-x"

	reg.Grant(holder, "cap-a", "target-1", 5*time.Minute)
	reg.Grant(holder, "cap-b", "target-2", 15*time.Minute)
	reg.Grant(holder, "cap-c", "target-3", 30*time.Minute)

	// Initially all 3 are active
	active := reg.ActiveGrants(holder)
	if len(active) != 3 {
		t.Fatalf("expected 3 active grants, got %d", len(active))
	}

	// At 10 minutes: cap-a is expired, cap-b and cap-c active
	currentTime = baseTime.Add(10 * time.Minute)
	if reg.IsGranted(holder, "cap-a", "target-1") {
		t.Errorf("cap-a should have dropped")
	}
	if !reg.IsGranted(holder, "cap-b", "target-2") {
		t.Errorf("cap-b should remain granted")
	}
	if !reg.IsGranted(holder, "cap-c", "target-3") {
		t.Errorf("cap-c should remain granted")
	}

	active = reg.ActiveGrants(holder)
	if len(active) != 2 {
		t.Fatalf("expected 2 active grants, got %d", len(active))
	}
	if active[0].Capability != "cap-b" || active[1].Capability != "cap-c" {
		t.Errorf("unexpected active grants: %+v", active)
	}

	// At 20 minutes: cap-a and cap-b expired, cap-c active
	currentTime = baseTime.Add(20 * time.Minute)
	if reg.IsGranted(holder, "cap-b", "target-2") {
		t.Errorf("cap-b should have dropped")
	}
	active = reg.ActiveGrants(holder)
	if len(active) != 1 || active[0].Capability != "cap-c" {
		t.Fatalf("expected only cap-c active, got %+v", active)
	}

	// At 35 minutes: all expired
	currentTime = baseTime.Add(35 * time.Minute)
	if reg.IsGranted(holder, "cap-c", "target-3") {
		t.Errorf("cap-c should have dropped")
	}
	active = reg.ActiveGrants(holder)
	if len(active) != 0 {
		t.Fatalf("expected 0 active grants, got %d", len(active))
	}

	// Zero and negative TTL should drop immediately
	reg.Grant(holder, "zero-ttl", "target-z", 0)
	reg.Grant(holder, "neg-ttl", "target-n", -1*time.Second)

	if reg.IsGranted(holder, "zero-ttl", "target-z") {
		t.Errorf("zero TTL should not be granted")
	}
	if reg.IsGranted(holder, "neg-ttl", "target-n") {
		t.Errorf("negative TTL should not be granted")
	}
}

func TestTTLGrantPruning(t *testing.T) {
	reg := NewTTLGrantRegistry()
	baseTime := time.Date(2026, 9, 3, 14, 0, 0, 0, time.UTC)
	currentTime := baseTime

	reg.now = func() time.Time {
		return currentTime
	}

	// Add 3 grants for holder1, 2 for holder2
	reg.Grant("holder1", "read", "file1", 5*time.Minute)
	reg.Grant("holder1", "write", "file2", 15*time.Minute)
	reg.Grant("holder1", "exec", "bin1", 30*time.Minute)

	reg.Grant("holder2", "admin", "cluster", 5*time.Minute)
	reg.Grant("holder2", "audit", "logs", 25*time.Minute)

	// Advance time to 10m: holder1/read and holder2/admin are expired (2 grants)
	currentTime = baseTime.Add(10 * time.Minute)

	pruned := reg.PruneExpired()
	if pruned != 2 {
		t.Fatalf("expected 2 pruned grants, got %d", pruned)
	}

	// Pruning again immediately should prune 0
	prunedSecond := reg.PruneExpired()
	if prunedSecond != 0 {
		t.Fatalf("expected 0 pruned grants on second pass, got %d", prunedSecond)
	}

	// Active grants remain
	h1Active := reg.ActiveGrants("holder1")
	if len(h1Active) != 2 {
		t.Fatalf("expected 2 active grants for holder1, got %d", len(h1Active))
	}
	h2Active := reg.ActiveGrants("holder2")
	if len(h2Active) != 1 {
		t.Fatalf("expected 1 active grant for holder2, got %d", len(h2Active))
	}

	// Advance past all expirations
	currentTime = baseTime.Add(1 * time.Hour)
	prunedFinal := reg.PruneExpired()
	if prunedFinal != 3 {
		t.Fatalf("expected remaining 3 grants to be pruned, got %d", prunedFinal)
	}

	if len(reg.ActiveGrants("holder1")) != 0 || len(reg.ActiveGrants("holder2")) != 0 {
		t.Fatalf("expected all active grants to be empty after total pruning")
	}
}

func TestTTLGrantRevocation(t *testing.T) {
	reg := NewTTLGrantRegistry()
	holder := "sec-operator"
	capName := "debug:attach"
	target := "pid:1234"

	// Revoke on non-existent grant returns false
	if reg.Revoke(holder, capName, target) {
		t.Errorf("expected Revoke on non-existent grant to return false")
	}

	// Grant and then revoke with wrong attributes
	reg.Grant(holder, capName, target, 10*time.Minute)

	if reg.Revoke("other-holder", capName, target) {
		t.Errorf("expected Revoke with wrong holder to return false")
	}
	if reg.Revoke(holder, "other-cap", target) {
		t.Errorf("expected Revoke with wrong cap to return false")
	}
	if reg.Revoke(holder, capName, "other-target") {
		t.Errorf("expected Revoke with wrong target to return false")
	}

	// Verify original is still granted
	if !reg.IsGranted(holder, capName, target) {
		t.Fatalf("original grant should still be active")
	}

	// Proper revocation
	if !reg.Revoke(holder, capName, target) {
		t.Fatalf("expected Revoke to return true")
	}
	if reg.IsGranted(holder, capName, target) {
		t.Fatalf("expected IsGranted to be false after Revoke")
	}

	// Refresh/re-grant after revocation
	reg.Grant(holder, capName, target, 5*time.Minute)
	if !reg.IsGranted(holder, capName, target) {
		t.Fatalf("expected grant to be re-established after re-granting")
	}
	if !reg.Revoke(holder, capName, target) {
		t.Fatalf("expected Revoke on re-established grant to succeed")
	}
}

func TestTTLGrantThreadSafety(t *testing.T) {
	reg := NewTTLGrantRegistry()
	const numGoroutines = 20
	const opsPerGoroutine = 200

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			holder := fmt.Sprintf("worker-%d", id%5)
			capName := fmt.Sprintf("cap-%d", id%3)
			target := fmt.Sprintf("target-%d", id%4)

			for j := 0; j < opsPerGoroutine; j++ {
				switch j % 5 {
				case 0:
					reg.Grant(holder, capName, target, 50*time.Millisecond)
				case 1:
					_ = reg.IsGranted(holder, capName, target)
				case 2:
					_ = reg.ActiveGrants(holder)
				case 3:
					_ = reg.Revoke(holder, capName, target)
				case 4:
					_ = reg.PruneExpired()
				}
			}
		}(i)
	}

	wg.Wait()
}

func TestTTLGrantNoForbiddenTokens(t *testing.T) {
	forbidden := []string{"context", "ctx", "render", "guard", "gate", "loop", "session"}

	checkToken := func(name string) {
		lower := strings.ToLower(name)
		for _, f := range forbidden {
			if strings.Contains(lower, f) {
				t.Fatalf("exported symbol %q contains forbidden token %q", name, f)
			}
		}
	}

	// Check TTLGrant struct and its fields
	grantType := reflect.TypeOf(TTLGrant{})
	checkToken(grantType.Name())
	for i := 0; i < grantType.NumField(); i++ {
		field := grantType.Field(i)
		if field.IsExported() {
			checkToken(field.Name)
		}
	}

	// Check TTLGrantRegistry and its methods
	regType := reflect.TypeOf(&TTLGrantRegistry{})
	checkToken(regType.Elem().Name())
	for i := 0; i < regType.NumMethod(); i++ {
		method := regType.Method(i)
		if method.IsExported() {
			checkToken(method.Name)
		}
	}
}
