package accounts

import (
	"net/http"
	"testing"
	"time"
)

func benchmarkRegistry() Registry {
	return Registry{
		Version: RegistryVersion,
		Roles: map[string]string{
			RoleAnchor: "seat-1",
			RoleActive: "seat-2",
		},
		Homes: []Home{
			{Name: "seat-1", Dir: "/homes/seat-1", Status: StatusActive, Identity: Identity{AccountUUID: "acc-1", Email: "user1@example.com", Exists: true, HasCreds: true}},
			{Name: "seat-2", Dir: "/homes/seat-2", Status: StatusActive, Identity: Identity{AccountUUID: "acc-2", Email: "user2@example.com", Exists: true, HasCreds: true}},
			{Name: "seat-3", Dir: "/homes/seat-3", Status: StatusActive, Identity: Identity{AccountUUID: "acc-3", Email: "user3@example.com", Exists: true, HasCreds: true}},
			{Name: "seat-4", Dir: "/homes/seat-4", Status: StatusActive, Identity: Identity{AccountUUID: "acc-1", Email: "user1@example.com", Exists: true, HasCreds: true}},
			{Name: "seat-5", Dir: "/homes/seat-5", Status: StatusActive, Identity: Identity{AccountUUID: "acc-2", Email: "user2@example.com", Exists: true, HasCreds: true}},
			{Name: "tomb-1", Status: StatusTombstoned, RehomeTo: "seat-1"},
			{Name: "tomb-2", Status: StatusTombstoned, RehomeTo: "tomb-1"},
		},
	}
}

func BenchmarkRegistryResolve(b *testing.B) {
	reg := benchmarkRegistry()

	b.Run("Active", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			h, chain, err := reg.Resolve("seat-1")
			if err != nil || h.Name != "seat-1" || len(chain) != 0 {
				b.Fatalf("unexpected resolve: %v", err)
			}
		}
	})

	b.Run("Rehome", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			h, chain, err := reg.Resolve("tomb-2")
			if err != nil || h.Name != "seat-1" || len(chain) != 2 {
				b.Fatalf("unexpected resolve: %v", err)
			}
		}
	})
}

func BenchmarkRegistryValidate(b *testing.B) {
	reg := benchmarkRegistry()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := reg.Validate(); err != nil {
			b.Fatalf("validation failed: %v", err)
		}
	}
}

func BenchmarkRegistryServeAt(b *testing.B) {
	reg := benchmarkRegistry()
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	now := time.Now()

	b.Run("Active", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			h, _, _, err := reg.ServeAt("seat-1", cd, now)
			if err != nil || h.Name != "seat-1" {
				b.Fatalf("serve failed: %v", err)
			}
		}
	})

	cd.Cool(UUIDBucketKey("acc-1"), CooldownRateLimit, "rate limit", now, now.Add(10*time.Minute))
	b.Run("CooledDownFallforward", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			h, chain, _, err := reg.ServeAt("seat-1", cd, now)
			if err != nil || h.Name != "seat-2" || len(chain) == 0 {
				b.Fatalf("serve failed: %v", err)
			}
		}
	})
}

func BenchmarkRotationPlan(b *testing.B) {
	reg := benchmarkRegistry()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := reg.RotationPlan()
		if len(res.Pool) == 0 {
			b.Fatal("empty rotation pool")
		}
	}
}

func BenchmarkNextInRotation(b *testing.B) {
	reg := benchmarkRegistry()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seat, ok := reg.NextInRotation("seat-1")
		if !ok || seat.Name == "" {
			b.Fatal("next rotation failed")
		}
	}
}

func BenchmarkCooldownStore(b *testing.B) {
	now := time.Now()

	b.Run("CooledDown", func(b *testing.B) {
		cd := &CooldownStore{entries: map[string]CooldownEntry{
			"acc-1": {Account: "acc-1", ResetAt: now.Add(time.Hour)},
		}}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, cooled := cd.CooledDown("acc-1", now); !cooled {
				b.Fatal("expected cooled")
			}
		}
	})

	b.Run("Cool", func(b *testing.B) {
		cd := &CooldownStore{entries: map[string]CooldownEntry{}}
		reset := now.Add(30 * time.Minute)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cd.Cool("acc-1", CooldownRateLimit, "limit", now, reset)
		}
	})

	b.Run("Prune", func(b *testing.B) {
		cd := &CooldownStore{entries: map[string]CooldownEntry{
			"stale-1": {Account: "stale-1", ResetAt: now.Add(-time.Hour)},
			"stale-2": {Account: "stale-2", ResetAt: now.Add(-2 * time.Hour)},
			"fresh-1": {Account: "fresh-1", ResetAt: now.Add(time.Hour)},
		}}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cd.Prune(now)
		}
	})
}

func BenchmarkRegistryJSON(b *testing.B) {
	reg := benchmarkRegistry()
	raw := reg.JSON()

	b.Run("Marshal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = reg.JSON()
		}
	})

	b.Run("Unmarshal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			parsed, err := ParseRegistry(raw)
			if err != nil || len(parsed.Homes) == 0 {
				b.Fatalf("parse failed: %v", err)
			}
		}
	})
}

func BenchmarkClassifySeatHealth(b *testing.B) {
	now := time.Now()
	body429 := []byte(`{"error":{"type":"rate_limit_error","message":"Number of request tokens has exceeded your daily limit."}}`)
	body403Wall := []byte(`{"error":{"type":"permission_error","message":"OAuth authentication is currently not allowed for this organization."}}`)
	hdr429 := http.Header{"Retry-After": []string{"60"}}

	b.Run("StatusOK", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			h := ClassifySeatHealth(http.StatusOK, nil, nil, now)
			if h != SeatHealthReady {
				b.Fatalf("unexpected health: %v", h)
			}
		}
	})

	b.Run("StatusRateLimit", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			h := ClassifySeatHealth(http.StatusTooManyRequests, body429, hdr429, now)
			if h != SeatHealthUsageLimited {
				b.Fatalf("unexpected health: %v", h)
			}
		}
	})

	b.Run("StatusOrgWall", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			h := ClassifySeatHealth(http.StatusForbidden, body403Wall, nil, now)
			if h != SeatHealthOrgAuthWall {
				b.Fatalf("unexpected health: %v", h)
			}
		}
	})
}
