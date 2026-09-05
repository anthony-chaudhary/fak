package accounts

import (
	"fmt"
	"testing"
	"time"
)

// benchFleetRegistry builds a realistic fleet registry of n seats with a mixture of
// active, reserved, tombstoned (including multi-hop rehome chains), and duplicated accounts.
func benchFleetRegistry(n int) Registry {
	homes := make([]Home, n)
	tru := true
	for i := 0; i < n; i++ {
		name := fmt.Sprintf("seat-%02d", i)
		acctIdx := i / 2 // every 2 seats share an account UUID -> duplicate detection
		uuid := fmt.Sprintf("u-uuid-%02d", acctIdx)
		email := fmt.Sprintf("seat%02d@example.test", i)
		dir := fmt.Sprintf("/home/.claude-%s", name)
		status := StatusActive
		rehomeTo := ""

		// Include realistic tombstone rehome chains at the tail
		if i == n-2 {
			status = StatusTombstoned
			rehomeTo = "seat-00"
			dir = dir + ".DELETED"
		} else if i == n-1 {
			status = StatusTombstoned
			rehomeTo = fmt.Sprintf("seat-%02d", n-2) // transitive rehome
			dir = dir + ".DELETED"
		}

		homes[i] = Home{
			Name:     name,
			Dir:      dir,
			Status:   status,
			RehomeTo: rehomeTo,
			Reserved: i%5 == 4,
			Enabled:  &tru,
			Identity: Identity{
				Email:       email,
				AccountUUID: uuid,
				HasCreds:    i%7 != 6,
				Exists:      true,
			},
		}
	}

	return Registry{
		Version: RegistryVersion,
		Roles: map[string]string{
			RoleAnchor: "seat-00",
			RoleActive: "seat-01",
		},
		Homes: homes,
		Views: map[string]ViewConfig{
			"dos": {
				BlockOrder: []string{"rotation", "defaults"},
				Blocks: map[string]any{
					"rotation": map[string]any{"order": "by_reset", "near_cap_util": 0.95},
					"defaults": map[string]any{
						"settings": map[string]any{
							"model":       "opus",
							"effortLevel": "xhigh",
						},
					},
				},
			},
			"job": {
				BlockOrder: []string{"defaults", "rotation", "launch"},
				Blocks: map[string]any{
					"rotation": map[string]any{"order": "by_reset", "avoid_reserved": true},
					"launch": map[string]any{
						"bypass_permissions": true,
					},
				},
			},
		},
	}
}

func BenchmarkResolve(b *testing.B) {
	reg := fixture()

	b.Run("Active", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h, _, err := reg.Resolve("gem8-seat")
			if err != nil || h.Name == "" {
				b.Fatal(err)
			}
		}
	})

	b.Run("Tombstone1Hop", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h, chain, err := reg.Resolve("q")
			if err != nil || h.Name == "" || len(chain) == 0 {
				b.Fatal(err)
			}
		}
	})

	b.Run("TombstoneTransitive", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h, chain, err := reg.Resolve("old")
			if err != nil || h.Name == "" || len(chain) != 2 {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkServeAt(b *testing.B) {
	reg := cooldownServeFixture()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.Cool(UUIDBucketKey("u-sink"), CooldownUsageLimit, "weekly limit", now, now.Add(2*time.Hour))

	b.Run("DirectServeable", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h, _, _, err := reg.ServeAt("anchor-seat", cd, now)
			if err != nil || h.Name == "" {
				b.Fatal(err)
			}
		}
	})

	b.Run("WalkPastCooled", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h, _, _, err := reg.ServeAt("gone", cd, now)
			if err != nil || h.Name != "anchor-seat" {
				b.Fatal(err)
			}
		}
	})

	allCooledReg := cooldownServeFixture()
	allCooledStore := &CooldownStore{entries: map[string]CooldownEntry{}}
	allCooledStore.Cool(UUIDBucketKey("u-anchor"), CooldownUsageLimit, "limit", now, now.Add(time.Hour))
	allCooledStore.Cool(UUIDBucketKey("u-sink"), CooldownUsageLimit, "limit", now, now.Add(2*time.Hour))

	b.Run("DegradeToSoonestReset", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h, _, entry, err := allCooledReg.ServeAt("gone", allCooledStore, now)
			if err != nil || h.Name == "" || entry == nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkValidate(b *testing.B) {
	small := fixture()
	fleet := benchFleetRegistry(50)

	b.Run("Small_5Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := small.Validate(); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Fleet_50Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := fleet.Validate(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkReconcile(b *testing.B) {
	small := benchFleetRegistry(10)
	fleet := benchFleetRegistry(50)

	b.Run("10_Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := small.Reconcile()
			if len(res) == 0 {
				b.Fatal("empty reconcile result")
			}
		}
	})

	b.Run("Fleet_50Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := fleet.Reconcile()
			if len(res) == 0 {
				b.Fatal("empty reconcile result")
			}
		}
	})
}

func BenchmarkRotationPlan(b *testing.B) {
	small := benchFleetRegistry(10)
	fleet := benchFleetRegistry(50)
	hr := make(RotationHeadroom, 25)
	for i := 0; i < 25; i++ {
		hr[UUIDBucketKey(fmt.Sprintf("u-uuid-%02d", i))] = float64(100 - i)
	}

	b.Run("StableByName_10Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := small.RotationPlan()
			if len(res.Pool) == 0 {
				b.Fatal("empty pool")
			}
		}
	})

	b.Run("StableByName_50Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := fleet.RotationPlan()
			if len(res.Pool) == 0 {
				b.Fatal("empty pool")
			}
		}
	})

	b.Run("WithHeadroom_50Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res := fleet.RotationPlanWithHeadroom(hr)
			if len(res.Pool) == 0 {
				b.Fatal("empty pool")
			}
		}
	})
}

func BenchmarkNextInRotation(b *testing.B) {
	fleet := benchFleetRegistry(50)
	hr := make(RotationHeadroom, 25)
	for i := 0; i < 25; i++ {
		hr[UUIDBucketKey(fmt.Sprintf("u-uuid-%02d", i))] = float64(100 - i)
	}

	b.Run("StableByName", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s, ok := fleet.NextInRotation("seat-00")
			if !ok || s.Name == "" {
				b.Fatal("NextInRotation failed")
			}
		}
	})

	b.Run("WithHeadroom", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s, ok := fleet.NextInRotationWithHeadroom("seat-00", hr)
			if !ok || s.Name == "" {
				b.Fatal("NextInRotationWithHeadroom failed")
			}
		}
	})

	b.Run("NextRotationDecision", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			d := fleet.NextRotationDecision("seat-00", hr)
			if !d.OK || d.Seat.Name == "" {
				b.Fatal("NextRotationDecision failed")
			}
		}
	})
}

func BenchmarkLoginReport(b *testing.B) {
	fleet := benchFleetRegistry(50)
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	for i := 0; i < 10; i++ {
		cd.Cool(UUIDBucketKey(fmt.Sprintf("u-uuid-%02d", i)), CooldownUsageLimit, "limit", now, now.Add(2*time.Hour))
	}

	b.Run("WithoutCooldown_50Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rep := fleet.LoginReport()
			if len(rep.Seats) != 50 {
				b.Fatalf("got %d seats, want 50", len(rep.Seats))
			}
		}
	})

	b.Run("WithCooldown_50Seats", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rep := fleet.LoginReportAt(cd, now)
			if len(rep.Seats) != 50 {
				b.Fatalf("got %d seats, want 50", len(rep.Seats))
			}
		}
	})

	b.Run("WithoutTombstoned_50Seats", func(b *testing.B) {
		rep := fleet.LoginReportAt(cd, now)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			filtered := rep.WithoutTombstoned()
			if len(filtered.Seats) == 0 {
				b.Fatal("empty filtered seats")
			}
		}
	})
}

func BenchmarkCooldownStore(b *testing.B) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)

	b.Run("UpdateOverload", func(b *testing.B) {
		store := &CooldownStore{entries: map[string]CooldownEntry{}}
		reset := now.Add(time.Hour)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			acct := fmt.Sprintf("acct-%04d", i%100)
			store.UpdateOverload(acct, "usage", CooldownUsageLimit, true, "limit reached", now, reset)
		}
	})

	b.Run("CooledDown_Hit", func(b *testing.B) {
		store := &CooldownStore{entries: map[string]CooldownEntry{}}
		store.Cool("target-acct", CooldownUsageLimit, "limit", now, now.Add(time.Hour))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entry, cooled := store.CooledDown("target-acct", now)
			if !cooled || entry.Account == "" {
				b.Fatal("expected cooled")
			}
		}
	})

	b.Run("CooledDown_Miss", func(b *testing.B) {
		store := &CooldownStore{entries: map[string]CooldownEntry{}}
		store.Cool("target-acct", CooldownUsageLimit, "limit", now, now.Add(time.Hour))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, cooled := store.CooledDown("other-acct", now)
			if cooled {
				b.Fatal("unexpected cooled")
			}
		}
	})

	b.Run("Active_100Entries", func(b *testing.B) {
		store := &CooldownStore{entries: map[string]CooldownEntry{}}
		for i := 0; i < 100; i++ {
			acct := fmt.Sprintf("acct-%03d", i)
			store.Cool(acct, CooldownUsageLimit, "limit", now, now.Add(time.Hour))
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entries := store.Active(now)
			if len(entries) != 100 {
				b.Fatalf("got %d active, want 100", len(entries))
			}
		}
	})

	b.Run("Prune", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			store := &CooldownStore{entries: map[string]CooldownEntry{}}
			for j := 0; j < 50; j++ {
				store.Cool(fmt.Sprintf("live-%02d", j), CooldownUsageLimit, "limit", now, now.Add(time.Hour))
				store.Cool(fmt.Sprintf("expired-%02d", j), CooldownUsageLimit, "limit", now.Add(-2*time.Hour), now.Add(-time.Hour))
			}
			b.StartTimer()
			pruned := store.Prune(now)
			if pruned != 50 {
				b.Fatalf("got %d pruned, want 50", pruned)
			}
		}
	})
}

func BenchmarkResetParsing(b *testing.B) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	msgAbsolute := "usage limit reached; resets at 2026-07-14T15:30:00Z"
	msgRelativeWait := "weekly limit reached; announced_wait≈1h7m"
	msgRetryAfter := "Too many requests; retry-after: 300"
	msgWeeklyFloor := "organization weekly limit reached, please contact support"

	b.Run("ParseReset_AbsoluteRFC3339", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			at := ParseReset(msgAbsolute)
			if at.IsZero() {
				b.Fatal("failed to parse reset")
			}
		}
	})

	b.Run("ResolveReset_RelativeWait", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			at := ResolveReset(msgRelativeWait, now)
			if at.IsZero() {
				b.Fatal("failed to resolve relative reset")
			}
		}
	})

	b.Run("ResolveReset_RetryAfterSecs", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			at := ResolveReset(msgRetryAfter, now)
			if at.IsZero() {
				b.Fatal("failed to resolve retry after")
			}
		}
	})

	b.Run("ResolveCooldownReset_WeeklyFloor", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			at := ResolveCooldownReset(msgWeeklyFloor, now)
			if at.IsZero() {
				b.Fatal("failed to resolve weekly floor")
			}
		}
	})

	b.Run("IsWeeklyLimit", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !IsWeeklyLimit(msgWeeklyFloor) {
				b.Fatal("expected weekly limit")
			}
		}
	})
}

func BenchmarkRegistryJSON(b *testing.B) {
	fleet := benchFleetRegistry(50)
	raw := fleet.JSON()

	b.Run("Marshal_50Seats", func(b *testing.B) {
		b.SetBytes(int64(len(raw)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			data := fleet.JSON()
			if len(data) == 0 {
				b.Fatal("empty JSON")
			}
		}
	})

	b.Run("Unmarshal_50Seats", func(b *testing.B) {
		b.SetBytes(int64(len(raw)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			reg, err := ParseRegistry(raw)
			if err != nil || len(reg.Homes) != 50 {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkRenderView(b *testing.B) {
	reg := viewFixture()

	b.Run("DosView", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := reg.RenderView(ViewDos)
			if err != nil || len(out) == 0 {
				b.Fatal(err)
			}
		}
	})

	b.Run("JobView", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			out, err := reg.RenderView(ViewJob)
			if err != nil || len(out) == 0 {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkAccountKey(b *testing.B) {
	idUUID := Identity{AccountUUID: "12345678-abcd-ef01-2345-6789abcdef01"}
	idTok := Identity{TokenFP: "a1b2c3d4e5f6"}
	idAPI := Identity{APIKeyEnv: "ANTHROPIC_API_KEY"}

	b.Run("UUIDBucketKey", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			k := UUIDBucketKey("12345678-abcd-ef01-2345-6789abcdef01")
			if k == "" {
				b.Fatal("empty key")
			}
		}
	})

	b.Run("AccountKey_UUID", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			k := idUUID.AccountKey()
			if k == "" {
				b.Fatal("empty key")
			}
		}
	})

	b.Run("AccountKey_TokenFP", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			k := idTok.AccountKey()
			if k == "" {
				b.Fatal("empty key")
			}
		}
	})

	b.Run("AccountKey_APIKey", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			k := idAPI.AccountKey()
			if k == "" {
				b.Fatal("empty key")
			}
		}
	})
}

func BenchmarkNameLie(b *testing.B) {
	matching := Home{
		Name:     "gem8-netra",
		Identity: Identity{Email: "gem8@netra.test"},
	}
	mismatch := Home{
		Name:     "q-seat",
		Identity: Identity{Email: "gem8@netra.test"},
	}

	b.Run("Matching", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if matching.NameLie() {
				b.Fatal("expected no name lie")
			}
		}
	})

	b.Run("Mismatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !mismatch.NameLie() {
				b.Fatal("expected name lie")
			}
		}
	})
}
