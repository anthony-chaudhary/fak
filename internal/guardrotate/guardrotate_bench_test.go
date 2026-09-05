package guardrotate

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

var (
	benchDirSink     string
	benchNoteSink    Note
	benchBoolSink    bool
	benchHomeSink    accounts.Home
	benchEntrySink   accounts.CooldownEntry
	benchExplainSink string
)

func benchStore(b *testing.B, cooledAt, resetAt time.Time, keys ...string) *accounts.CooldownStore {
	b.Helper()
	s, err := accounts.LoadCooldownStore(filepath.Join(b.TempDir(), "cd.json"))
	if err != nil {
		b.Fatalf("load cooldown store: %v", err)
	}
	for _, k := range keys {
		s.Cool(k, accounts.CooldownUsageLimit, "weekly", cooledAt, resetAt)
	}
	return s
}

func makeSyntheticRegistry(seats int) (accounts.Registry, accounts.RotationHeadroom) {
	homes := make([]accounts.Home, seats)
	hr := make(accounts.RotationHeadroom, seats)
	for i := 0; i < seats; i++ {
		name := fmt.Sprintf("seat-%03d", i)
		uuid := fmt.Sprintf("u-%03d", i)
		dir := fmt.Sprintf("/var/empty/.claude-%03d", i)
		homes[i] = accounts.Home{
			Name:     name,
			Dir:      dir,
			Status:   accounts.StatusActive,
			Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: uuid, Email: name + "@test.local"},
		}
		if i == 0 {
			hr["uuid:"+uuid] = -1.0
		} else {
			hr["uuid:"+uuid] = 1.0 + float64(i%10)*0.1
		}
	}
	return accounts.Registry{Homes: homes}, hr
}

func BenchmarkPlan(b *testing.B) {
	now := time.Date(2026, 7, 7, 12, 0, 0, 0, time.UTC)
	resetDistant := now.Add(2 * time.Hour)
	resetImminent := now.Add(5 * time.Minute)

	reg2, hr2 := makeSyntheticRegistry(2)
	storeCooledAlice := benchStore(b, now, resetDistant, "uuid:u-000")
	storeCooledAliceImminent := benchStore(b, now, resetImminent, "uuid:u-000")
	storeEmpty := benchStore(b, now, resetDistant)

	b.Run("WarmNoop", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dir, note, ok := Plan(reg2, storeEmpty, hr2, reg2.Homes[0].Dir, now)
			benchDirSink = dir
			benchNoteSink = note
			benchBoolSink = ok
		}
	})

	b.Run("RotateOfferable", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dir, note, ok := Plan(reg2, storeCooledAlice, hr2, reg2.Homes[0].Dir, now)
			benchDirSink = dir
			benchNoteSink = note
			benchBoolSink = ok
		}
	})

	b.Run("RotateUnknownDistant", func(b *testing.B) {
		hrUnknown := accounts.RotationHeadroom{"uuid:u-000": -1.0, "uuid:u-001": 0.0}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dir, note, ok := Plan(reg2, storeCooledAlice, hrUnknown, reg2.Homes[0].Dir, now)
			benchDirSink = dir
			benchNoteSink = note
			benchBoolSink = ok
		}
	})

	b.Run("ImminentResetTieBreak", func(b *testing.B) {
		hrUnknown := accounts.RotationHeadroom{"uuid:u-000": -1.0, "uuid:u-001": 0.0}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dir, note, ok := Plan(reg2, storeCooledAliceImminent, hrUnknown, reg2.Homes[0].Dir, now)
			benchDirSink = dir
			benchNoteSink = note
			benchBoolSink = ok
		}
	})

	b.Run("ServeAtFallForward", func(b *testing.B) {
		bob := grHome("bob", "/var/empty/.claude-bob", "u-bob")
		bob.RehomeTo = "carol"
		regFall := accounts.Registry{Homes: []accounts.Home{
			grHome("alice", "/var/empty/.claude-alice", "u-alice"),
			bob,
			grHome("carol", "/var/empty/.claude-carol", "u-carol"),
		}}
		storeFall := benchStore(b, now, resetDistant, "uuid:u-alice", "uuid:u-bob")
		hrFall := accounts.RotationHeadroom{"uuid:u-alice": -1.0, "uuid:u-bob": 1.5, "uuid:u-carol": 1.2}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			dir, note, ok := Plan(regFall, storeFall, hrFall, "/var/empty/.claude-alice", now)
			benchDirSink = dir
			benchNoteSink = note
			benchBoolSink = ok
		}
	})

	for _, count := range []int{5, 20, 100} {
		regN, hrN := makeSyntheticRegistry(count)
		storeN := benchStore(b, now, resetDistant, "uuid:u-000")
		b.Run(fmt.Sprintf("Scaling_%d_seats", count), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dir, note, ok := Plan(regN, storeN, hrN, regN.Homes[0].Dir, now)
				benchDirSink = dir
				benchNoteSink = note
				benchBoolSink = ok
			}
		})
	}
}

func BenchmarkHomeForDir(b *testing.B) {
	reg5, _ := makeSyntheticRegistry(5)
	reg20, _ := makeSyntheticRegistry(20)
	reg100, _ := makeSyntheticRegistry(100)

	b.Run("ExactHitFirst", func(b *testing.B) {
		dir := reg20.Homes[0].Dir
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			home, ok := HomeForDir(reg20, dir)
			benchHomeSink = home
			benchBoolSink = ok
		}
	})

	b.Run("ExactHitLast", func(b *testing.B) {
		dir := reg20.Homes[len(reg20.Homes)-1].Dir
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			home, ok := HomeForDir(reg20, dir)
			benchHomeSink = home
			benchBoolSink = ok
		}
	})

	b.Run("Miss", func(b *testing.B) {
		dir := "/var/empty/.claude-nonexistent"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			home, ok := HomeForDir(reg20, dir)
			benchHomeSink = home
			benchBoolSink = ok
		}
	})

	b.Run("NormalizedHit", func(b *testing.B) {
		dir := reg20.Homes[0].Dir + "/../" + filepath.Base(reg20.Homes[0].Dir) + "/"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			home, ok := HomeForDir(reg20, dir)
			benchHomeSink = home
			benchBoolSink = ok
		}
	})

	for _, count := range []int{5, 20, 100} {
		var r accounts.Registry
		switch count {
		case 5:
			r = reg5
		case 20:
			r = reg20
		case 100:
			r = reg100
		}
		dir := r.Homes[count-1].Dir
		b.Run(fmt.Sprintf("Scaling_%d_homes", count), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				home, ok := HomeForDir(r, dir)
				benchHomeSink = home
				benchBoolSink = ok
			}
		})
	}
}

func BenchmarkWaitResetHorizon(b *testing.B) {
	now := time.Date(2026, 7, 7, 12, 0, 0, 0, time.UTC)
	imminent := now.Add(5 * time.Minute)
	exact := now.Add(WaitResetHorizon)
	distant := now.Add(2 * time.Hour)
	zero := time.Time{}

	b.Run("Imminent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = resetImminent(imminent, now)
		}
	})

	b.Run("ExactHorizon", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = resetImminent(exact, now)
		}
	})

	b.Run("Distant", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = resetImminent(distant, now)
		}
	})

	b.Run("Zero", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = resetImminent(zero, now)
		}
	})
}

func BenchmarkPersistCooldownForRehome(b *testing.B) {
	now := time.Date(2026, 7, 7, 12, 0, 0, 0, time.UTC)
	store := benchStore(b, now, now)

	b.Run("ExplicitReset", func(b *testing.B) {
		resetSrc := "resets at 2026-07-07T17:00:00Z"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entry, ok := PersistCooldownForRehome(store, "uuid:u-alice", resetSrc, resetSrc, true, now)
			benchEntrySink = entry
			benchBoolSink = ok
		}
	})

	b.Run("DefaultWindow", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			entry, ok := PersistCooldownForRehome(store, "uuid:u-alice", "rehomed_seat", "rehomed_seat", true, now)
			benchEntrySink = entry
			benchBoolSink = ok
		}
	})
}

func BenchmarkNoteExplain(b *testing.B) {
	reset := time.Date(2026, 7, 7, 17, 0, 0, 0, time.UTC)
	room := 1.5

	b.Run("Base", func(b *testing.B) {
		note := Note{From: "alice", To: "bob"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchExplainSink = note.Explain()
		}
	})

	b.Run("WithResetAndHeadroom", func(b *testing.B) {
		note := Note{From: "alice", To: "bob", ResetAt: reset, Headroom: &room}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchExplainSink = note.Explain()
		}
	})
}

func BenchmarkNormalizeDir(b *testing.B) {
	cleanDir := "/var/empty/.claude"
	dirtyDir := "/var/empty/./foo/../.claude//"

	b.Run("Clean", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDirSink = NormalizeDir(cleanDir)
		}
	})

	b.Run("Dirty", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchDirSink = NormalizeDir(dirtyDir)
		}
	})
}
