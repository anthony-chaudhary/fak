package goalpark

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"testing"
	"time"
)

func BenchmarkPark(b *testing.B) {
	dir := b.TempDir()
	store := Store{Dir: dir}
	now := time.Unix(1_800_000_000, 0)
	base := Record{
		Goal:        "goal-bench",
		Lane:        "guard",
		Account:     "seat-alpha",
		Reason:      "LONG_RETRY_AFTER",
		ParkedAt:    now.Unix(),
		ParkedUntil: now.Unix() + 7200,
		Command:     []string{"fak", "guard", "--", "claude"},
	}

	b.Run("overwrite", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := store.Park(base); err != nil {
				b.Fatalf("Park: %v", err)
			}
		}
	})

	b.Run("distinct", func(b *testing.B) {
		subDir := b.TempDir()
		distinctStore := Store{Dir: subDir}
		rec := base
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rec.Goal = fmt.Sprintf("goal-%d", i)
			if err := distinctStore.Park(rec); err != nil {
				b.Fatalf("Park: %v", err)
			}
		}
	})
}

func BenchmarkResolve(b *testing.B) {
	dir := b.TempDir()
	store := Store{Dir: dir}
	now := time.Unix(1_800_000_000, 0)
	rec := Record{
		Goal:        "goal-resolve",
		Account:     "seat-alpha",
		Reason:      "LONG_RETRY_AFTER",
		ParkedAt:    now.Unix(),
		ParkedUntil: now.Unix() + 7200,
		Command:     []string{"fak", "guard", "--", "claude"},
	}
	if err := store.Park(rec); err != nil {
		b.Fatalf("setup Park: %v", err)
	}

	b.Run("blocked", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r, blocked := store.Resolve("goal-resolve", "seat-alpha", "supervisor", now.Add(time.Minute))
			if !blocked || r.Goal != "goal-resolve" {
				b.Fatalf("unexpected resolve verdict: blocked=%v rec=%+v", blocked, r)
			}
		}
	})

	b.Run("sibling_pass_through", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			r, blocked := store.Resolve("goal-resolve", "seat-beta", "supervisor", now.Add(time.Minute))
			if blocked || r.Goal != "goal-resolve" {
				b.Fatalf("unexpected sibling resolve verdict: blocked=%v rec=%+v", blocked, r)
			}
		}
	})

	b.Run("missing_fail_open", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, blocked := store.Resolve("missing-goal", "seat-alpha", "supervisor", now)
			if blocked {
				b.Fatal("missing goal unexpectedly blocked")
			}
		}
	})
}

func BenchmarkRelease(b *testing.B) {
	b.Run("fresh", func(b *testing.B) {
		dir := b.TempDir()
		store := Store{Dir: dir}
		now := time.Unix(1_800_000_000, 0)
		rec := Record{
			Schema:      Schema,
			Goal:        "bench-release",
			Account:     "seat-alpha",
			ParkedAt:    now.Unix(),
			ParkedUntil: now.Unix() + 7200,
			Command:     []string{"fak", "guard", "--", "claude"},
		}
		data, err := json.MarshalIndent(rec, "", "  ")
		if err != nil {
			b.Fatalf("marshal: %v", err)
		}
		data = append(data, '\n')

		goals := make([]string, b.N)
		for i := 0; i < b.N; i++ {
			goals[i] = fmt.Sprintf("release-goal-%d", i)
			if err := os.WriteFile(store.path(goals[i]), data, 0o600); err != nil {
				b.Fatalf("setup WriteFile: %v", err)
			}
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := store.Release(goals[i], "guard", "seat enrolled", now); err != nil {
				b.Fatalf("Release: %v", err)
			}
		}
	})

	b.Run("already_claimed", func(b *testing.B) {
		dir := b.TempDir()
		store := Store{Dir: dir}
		now := time.Unix(1_800_000_000, 0)
		const goal = "claimed-goal"
		rec := Record{
			Goal:        goal,
			Account:     "seat-alpha",
			ParkedAt:    now.Unix(),
			ParkedUntil: now.Unix() + 7200,
			Command:     []string{"fak", "guard", "--", "claude"},
		}
		if err := store.Park(rec); err != nil {
			b.Fatalf("setup Park: %v", err)
		}
		if _, err := store.Release(goal, "winner", "seat enrolled", now); err != nil {
			b.Fatalf("initial Release: %v", err)
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := store.Release(goal, "racer", "seat enrolled", now)
			if !errors.Is(err, ErrClaimed) {
				b.Fatalf("Release got %v, want ErrClaimed", err)
			}
		}
	})
}

func BenchmarkClaimDue(b *testing.B) {
	b.Run("due", func(b *testing.B) {
		dir := b.TempDir()
		store := Store{Dir: dir}
		now := time.Unix(1_800_000_000, 0)
		rec := Record{
			Schema:      Schema,
			Goal:        "bench-due",
			Account:     "seat-alpha",
			ParkedAt:    now.Unix() - 7200,
			ParkedUntil: now.Unix() - 3600,
			Command:     []string{"fak", "guard", "--", "claude"},
		}
		data, err := json.MarshalIndent(rec, "", "  ")
		if err != nil {
			b.Fatalf("marshal: %v", err)
		}
		data = append(data, '\n')

		goals := make([]string, b.N)
		for i := 0; i < b.N; i++ {
			goals[i] = fmt.Sprintf("due-goal-%d", i)
			if err := os.WriteFile(store.path(goals[i]), data, 0o600); err != nil {
				b.Fatalf("setup WriteFile: %v", err)
			}
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := store.ClaimDue(goals[i], "supervisor", now); err != nil {
				b.Fatalf("ClaimDue: %v", err)
			}
		}
	})

	b.Run("not_due", func(b *testing.B) {
		dir := b.TempDir()
		store := Store{Dir: dir}
		now := time.Unix(1_800_000_000, 0)
		const goal = "not-due-goal"
		rec := Record{
			Goal:        goal,
			Account:     "seat-alpha",
			ParkedAt:    now.Unix(),
			ParkedUntil: now.Unix() + 7200,
			Command:     []string{"fak", "guard", "--", "claude"},
		}
		if err := store.Park(rec); err != nil {
			b.Fatalf("setup Park: %v", err)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := store.ClaimDue(goal, "supervisor", now)
			if !errors.Is(err, ErrNotDue) {
				b.Fatalf("ClaimDue got %v, want ErrNotDue", err)
			}
		}
	})
}

func BenchmarkLoad(b *testing.B) {
	dir := b.TempDir()
	store := Store{Dir: dir}
	now := time.Unix(1_800_000_000, 0)
	rec := Record{
		Goal:        "load-goal",
		Lane:        "guard",
		Account:     "seat-alpha",
		Reason:      "LONG_RETRY_AFTER",
		ParkedAt:    now.Unix(),
		ParkedUntil: now.Unix() + 7200,
		Command:     []string{"fak", "guard", "--", "claude"},
	}
	if err := store.Park(rec); err != nil {
		b.Fatalf("setup Park: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r, err := store.Load("load-goal")
		if err != nil {
			b.Fatalf("Load: %v", err)
		}
		if r.Goal != "load-goal" {
			b.Fatalf("unexpected goal: %s", r.Goal)
		}
	}
}

func BenchmarkList(b *testing.B) {
	dir := b.TempDir()
	store := Store{Dir: dir}
	now := time.Unix(1_800_000_000, 0)
	for i := 0; i < 20; i++ {
		rec := Record{
			Goal:        fmt.Sprintf("list-goal-%d", i),
			Lane:        "guard",
			Account:     fmt.Sprintf("seat-%d", i%3),
			Reason:      "LONG_RETRY_AFTER",
			ParkedAt:    now.Unix(),
			ParkedUntil: now.Unix() + 7200,
			Command:     []string{"fak", "guard", "--", "claude"},
		}
		if err := store.Park(rec); err != nil {
			b.Fatalf("setup Park: %v", err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		records, err := store.List()
		if err != nil {
			b.Fatalf("List: %v", err)
		}
		if len(records) != 20 {
			b.Fatalf("expected 20 records, got %d", len(records))
		}
	}
}

func BenchmarkAdmitProbe(b *testing.B) {
	dir := b.TempDir()
	store := Store{Dir: dir}
	now := time.Unix(1_800_000_000, 0)
	rec := Record{
		Schema:      Schema,
		Goal:        "bench-probe",
		Account:     "seat-alpha",
		ParkedAt:    now.Unix(),
		ParkedUntil: now.Unix() + 14400,
		Command:     []string{"fak", "guard", "--", "claude"},
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		b.Fatalf("marshal: %v", err)
	}
	data = append(data, '\n')

	goals := make([]string, b.N)
	for i := 0; i < b.N; i++ {
		goals[i] = fmt.Sprintf("probe-goal-%d", i)
		if err := os.WriteFile(store.path(goals[i]), data, 0o600); err != nil {
			b.Fatalf("setup WriteFile: %v", err)
		}
	}

	probeTime := now.Add(time.Hour)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, ok := store.AdmitProbe(goals[i], "prober", probeTime)
		if !ok {
			b.Fatalf("AdmitProbe failed for %s", goals[i])
		}
	}
}

func BenchmarkBlocks(b *testing.B) {
	now := time.Unix(1_800_000_000, 0)
	rec := Record{
		Schema:      Schema,
		Goal:        "blocks-goal",
		Account:     "seat-alpha",
		ParkedAt:    now.Unix(),
		ParkedUntil: now.Unix() + 3600,
	}

	b.Run("matching_account_blocked", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !rec.Blocks("seat-alpha", now) {
				b.Fatal("expected Blocks=true")
			}
		}
	})

	b.Run("sibling_account_unblocked", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if rec.Blocks("seat-beta", now) {
				b.Fatal("expected Blocks=false")
			}
		}
	})

	b.Run("whitespace_trimmed_match", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !rec.Blocks("  seat-alpha  ", now) {
				b.Fatal("expected Blocks=true")
			}
		}
	})
}

func BenchmarkParseRetryAfter(b *testing.B) {
	now := time.Unix(1_800_000_000, 0)

	b.Run("seconds", func(b *testing.B) {
		const val = "7200"
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			t, ok := ParseRetryAfter(val, now)
			if !ok || t.Unix() != now.Unix()+7200 {
				b.Fatalf("ParseRetryAfter: %v, %v", t, ok)
			}
		}
	})

	b.Run("http_date", func(b *testing.B) {
		val := now.Add(2 * time.Hour).UTC().Format(http.TimeFormat)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			t, ok := ParseRetryAfter(val, now)
			if !ok || t.Unix() != now.Unix()+7200 {
				b.Fatalf("ParseRetryAfter: %v, %v", t, ok)
			}
		}
	})
}
