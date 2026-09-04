package accountprobe

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func BenchmarkLedgerAppend(b *testing.B) {
	rd := b.TempDir()
	entry := LedgerEntry{
		TS:          "2026-08-14T12:00:00Z",
		Account:     ".claude-seat-01",
		Tag:         "seat-01",
		Status:      "ACCESS",
		BlockReason: "organization inference disabled",
		Reason:      "paired_baseline_provider_access",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := AppendLedger(rd, entry); err != nil {
			b.Fatalf("AppendLedger failed: %v", err)
		}
	}
}

func BenchmarkLedgerRead(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_records", count), func(b *testing.B) {
			rd := b.TempDir()
			var buf strings.Builder
			for i := 0; i < count; i++ {
				buf.WriteString(fmt.Sprintf(`{"ts":"2026-08-14T12:00:00Z","account":"seat-%03d","status":"OK","tag":"seat-%03d","reason":"baseline_access"}`+"\n", i%25, i%25))
			}
			path := ProbeLedgerPath(rd)
			if err := os.MkdirAll(rd, 0o755); err != nil {
				b.Fatal(err)
			}
			if err := os.WriteFile(path, []byte(buf.String()), 0o644); err != nil {
				b.Fatal(err)
			}

			b.SetBytes(int64(buf.Len()))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				entries := ReadLedger(path)
				if len(entries) != count {
					b.Fatalf("got %d entries, want %d", len(entries), count)
				}
			}
		})
	}
}

func BenchmarkLastProbeByAccount(b *testing.B) {
	for _, records := range []int{50, 250, 1000} {
		b.Run(fmt.Sprintf("%d_records", records), func(b *testing.B) {
			rd := b.TempDir()
			var buf strings.Builder
			for i := 0; i < records; i++ {
				buf.WriteString(fmt.Sprintf(`{"ts":"2026-08-14T12:00:00Z","account":"seat-%02d","status":"OK"}`+"\n", i%20))
			}
			if err := os.MkdirAll(rd, 0o755); err != nil {
				b.Fatal(err)
			}
			if err := os.WriteFile(ProbeLedgerPath(rd), []byte(buf.String()), 0o644); err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				latest := LastProbeByAccount(rd)
				if len(latest) != 20 {
					b.Fatalf("got %d accounts, want 20", len(latest))
				}
			}
		})
	}
}

func BenchmarkCoverageProbe(b *testing.B) {
	rd := b.TempDir()
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	var buf strings.Builder
	for i := 0; i < 50; i++ {
		var age time.Duration
		if i%2 == 0 {
			age = 10 * time.Minute
		} else {
			age = 48 * time.Hour
		}
		ts := now.Add(-age).Format(time.RFC3339)
		buf.WriteString(fmt.Sprintf(`{"ts":%q,"account":"seat-%03d","status":"OK"}`+"\n", ts, i))
	}
	if err := os.MkdirAll(rd, 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(ProbeLedgerPath(rd), []byte(buf.String()), 0o644); err != nil {
		b.Fatal(err)
	}

	for _, seatCount := range []int{10, 50, 100} {
		seats := make([]string, seatCount)
		for i := 0; i < seatCount; i++ {
			seats[i] = fmt.Sprintf("seat-%03d", i)
		}
		b.Run(fmt.Sprintf("%d_seats", seatCount), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rep := GradeSeats(seats, rd, now)
				if len(rep.Seats) != seatCount {
					b.Fatalf("got %d seats, want %d", len(rep.Seats), seatCount)
				}
			}
		})
	}
}

func BenchmarkGradeSeat(b *testing.B) {
	rd := b.TempDir()
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	writeBenchLedger(b, rd,
		probeRow("fresh-seat", "OK", now.Add(-10*time.Minute)),
		probeRow("stale-seat", "LIMIT", now.Add(-48*time.Hour)),
	)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cov := GradeSeat("fresh-seat", rd, now)
		if cov.Health != SeatHealthFresh {
			b.Fatalf("health = %q, want probe-fresh", cov.Health)
		}
	}
}

func BenchmarkCoverageReportNote(b *testing.B) {
	rep := CoverageReport{
		LedgerPath:     "/fleet/registry/probe_ledger.jsonl",
		LedgerAccounts: 20,
		Sufficient:     true,
		Fresh:          10,
		Stale:          5,
		Never:          5,
		Seats:          make([]SeatCoverage, 0, 20),
	}
	for i := 0; i < 10; i++ {
		rep.Seats = append(rep.Seats, SeatCoverage{
			Account: fmt.Sprintf("seat-fresh-%02d", i),
			Health:  SeatHealthFresh,
			AgeMin:  float64(5 + i),
			HasAge:  true,
			Status:  "OK",
		})
	}
	for i := 0; i < 5; i++ {
		rep.Seats = append(rep.Seats, SeatCoverage{
			Account: fmt.Sprintf("seat-stale-%02d", i),
			Health:  SeatHealthStale,
			AgeMin:  float64(2880 + i*60),
			HasAge:  true,
			Status:  "LIMIT",
		})
	}
	for i := 0; i < 5; i++ {
		rep.Seats = append(rep.Seats, SeatCoverage{
			Account: fmt.Sprintf("seat-never-%02d", i),
			Health:  SeatHealthNever,
		})
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		note := rep.Note()
		if note == "" {
			b.Fatal("unexpected empty note")
		}
	}
}

func BenchmarkRecentProbeAgeMin(b *testing.B) {
	rd := b.TempDir()
	now := time.Date(2026, 8, 15, 12, 0, 0, 0, time.UTC)
	var buf strings.Builder
	for i := 0; i < 50; i++ {
		buf.WriteString(fmt.Sprintf(`{"ts":%q,"account":"seat-%02d","status":"OK"}`+"\n",
			now.Add(-time.Duration(i*10)*time.Minute).Format(time.RFC3339), i))
	}
	if err := os.MkdirAll(rd, 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(ProbeLedgerPath(rd), []byte(buf.String()), 0o644); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		age := RecentProbeAgeMin("seat-10", rd, now)
		if age == nil {
			b.Fatal("nil age")
		}
	}
}

func BenchmarkParseLedgerTime(b *testing.B) {
	formats := []struct {
		name string
		raw  string
	}{
		{"rfc3339_z", "2026-08-14T12:00:00Z"},
		{"rfc3339_subsecond", "2026-08-14T12:00:00.123456789Z"},
		{"offset", "2026-08-14T12:00:00+02:00"},
		{"naive", "2026-08-14T12:00:00"},
	}
	for _, tc := range formats {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				t := parseLedgerTime(tc.raw)
				if t == nil {
					b.Fatalf("failed to parse %q", tc.raw)
				}
			}
		})
	}
}

func BenchmarkResolveRegDir(b *testing.B) {
	root := b.TempDir()
	userHome := filepath.Join(root, "localappdata")
	cloneRoot := filepath.Join(root, "clone")
	userReg := filepath.Join(userHome, "Fleet", "registry")
	cloneReg := filepath.Join(cloneRoot, "tools", "_registry")

	if err := os.MkdirAll(userReg, 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.MkdirAll(cloneReg, 0o755); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(userReg, ledgerFile), []byte(`{"ts":"2026-08-14T12:00:00Z","account":"a","status":"OK"}`+"\n"), 0o644); err != nil {
		b.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cloneReg, sessionsFile), []byte(`{"sessions":[]}`), 0o644); err != nil {
		b.Fatal(err)
	}

	b.Setenv("FLEET_REG_DIR", "")
	b.Setenv("FLEET_STATE_DIR", "")
	b.Setenv("LOCALAPPDATA", userHome)
	b.Chdir(cloneRoot)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := ResolveRegDir()
		if c.Dir == "" {
			b.Fatal("empty dir")
		}
	}
}

func writeBenchLedger(b *testing.B, rd string, lines ...string) {
	b.Helper()
	if err := os.MkdirAll(rd, 0o755); err != nil {
		b.Fatal(err)
	}
	body := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(ProbeLedgerPath(rd), []byte(body), 0o644); err != nil {
		b.Fatal(err)
	}
}
