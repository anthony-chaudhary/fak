package billing_test

import (
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/billing"
)

// 2026-12-31 and 2027-01-01 both fall in ISO week 2026-W53 but in different
// calendar months; 2027-01-04 opens ISO week 2027-W01.
var (
	isoW53Dec31 = time.Date(2026, 12, 31, 12, 0, 0, 0, time.UTC)
	isoW53Jan01 = time.Date(2027, 1, 1, 12, 0, 0, 0, time.UTC)
	isoW01Jan04 = time.Date(2027, 1, 4, 12, 0, 0, 0, time.UTC)
)

// fak-test:runtime fast est=1ms lane=default
func TestSpendCapsWeeklySharedAcrossYearEndWhileMonthlyResets(t *testing.T) {
	caps := billing.NewSpendCaps(billing.CapConfig{WeeklyTokenCap: 1_000, MonthlyTokenCap: 1_000})
	p := billing.Principal{TenantID: "t"}

	caps.Add(p, 300, isoW53Dec31)
	caps.Add(p, 200, isoW53Jan01)

	_, weekly, monthly, _ := caps.SpentWindows(p, isoW53Jan01)
	if weekly != 500 {
		t.Fatalf("weekly spend on 2027-01-01 = %d, want 500 shared across ISO 2026-W53", weekly)
	}
	if monthly != 200 {
		t.Fatalf("monthly spend on 2027-01-01 = %d, want 200 after the month reset", monthly)
	}

	_, weekly, monthly, _ = caps.SpentWindows(p, isoW01Jan04)
	if weekly != 0 {
		t.Fatalf("weekly spend in 2027-W01 = %d, want 0 after the week reset", weekly)
	}
	if monthly != 200 {
		t.Fatalf("monthly spend on 2027-01-04 = %d, want 200 still in January", monthly)
	}

	caps.Add(p, 50, isoW01Jan04)
	_, weekly, monthly, _ = caps.SpentWindows(p, isoW01Jan04)
	if weekly != 50 || monthly != 250 {
		t.Fatalf("after 2027-W01 add weekly=%d monthly=%d, want 50/250", weekly, monthly)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpendCapsWeeklyBreachAcrossYearEndThenClears(t *testing.T) {
	caps := billing.NewSpendCaps(billing.CapConfig{WeeklyTokenCap: 500})
	p := billing.Principal{TenantID: "t"}

	caps.Add(p, 300, isoW53Dec31)
	if caps.Breached(p, isoW53Jan01) {
		t.Fatal("300 of 500 weekly tokens reports breached")
	}
	caps.Add(p, 200, isoW53Jan01)
	if !caps.Breached(p, isoW53Jan01) {
		t.Fatal("500 of 500 weekly tokens across the year boundary does not breach")
	}
	if caps.Breached(p, isoW01Jan04) {
		t.Fatal("a new ISO week still reports the previous week's breach")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpendCapsMonthlyBreachResetsAtMonthBoundary(t *testing.T) {
	caps := billing.NewSpendCaps(billing.CapConfig{MonthlyTokenCap: 400})
	p := billing.Principal{TenantID: "t"}

	caps.Add(p, 400, isoW53Dec31)
	if !caps.Breached(p, isoW53Dec31) {
		t.Fatal("400 of 400 monthly tokens does not breach")
	}
	if caps.Breached(p, isoW53Jan01) {
		t.Fatal("January still reports December's monthly breach")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpendCapsUnconfiguredWindowsNeverBreach(t *testing.T) {
	caps := billing.NewSpendCaps(billing.CapConfig{})
	p := billing.Principal{TenantID: "t", SessionID: "s"}

	caps.Add(p, 1<<40, isoW53Dec31)
	if caps.Breached(p, isoW53Dec31) {
		t.Fatal("a config with no ceilings breached")
	}
	daily, weekly, monthly, session := caps.SpentWindows(p, isoW53Dec31)
	if daily != 0 || weekly != 0 || monthly != 0 || session != 0 {
		t.Fatalf("untracked axes report spend daily=%d weekly=%d monthly=%d session=%d", daily, weekly, monthly, session)
	}

	dailyOnly := billing.NewSpendCaps(billing.CapConfig{DailyTokenCap: 1 << 50})
	dailyOnly.Add(p, 1<<40, isoW53Dec31)
	if dailyOnly.Breached(p, isoW53Dec31) {
		t.Fatal("a daily ceiling far above spend breached through an unconfigured weekly or monthly axis")
	}
	_, weekly, monthly, _ = dailyOnly.SpentWindows(p, isoW53Dec31)
	if weekly != 0 || monthly != 0 {
		t.Fatalf("unconfigured weekly/monthly tracked spend weekly=%d monthly=%d", weekly, monthly)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpendCapsDailyBehaviourUnchangedByWindows(t *testing.T) {
	caps := billing.NewSpendCaps(billing.CapConfig{DailyTokenCap: 100, WeeklyTokenCap: 1_000, MonthlyTokenCap: 1_000})
	p := billing.Principal{TenantID: "t"}

	caps.Add(p, 100, isoW53Dec31)
	if !caps.Breached(p, isoW53Dec31) {
		t.Fatal("100 of 100 daily tokens does not breach")
	}
	if caps.Breached(p, isoW53Jan01) {
		t.Fatal("the next day still breached although weekly and monthly are under their ceilings")
	}
	daily, _ := caps.Spent(p, isoW53Dec31)
	dailyW, weekly, monthly, _ := caps.SpentWindows(p, isoW53Dec31)
	if daily != 100 || dailyW != daily {
		t.Fatalf("Spent daily=%d SpentWindows daily=%d, want 100 from both", daily, dailyW)
	}
	if weekly != 100 || monthly != 100 {
		t.Fatalf("weekly=%d monthly=%d, want 100/100", weekly, monthly)
	}
	if got, _ := caps.Spent(p, isoW53Jan01); got != 0 {
		t.Fatalf("daily spend on the next day = %d, want 0", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSpendCapsWindowsConcurrentAdd(t *testing.T) {
	caps := billing.NewSpendCaps(billing.CapConfig{
		DailyTokenCap: 1_000_000, WeeklyTokenCap: 1_000_000, MonthlyTokenCap: 1_000_000, SessionTokenCap: 1_000_000,
	})
	p := billing.Principal{TenantID: "t", SessionID: "s"}

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 500; i++ {
				caps.Add(p, 1, isoW53Dec31)
				_ = caps.Breached(p, isoW53Dec31)
				_, _, _, _ = caps.SpentWindows(p, isoW53Dec31)
			}
		}()
	}
	wg.Wait()

	daily, weekly, monthly, session := caps.SpentWindows(p, isoW53Dec31)
	if daily != 4000 || weekly != 4000 || monthly != 4000 || session != 4000 {
		t.Fatalf("concurrent totals daily=%d weekly=%d monthly=%d session=%d, want 4000 each", daily, weekly, monthly, session)
	}
	if caps.Breached(p, isoW53Dec31) {
		t.Fatal("totals far below every ceiling report breached")
	}
}
