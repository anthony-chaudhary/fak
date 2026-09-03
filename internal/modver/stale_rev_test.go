package modver

import (
	"testing"
)

func TestExtractModuleRevClaims(t *testing.T) {
	text := `Recalled context:
We verified that internal/gateway@r652+g1f75c56d fixed the timeout issue.
Also checked cmd/fak@r100 and internal/modver@r42+gabcdef12.
Malformed tokens like foo@bar and @r10 should be ignored.
Duplicate reference: internal/gateway@r652+g1f75c56d should be deduplicated.`

	claims := ExtractModuleRevClaims(text)
	if len(claims) != 3 {
		t.Fatalf("len(claims) = %d, want 3: %+v", len(claims), claims)
	}

	expected := map[string]struct {
		rev    int
		commit string
	}{
		"internal/gateway": {rev: 652, commit: "1f75c56d"},
		"cmd/fak":          {rev: 100, commit: ""},
		"internal/modver":  {rev: 42, commit: "abcdef12"},
	}

	for _, c := range claims {
		exp, ok := expected[c.Module]
		if !ok {
			t.Errorf("unexpected module claim %q", c.Module)
			continue
		}
		if c.Rev != exp.rev {
			t.Errorf("module %q rev = %d, want %d", c.Module, c.Rev, exp.rev)
		}
		if c.Commit != exp.commit {
			t.Errorf("module %q commit = %q, want %q", c.Module, c.Commit, exp.commit)
		}
	}
}

func TestCheckModuleRevClaims(t *testing.T) {
	live := &Report{
		Head:       "abc1234",
		AppVersion: "0.45.0",
		Modules: []Module{
			{Name: "internal/gateway", Rev: 652, LastCommit: "1f75c56d"},
			{Name: "cmd/fak", Rev: 120, LastCommit: "9876fedc"},
			{Name: "internal/modver", Rev: 42, LastCommit: "abcdef12"},
		},
	}

	t.Run("empty claims", func(t *testing.T) {
		adv := CheckModuleRevClaims(nil, live)
		if !adv.Advisory || adv.StaleCount != 0 || adv.ReasonClass != "" {
			t.Fatalf("unexpected advisory for empty claims: %+v", adv)
		}
	})

	t.Run("fresh claims", func(t *testing.T) {
		claims := []ModuleRevClaim{
			{Module: "internal/gateway", Rev: 652},
			{Module: "internal/modver", Rev: 45}, // newer than live (e.g. forward branch)
		}
		adv := CheckModuleRevClaims(claims, live)
		if adv.StaleCount != 0 {
			t.Errorf("StaleCount = %d, want 0", adv.StaleCount)
		}
		if adv.ReasonClass != "" {
			t.Errorf("ReasonClass = %q, want empty", adv.ReasonClass)
		}
		for _, f := range adv.Findings {
			if f.Status != ModuleRevFresh {
				t.Errorf("module %s status = %v, want fresh", f.Claim.Module, f.Status)
			}
		}
	})

	t.Run("stale claim triggers advisory MODULE_REV_STALE", func(t *testing.T) {
		claims := []ModuleRevClaim{
			{Module: "internal/gateway", Rev: 652}, // fresh
			{Module: "cmd/fak", Rev: 100},          // stale (live is 120)
			{Module: "unknown/module", Rev: 5},     // unverifiable
		}
		adv := CheckModuleRevClaims(claims, live)
		if !adv.Advisory {
			t.Error("expected Advisory=true (advisory-first policy)")
		}
		if adv.StaleCount != 1 {
			t.Fatalf("StaleCount = %d, want 1", adv.StaleCount)
		}
		if adv.ReasonClass != ReasonModuleRevStale {
			t.Fatalf("ReasonClass = %q, want %q", adv.ReasonClass, ReasonModuleRevStale)
		}

		findingMap := make(map[string]ModuleRevFinding)
		for _, f := range adv.Findings {
			findingMap[f.Claim.Module] = f
		}

		gw := findingMap["internal/gateway"]
		if gw.Status != ModuleRevFresh {
			t.Errorf("gateway status = %v, want fresh", gw.Status)
		}

		fak := findingMap["cmd/fak"]
		if fak.Status != ModuleRevStale {
			t.Errorf("cmd/fak status = %v, want stale", fak.Status)
		}
		if fak.ReasonClass != ReasonModuleRevStale {
			t.Errorf("cmd/fak reason class = %q, want %q", fak.ReasonClass, ReasonModuleRevStale)
		}
		if fak.CurrentRev != 120 {
			t.Errorf("cmd/fak current rev = %d, want 120", fak.CurrentRev)
		}

		unk := findingMap["unknown/module"]
		if unk.Status != ModuleRevUnverifiable {
			t.Errorf("unknown module status = %v, want unverifiable", unk.Status)
		}
	})
}

func TestCheckRecallText(t *testing.T) {
	live := &Report{
		Head: "headsha",
		Modules: []Module{
			{Name: "internal/adjudicator", Rev: 85},
			{Name: "internal/policy", Rev: 40},
		},
	}

	text := "Past run recalled that internal/adjudicator@r80 fixed decision drift and internal/policy@r40 held."
	adv := CheckRecallText(text, live)

	if adv.StaleCount != 1 {
		t.Fatalf("StaleCount = %d, want 1", adv.StaleCount)
	}
	if adv.ReasonClass != "MODULE_REV_STALE" {
		t.Fatalf("ReasonClass = %q, want MODULE_REV_STALE", adv.ReasonClass)
	}
}
