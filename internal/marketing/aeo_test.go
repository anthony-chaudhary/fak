package marketing

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func aeoShips() []Ship {
	return []Ship{
		{SHA: "aaaa1111", Leaf: "gateway", Kind: "trailer", Subject: "feat(gateway): add reclaim path (fak gateway)", Date: time.Date(2026, 6, 28, 0, 0, 0, 0, time.UTC)},
		{SHA: "bbbb2222", Leaf: "model", Kind: "direct", Subject: "fak/model: implement Q4_K reducer", Date: time.Date(2026, 6, 27, 0, 0, 0, 0, time.UTC)},
	}
}

func TestUpdatesFeedIsValidItemListWithShas(t *testing.T) {
	b, err := UpdatesFeed(aeoShips(), time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatalf("UpdatesFeed: %v", err)
	}
	var feed map[string]any
	if err := json.Unmarshal(b, &feed); err != nil {
		t.Fatalf("feed is not valid JSON: %v\n%s", err, b)
	}
	if feed["@type"] != "ItemList" {
		t.Errorf("@type = %v, want ItemList", feed["@type"])
	}
	items, ok := feed["itemListElement"].([]any)
	if !ok || len(items) != 2 {
		t.Fatalf("itemListElement = %v, want 2 items", feed["itemListElement"])
	}
	// every item must cite its commit sha (the witness)
	s := string(b)
	for _, sha := range []string{"aaaa1111", "bbbb2222"} {
		if !strings.Contains(s, sha) {
			t.Errorf("feed missing commit sha %q:\n%s", sha, s)
		}
	}
	// newest-first: gateway (06-28) at position 1
	first := items[0].(map[string]any)
	if first["position"].(float64) != 1 {
		t.Errorf("first item position = %v, want 1", first["position"])
	}
}

func TestUpdatesFeedEmptyIsValid(t *testing.T) {
	b, err := UpdatesFeed(nil, time.Time{})
	if err != nil {
		t.Fatalf("UpdatesFeed(nil): %v", err)
	}
	var feed map[string]any
	if err := json.Unmarshal(b, &feed); err != nil {
		t.Fatalf("empty feed invalid JSON: %v", err)
	}
	if feed["@type"] != "ItemList" {
		t.Errorf("empty feed @type = %v, want ItemList", feed["@type"])
	}
}

func TestWhatsNewMarkdownCitesShaAndIsStable(t *testing.T) {
	md1 := WhatsNewMarkdown(aeoShips())
	md2 := WhatsNewMarkdown(aeoShips())
	if md1 != md2 {
		t.Error("WhatsNewMarkdown not stable across calls (idempotence broken)")
	}
	for _, sha := range []string{"aaaa1111", "bbbb2222"} {
		if !strings.Contains(md1, sha) {
			t.Errorf("What's-new missing sha %q:\n%s", sha, md1)
		}
	}
	if !strings.Contains(md1, "2026-06-28") {
		t.Errorf("What's-new missing date:\n%s", md1)
	}
	// newest first
	if strings.Index(md1, "aaaa1111") > strings.Index(md1, "bbbb2222") {
		t.Errorf("What's-new not newest-first:\n%s", md1)
	}
}

func TestWhatsNewMarkdownEmptyIsHonest(t *testing.T) {
	md := WhatsNewMarkdown(nil)
	if !strings.Contains(strings.ToLower(md), "no witnessed ships") {
		t.Errorf("empty What's-new should say so, got: %q", md)
	}
}

func TestLlmsUpdatesTextHasHeaderAndBody(t *testing.T) {
	txt := LlmsUpdatesText(aeoShips(), time.Date(2026, 6, 28, 12, 0, 0, 0, time.UTC))
	if !strings.Contains(txt, "# fak — what shipped") {
		t.Errorf("llms-updates missing header:\n%s", txt)
	}
	if !strings.Contains(txt, "aaaa1111") {
		t.Errorf("llms-updates missing a ship sha:\n%s", txt)
	}
	if !strings.Contains(txt, "Updated: 2026-06-28") {
		t.Errorf("llms-updates missing the updated stamp:\n%s", txt)
	}
}
