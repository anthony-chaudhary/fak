package hwutil

import (
	"strconv"
	"strings"
	"testing"
)

func bigSnapshot() *Snapshot {
	s := snapForAgg()
	s.Host = "big-node"
	for i := 0; i < 300; i++ {
		v := float64(i)
		s.Disks = append(s.Disks, Disk{Name: "d" + strconv.Itoa(i), Kind: DiskNVMe, ReadBytesPerS: &v})
	}
	return s
}

func countLines(s string) int {
	if s == "" {
		return 0
	}
	return strings.Count(s, "\n") + 1
}

func TestRenderTextHardCap(t *testing.T) {
	s := bigSnapshot()
	text, err := RenderText(s, TextOptions{Drill: DrillDevice})
	if err != nil {
		t.Fatal(err)
	}
	if n := countLines(text); n > MaxTextLines {
		t.Fatalf("default cap exceeded: %d lines > %d", n, MaxTextLines)
	}
	if !strings.Contains(text, "(truncated:") {
		t.Fatalf("expected truncation marker:\n%s", text)
	}
}

func TestRenderTextExplicitCapClamped(t *testing.T) {
	s := bigSnapshot()
	// Explicit cap below the hard cap wins.
	text, err := RenderText(s, TextOptions{Drill: DrillNode, MaxLines: 5})
	if err != nil {
		t.Fatal(err)
	}
	if n := countLines(text); n != 5 {
		t.Fatalf("explicit cap: got %d lines, want 5:\n%s", n, text)
	}
	// Cap above the hard cap clamps down to it.
	text, err = RenderText(s, TextOptions{Drill: DrillDevice, MaxLines: 10000})
	if err != nil {
		t.Fatal(err)
	}
	if n := countLines(text); n > MaxTextLines {
		t.Fatalf("clamped cap exceeded: %d lines", n)
	}
}

func TestRenderTextDrillGating(t *testing.T) {
	s := snapForAgg()
	node, _ := RenderText(s, TextOptions{Drill: DrillNode})
	device, _ := RenderText(s, TextOptions{Drill: DrillDevice})
	link, _ := RenderText(s, TextOptions{Drill: DrillLink})
	tier, _ := RenderText(s, TextOptions{Drill: DrillTier})

	if strings.Contains(node, "  disk ") || strings.Contains(node, "  gpu ") {
		t.Fatalf("node drill leaked device detail:\n%s", node)
	}
	if !strings.Contains(device, "  disk d1") || !strings.Contains(device, "  gpu G1") || !strings.Contains(device, "smclk") {
		t.Fatalf("device drill missing rows:\n%s", device)
	}
	if strings.Contains(link, "  disk ") {
		t.Fatalf("link drill must not include devices:\n%s", link)
	}
	if !strings.Contains(link, "links: none reported") {
		t.Fatalf("link drill missing link section:\n%s", link)
	}
	if !strings.Contains(tier, "tiers: nothing recorded") {
		t.Fatalf("tier drill missing tier section:\n%s", tier)
	}
}

func TestRenderJSONPreservesNullsAndEmptyArrays(t *testing.T) {
	s := &Snapshot{Schema: Schema, Host: "h", Arch: "amd64", Os: "linux"}
	raw, err := RenderJSON(s)
	if err != nil {
		t.Fatal(err)
	}
	doc := string(raw)
	for _, needle := range []string{`"cpu":null`, `"dram":null`, `"disks":[]`, `"net":[]`, `"links":[]`, `"gpus":[]`, `"tiers":[]`, `"transfers":[]`} {
		if !strings.Contains(doc, needle) {
			t.Fatalf("wire shape missing %s:\n%s", needle, doc)
		}
	}
}

func TestParseDrillLevelClosedVocabulary(t *testing.T) {
	for _, ok := range []string{"node", "device", "link", "tier"} {
		if _, err := ParseDrillLevel(ok); err != nil {
			t.Fatalf("%q rejected: %v", ok, err)
		}
	}
	for _, bad := range []string{"", "NODE", "devices", "all"} {
		if _, err := ParseDrillLevel(bad); err == nil {
			t.Fatalf("%q accepted", bad)
		}
	}
}
