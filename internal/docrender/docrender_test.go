package docrender

import "testing"

func TestReady(t *testing.T) {
	if !Ready() {
		t.Fatal("Ready() should report true for the generated skeleton")
	}
}

func TestScanAlternateThematicBreaks(t *testing.T) {
	for _, source := range []string{"***", "* * *", "___", "_ _ _"} {
		items := Scan(source)
		if len(items) != 1 || items[0].Construct != "alternate thematic break" {
			t.Fatalf("Scan(%q) = %+v", source, items)
		}
	}
	if items := Scan("* _ *"); len(items) != 0 {
		t.Fatalf("mixed markers = %+v", items)
	}
}
