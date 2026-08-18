package markerblock

import "testing"

func TestExtractAndSplice(t *testing.T) {
	const (
		begin = "<!-- begin -->"
		end   = "<!-- end -->"
		doc   = "head\n<!-- begin -->\nstale\n<!-- end -->\ntail\n"
	)
	got, ok := Extract(doc, begin, end)
	if !ok || got != "<!-- begin -->\nstale\n<!-- end -->" {
		t.Fatalf("Extract() = %q, %v", got, ok)
	}
	replaced, err := Splice(doc, begin, end, begin+"\nfresh\n"+end)
	if err != nil {
		t.Fatalf("Splice: %v", err)
	}
	want := "head\n<!-- begin -->\nfresh\n<!-- end -->\ntail\n"
	if replaced != want {
		t.Fatalf("Splice() = %q, want %q", replaced, want)
	}
}

func TestMissingMarkers(t *testing.T) {
	if _, ok := Extract("plain", "begin", "end"); ok {
		t.Fatal("Extract found absent markers")
	}
	if _, err := Splice("plain", "begin", "end", "new"); err == nil || err.Error() != "begin marker not found: begin" {
		t.Fatalf("missing begin error = %v", err)
	}
	if _, err := Splice("begin only", "begin", "end", "new"); err == nil || err.Error() != "end marker not found after begin marker: end" {
		t.Fatalf("missing end error = %v", err)
	}
}
