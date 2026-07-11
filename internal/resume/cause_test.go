package resume

import "testing"

func TestDominantCause(t *testing.T) {
	got := DominantCause([]string{"STOPPED_APIERR", "AUTH", "STOPPED_APIERR"})
	if got.Cause != "STOPPED_APIERR" || got.Share != "2/3" {
		t.Fatalf("got=%+v", got)
	}
}
func TestDominantCauseTieDeterministic(t *testing.T) {
	got := DominantCause([]string{"Z", "A"})
	if got.Cause != "A" {
		t.Fatalf("got=%+v", got)
	}
}
