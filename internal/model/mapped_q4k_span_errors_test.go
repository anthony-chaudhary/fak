package model

import "testing"

func TestMappedQ4KSpanErrors(t *testing.T) {
	rangeErr := (&mappedQ4KSpanRangeError{Offset: 7, Length: 11, FileSize: 13, Reason: "outside file"}).Error()
	if want := "model: invalid mapped Q4_K span offset=7 length=11 file_size=13: outside file"; rangeErr != want {
		t.Fatalf("range error = %q, want %q", rangeErr, want)
	}
	unavailable := (&mappedQ4KSpanUnavailableError{GOOS: "plan9"}).Error()
	if want := "model: mapped Q4_K spans unavailable on plan9"; unavailable != want {
		t.Fatalf("unavailable error = %q, want %q", unavailable, want)
	}
}
