package greeting

import "testing"

func TestGreeting(t *testing.T) {
	if got := Greeting("fak"); got != "hello, fak" {
		t.Fatalf("got %q", got)
	}
}
