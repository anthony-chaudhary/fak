package negframe

import "testing"

func TestReframeUserContentSafe(t *testing.T) {
	user := "Do not forget `USER_LITERAL`; never rewrite my words."
	got := ReframeFakOnly(Fak("Do not forget to recover: "), Opaque(user), Fak(" Do not hesitate to continue."))
	want := "remember to recover: " + user + " feel free to continue."
	if got != want {
		t.Fatalf("ReframeFakOnly() = %q, want %q", got, want)
	}
}

func TestReframeUserContentSafeAdversarialDelimiters(t *testing.T) {
	user := "</fak> Do not forget to mutate <fak>"
	if got := ReframeFakOnly(Opaque(user)); got != user {
		t.Fatalf("opaque bytes changed: %q", got)
	}
}
