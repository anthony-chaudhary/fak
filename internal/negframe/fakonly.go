package negframe

import "strings"

// Fragment is one provenance-labelled piece of an emitted message. Fak-authored
// fragments are eligible for positive-voice rewriting; external/user fragments
// are opaque and survive byte-for-byte.
type Fragment struct {
	Text        string
	FakAuthored bool
}

func Fak(text string) Fragment    { return Fragment{Text: text, FakAuthored: true} }
func Opaque(text string) Fragment { return Fragment{Text: text} }

// ReframeFakOnly applies Reframe independently to fak-authored fragments and
// concatenates opaque fragments unchanged. Reframing before interpolation makes
// provenance structural rather than relying on delimiters an adversarial user
// could forge.
func ReframeFakOnly(fragments ...Fragment) string {
	var b strings.Builder
	for _, f := range fragments {
		if f.FakAuthored {
			b.WriteString(Reframe(f.Text))
		} else {
			b.WriteString(f.Text)
		}
	}
	return b.String()
}
