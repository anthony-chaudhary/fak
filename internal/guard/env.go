package guard

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/httptrust"
	"github.com/anthony-chaudhary/fak/internal/secretload"
)

// landlockChildEnv builds the environment for the agent the Landlock trampoline
// execs: the deny-by-default sandbox allow-list, plus the caller's explicit
// entries, plus (#8172) the per-runtime trust variables derived from a declared
// corporate CA bundle.
//
// The derivation belongs here rather than in the allow-list because the two answer
// different questions. The allow-list decides whether a variable the OPERATOR set
// may cross the boundary; this decides what to set when they set nothing, which is
// the common case precisely because no operator knows all six variable names off
// hand. httptrust.ChildEnv skips every name already present, so an operator's own
// value is never overridden — and with no bundle declared it returns nothing and
// this is the historical call unchanged.
func landlockChildEnv(extra ...string) []string {
	environ := os.Environ()
	if src := httptrust.Resolve(); src.Usable() {
		extra = append(httptrust.ChildEnv(src.Path, environ), extra...)
	}
	return secretload.SandboxEnv(environ, extra...)
}
