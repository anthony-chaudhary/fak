package main

// Fixture dispatch switch for internal/ctxplans tests. testdata is ignored by the
// go tool, so this need not build — it is read as text by dispatchVerbs.

import "os"

func main() {
	switch os.Args[1] {
	case "session": // context verb — DECLARED (session.go)
		cmdSession(os.Args[2:])
	case "vcache": // context verb — DECLARED (vcache.go)
		cmdVCache(os.Args[2:])
	case "guard": // non-context NAME — DECLARED via directive, so it joins the population
		cmdGuard(os.Args[2:])
	case "headroom": // context verb — UNDECLARED (debt)
		cmdHeadroom(os.Args[2:])
	case "recall": // context verb — PARTIAL directive only (missing warms=) → UNDECLARED (debt)
		cmdRecall(os.Args[2:])
	case "widget": // non-context, undeclared — ignored (not a context surface)
		cmdWidget(os.Args[2:])
	}
}
