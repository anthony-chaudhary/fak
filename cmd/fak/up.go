package main

// cmdUp is the product entry point for the unified deployable runtime. Serve
// remains the implementation and flag authority so the two commands cannot
// drift into different engines, policy, metrics, or session lifecycles.
func cmdUp(argv []string) {
	cmdServe(argv)
}
