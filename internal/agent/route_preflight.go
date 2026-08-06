package agent

// ResolveToolRoute applies RunOptions to one tool and returns the exact engine route
// the owned loop will bind before kernel submit. It is the side-effect-free preflight
// used by command wiring witnesses; runtime dispatch uses the same resolveToolEngine.
func ResolveToolRoute(tool string, opts ...RunOption) (string, error) {
	return resolveRunConfig(opts).resolveToolEngine(tool)
}
