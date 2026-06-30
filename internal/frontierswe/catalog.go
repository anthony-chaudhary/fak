package frontierswe

import "sort"

// The canonical FrontierSWE task roster: the exact 17 task names and their
// scoring family, mirroring the IMPLEMENTATION / PERFORMANCE / ML_RESEARCH
// lists in upstream scripts/score_from_reward.py. These are the source of
// truth for ScoringCategory; task_test.go cross-checks them against the
// upstream lists so a drift between this catalog and the scorer fails the build.

// implementationTasks is the upstream IMPLEMENTATION list (5 tasks).
var implementationTasks = []string{
	"git-to-zig",
	"dart-style-haskell",
	"lua-native-compiler",
	"postgres-sqlite-wire-adapter",
	"modular-stack-wan21",
}

// performanceTasks is the upstream PERFORMANCE list (9 tasks).
var performanceTasks = []string{
	"libexpat-to-x86asm",
	"ffmpeg-swscale-rewrite",
	"pyright-type-checking-optimization",
	"granite-mamba2-inference-optimization",
	"notebook-compression",
	"revideo-perf-opt",
	"cranelift-codegen-opt",
	"dependent-type-checker",
	"inference-system-optimization",
}

// mlResearchTasks is the upstream ML_RESEARCH list (3 tasks).
var mlResearchTasks = []string{
	"pcqm4mv2-autoresearch",
	"frogsgame-rl",
	"optimizer-design",
}

// catalog maps every known task name to its scoring Category. Built once at
// package init from the three upstream lists.
var catalog = func() map[string]Category {
	m := make(map[string]Category, 17)
	for _, n := range implementationTasks {
		m[n] = CategoryImplementation
	}
	for _, n := range performanceTasks {
		m[n] = CategoryPerformance
	}
	for _, n := range mlResearchTasks {
		m[n] = CategoryMLResearch
	}
	return m
}()

// CategoryOf returns the scoring Category for a task name and whether the name
// is one of the 17 known FrontierSWE tasks. An unknown name returns ("", false).
func CategoryOf(name string) (Category, bool) {
	c, ok := catalog[name]
	return c, ok
}

// Catalog returns the full task->category roster as a fresh map, so callers may
// inspect or iterate it without mutating the package's source of truth.
func Catalog() map[string]Category {
	out := make(map[string]Category, len(catalog))
	for k, v := range catalog {
		out[k] = v
	}
	return out
}

// TaskNames returns all 17 known task names in sorted order (deterministic,
// independent of map iteration).
func TaskNames() []string {
	out := make([]string, 0, len(catalog))
	for n := range catalog {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// TaskNamesByCategory returns the known task names in category c, in the
// upstream list order. An unknown category returns nil.
func TaskNamesByCategory(c Category) []string {
	switch c {
	case CategoryImplementation:
		return append([]string(nil), implementationTasks...)
	case CategoryPerformance:
		return append([]string(nil), performanceTasks...)
	case CategoryMLResearch:
		return append([]string(nil), mlResearchTasks...)
	default:
		return nil
	}
}
