package testroute

// Kind is the closed set of test execution routes the environment preflight can
// recommend without doing any probing itself.
type Kind string

const (
	KindNative      Kind = "native"
	KindWSL         Kind = "wsl"
	KindCI          Kind = "ci"
	KindUnavailable Kind = "unavailable"
)

const ArgsPlaceholder = "{go_test_args}"

// Probe is the caller-supplied host evidence. It is deliberately data-only: this
// package never shells out, checks PATH, reaches the network, or reads the host.
type Probe struct {
	GOOS              string
	NativeTestAllowed bool
	WSLPresent        bool
	CIReachable       bool
}

// Route is the pure routing decision. CommandTemplate names the command prefix and
// includes ArgsPlaceholder where the caller should splice the resolved go-test args.
type Route struct {
	Kind            Kind     `json:"kind"`
	CommandTemplate []string `json:"command_template"`
	Reason          string   `json:"reason"`
}

// Decide folds a probe into the cheapest usable test route. Native execution wins
// whenever the caller says it is allowed; Windows with native test binaries blocked
// prefers the repo's WSL wrapper; CI is the last executable fallback.
func Decide(p Probe) Route {
	if p.NativeTestAllowed {
		return Route{
			Kind:            KindNative,
			CommandTemplate: []string{"go", "test", ArgsPlaceholder},
			Reason:          "native go test is allowed on this host",
		}
	}
	if p.GOOS == "windows" && p.WSLPresent {
		return Route{
			Kind:            KindWSL,
			CommandTemplate: []string{"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "test.ps1", ArgsPlaceholder},
			Reason:          "windows native go test is blocked; route through WSL via test.ps1",
		}
	}
	if p.CIReachable {
		return Route{
			Kind:            KindCI,
			CommandTemplate: []string{"gh", "workflow", "run", "ci.yml", ArgsPlaceholder},
			Reason:          "local test routes unavailable; use CI as the fallback witness",
		}
	}
	return Route{
		Kind:            KindUnavailable,
		CommandTemplate: nil,
		Reason:          "no allowed native, WSL, or CI test route is available",
	}
}

// Command expands a route template by replacing ArgsPlaceholder with args.
func Command(template, args []string) []string {
	out := make([]string, 0, len(template)+len(args))
	for _, part := range template {
		if part == ArgsPlaceholder {
			out = append(out, args...)
			continue
		}
		out = append(out, part)
	}
	return out
}
