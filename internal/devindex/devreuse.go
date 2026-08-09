package devindex

// DevReuse is independent of binary ownership. It distinguishes reusable
// development patterns from machinery specific to fak or its private lab.
type DevReuse string

const (
	DevReuseNA         DevReuse = "not-applicable"
	DevReusePortable   DevReuse = "portable-pattern"
	DevReuseMaintainer DevReuse = "fak-maintainer"
	DevReuseLab        DevReuse = "lab-operations"
)

func (r DevReuse) valid() bool {
	return r == DevReuseNA || r == DevReusePortable || r == DevReuseMaintainer || r == DevReuseLab
}

var portableDevPatterns = map[string]string{
	"affected":     "dependency-aware affected-test selection is reusable across repositories",
	"blockers":     "typed blocker discovery is a reusable multi-agent coordination pattern",
	"buildcheck":   "peer-dirty-safe compile checking is a reusable shared-checkout pattern",
	"ci-preflight": "clean-tip verification separates committed evidence from working-tree state",
	"commit":       "path-scoped serialized commits are a reusable shared-checkout pattern",
	"dispatch":     "contract-based work dispatch is a reusable multi-agent pattern",
	"done":         "witnessed completion is a reusable agent-workflow pattern",
	"hooks":        "tool-call and git-hook enforcement is a reusable policy pattern",
	"issue":        "contract-shaped issue creation and fan-out are reusable planning patterns",
	"project":      "typed project decomposition is reusable beyond this repository",
	"sweep":        "ownership-aware dirty-tree grouping is a reusable shared-checkout pattern",
	"task":         "typed task contracts are reusable beyond this repository",
	"tasks":        "typed task portfolio views are reusable beyond this repository",
	"validate":     "explicit-path overlay validation is a reusable shared-checkout pattern",
	"workspace":    "workspace state inspection is a reusable agent orientation pattern",
	"worktree":     "leased detached worker isolation is a reusable concurrency pattern",
}

var labDevCommands = map[string]string{
	"amd-gpu-facts":  "hardware-lab inventory command",
	"claude-mac-fak": "fak lab-host control command",
	"cluster":        "fak compute-fleet operation",
	"fleet":          "fak compute-fleet operation",
	"fleet-accounts": "fak compute-fleet account operation",
	"fleet-trend":    "fak compute-fleet telemetry",
	"fleetcap":       "fak compute-fleet capacity operation",
	"lab":            "fak private-lab operation",
	"macbench":       "fak lab-host benchmark operation",
	"node":           "fak compute-node operation",
	"node-compare":   "fak compute-node comparison",
	"nodeusage":      "fak compute-node telemetry",
	"nightrun":       "fak lab/fleet scheduled operation",
}

// ClassifyDevReuse returns a total reuse classification. Portable means that
// the concept is suitable for examples and dogfood; it does not make the
// current repository-bound command a stable adopter API.
func ClassifyDevReuse(name string, owner CommandOwner) (DevReuse, string) {
	if owner != OwnerDev {
		return DevReuseNA, "runtime ownership is outside the development-reuse axis"
	}
	if rationale, ok := portableDevPatterns[name]; ok {
		return DevReusePortable, rationale
	}
	if rationale, ok := labDevCommands[name]; ok {
		return DevReuseLab, rationale
	}
	return DevReuseMaintainer, "implementation maintains, measures, releases, or operates fak itself"
}
