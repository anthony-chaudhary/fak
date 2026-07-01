package devindex

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Orientation is the task-scoped convention bundle for one requested path or glob.
// It answers the facts an agent normally has to infer before editing: lane, arch
// tier, owning test target, commit stamp, and any live lease that overlaps the tree.
type Orientation struct {
	Path       string             `json:"path"`
	Lane       string             `json:"lane,omitempty"`
	LaneTree   []string           `json:"lane_tree,omitempty"`
	Tier       *int               `json:"tier,omitempty"`
	TierName   string             `json:"tier_name,omitempty"`
	TestTarget string             `json:"owning_test_target,omitempty"`
	Stamp      string             `json:"stamp,omitempty"`
	LiveLeases []OrientationLease `json:"live_leases,omitempty"`
}

// OrientationLease is the live-lease projection devindex needs. The CLI fills it
// from internal/leaseref; tests can inject rows without reaching git.
type OrientationLease struct {
	ID         string   `json:"id"`
	Holder     string   `json:"holder,omitempty"`
	Tree       []string `json:"tree,omitempty"`
	TTLSeconds int64    `json:"ttl_seconds,omitempty"`
}

var tierEntryRE = regexp.MustCompile(`"([A-Za-z0-9][\w.\-]*)"\s*:\s*(\d+)`)

func (c *Catalog) parseTiers(text string) {
	inTable := false
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if !inTable {
			if strings.Contains(line, "var tier = map[string]int{") {
				inTable = true
			}
			continue
		}
		if line == "}" {
			break
		}
		for _, m := range tierEntryRE.FindAllStringSubmatch(line, -1) {
			n, err := strconv.Atoi(m[2])
			if err != nil {
				continue
			}
			c.tiers[strings.ToLower(m[1])] = n
		}
	}
}

// Orient resolves each requested path/glob to its local development conventions.
func (c *Catalog) Orient(paths []string, leases []OrientationLease) []Orientation {
	out := make([]Orientation, 0, len(paths))
	for _, raw := range paths {
		path := strings.TrimSpace(raw)
		if path == "" {
			continue
		}
		lane := c.LaneForPath(path)
		o := Orientation{
			Path:       path,
			Lane:       lane,
			Stamp:      c.SuggestStamp(path),
			TestTarget: owningTestTarget(path),
		}
		if leaf, ok := c.LeafByName(lane); ok {
			o.LaneTree = splitLaneTree(leaf.Tree)
		}
		if tier, ok := c.TierForPath(path); ok {
			t := tier
			o.Tier = &t
			o.TierName = TierName(tier)
		}
		requestTree := []string{path}
		for _, lease := range leases {
			if orientationTreesOverlap(requestTree, lease.Tree) {
				o.LiveLeases = append(o.LiveLeases, cloneOrientationLease(lease))
			}
		}
		sort.SliceStable(o.LiveLeases, func(i, j int) bool {
			return o.LiveLeases[i].ID < o.LiveLeases[j].ID
		})
		out = append(out, o)
	}
	return out
}

// TierForPath returns the architest tier for internal/<leaf> paths. Non-internal
// paths have no arch tier and return ok=false.
func (c *Catalog) TierForPath(path string) (tier int, ok bool) {
	leaf := internalLeaf(path)
	if leaf == "" {
		return 0, false
	}
	tier, ok = c.tiers[leaf]
	return tier, ok
}

// TierName renders architest's numeric layer into its review vocabulary.
func TierName(tier int) string {
	switch tier {
	case 0:
		return "root"
	case 1:
		return "foundation"
	case 2:
		return "mechanism"
	case 3:
		return "composer"
	case 4:
		return "integrator"
	default:
		return "unknown"
	}
}

func splitLaneTree(tree string) []string {
	var out []string
	for _, part := range strings.Split(tree, ",") {
		if s := strings.TrimSpace(part); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func owningTestTarget(path string) string {
	p := orientTreePrefix(path)
	if p == "" || p == "**" {
		return ""
	}
	seg := strings.Split(p, "/")
	if len(seg) < 2 {
		return ""
	}
	switch seg[0] {
	case "internal":
		return "go test ./internal/" + seg[1]
	case "cmd":
		return "go test ./cmd/" + seg[1]
	default:
		return ""
	}
}

func internalLeaf(path string) string {
	p := orientTreePrefix(path)
	seg := strings.Split(p, "/")
	if len(seg) >= 2 && seg[0] == "internal" {
		return strings.ToLower(seg[1])
	}
	return ""
}

func cloneOrientationLease(lease OrientationLease) OrientationLease {
	lease.Tree = append([]string(nil), lease.Tree...)
	return lease
}

func orientationTreesOverlap(a, b []string) bool {
	ta, tb := cleanOrientationTrees(a), cleanOrientationTrees(b)
	if len(ta) == 0 || len(tb) == 0 {
		return true
	}
	for _, x := range ta {
		for _, y := range tb {
			if orientationTreeOverlap(x, y) {
				return true
			}
		}
	}
	return false
}

func cleanOrientationTrees(xs []string) []string {
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		if p := orientTreePrefix(x); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func orientationTreeOverlap(a, b string) bool {
	a, b = orientTreePrefix(a), orientTreePrefix(b)
	if a == "" || b == "" {
		return true
	}
	if a == "**" || b == "**" {
		return true
	}
	return a == b || strings.HasPrefix(a, b+"/") || strings.HasPrefix(b, a+"/")
}

func orientTreePrefix(path string) string {
	p := normPath(strings.TrimSpace(path))
	p = strings.TrimSuffix(p, "/")
	if p == "" {
		return ""
	}
	if p == "**" || p == "**/*" {
		return "**"
	}
	if i := strings.IndexAny(p, "*?["); i >= 0 {
		if slash := strings.LastIndex(p[:i], "/"); slash >= 0 {
			p = p[:slash]
		} else {
			return "**"
		}
	}
	p = strings.TrimSuffix(p, "/**")
	p = strings.TrimSuffix(p, "/*")
	return strings.TrimSuffix(p, "/")
}
