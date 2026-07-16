package egresslist

import (
	"embed"
	"io/fs"
	"sort"
	"strings"
)

// bundledLists holds the community filter lists that ship inside the binary. Bundling —
// rather than fetching at runtime — is the spine's deliberate choice: a policy that
// names a block_list must resolve deterministically and offline, with no network fetch
// on the decision path and no surprise when the upstream feed is down. The refreshable,
// network-backed catalog (pull the current StevenBlack/EasyList revisions on a cadence,
// like a browser ad blocker updates its subscriptions) is layered ON TOP of this by the
// expansion tickets; it will drop refreshed files into the same registry shape.
//
//go:embed lists/*.txt
var bundledLists embed.FS

// BundledListNames returns the names a policy may reference in egress.block_lists, sorted
// for stable output. A name is the file stem: "lists/sample-malware.txt" -> "sample-malware".
func BundledListNames() []string {
	entries, err := fs.ReadDir(bundledLists, "lists")
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".txt") {
			continue
		}
		names = append(names, strings.TrimSuffix(e.Name(), ".txt"))
	}
	sort.Strings(names)
	return names
}

// BundledList returns the raw text of a bundled filter list by name. ok is false for a
// name that does not resolve to a shipped list, so the policy loader can fail LOUD on an
// unknown block_lists reference rather than silently compiling an empty (all-permissive)
// list — a dropped block list is a security regression, not a no-op.
func BundledList(name string) (text string, ok bool) {
	if name == "" || strings.ContainsAny(name, "/\\.") {
		return "", false // names are bare stems; reject path-shaped input.
	}
	b, err := fs.ReadFile(bundledLists, "lists/"+name+".txt")
	if err != nil {
		return "", false
	}
	return string(b), true
}
