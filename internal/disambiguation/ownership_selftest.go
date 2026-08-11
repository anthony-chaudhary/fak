package disambiguation

import (
	"os"
	"path/filepath"
	"strings"
)

type OwnershipSelfCheckReport struct {
	OK              bool   `json:"ok"`
	AcceptedFixture bool   `json:"accepted_fixture"`
	RejectedLeaf    bool   `json:"rejected_owner_leaf"`
	RejectedLane    bool   `json:"rejected_dispatch_lane"`
	LeafSource      string `json:"leaf_source"`
	LaneSource      string `json:"lane_source"`
}

// OwnershipSelfCheck proves both admission edges using repository-shaped public
// fixtures. The report is JSON-safe for the CLI selfcheck seam.
func OwnershipSelfCheck() OwnershipSelfCheckReport {
	report := OwnershipSelfCheckReport{LeafSource: "internal/<leaf> and cmd/<leaf> directories", LaneSource: "dos.toml [lanes] and [lanes.trees]"}
	root, err := os.MkdirTemp("", "fak-disambiguation-ownership-")
	if err != nil {
		return report
	}
	defer os.RemoveAll(root)
	_ = os.MkdirAll(filepath.Join(root, "internal", "canon"), 0755)
	_ = os.MkdirAll(filepath.Join(root, "cmd", "fak"), 0755)
	_ = os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes]\nconcurrent = [\"canon\", \"cmd\"]\n[lanes.trees]\ncanon = [\"internal/canon/**\"]\ncmd = [\"cmd/**\"]\n"), 0644)
	manifests, err := LoadPublicManifests(root)
	if err != nil {
		return report
	}
	accepted := selfCheckEntries()
	_, acceptedErr := NewAdmittedIndex(accepted, manifests)
	report.AcceptedFixture = acceptedErr == nil
	badLeaf := selfCheckEntries()
	badLeaf[0].Owner.Leaf = "private-canon"
	if err := AdmitOwnership(badLeaf, manifests); err != nil && strings.Contains(err.Error(), "owner leaf") {
		report.RejectedLeaf = true
	}
	badLane := selfCheckEntries()
	badLane[0].Owner.Lane = "private-lane"
	if err := AdmitOwnership(badLane, manifests); err != nil && strings.Contains(err.Error(), "dispatch lane") {
		report.RejectedLane = true
	}
	report.OK = report.AcceptedFixture && report.RejectedLeaf && report.RejectedLane
	return report
}

func selfCheckEntries() []Entry {
	entries := append([]Entry(nil), publicEntries[:2]...)
	for i := range entries {
		entries[i].Owner = Owner{Leaf: "canon", Lane: "canon"}
	}
	return entries
}
