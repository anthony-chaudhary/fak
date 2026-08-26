package issue8833witness

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type packet struct {
	Decision               string            `json:"decision"`
	Files                  map[string]string `json:"files_sha256"`
	Quality                 bool              `json:"quality"`
	Lifecycle               bool              `json:"lifecycle"`
	Identity                bool              `json:"identity"`
	ControlMedianNetDecode  float64           `json:"control_median_net_decode"`
	CandidateMedianNetDecode float64          `json:"candidate_median_net_decode"`
	ControlEvents           int               `json:"control_events"`
	CandidateEvents         int               `json:"candidate_events"`
	CatastrophicRegression  bool              `json:"catastrophic_regression"`
}

func TestImmutableWitnessPacket(t *testing.T) {
	body, err := os.ReadFile("packet.json")
	if os.IsNotExist(err) {
		t.Skip("issue #8833: no real immutable receipt packet; REJECT and selector remains off")
	}
	if err != nil {
		t.Fatal(err)
	}
	var p packet
	if err := json.Unmarshal(body, &p); err != nil {
		t.Fatal(err)
	}
	for name, want := range p.Files {
		body, err := os.ReadFile(filepath.Clean(name))
		if err != nil {
			t.Fatalf("receipt %q: %v", name, err)
		}
		got := sha256.Sum256(body)
		if hex.EncodeToString(got[:]) != want {
			t.Fatalf("receipt %q SHA-256 mismatch", name)
		}
	}
	keep := p.Quality && p.Lifecycle && p.Identity &&
		p.CandidateMedianNetDecode > p.ControlMedianNetDecode &&
		p.CandidateEvents < p.ControlEvents && !p.CatastrophicRegression
	if p.Decision == "KEEP" && !keep {
		t.Fatal("KEEP requires all quality/lifecycle/identity/performance/event/regression gates")
	}
	if p.Decision != "KEEP" && p.Decision != "REJECT" {
		t.Fatalf("decision %q must be KEEP or REJECT", p.Decision)
	}
}
