package configguide

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
)

func TestEveryPostureIsMinimalAndRoundTrips(t *testing.T) {
	if got, want := Names(), []string{"default", "long-session", "team-gateway", "hardened"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("Names = %v, want %v", got, want)
	}
	for _, name := range Names() {
		t.Run(name, func(t *testing.T) {
			result, err := Guide(Options{Posture: name})
			if err != nil {
				t.Fatal(err)
			}
			if result.Schema != Schema || result.Posture != name || result.Summary == "" {
				t.Fatalf("incomplete result: %+v", result)
			}
			if name == "default" {
				if result.NeedsConfig || result.Manifest != "" || len(result.Changes) != 0 || result.Run != "fak serve" {
					t.Fatalf("default must require no file: %+v", result)
				}
				return
			}
			if !result.NeedsConfig || result.Manifest == "" || len(result.Changes) == 0 {
				t.Fatalf("non-default posture has no minimal delta: %+v", result)
			}
			m, err := deploymanifest.Parse([]byte(result.Manifest))
			if err != nil {
				t.Fatalf("manifest does not round-trip: %v\n%s", err, result.Manifest)
			}
			if got := len(m.DeclaredKeys()); got != len(result.Changes) {
				t.Fatalf("declared keys = %d, changes = %d; generated output is not a pure delta", got, len(result.Changes))
			}
			for _, change := range result.Changes {
				if change.Why == "" || change.EquivalentFlag == "" {
					t.Errorf("change lacks explanation or equivalent flag: %+v", change)
				}
			}
		})
	}
}

func TestTeamGatewayAcceptsUserOpinions(t *testing.T) {
	result, err := Guide(Options{Posture: "team-gateway", KeyEnv: "OUR_TOKEN", Bind: "10.2.3.4:9443"})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`require_key_env = "OUR_TOKEN"`, `bind = "10.2.3.4:9443"`} {
		if !strings.Contains(result.Manifest, want) {
			t.Errorf("manifest missing %q:\n%s", want, result.Manifest)
		}
	}
}

func TestLongSessionAcceptsBudgetOpinion(t *testing.T) {
	result, err := Guide(Options{Posture: "long-session", Budget: 240000})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Manifest, "default_tokens = 240000") {
		t.Fatalf("custom budget missing:\n%s", result.Manifest)
	}
}

func TestUnknownPostureNamesChoices(t *testing.T) {
	_, err := Guide(Options{Posture: "surprise"})
	if err == nil || !strings.Contains(err.Error(), "default, long-session, team-gateway, hardened") {
		t.Fatalf("error = %v, want discoverable choices", err)
	}
}
