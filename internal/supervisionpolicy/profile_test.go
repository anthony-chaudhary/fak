package supervisionpolicy

import (
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
)

func pointer[T any](value T) *T { return &value }

func baselineLayers() ProfileLayers {
	capabilities := []string{"agent.run", "serve.read"}
	secrets := []SecretHandle{{Name: "model", Handle: "secret://model"}}
	return ProfileLayers{
		CompiledDefaults: ProfilePatch{
			Operational:  OperationalPatch{LogLevel: pointer("info"), ShutdownGrace: pointer(30 * time.Second)},
			Capabilities: &capabilities,
			Resources:    ResourcePatch{CPUUnits: pointer(uint64(8)), MemoryBytes: pointer(uint64(16 << 30)), Processes: pointer(uint32(32))},
			Secrets:      &secrets,
			Restart:      RestartPatch{MaxRestarts: pointer(uint32(12)), Window: pointer(10 * time.Minute)},
		},
	}
}

func TestResolveProfileRejectsPrivilegeWidening(t *testing.T) {
	layers := baselineLayers()
	widened := []string{"agent.run", "serve.read", "admin.root"}
	layers.ProcessClass.Capabilities = &widened

	_, _, err := ResolveProfile(layers)
	if !errors.Is(err, ErrPrivilegeWidening) {
		t.Fatalf("ResolveProfile error = %v, want ErrPrivilegeWidening", err)
	}
}

func TestResolveProfileExplicitCapabilityGrantAllowsDescendantExpansion(t *testing.T) {
	layers := baselineLayers()
	layers.Installation.GrantCapabilities = []string{"serve.read"}
	reduced := []string{"agent.run"}
	layers.ProcessClass.Capabilities = &reduced
	reexpanded := []string{"agent.run", "serve.read"}
	layers.ParentDomain.Capabilities = &reexpanded

	profile, _, err := ResolveProfile(layers)
	if err != nil {
		t.Fatalf("ResolveProfile: %v", err)
	}
	if !reflect.DeepEqual(profile.Capabilities, []string{"agent.run", "serve.read"}) {
		t.Fatalf("capabilities = %v", profile.Capabilities)
	}
}

func TestResolveProfileRejectsResourceWidening(t *testing.T) {
	layers := baselineLayers()
	layers.ParentDomain.Resources.MemoryBytes = pointer(uint64(17 << 30))

	_, _, err := ResolveProfile(layers)
	if !errors.Is(err, ErrResourceWidening) {
		t.Fatalf("ResolveProfile error = %v, want ErrResourceWidening", err)
	}
}

func TestResolveProfileRestartInheritance(t *testing.T) {
	for name, mutate := range map[string]func(*ProfileLayers){
		"rejects max restarts widening": func(layers *ProfileLayers) {
			layers.ProcessClass.Restart.MaxRestarts = pointer(uint32(13))
		},
		"rejects shorter window": func(layers *ProfileLayers) {
			layers.ProcessClass.Restart.Window = pointer(9 * time.Minute)
		},
	} {
		t.Run(name, func(t *testing.T) {
			layers := baselineLayers()
			mutate(&layers)
			_, _, err := ResolveProfile(layers)
			if !errors.Is(err, ErrRestartWidening) {
				t.Fatalf("ResolveProfile error = %v, want ErrRestartWidening", err)
			}
		})
	}

	t.Run("accepts narrowing and deterministic partial inheritance", func(t *testing.T) {
		layers := baselineLayers()
		layers.ProcessClass.Restart.MaxRestarts = pointer(uint32(8))
		layers.ParentDomain.Restart.Window = pointer(20 * time.Minute)

		first, firstDigest, err := ResolveProfile(layers)
		if err != nil {
			t.Fatalf("first ResolveProfile: %v", err)
		}
		second, secondDigest, err := ResolveProfile(layers)
		if err != nil {
			t.Fatalf("second ResolveProfile: %v", err)
		}
		want := RestartBudget{MaxRestarts: 8, Window: 20 * time.Minute}
		if first.Restart != want || second.Restart != want || firstDigest != secondDigest {
			t.Fatalf("partial restart inheritance = %+v/%q then %+v/%q, want %+v with stable digest", first.Restart, firstDigest, second.Restart, secondDigest, want)
		}
	})
}

func TestValidateIdentityTransition(t *testing.T) {
	admitted := MemberIdentity{Member: "agent-a", Generation: 2, Class: ProcessClassRegularAgent, ParentDomain: "agents"}
	for name, next := range map[string]MemberIdentity{
		"member":     {Member: "agent-b", Generation: 3, Class: ProcessClassRegularAgent, ParentDomain: "agents"},
		"class":      {Member: "agent-a", Generation: 3, Class: ProcessClassSubagent, ParentDomain: "agents"},
		"domain":     {Member: "agent-a", Generation: 3, Class: ProcessClassRegularAgent, ParentDomain: "other"},
		"regression": {Member: "agent-a", Generation: 1, Class: ProcessClassRegularAgent, ParentDomain: "agents"},
	} {
		t.Run("rejects "+name, func(t *testing.T) {
			if err := ValidateIdentityTransition(admitted, next); !errors.Is(err, ErrIdentityMutation) {
				t.Fatalf("error = %v, want ErrIdentityMutation", err)
			}
		})
	}
	for name, generation := range map[string]uint64{"idempotent unchanged generation": 2, "generation advancement": 3} {
		t.Run("accepts "+name, func(t *testing.T) {
			next := admitted
			next.Generation = generation
			if err := ValidateIdentityTransition(admitted, next); err != nil {
				t.Fatalf("ValidateIdentityTransition: %v", err)
			}
		})
	}
}

func TestResolveProfileIsDeterministicAndUsesDeclaredOrder(t *testing.T) {
	layers := baselineLayers()
	layers.Installation.Operational.LogLevel = pointer("warn")
	layers.ProcessClass.Operational.LogLevel = pointer("debug")
	layers.ParentDomain.Operational.ShutdownGrace = pointer(15 * time.Second)
	layers.Instance.Operational.LogLevel = pointer("error")
	unorderedCapabilities := []string{"serve.read", "agent.run"}
	layers.CompiledDefaults.Capabilities = &unorderedCapabilities
	unorderedSecrets := []SecretHandle{{Name: "zeta", Handle: "secret://z"}, {Name: "alpha", Handle: "secret://a"}}
	layers.CompiledDefaults.Secrets = &unorderedSecrets

	first, firstDigest, err := ResolveProfile(layers)
	if err != nil {
		t.Fatalf("first ResolveProfile: %v", err)
	}
	second, secondDigest, err := ResolveProfile(layers)
	if err != nil {
		t.Fatalf("second ResolveProfile: %v", err)
	}
	if !reflect.DeepEqual(first, second) || firstDigest != secondDigest || firstDigest == "" {
		t.Fatalf("resolution not deterministic: first=%+v/%q second=%+v/%q", first, firstDigest, second, secondDigest)
	}
	if first.Operational.LogLevel != "error" || first.Operational.ShutdownGrace != 15*time.Second {
		t.Fatalf("nearest explicit scalars not applied: %+v", first.Operational)
	}
	if !reflect.DeepEqual(first.Capabilities, []string{"agent.run", "serve.read"}) || first.Secrets[0].Name != "alpha" {
		t.Fatalf("profile not canonical: %+v", first)
	}
}

func validTopology() Topology {
	budget := RestartBudget{MaxRestarts: 10, Window: 10 * time.Minute}
	childBudget := RestartBudget{MaxRestarts: 5, Window: 15 * time.Minute}
	return Topology{
		Domains: []DomainSpec{
			{ID: "root", Plane: FaultPlaneRoot, RestartBudget: budget},
			{ID: "agents", Parent: "root", Plane: FaultPlaneAgent, RestartBudget: childBudget},
			{ID: "serving", Parent: "root", Plane: FaultPlaneServing, RestartBudget: childBudget},
		},
		Members: []MemberSpec{
			{Identity: MemberIdentity{Member: "root", Generation: 1, Class: ProcessClassRootService, ParentDomain: "root"}},
			{Identity: MemberIdentity{Member: "agent", Generation: 1, Class: ProcessClassRegularAgent, ParentDomain: "agents"}, Parent: "root"},
			{Identity: MemberIdentity{Member: "subagent", Generation: 1, Class: ProcessClassSubagent, ParentDomain: "agents"}, Parent: "agent"},
			{Identity: MemberIdentity{Member: "controller", Generation: 1, Class: ProcessClassServeController, ParentDomain: "serving"}, Parent: "root"},
			{Identity: MemberIdentity{Member: "proxy", Generation: 1, Class: ProcessClassServeProxy, ParentDomain: "serving"}, Parent: "controller"},
			{Identity: MemberIdentity{Member: "replica", Generation: 1, Class: ProcessClassServeReplica, ParentDomain: "serving"}, Parent: "controller"},
		},
	}
}

func TestValidateTopologyAcceptsDeclaredTopology(t *testing.T) {
	if err := ValidateTopology(validTopology()); err != nil {
		t.Fatalf("ValidateTopology: %v", err)
	}
}

func TestValidateTopologyRejectsMalformedOrUndeclaredTopology(t *testing.T) {
	for name, mutate := range map[string]func(*Topology){
		"unknown class":            func(topology *Topology) { topology.Members[1].Identity.Class = "worker-ish" },
		"undeclared domain":        func(topology *Topology) { topology.Members[1].Identity.ParentDomain = "missing" },
		"undeclared parent member": func(topology *Topology) { topology.Members[2].Parent = "missing" },
		"invalid class edge":       func(topology *Topology) { topology.Members[4].Parent = "agent" },
		"agent in serving domain":  func(topology *Topology) { topology.Members[2].Identity.ParentDomain = "serving" },
		"controller in agent domain": func(topology *Topology) {
			topology.Members[3].Identity.ParentDomain = "agents"
		},
		"domain budget widening": func(topology *Topology) { topology.Domains[1].RestartBudget.MaxRestarts = 11 },
		"duplicate root": func(topology *Topology) {
			topology.Members[1].Identity.Class = ProcessClassRootService
			topology.Members[1].Parent = ""
		},
	} {
		t.Run(name, func(t *testing.T) {
			topology := validTopology()
			mutate(&topology)
			if err := ValidateTopology(topology); !errors.Is(err, ErrInvalidTopology) {
				t.Fatalf("error = %v, want ErrInvalidTopology", err)
			}
		})
	}
}

func TestValidateTopologyRejectsDomainCycle(t *testing.T) {
	topology := validTopology()
	topology.Domains[0].Parent = "agents"
	topology.Domains[1].Plane = FaultPlaneRoot
	topology.Domains[1].RestartBudget = topology.Domains[0].RestartBudget
	if err := ValidateTopology(topology); !errors.Is(err, ErrInvalidTopology) || !strings.Contains(err.Error(), "domain cycle") {
		t.Fatalf("error = %v, want domain-cycle ErrInvalidTopology", err)
	}
}

func TestValidateTopologyRejectsMemberCycle(t *testing.T) {
	topology := validTopology()
	// Cycle detection precedes edge-role validation so corrupt graphs are classified as cycles.
	topology.Members[0].Parent = "agent"
	if err := ValidateTopology(topology); !errors.Is(err, ErrInvalidTopology) || !strings.Contains(err.Error(), "member cycle") {
		t.Fatalf("error = %v, want member-cycle ErrInvalidTopology", err)
	}
}

func TestRequiredProcessClassesReturnsCopy(t *testing.T) {
	classes := RequiredProcessClasses()
	classes[0] = "invented"
	if knownProcessClass("invented") || RequiredProcessClasses()[0] != ProcessClassRootService {
		t.Fatal("caller mutation changed closed process-class vocabulary")
	}
}
