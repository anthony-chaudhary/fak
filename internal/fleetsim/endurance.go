package fleetsim

import "github.com/anthony-chaudhary/fak/internal/superloop"

// EnduranceConfig describes a deterministic overnight liveness replay. It models
// orchestration decisions, not provider quality or token economics.
type EnduranceConfig struct {
	Issues          int
	Workers         int
	ClosePerCycle   int
	RefusalCycles   map[int]bool
	OwnedWIPCycles  map[int]bool
	PeerWIPCycles   map[int]bool
	AbandonedCycles map[int]bool
	MaxCycles       int
}

// EnduranceCycle captures the liveness fact and selected recovery rung for one tick.
type EnduranceCycle struct {
	Cycle          int    `json:"cycle"`
	OpenBefore     int    `json:"open_before"`
	Closed         int    `json:"closed"`
	OpenAfter      int    `json:"open_after"`
	NoProgress     int    `json:"no_progress_streak"`
	Stage          string `json:"stage"`
	Residual       string `json:"residual,omitempty"`
	KeepGoing      bool   `json:"keep_going"`
	TouchedPeerWIP bool   `json:"touched_peer_wip"`
}

// EnduranceReport is the machine-readable proof summary for a sustained replay.
type EnduranceReport struct {
	Schema          string           `json:"schema"`
	Workers         int              `json:"workers"`
	InitialIssues   int              `json:"initial_issues"`
	ClosedIssues    int              `json:"closed_issues"`
	Cycles          []EnduranceCycle `json:"cycles"`
	PrematureDone   bool             `json:"premature_done"`
	TouchedPeerWIP  bool             `json:"touched_peer_wip"`
	MaxNoProgress   int              `json:"max_no_progress_streak"`
	TerminalStage   string           `json:"terminal_stage"`
	EventuallyDrain bool             `json:"eventually_drain"`
}

// ReplayEndurance composes issue backlog, residual-WIP, refusal, reset, and bounded
// escalation rules without launching workers or mutating a repository.
//
// Invariant: endurance simulation replay is fail-closed and monotonic.
// Backlog progress strictly requires witnessed issue closes; unresolved peer WIP
// or active owner work stalls progress and triggers deterministic escalation.
//
// Guard: premature done state is flagged if cycles terminate while issues remain open.
func ReplayEndurance(cfg EnduranceConfig) EnduranceReport {
	if cfg.Workers < 1 {
		cfg.Workers = 1
	}
	if cfg.ClosePerCycle < 1 {
		cfg.ClosePerCycle = cfg.Workers
	}
	if cfg.MaxCycles < 1 {
		cfg.MaxCycles = cfg.Issues*2 + 32
	}
	rep := EnduranceReport{Schema: "fak.fleetsim.endurance.v1", Workers: cfg.Workers, InitialIssues: cfg.Issues}
	open, streak := cfg.Issues, 0
	for cycle := 1; cycle <= cfg.MaxCycles; cycle++ {
		row := EnduranceCycle{Cycle: cycle, OpenBefore: open}
		switch {
		case cfg.PeerWIPCycles[cycle]:
			row.Residual = "peer-active-wip"
			// Waiting is deliberate: peer work remains live, but this fleet never steals it.
		case cfg.OwnedWIPCycles[cycle]:
			row.Residual = "owned-reconcile"
		case cfg.AbandonedCycles[cycle]:
			row.Residual = "abandoned-recover"
		case cfg.RefusalCycles[cycle]:
			row.Residual = "transient-refusal"
		case open > 0:
			row.Closed = min(cfg.ClosePerCycle, open)
			open -= row.Closed
		}
		if row.Closed > 0 {
			streak = 0
		} else if open > 0 || row.Residual != "" {
			streak++
		}
		stage := superloop.EscalateNoProgress(streak)
		row.NoProgress, row.Stage, row.OpenAfter = streak, stage.Name, open
		row.KeepGoing = open > 0 || row.Residual != "" || enduranceFutureResidual(cfg, cycle)
		if !row.KeepGoing && open > 0 {
			rep.PrematureDone = true
		}
		if row.TouchedPeerWIP {
			rep.TouchedPeerWIP = true
		}
		if streak > rep.MaxNoProgress {
			rep.MaxNoProgress = streak
		}
		rep.ClosedIssues += row.Closed
		rep.Cycles = append(rep.Cycles, row)
		rep.TerminalStage = stage.Name
		if !row.KeepGoing {
			rep.EventuallyDrain = open == 0
			break
		}
	}
	return rep
}

func enduranceFutureResidual(cfg EnduranceConfig, cycle int) bool {
	for next := range cfg.RefusalCycles {
		if next > cycle {
			return true
		}
	}
	for next := range cfg.OwnedWIPCycles {
		if next > cycle {
			return true
		}
	}
	for next := range cfg.PeerWIPCycles {
		if next > cycle {
			return true
		}
	}
	for next := range cfg.AbandonedCycles {
		if next > cycle {
			return true
		}
	}
	return false
}
