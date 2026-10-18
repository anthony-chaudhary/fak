package gateway

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// dflash2Selected preserves historical selectors unless DFlash2 participates.
// Explicit conflicting ambient selectors cannot turn a selected sidecar into an
// ordinary planner, or silently substitute another speculation mechanism.
func dflash2Selected(cfg Config) (bool, error) {
	a, b := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_SPECULATIVE"))), strings.ToLower(strings.TrimSpace(os.Getenv("SPECULATIVE")))
	c := strings.ToLower(strings.TrimSpace(cfg.SpeculativeMode))
	relevant := a == "dflash2" || b == "dflash2" || c == "dflash2" || cfg.DFlash2Target != nil
	if !relevant {
		return false, nil
	}
	mode := c
	if a != "" {
		mode = a
	}
	if b != "" {
		if a != "" && a != b {
			return false, errors.New("gateway: conflicting DFlash2 environment selectors")
		}
		mode = b
	}
	if mode != "dflash2" {
		return false, errors.New("gateway: DFlash2 target requires an explicit matching selector")
	}
	if c != "" && c != "dflash2" {
		return false, errors.New("gateway: DFlash2 conflicts with structural selector")
	}
	return true, nil
}

func validateDFlash2Config(cfg Config) error {
	selected, err := dflash2Selected(cfg)
	if err != nil || !selected {
		return err
	}
	if cfg.DFlash2Target == nil {
		return errors.New("gateway: DFlash2 requires an authenticated target owner")
	}
	if strings.TrimSpace(cfg.BaseURL) != "" || len(cfg.ReplicaBaseURLs) != 0 || cfg.ExpertParallelRanks > 1 {
		return errors.New("gateway: DFlash2 requires an exclusive local dense target")
	}
	return cfg.DFlash2Target.ValidateBinding(cfg.InKernelModel, cfg.Tokenizer, cfg.Backend, cfg.InKernelQ4K, cfg.Metal)
}

// Construction cannot silently fall back if the admitted binding changes between
// validation and planner creation. Requests see the exact binding error.
type dflash2RefusedPlanner struct {
	model string
	err   error
}

func (p *dflash2RefusedPlanner) Model() string { return p.model }
func (p *dflash2RefusedPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return nil, fmt.Errorf("gateway: DFlash2 binding refused: %w", p.err)
}
