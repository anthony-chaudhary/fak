package gateway

import (
	"net/http"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	// HeaderSpeculative is the response header reporting the speculative decoding engine used.
	HeaderSpeculative = "x-fak-speculative"
	// SpeculativeMTPMetal indicates speculative decoding via the resident Metal MTP coordinator.
	SpeculativeMTPMetal = "mtp-metal"
	// SpeculativeMTPVulkan indicates observed execution by the request-bound resident Vulkan MTP route.
	SpeculativeMTPVulkan = "mtp-vulkan"
	// HeaderSpeculativeDowngrade reports a typed ordinary-decode downgrade for a selected route.
	HeaderSpeculativeDowngrade = "x-fak-speculative-downgrade"
)

// isMetalMTPActive reports whether Metal MTP speculative decoding is configured
// and active for the specified request model on this server.
func (s *Server) isMetalMTPActive(reqModel string) bool {
	if s == nil {
		return false
	}
	p := s.plannerForRequest(reqModel)
	if coord := metalMTPCoordinatorFromPlanner(p); coord != nil {
		return true
	}
	return s.metalMTPCoord != nil
}

// activeMetalMTPCoordinator extracts the active MetalMTPCoordinator from Server or its planner.
func (s *Server) activeMetalMTPCoordinator() *model.MetalMTPCoordinator {
	if s == nil {
		return nil
	}
	if s.metalMTPCoord != nil {
		return s.metalMTPCoord
	}
	return metalMTPCoordinatorFromPlanner(s.planner)
}

// MetalMTPCoordinator returns the active MetalMTPCoordinator for the server, if any.
func (s *Server) MetalMTPCoordinator() *model.MetalMTPCoordinator {
	if s == nil {
		return nil
	}
	if coord := metalMTPCoordinatorFromPlanner(s.planner); coord != nil {
		return coord
	}
	return s.metalMTPCoord
}

// SetMetalMTPCoordinator configures an explicit MetalMTPCoordinator on the server
// and its underlying in-kernel planner.
func (s *Server) SetMetalMTPCoordinator(c *model.MetalMTPCoordinator) {
	if s == nil {
		return
	}
	s.metalMTPCoord = c
	if s.planner != nil {
		if ikp, ok := s.planner.(interface {
			SetMetalMTPCoordinator(*model.MetalMTPCoordinator)
		}); ok {
			ikp.SetMetalMTPCoordinator(c)
		} else if dp, ok := s.planner.(*DualPlanner); ok {
			if ikp, ok := dp.Local().(interface {
				SetMetalMTPCoordinator(*model.MetalMTPCoordinator)
			}); ok {
				ikp.SetMetalMTPCoordinator(c)
			}
		}
	}
}

// EnableMetalMTP enables the Metal MTP execution loop on the server and its underlying in-kernel planner.
func (s *Server) EnableMetalMTP(cfg ...model.MetalMTPConfig) error {
	coord, err := model.NewMetalMTPCoordinator(nil, cfg...)
	if err != nil {
		return err
	}
	s.SetMetalMTPCoordinator(coord)
	return nil
}

// DisableMetalMTP disables the Metal MTP execution loop on the server and its underlying in-kernel planner.
func (s *Server) DisableMetalMTP() {
	if s == nil {
		return
	}
	if s.metalMTPCoord != nil {
		_ = s.metalMTPCoord.Close()
		s.metalMTPCoord = nil
	}
	if s.planner != nil {
		if ikp, ok := s.planner.(interface {
			DisableMetalMTP()
		}); ok {
			ikp.DisableMetalMTP()
		} else if dp, ok := s.planner.(*DualPlanner); ok {
			if ikp, ok := dp.Local().(interface {
				DisableMetalMTP()
			}); ok {
				ikp.DisableMetalMTP()
			}
		}
	}
}

// plannerForRequest returns the sub-planner serving reqModel, resolving DualPlanner routing if present.
func (s *Server) plannerForRequest(reqModel string) agent.Planner {
	if s == nil || s.planner == nil {
		return nil
	}
	if dp, ok := s.planner.(*DualPlanner); ok {
		if dp.RoutesLocal(reqModel) {
			return dp.Local()
		}
		return dp.Proxy()
	}
	return s.planner
}

// metalMTPCoordinatorFromPlanner extracts the MetalMTPCoordinator from an agent.Planner if supported.
func metalMTPCoordinatorFromPlanner(p agent.Planner) *model.MetalMTPCoordinator {
	if p == nil {
		return nil
	}
	if m, ok := p.(interface {
		MetalMTPCoordinator() *model.MetalMTPCoordinator
	}); ok {
		return m.MetalMTPCoordinator()
	}
	return nil
}

func (s *Server) isVulkanMTPEnabled(reqModel string) bool {
	p := s.plannerForRequest(reqModel)
	if p == nil {
		return false
	}
	enabled, ok := p.(interface{ VulkanMTPEnabled() bool })
	return ok && enabled.VulkanMTPEnabled()
}

// shouldEnableMetalMTP reports whether the in-kernel chat planner should initialize
// and activate the resident Metal MTP coordinator.
func shouldEnableMetalMTP(cfg Config) bool {
	if cfg.DisableMetalMTP {
		return false
	}
	if !cfg.Metal || cfg.Backend != nil || cfg.InKernelModel == nil {
		return false
	}
	if !isModelHybrid(cfg.InKernelModel) {
		return false
	}
	return cfg.MetalMTP || speculativeModeSelected("mtp")
}

// shouldEnableVulkanMTP admits only the supported Qwen3.8 resident
// draft/target envelope. The request-bound constructor repeats these checks and
// produces a typed ordinary-decode downgrade if live session state is unsuitable.
func shouldEnableVulkanMTP(cfg Config) bool {
	if cfg.InKernelModel == nil || cfg.Backend == nil || cfg.Metal || !cfg.InKernelQ4K || !vulkanMTPModeSelected() {
		return false
	}
	if !isModelHybrid(cfg.InKernelModel) || !strings.EqualFold(strings.TrimSpace(cfg.Backend.Name()), "vulkan") {
		return false
	}
	mode, err := cfg.InKernelModel.Qwen35MTPMode(false)
	if err != nil || !mode.Enabled {
		return false
	}
	draft, ok := cfg.Backend.(compute.Qwen35MTPDraftBackend)
	if !ok || draft.Qwen35MTPDraftPath() != compute.Qwen35MTPDraftPath {
		return false
	}
	sequence, ok := cfg.Backend.(model.Qwen35SequencePrefillBackend)
	if !ok || sequence.Qwen35SequencePrefillPath() != compute.Qwen35SequencePrefillPath {
		return false
	}
	raw, ok := cfg.Backend.(compute.Qwen35SequenceRawHiddenBackend)
	if !ok || raw.Qwen35SequenceRawHiddenPath() != compute.Qwen35SequenceRawHiddenPath {
		return false
	}
	allLogits, ok := cfg.Backend.(compute.Qwen35SequenceAllLogitsBackend)
	if !ok || allLogits.Qwen35SequenceAllLogitsPath() != compute.Qwen35SequenceAllLogitsPath {
		return false
	}
	replay, ok := cfg.Backend.(compute.Qwen35SequencePrefixReplayBackend)
	return ok && replay.Qwen35SequencePrefixReplayPath() == compute.Qwen35SequencePrefixReplayPath
}

// shouldEnableNGramSpeculative reports whether the operator explicitly selected
// prompt n-gram speculation for the supported resident Qwen Vulkan route.
func shouldEnableNGramSpeculative(cfg Config) bool {
	if cfg.InKernelModel == nil || cfg.Backend == nil || cfg.Metal || !cfg.InKernelQ4K {
		return false
	}
	if !isModelHybrid(cfg.InKernelModel) || !strings.EqualFold(strings.TrimSpace(cfg.Backend.Name()), "vulkan") {
		return false
	}
	backend, ok := cfg.Backend.(compute.Qwen35SequenceAllLogitsBackend)
	if !ok || backend.Qwen35SequenceAllLogitsPath() != compute.Qwen35SequenceAllLogitsPath {
		return false
	}
	return speculativeModeSelected("ngram")
}

func speculativeModeSelected(mode string) bool {
	want := strings.ToLower(strings.TrimSpace(mode))
	primary := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_SPECULATIVE")))
	alternate := strings.ToLower(strings.TrimSpace(os.Getenv("SPECULATIVE")))
	return want != "" && (primary == want || alternate == want)
}

func vulkanMTPModeSelected() bool {
	primary := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_SPECULATIVE")))
	alternate := strings.ToLower(strings.TrimSpace(os.Getenv("SPECULATIVE")))
	if primary != "" && alternate != "" && primary != alternate {
		return false
	}
	return speculativeModeSelected("mtp")
}

// isModelHybrid checks whether the model has linear attention layers (Qwen 3.5/3.8 hybrid architecture).
func isModelHybrid(m *model.Model) bool {
	if m == nil {
		return false
	}
	for _, lt := range m.Cfg.LayerTypes {
		if lt == "linear_attention" {
			return true
		}
	}
	return false
}

// applyChatCompletionSpeculativeHeaders stamps speculative decoding headers onto
// response writers before status and payload bytes leave the wire.
func applyChatCompletionSpeculativeHeaders(w http.ResponseWriter, s *Server, reqModel string) {
	if s == nil || w == nil {
		return
	}
	if s.isMetalMTPActive(reqModel) {
		w.Header().Set(HeaderSpeculative, SpeculativeMTPMetal)
	}
}

func applyVulkanMTPExecutionHeaders(w http.ResponseWriter, execution *agent.VulkanMTPExecution) {
	if w == nil || execution == nil || execution.Engine != SpeculativeMTPVulkan {
		return
	}
	if execution.Used {
		w.Header().Set(HeaderSpeculative, SpeculativeMTPVulkan)
	}
	if execution.DowngradeReason != "" {
		w.Header().Set(HeaderSpeculativeDowngrade, execution.DowngradeReason)
	}
}

func chatRequestVulkanMTPEligible(req ChatRequest) bool {
	return (req.Temperature == nil || *req.Temperature <= 0) && len(req.LogitBias) == 0 &&
		(req.FrequencyPenalty == nil || *req.FrequencyPenalty == 0) &&
		(req.PresencePenalty == nil || *req.PresencePenalty == 0)
}
