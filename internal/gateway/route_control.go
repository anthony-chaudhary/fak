package gateway

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/modelroute/inputtrigger"
)

const (
	InputTriggerRouteReceiptSchema = "fak.input-trigger-route/1"
	TurnIngressEngine              = "fak_native"
	TurnIngressModel               = "Qwen3.8"
	TurnIngressRouteIdentity       = TurnIngressEngine + "/" + TurnIngressModel
	inputTriggerProvenanceLabel    = "input_trigger_provenance"
)

var errInvalidExplicitInputTrigger = errors.New("gateway: invalid explicit input trigger")

// InputTriggerRouteReceipt binds one immutable ingress trigger to the exact
// content-addressed request-route decision it produced.
type InputTriggerRouteReceipt struct {
	trigger inputtrigger.InputTrigger
	route   modelroute.AuditRecord
}

func (r InputTriggerRouteReceipt) Trigger() inputtrigger.InputTrigger { return r.trigger }
func (r InputTriggerRouteReceipt) Route() modelroute.AuditRecord {
	return cloneInputTriggerRouteAudit(r.route)
}
func (InputTriggerRouteReceipt) Engine() string        { return TurnIngressEngine }
func (InputTriggerRouteReceipt) Model() string         { return TurnIngressModel }
func (InputTriggerRouteReceipt) RouteIdentity() string { return TurnIngressRouteIdentity }

type inputTriggerRouteReceiptWire struct {
	Schema        string                 `json:"schema"`
	Engine        string                 `json:"engine"`
	Model         string                 `json:"model"`
	RouteIdentity string                 `json:"route_identity"`
	InputTrigger  json.RawMessage        `json:"input_trigger"`
	Route         modelroute.AuditRecord `json:"route"`
}

func (r InputTriggerRouteReceipt) MarshalJSON() ([]byte, error) {
	if err := validateInputTriggerRouteReceipt(r); err != nil {
		return nil, err
	}
	trigger, err := json.Marshal(r.trigger)
	if err != nil {
		return nil, err
	}
	return json.Marshal(inputTriggerRouteReceiptWire{
		Schema:        InputTriggerRouteReceiptSchema,
		Engine:        TurnIngressEngine,
		Model:         TurnIngressModel,
		RouteIdentity: TurnIngressRouteIdentity,
		InputTrigger:  trigger,
		Route:         cloneInputTriggerRouteAudit(r.route),
	})
}

func (r *InputTriggerRouteReceipt) UnmarshalJSON(data []byte) error {
	replayed, err := ReplayInputTriggerRouteReceipt(data)
	if err != nil {
		return err
	}
	*r = replayed
	return nil
}

// ReplayInputTriggerRouteReceipt restores only the bounded trigger receipt and
// content-addressed route audit; it never re-reads prompt or tool-result bytes.
func ReplayInputTriggerRouteReceipt(data []byte) (InputTriggerRouteReceipt, error) {
	var wire inputTriggerRouteReceiptWire
	if err := decodeInputTriggerReceiptStrict(data, &wire); err != nil {
		return InputTriggerRouteReceipt{}, fmt.Errorf("gateway: replay input trigger route receipt: %w", err)
	}
	if wire.Schema != InputTriggerRouteReceiptSchema ||
		wire.Engine != TurnIngressEngine ||
		wire.Model != TurnIngressModel ||
		wire.RouteIdentity != TurnIngressRouteIdentity {
		return InputTriggerRouteReceipt{}, errors.New("gateway: invalid input trigger route receipt identity")
	}
	trigger, err := inputtrigger.Parse(wire.InputTrigger)
	if err != nil {
		return InputTriggerRouteReceipt{}, err
	}
	receipt := InputTriggerRouteReceipt{
		trigger: trigger,
		route:   cloneInputTriggerRouteAudit(wire.Route),
	}
	if err := validateInputTriggerRouteReceipt(receipt); err != nil {
		return InputTriggerRouteReceipt{}, err
	}
	return receipt, nil
}

// admitAndRouteChatInputTrigger runs on the untouched chat envelope before any
// transform, planner selection, or model call. Trigger metadata is gone before
// the modelroute Subject is constructed.
func (s *Server) admitAndRouteChatInputTrigger(req ChatRequest) (*InputTriggerRouteReceipt, string, error) {
	turn := make([]inputtrigger.Message, len(req.Messages))
	for i, message := range req.Messages {
		turn[i] = inputtrigger.Message{
			Role:       inputtrigger.Role(message.Role),
			Content:    message.Content,
			ToolCallID: message.ToolCallID,
		}
	}
	var explicit *inputtrigger.Explicit
	if req.Fak != nil {
		explicit = req.Fak.InputTrigger
	}
	trigger, err := inputtrigger.AdmitTurn(turn, explicit)
	if err != nil {
		return nil, "", fmt.Errorf("%w: %v", errInvalidExplicitInputTrigger, err)
	}
	if s.route == nil {
		return nil, "", nil
	}
	manifest := s.route.Manifest()
	if manifest == nil {
		return nil, "", errors.New("gateway: route manifest unavailable")
	}
	// Tool-only manifests predate request ingress routing. An input-trigger rule
	// opts the request route in without changing those manifests' dispatch.
	if !hasInputTriggerRequestRoute(manifest) {
		return nil, "", nil
	}
	subject := modelroute.Subject{
		Aspect:       modelroute.AspectRequest,
		InputTrigger: trigger.Classification(),
		Labels: map[string]string{
			inputTriggerProvenanceLabel: string(trigger.Provenance()),
		},
	}
	began := time.Now()
	audit := manifest.NewAuditRecord(subject)
	s.metrics.observeRouteDecision(audit.Version, audit.Decision, time.Since(began))
	if !singleNativeTurnRoute(audit.Decision.Plan) {
		return nil, "", errors.New("gateway: request input-trigger route is not a single fak_native/Qwen3.8 execution")
	}
	model := audit.Decision.Plan.Primary()
	receipt := InputTriggerRouteReceipt{
		trigger: trigger,
		route:   cloneInputTriggerRouteAudit(audit),
	}
	if err := validateInputTriggerRouteReceipt(receipt); err != nil {
		return nil, "", err
	}
	return &receipt, model, nil
}

func hasInputTriggerRequestRoute(manifest *modelroute.Manifest) bool {
	for _, rule := range manifest.Rules {
		if rule.Match.InputTrigger != "" &&
			(rule.Match.Aspect == "" || rule.Match.Aspect == modelroute.AspectRequest) {
			return true
		}
	}
	return false
}

func validateInputTriggerRouteReceipt(receipt InputTriggerRouteReceipt) error {
	if err := receipt.trigger.Validate(); err != nil {
		return err
	}
	if err := receipt.route.Verify(); err != nil {
		return err
	}
	subject := receipt.route.Decision.Subject
	if subject.Aspect != modelroute.AspectRequest {
		return errors.New("gateway: input trigger route receipt has non-request aspect")
	}
	if subject.InputTrigger != receipt.trigger.Classification() {
		return errors.New("gateway: input trigger route receipt classification mismatch")
	}
	if subject.Labels[inputTriggerProvenanceLabel] != string(receipt.trigger.Provenance()) {
		return errors.New("gateway: input trigger route receipt provenance mismatch")
	}
	if len(subject.Labels) != 1 || subject.AgentOperation != nil {
		return errors.New("gateway: input trigger route receipt carries unrelated request metadata")
	}
	if !singleNativeTurnRoute(receipt.route.Decision.Plan) {
		return errors.New("gateway: input trigger route receipt contains a fallback route")
	}
	return nil
}

func singleNativeTurnRoute(plan modelroute.Plan) bool {
	return len(plan.Members) == 1 && plan.Members[0].Model == TurnIngressModel && plan.Scout == ""
}

func cloneInputTriggerRouteAudit(in modelroute.AuditRecord) modelroute.AuditRecord {
	out := in
	out.Decision.Subject.Labels = make(map[string]string, len(in.Decision.Subject.Labels))
	for key, value := range in.Decision.Subject.Labels {
		out.Decision.Subject.Labels[key] = value
	}
	out.Decision.Plan.Members = append([]modelroute.Member(nil), in.Decision.Plan.Members...)
	return out
}

func decodeInputTriggerReceiptStrict(data []byte, target any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values are not allowed")
		}
		return err
	}
	return nil
}

// RouteWatcher exposes the installed route-manifest watcher to trusted control-plane
// adapters. A nil result means hot reload was not configured for this server.
func (s *Server) RouteWatcher() *modelroute.Watcher {
	if s == nil {
		return nil
	}
	return s.currentRouteWatcher()
}
