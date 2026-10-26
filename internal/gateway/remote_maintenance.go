package gateway

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/remotemaintresult"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

const remoteMaintenanceTool = "fak_remote_maintenance"

// SetRemoteMaintenanceAuthority installs the trusted host authority used by the
// native maintenance entry point. A nil authority restores lazy loading from
// the operator-owned default configuration.
func (s *Server) SetRemoteMaintenanceAuthority(authority remotemaint.Authority) {
	if s == nil {
		return
	}
	s.remoteMaintenanceMu.Lock()
	s.remoteMaintenanceAuthority = authority
	s.remoteMaintenanceMu.Unlock()
}

func (s *Server) remoteMaintenanceAuthorityForCall() (remotemaint.Authority, error) {
	s.remoteMaintenanceMu.RLock()
	authority := s.remoteMaintenanceAuthority
	s.remoteMaintenanceMu.RUnlock()
	if authority != nil {
		return authority, nil
	}
	configPath, err := remotemaint.DefaultConfigPath()
	if err != nil {
		return nil, fmt.Errorf("remote maintenance authority: %w", err)
	}
	authority, err = remotemaint.LoadAuthority(configPath)
	if err != nil {
		return nil, fmt.Errorf("remote maintenance authority: %w", err)
	}
	return authority, nil
}

func validateRemoteMaintenanceRequest(ctx context.Context, call *abi.ToolCall) (remotemaint.Request, error) {
	// read_only is an untrusted syscall hint. This operation always carries
	// mutating flags on the actual kernel call that Decide sees.
	if call.Meta == nil {
		call.Meta = make(map[string]string)
	}
	call.Meta["readOnlyHint"] = "false"
	call.Meta["idempotentHint"] = "false"
	call.Meta["destructive"] = "true"
	return remotemaint.ParseRequest(resolveBytes(ctx, call.Args))
}

func (s *Server) prepareRemoteMaintenance(ctx context.Context, request remotemaint.Request) (*remotemaint.Prepared, error) {
	authority, err := s.remoteMaintenanceAuthorityForCall()
	if err != nil {
		return nil, err
	}
	return remotemaint.Prepare(ctx, authority, request)
}

func (s *Server) remoteMaintenanceScopeCalls(ctx context.Context, wireCall *abi.ToolCall, prepared *remotemaint.Prepared) ([]*abi.ToolCall, error) {
	grant := prepared.Grant()
	facts, err := adjudicator.ProjectRemoteMaintenancePolicyFacts(grant)
	if err != nil {
		return nil, err
	}
	fullFactsJSON, err := json.Marshal(facts)
	if err != nil {
		return nil, err
	}
	wireHash := sha256.Sum256(resolveBytes(ctx, wireCall.Args))
	factsHash := sha256.Sum256(fullFactsJSON)
	calls := make([]*abi.ToolCall, 0, len(facts))
	for _, fact := range facts {
		factJSON, err := json.Marshal(fact)
		if err != nil {
			return nil, err
		}
		ref, err := abi.ActiveResolver().Put(ctx, factJSON)
		if err != nil {
			return nil, fmt.Errorf("resolver: %w", err)
		}
		scopeCall := *wireCall
		// The original wire alias has already earned kernel ALLOW before this
		// helper is reached. Canonicalize only the derived trusted-scope call so
		// write-shaped path floors cannot be skipped by a policy-recognized alias.
		scopeCall.Tool = remoteMaintenanceTool
		scopeCall.Args = ref
		scopeCall.Meta = make(map[string]string, len(wireCall.Meta)+4)
		for key, value := range wireCall.Meta {
			scopeCall.Meta[key] = value
		}
		scopeCall.Meta["remote_maintenance_wire_args_sha256"] = fmt.Sprintf("%x", wireHash)
		scopeCall.Meta["remote_maintenance_grant_sha256"] = remotemaint.GrantDigest(grant)
		scopeCall.Meta["remote_maintenance_policy_facts_sha256"] = fmt.Sprintf("%x", factsHash)
		scopeCall.Meta["remote_maintenance_resource_index"] = fmt.Sprintf("%d", fact.ResourceIndex)
		calls = append(calls, &scopeCall)
	}
	return calls, nil
}

func (s *Server) adjudicateRemoteMaintenance(ctx context.Context, call *abi.ToolCall) (WireVerdict, error) {
	request, err := validateRemoteMaintenanceRequest(ctx, call)
	if err != nil {
		return WireVerdict{}, err
	}
	initial := s.k.Decide(ctx, call)
	if initial.Kind != abi.VerdictAllow {
		return renderVerdict(initial, nil), nil
	}
	prepared, err := s.prepareRemoteMaintenance(ctx, request)
	if err != nil {
		return WireVerdict{}, err
	}
	scopeCalls, err := s.remoteMaintenanceScopeCalls(ctx, call, prepared)
	if err != nil {
		return WireVerdict{}, err
	}
	for _, scopeCall := range scopeCalls {
		verdict := s.k.Decide(ctx, scopeCall)
		if verdict.Kind != abi.VerdictAllow {
			return renderVerdict(verdict, nil), nil
		}
	}
	return renderVerdict(initial, nil), nil
}

func (s *Server) syscallRemoteMaintenance(ctx context.Context, call *abi.ToolCall) (WireVerdict, *ResultEnvelope, error) {
	request, err := validateRemoteMaintenanceRequest(ctx, call)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	if call.SeqNo == 0 {
		call.SeqNo = s.nextOriginSeq()
	}
	verdict := s.k.Decide(ctx, call)
	if verdict.Kind != abi.VerdictAllow {
		return renderVerdict(verdict, nil), nil, nil
	}
	prepared, err := s.prepareRemoteMaintenance(ctx, request)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	scopeCalls, err := s.remoteMaintenanceScopeCalls(ctx, call, prepared)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	for _, scopeCall := range scopeCalls {
		verdict = s.k.Decide(ctx, scopeCall)
		if verdict.Kind != abi.VerdictAllow {
			return renderVerdict(verdict, nil), nil, nil
		}
	}
	scopeCall := scopeCalls[len(scopeCalls)-1]
	knownGrant := prepared.Grant()
	attemptedAt := time.Now().UTC()
	safeUnknownReceipt := remoteMaintenanceUnknownEffectReceipt(knownGrant, attemptedAt, attemptedAt)
	receipt, executeErr := prepared.Execute(ctx)
	structuredFailure := executeErr != nil && receipt.GrantDigest != ""
	if executeErr != nil && !structuredFailure {
		// A zero receipt means execution did not produce bound effect evidence.
		return renderVerdict(verdict, nil), nil, executeErr
	}
	body, err := json.Marshal(receipt)
	if err != nil {
		return renderVerdict(verdict, nil), nil, err
	}
	result := &abi.Result{
		Call:   scopeCall,
		Status: abi.StatusOK,
		Payload: abi.Ref{
			Kind: abi.RefInline, Inline: body, Len: int64(len(body)),
			Taint: abi.TaintTainted, Scope: abi.ScopeAgent,
		},
		Meta: map[string]string{
			"engine": "fak-mcp", "served_by": "native", "tool": remoteMaintenanceTool,
			"remote_maintenance_grant":         receipt.GrantDigest,
			"remote_maintenance_authorization": verdictKindName(verdict.Kind),
		},
	}
	executionError := structuredFailure || receipt.Outcome == remotemaint.OutcomeRolledBack || receipt.Outcome == remotemaint.OutcomeUnknownEffect || receipt.Outcome == remotemaint.OutcomeRollbackFailed
	if executionError {
		result.Status = abi.StatusError
		result.Meta["remote_maintenance_outcome"] = receipt.Outcome
		result.Meta["remote_maintenance_backup"] = string(receipt.BackupDisposition)
	}
	admissionCtx := remotemaintresult.Bind(ctx, scopeCall, knownGrant, body)
	admitted := s.k.AdmitResult(admissionCtx, scopeCall, result)
	s.rememberOriginSeq(call.TraceID, call.Tool, string(resolveBytes(ctx, call.Args)), call.SeqNo)
	if result.Meta == nil {
		result.Meta = make(map[string]string)
	}
	result.Meta["remote_maintenance_authorization"] = verdictKindName(verdict.Kind)
	result.Meta["remote_maintenance_authorization_by"] = verdict.By
	result.Meta["remote_maintenance_result_admission"] = verdictKindName(admitted.Kind)
	result.Meta["remote_maintenance_result_admission_by"] = admitted.By
	if admitted.Reason != abi.ReasonNone {
		result.Meta["remote_maintenance_result_admission_reason"] = abi.ReasonName(admitted.Reason)
	}
	if admitted.Kind == abi.VerdictDeny {
		// Execution was already attempted. Keep the original authorization and
		// replace the denied result only with identity known from Prepared before
		// the effect boundary; the rejected body is never re-exposed.
		receipt = safeUnknownReceipt
		body, marshalErr := json.Marshal(receipt)
		if marshalErr != nil {
			return renderVerdict(verdict, nil), nil, marshalErr
		}
		result.Status = abi.StatusError
		result.Payload = abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body)), Taint: abi.TaintTainted, Scope: abi.ScopeAgent}
		result.Meta["engine"] = "fak-mcp"
		result.Meta["served_by"] = "native"
		result.Meta["tool"] = remoteMaintenanceTool
		result.Meta["remote_maintenance_grant"] = receipt.GrantDigest
		result.Meta["remote_maintenance_outcome"] = receipt.Outcome
		result.Meta["remote_maintenance_backup"] = string(receipt.BackupDisposition)
		result.Meta["remote_maintenance_effect"] = "attempted_unknown"
	}
	// AdmitResult may replace the payload with a quarantine pointer, a transform,
	// or a deny-as-value. Resolve only the post-admission Result; never retain the
	// original receipt bytes as a side channel around that floor.
	envelope := &ResultEnvelope{Status: statusName(result.Status), Content: string(resolveBytes(ctx, result.Payload)), Meta: result.Meta}
	// The wire verdict is the pre-effect authorization. Result admission remains
	// separately visible in the envelope metadata and never rewrites that fact.
	wireVerdict := renderVerdict(verdict, nil)
	if executionError || admitted.Kind == abi.VerdictDeny {
		wireVerdict.Detail = map[string]string{
			"execution":        "attempted",
			"outcome":          receipt.Outcome,
			"result_admission": verdictKindName(admitted.Kind),
		}
	}
	return wireVerdict, envelope, nil
}

func remoteMaintenanceUnknownEffectReceipt(grant remotemaint.Grant, startedAt, endedAt time.Time) remotemaint.Receipt {
	return remotemaint.Receipt{
		Schema: remotemaint.ReceiptSchema, GrantDigest: remotemaint.GrantDigest(grant), ProfileID: grant.ProfileID,
		OperationDigest: grant.OperationDigest, ContextID: grant.ContextID, Target: grant.Target, HostKey: grant.HostKey,
		Operation: grant.Operation, BootID: grant.BootID, StartedAt: startedAt.UTC(), EndedAt: endedAt.UTC(),
		Outcome: remotemaint.OutcomeUnknownEffect, Verified: false, BackupDisposition: remotemaint.BackupUnknown,
	}
}

// callRemoteMaintenance is the direct MCP bridge. The direct tool and a nested
// fak_syscall therefore share the same kernel decision, execution, and result
// admission path.
func (s *Server) callRemoteMaintenance(ctx context.Context, raw json.RawMessage) (SyscallResponse, error) {
	traceID := s.traceFor("")
	s.bindTraceOwner(traceID, principalFromContext(ctx))
	verdict, result, err := s.syscall(ctx, remoteMaintenanceTool, string(raw), false, "", traceID)
	if err != nil {
		return SyscallResponse{}, err
	}
	return SyscallResponse{Verdict: verdict, Result: result, TraceID: traceID}, nil
}
