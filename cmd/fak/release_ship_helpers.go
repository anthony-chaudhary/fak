package main

import (
	"encoding/json"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/workdelivery"
)

// releaseShipPayloadOK reads the "ok" verdict out of a release-ship gate payload. An absent
// payload is NOT ok: a gate that never ran must never read as a gate that passed -- which is
// the whole reason these two gates need a reader of their own rather than a bare
// payload["ok"] assertion. The source-CI and the target-ancestry gates publish the same
// shaped map and both can be skipped, so one reader serves both.
func releaseShipPayloadOK(payload map[string]any) bool {
	if payload == nil {
		return false
	}
	ok, _ := payload["ok"].(bool)
	return ok
}

func releaseShipNeedsTargetAncestry(opts releaseShipOptions) bool {
	source := strings.TrimSpace(opts.sourceBranch)
	target := strings.TrimSpace(opts.trunk)
	return source != "" && target != "" && source != target
}

func releaseShipDirectPromotionRequiresOpenPR(opts releaseShipOptions) bool {
	source := strings.TrimSpace(opts.sourceBranch)
	target := strings.TrimSpace(opts.trunk)
	return opts.execute && !opts.openPR && !opts.allowDirectPromotion && source != "" && target != "" && source != target
}

func releaseShipTargetAncestry(result *releaseShipResult, root string, opts releaseShipOptions, sourceSHA string) map[string]any {
	targetRef := opts.remote + "/" + opts.trunk
	if strings.TrimSpace(sourceSHA) == "" {
		return map[string]any{"ok": false, "status": "missing_source_sha", "target_ref": targetRef}
	}
	code, out := releaseShipCmd(result, root, "git", []string{"rev-parse", "--verify", targetRef + "^{commit}"}, nil, time.Minute)
	if code != 0 {
		return map[string]any{"ok": false, "status": "target_unresolvable", "target_ref": targetRef, "tail": tail(out)}
	}
	targetSHA := strings.TrimSpace(out)
	payload := map[string]any{
		"target_ref": targetRef,
		"target_sha": targetSHA,
		"source_ref": opts.sourceBranch,
		"source_sha": sourceSHA,
	}
	if sameSHA(targetSHA, sourceSHA) {
		payload["ok"] = true
		payload["status"] = "same"
		return payload
	}
	code, out = releaseShipCmd(result, root, "git", []string{"merge-base", "--is-ancestor", targetSHA, sourceSHA}, nil, time.Minute)
	switch code {
	case 0:
		payload["ok"] = true
		payload["status"] = "ancestor"
	case 1:
		payload["ok"] = false
		payload["status"] = "non_fast_forward"
	default:
		payload["ok"] = false
		payload["status"] = "ancestor_check_failed"
		payload["tail"] = tail(out)
	}
	return payload
}

func cleanupReleaseShipWorktree(root, wt string) map[string]any {
	code, out := releaseShipRunCommand(root, "git", []string{"worktree", "remove", "--force", wt}, nil, 5*time.Minute)
	payload := map[string]any{
		"ok":        code == 0,
		"path":      wt,
		"exit_code": code,
	}
	if code != 0 {
		payload["tail"] = tail(out)
	}
	pruneCode, pruneOut := releaseShipRunCommand(root, "git", []string{"worktree", "prune"}, nil, time.Minute)
	payload["prune_exit_code"] = pruneCode
	if pruneCode != 0 {
		payload["prune_tail"] = tail(pruneOut)
	}
	return payload
}

func readReleaseReadiness(path string) (workdelivery.WorkUnit, workdelivery.Receipt, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return workdelivery.WorkUnit{}, workdelivery.Receipt{}, err
	}
	var envelope struct {
		Unit          workdelivery.WorkUnit                `json:"unit"`
		Receipt       workdelivery.Receipt                 `json:"receipt"`
		Qualification *installedLaunchQualificationReceipt `json:"installed_launch_qualification,omitempty"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return workdelivery.WorkUnit{}, workdelivery.Receipt{}, err
	}
	if err := envelope.Unit.Validate(); err != nil {
		return workdelivery.WorkUnit{}, workdelivery.Receipt{}, err
	}
	if err := validateInstalledLaunchQualification(envelope.Qualification); err != nil {
		return workdelivery.WorkUnit{}, workdelivery.Receipt{}, err
	}
	return envelope.Unit, envelope.Receipt, nil
}
