package workflowoutcomestudy

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
)

const Schema = "fak-workflow-outcome-study/1"

type Usage struct {
	InputTokens       int64 `json:"input_tokens"`
	CachedInputTokens int64 `json:"cached_input_tokens"`
	OutputTokens      int64 `json:"output_tokens"`
}

type Task struct {
	ID     string   `json:"id"`
	Class  string   `json:"class"`
	Prompt string   `json:"prompt"`
	Rubric []string `json:"rubric"`
}

type ArmResult struct {
	TaskID         string `json:"task_id"`
	Arm            string `json:"arm"`
	CandidateID    string `json:"candidate_id"`
	ProducerID     string `json:"producer_id"`
	Model          string `json:"model"`
	Effort         string `json:"effort"`
	BudgetTokens   int64  `json:"budget_tokens"`
	Completed      bool   `json:"completed"`
	ElapsedMS      int64  `json:"elapsed_ms"`
	ArtifactDigest string `json:"artifact_digest"`
	Usage          Usage  `json:"usage"`
	Failure        string `json:"failure,omitempty"`
}

type Grade struct {
	TaskID      string `json:"task_id"`
	CandidateID string `json:"candidate_id"`
	GraderID    string `json:"grader_id"`
	Correctness int    `json:"correctness"`
	Usefulness  int    `json:"usefulness"`
	Rationale   string `json:"rationale"`
}

type Study struct {
	Schema  string      `json:"schema"`
	StudyID string      `json:"study_id"`
	Tasks   []Task      `json:"tasks"`
	Results []ArmResult `json:"results"`
	Grades  []Grade     `json:"grades"`
}

type ArmSummary struct {
	Completed   int   `json:"completed"`
	Failed      int   `json:"failed"`
	ElapsedMS   int64 `json:"elapsed_ms"`
	Usage       Usage `json:"usage"`
	Correctness int   `json:"correctness"`
	Usefulness  int   `json:"usefulness"`
	Grades      int   `json:"grades"`
}

type Report struct {
	Schema          string                `json:"schema"`
	StudyID         string                `json:"study_id"`
	TaskDigest      string                `json:"task_digest"`
	TaskCount       int                   `json:"task_count"`
	EquivalentPairs int                   `json:"equivalent_pairs"`
	CompletePairs   int                   `json:"complete_pairs"`
	BlindGrades     int                   `json:"blind_grades"`
	Arms            map[string]ArmSummary `json:"arms"`
	GainClaimReady  bool                  `json:"gain_claim_ready"`
	EvidenceNote    string                `json:"evidence_note"`
}

func Analyze(s Study) (Report, error) {
	if s.Schema != Schema || strings.TrimSpace(s.StudyID) == "" {
		return Report{}, errors.New("invalid workflow outcome study identity")
	}
	tasks := map[string]Task{}
	for _, task := range s.Tasks {
		if task.ID == "" || task.Prompt == "" || len(task.Rubric) == 0 {
			return Report{}, fmt.Errorf("task %q lacks prompt or rubric", task.ID)
		}
		if _, exists := tasks[task.ID]; exists {
			return Report{}, fmt.Errorf("duplicate task %q", task.ID)
		}
		tasks[task.ID] = task
	}
	byTask := map[string]map[string]ArmResult{}
	candidateArm := map[string]string{}
	producerByBlindID := map[string]string{}
	for _, row := range s.Results {
		if _, ok := tasks[row.TaskID]; !ok {
			return Report{}, fmt.Errorf("result names unknown task %q", row.TaskID)
		}
		if row.Arm != "direct" && row.Arm != "workflow" {
			return Report{}, fmt.Errorf("result %s has invalid arm %q", row.TaskID, row.Arm)
		}
		if row.CandidateID == "" || row.ProducerID == "" || row.Model == "" || row.Effort == "" || row.BudgetTokens <= 0 {
			return Report{}, fmt.Errorf("result %s/%s lacks frozen envelope, producer, or candidate", row.TaskID, row.Arm)
		}
		if row.CandidateID == "direct" || row.CandidateID == "workflow" || row.ElapsedMS < 0 || row.Usage.InputTokens < 0 || row.Usage.CachedInputTokens < 0 || row.Usage.OutputTokens < 0 {
			return Report{}, fmt.Errorf("result %s/%s has an unblinded candidate or negative measurement", row.TaskID, row.Arm)
		}
		if byTask[row.TaskID] == nil {
			byTask[row.TaskID] = map[string]ArmResult{}
		}
		if _, exists := byTask[row.TaskID][row.Arm]; exists {
			return Report{}, fmt.Errorf("duplicate result %s/%s", row.TaskID, row.Arm)
		}
		byTask[row.TaskID][row.Arm] = row
		if _, exists := candidateArm[row.TaskID+"\x00"+row.CandidateID]; exists {
			return Report{}, fmt.Errorf("duplicate candidate %s/%s", row.TaskID, row.CandidateID)
		}
		candidateArm[row.TaskID+"\x00"+row.CandidateID] = row.Arm
		producerByBlindID[row.TaskID+"\x00"+row.CandidateID] = row.ProducerID
	}
	report := Report{Schema: "fak-workflow-outcome-report/1", StudyID: s.StudyID, TaskDigest: taskDigest(s.Tasks), TaskCount: len(s.Tasks), Arms: map[string]ArmSummary{"direct": {}, "workflow": {}}, EvidenceNote: "gain claims require every frozen task to have equivalent completed arms and blind grades; provider usage is observed, not FAK-authored"}
	for id := range tasks {
		arms := byTask[id]
		direct, dok := arms["direct"]
		workflow, wok := arms["workflow"]
		if !dok || !wok {
			continue
		}
		if direct.Model != workflow.Model || direct.Effort != workflow.Effort || direct.BudgetTokens != workflow.BudgetTokens {
			return Report{}, fmt.Errorf("task %s arms do not share model/effort/budget envelope", id)
		}
		report.EquivalentPairs++
		if direct.Completed && workflow.Completed && direct.Failure == "" && workflow.Failure == "" && direct.ArtifactDigest != "" && workflow.ArtifactDigest != "" {
			report.CompletePairs++
		}
	}
	for _, row := range s.Results {
		summary := report.Arms[row.Arm]
		if row.Completed && row.Failure == "" {
			summary.Completed++
		} else {
			summary.Failed++
		}
		summary.ElapsedMS += row.ElapsedMS
		summary.Usage.InputTokens += row.Usage.InputTokens
		summary.Usage.CachedInputTokens += row.Usage.CachedInputTokens
		summary.Usage.OutputTokens += row.Usage.OutputTokens
		report.Arms[row.Arm] = summary
	}
	seenGrade := map[string]bool{}
	for _, grade := range s.Grades {
		arm, ok := candidateArm[grade.TaskID+"\x00"+grade.CandidateID]
		if !ok {
			return Report{}, fmt.Errorf("grade names unknown blinded candidate %s/%s", grade.TaskID, grade.CandidateID)
		}
		if grade.GraderID == producerByBlindID[grade.TaskID+"\x00"+grade.CandidateID] {
			return Report{}, fmt.Errorf("candidate %s/%s was self-graded", grade.TaskID, grade.CandidateID)
		}
		if grade.GraderID == "" || grade.Correctness < 0 || grade.Correctness > 4 || grade.Usefulness < 0 || grade.Usefulness > 4 || grade.Rationale == "" {
			return Report{}, fmt.Errorf("invalid grade %s/%s", grade.TaskID, grade.CandidateID)
		}
		key := grade.TaskID + "\x00" + grade.CandidateID
		if seenGrade[key] {
			return Report{}, fmt.Errorf("duplicate grade %s/%s", grade.TaskID, grade.CandidateID)
		}
		seenGrade[key] = true
		summary := report.Arms[arm]
		summary.Correctness += grade.Correctness
		summary.Usefulness += grade.Usefulness
		summary.Grades++
		report.Arms[arm] = summary
		report.BlindGrades++
	}
	report.GainClaimReady = len(tasks) > 0 && report.EquivalentPairs == len(tasks) && report.CompletePairs == len(tasks) && report.BlindGrades == 2*len(tasks)
	return report, nil
}

func taskDigest(tasks []Task) string {
	rows := make([]string, 0, len(tasks))
	for _, t := range tasks {
		rows = append(rows, t.ID+"\x00"+t.Class+"\x00"+t.Prompt+"\x00"+strings.Join(t.Rubric, "\x00"))
	}
	sort.Strings(rows)
	sum := sha256.Sum256([]byte(strings.Join(rows, "\n")))
	return "sha256:" + hex.EncodeToString(sum[:])
}
