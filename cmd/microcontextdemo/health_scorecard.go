package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

const healthSchema = "fak-microcontext-health-scorecard/1"

type healthGrade struct {
	Schema           string   `json:"schema"`
	Grade            string   `json:"grade"`
	Score            int      `json:"score"`
	Evidence         string   `json:"evidence"`
	Submitted        int      `json:"submitted"`
	Success          int      `json:"success"`
	Errors           int      `json:"errors"`
	Refusals         int      `json:"refusals"`
	FailureRate      float64  `json:"failure_rate"`
	VerificationRate float64  `json:"verification_rate"`
	UsefulPerSecond  float64  `json:"useful_per_wall_second"`
	Drift            string   `json:"drift"`
	Reasons          []string `json:"reasons"`
}

func gradeHealth(input string) (healthGrade, error) {
	b, e := os.ReadFile(input)
	if e != nil {
		return healthGrade{}, e
	}
	var l microagent.QualityLedger
	if e = json.Unmarshal(b, &l); e != nil {
		return healthGrade{}, e
	}
	if e = microagent.VerifyQualityLedger(l); e != nil {
		return healthGrade{}, e
	}
	r := healthGrade{Schema: healthSchema, Evidence: input, Submitted: l.Submitted, Success: l.Outcomes["success"], Errors: l.Outcomes["error"], Refusals: l.Outcomes["refusal"], UsefulPerSecond: l.ClaimFamilies.UsefulWork.PerWallSecond, Drift: "baseline-only; second comparable ledger required for trend"}
	r.FailureRate = float64(r.Errors+r.Refusals) / float64(r.Submitted)
	r.VerificationRate = float64(l.Verification.Passed) / float64(l.Submitted)
	r.Score = 100
	if r.Submitted < 1000 {
		r.Score -= 20
		r.Reasons = append(r.Reasons, "fewer than 1,000 submitted contexts")
	}
	if r.FailureRate > 0 {
		r.Score -= 40
		r.Reasons = append(r.Reasons, "nonzero terminal failure/refusal rate")
	}
	if r.VerificationRate < 1 {
		r.Score -= 40
		r.Reasons = append(r.Reasons, "independent verification below 100%")
	}
	if r.UsefulPerSecond <= 0 {
		r.Score -= 20
		r.Reasons = append(r.Reasons, "useful-work throughput absent")
	}
	switch {
	case r.Score >= 90:
		r.Grade = "A"
	case r.Score >= 75:
		r.Grade = "B"
	case r.Score >= 60:
		r.Grade = "C"
	default:
		r.Grade = "F"
	}
	if len(r.Reasons) == 0 {
		r.Reasons = []string{"1,000-context floor met", "zero terminal failures/refusals", "100% independent verification", "positive useful-work throughput"}
	}
	return r, verifyHealthGrade(r)
}
func verifyHealthGrade(r healthGrade) error {
	if r.Schema != healthSchema || r.Grade == "" || r.Score < 0 || r.Score > 100 || r.Submitted <= 0 || r.Success+r.Errors+r.Refusals != r.Submitted || r.FailureRate < 0 || r.VerificationRate <= 0 || r.UsefulPerSecond <= 0 || r.Drift == "" || len(r.Reasons) == 0 {
		return errors.New("microcontext health scorecard invariants failed")
	}
	return nil
}
func writeHealthGrade(input, output string) error {
	r, e := gradeHealth(input)
	if e != nil {
		return e
	}
	b, _ := json.MarshalIndent(r, "", "  ")
	if output == "-" {
		fmt.Println(string(b))
		return nil
	}
	return os.WriteFile(output, append(b, '\n'), 0644)
}
func verifyHealthArtifact(path string) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	var r healthGrade
	if e = json.Unmarshal(b, &r); e != nil {
		return e
	}
	return verifyHealthGrade(r)
}
