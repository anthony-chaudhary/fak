package dojo

import "fmt"

// claim_microcontext.go is the microcontext calibration cell (#11654):
// measuring microcontext_elision_ratio (the fraction of eligible raw tool-output tokens
// that microcontext elision shed to CAS) and client_resubmission_loop_rate (an intentional
// floor defending against client runaway re-submission loops induced by aggressive elision).

// The anchored literals for this cell.
var _ = RegisterClaim("microcontext", "microcontext_elision_ratio", claim(0.50,
	"seed theory (#11654): microcontext elision sheds ~50% of raw tool-output tokens across long subturns into CAS while preserving the working set; a genuine estimate the RSI loop recalibrates toward measured reality"))

var _ = RegisterClaim("microcontext", "client_resubmission_loop_rate", floor(0.0, true,
	"zero-tolerance safety floor (#11654): aggressive microcontext elision must NEVER induce client tool resubmission loops; any loop breaches this floor"))

// MicrocontextLedger is the reduced view of microcontext elision telemetry.
type MicrocontextLedger struct {
	// RawToolOutputTokens is the total volume of raw tool-result tokens before elision.
	RawToolOutputTokens int
	// ElidedToolOutputTokens is the volume of tool-result tokens paged out to CAS.
	ElidedToolOutputTokens int
	// ElisionRecorded indicates whether the telemetry source relays elision byte/token fields.
	ElisionRecorded bool
	// TotalSubturns is the total count of subturns observed.
	TotalSubturns int
	// ResubmissionLoops is the count of client tool resubmission loops detected after elisions.
	ResubmissionLoops int
	// LoopRecorded indicates whether client tool loop detection was active and recorded.
	LoopRecorded bool
}

// MicrocontextEpisodes folds the microcontext ledger into dojo ScoredInputs:
// one for microcontext_elision_ratio, and one for client_resubmission_loop_rate.
func MicrocontextEpisodes(led MicrocontextLedger) []ScoredInput {
	var episodes []ScoredInput

	// 1. microcontext_elision_ratio
	elisionPred := Registry.MustPredict("microcontext", "microcontext_elision_ratio", "fraction")
	if !led.ElisionRecorded || led.RawToolOutputTokens <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: elisionPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   led.RawToolOutputTokens,
				Source:   "no raw tool output tokens recorded in microcontext telemetry — elision ratio is UNMEASURED",
			},
		})
	} else {
		elided := led.ElidedToolOutputTokens
		if elided < 0 {
			elided = 0
		}
		if elided > led.RawToolOutputTokens {
			elided = led.RawToolOutputTokens
		}
		ratio := float64(elided) / float64(led.RawToolOutputTokens)
		episodes = append(episodes, ScoredInput{
			Prediction: elisionPred,
			Outcome: Outcome{
				Realized:   ratio,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     led.RawToolOutputTokens,
				Source: fmt.Sprintf("%d of %d raw tool tokens elided to CAS (WITNESSED)",
					elided, led.RawToolOutputTokens),
			},
		})
	}

	// 2. client_resubmission_loop_rate
	loopPred := Registry.MustPredict("microcontext", "client_resubmission_loop_rate", "fraction")
	if !led.LoopRecorded || led.TotalSubturns <= 0 {
		episodes = append(episodes, ScoredInput{
			Prediction: loopPred,
			Outcome: Outcome{
				Measured: false,
				Sample:   led.TotalSubturns,
				Source:   "no subturns recorded in loop monitoring telemetry — loop rate is UNMEASURED",
			},
		})
	} else {
		loops := led.ResubmissionLoops
		if loops < 0 {
			loops = 0
		}
		loopRate := float64(loops) / float64(led.TotalSubturns)
		episodes = append(episodes, ScoredInput{
			Prediction: loopPred,
			Outcome: Outcome{
				Realized:   loopRate,
				Provenance: Witnessed,
				Measured:   true,
				Sample:     led.TotalSubturns,
				Source: fmt.Sprintf("%d resubmission loop(s) across %d subturn(s) (WITNESSED)",
					loops, led.TotalSubturns),
			},
		})
	}

	return episodes
}
