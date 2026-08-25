package trajectory

// QwenMutationKind identifies an observed mutation-stream event.
type QwenMutationKind string

const (
	QwenMutationWrite   QwenMutationKind = "write"
	QwenMutationWitness QwenMutationKind = "witness"
)

// QwenMutationEvent is one ordered observation from a transcript.
type QwenMutationEvent struct {
	TranscriptID    string
	Target          string
	Kind            QwenMutationKind
	AccountedTokens uint64
	HypothesisID    string
}

// QwenMutationIntervention describes the non-enforcing response to detected churn.
type QwenMutationIntervention string

const QwenMutationObserveReproFirst QwenMutationIntervention = "observe-only: reproduce before intervening"

// QwenMutationChurn summarizes a maximal run of repeated unwitnessed writes.
type QwenMutationChurn struct {
	TranscriptID    string
	Target          string
	Count           int
	AccountedTokens uint64
	Intervention    QwenMutationIntervention
}

// DetectQwenMutationChurn finds consecutive writes by one transcript to one target
// under an unchanged hypothesis. Results retain the event-stream order. Any event
// outside the run, including a witness or a write to another target, ends the run.
func DetectQwenMutationChurn(events []QwenMutationEvent) []QwenMutationChurn {
	var churn []QwenMutationChurn
	var run QwenMutationEvent
	count := 0
	var tokens uint64

	flush := func() {
		if count >= 2 {
			churn = append(churn, QwenMutationChurn{
				TranscriptID:    run.TranscriptID,
				Target:          run.Target,
				Count:           count,
				AccountedTokens: tokens,
				Intervention:    QwenMutationObserveReproFirst,
			})
		}
		count = 0
		tokens = 0
	}

	for _, event := range events {
		if event.Kind != QwenMutationWrite {
			flush()
			continue
		}
		if count > 0 && (event.TranscriptID != run.TranscriptID || event.Target != run.Target || event.HypothesisID != run.HypothesisID) {
			flush()
		}
		if count == 0 {
			run = event
		}
		count++
		tokens += event.AccountedTokens
	}
	flush()

	return churn
}
