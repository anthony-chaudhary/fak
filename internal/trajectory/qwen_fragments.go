package trajectory

import (
	"fmt"
	"sort"
)

// QwenFragmentSourceKind identifies the transcript format that supplied a fragment.
type QwenFragmentSourceKind string

const (
	QwenFragmentClaude QwenFragmentSourceKind = "claude"
	QwenFragmentCodex  QwenFragmentSourceKind = "codex"
)

// QwenFragmentMessage carries the source-assigned message identity and its exact usage.
type QwenFragmentMessage struct {
	ID    string
	Usage QwenCanonicalUsage
}

// QwenFragment is one source file or split-file portion of a transcript.
type QwenFragment struct {
	SourceKind   QwenFragmentSourceKind
	SourcePath   string
	FragmentID   string
	TranscriptID string
	SessionID    string
	Messages     []QwenFragmentMessage
}

// QwenFragmentProvenance identifies every fragment contributing to a transcript.
type QwenFragmentProvenance struct {
	SourcePath string
	FragmentID string
}

// QwenCanonicalTranscript is one source-kind-specific transcript with deduplicated usage.
type QwenCanonicalTranscript struct {
	SourceKind QwenFragmentSourceKind
	Identity   string
	Usage      QwenCanonicalUsage
	Provenance []QwenFragmentProvenance
}

// QwenFragmentCanonicalization reports both the input and canonicalized cardinalities.
type QwenFragmentCanonicalization struct {
	RawFragmentCount         int
	CanonicalTranscriptCount int
	Transcripts              []QwenCanonicalTranscript
}

// QwenFragmentRefusalCode classifies identity failures without relying on error text.
type QwenFragmentRefusalCode string

const (
	QwenFragmentIdentityMissing   QwenFragmentRefusalCode = "missing_identity"
	QwenFragmentIdentityAmbiguous QwenFragmentRefusalCode = "ambiguous_identity"
)

// QwenFragmentRefusal is returned when source identity cannot be resolved exactly.
type QwenFragmentRefusal struct {
	Code          QwenFragmentRefusalCode
	FragmentIndex int
	MessageID     string
	Detail        string
}

func (r *QwenFragmentRefusal) Error() string {
	return fmt.Sprintf("qwen fragment %d: %s: %s", r.FragmentIndex, r.Code, r.Detail)
}

type qwenTranscriptAccumulator struct {
	transcript QwenCanonicalTranscript
	messages   map[string]QwenCanonicalUsage
}

// CanonicalizeQwenFragments groups fragments only by supplied source identity and rolls up
// usage once per source-assigned message ID. It never compares message content.
func CanonicalizeQwenFragments(fragments []QwenFragment) (QwenFragmentCanonicalization, error) {
	result := QwenFragmentCanonicalization{RawFragmentCount: len(fragments)}
	byIdentity := make(map[string]*qwenTranscriptAccumulator)

	for i, fragment := range fragments {
		identity, err := qwenFragmentIdentity(fragment, i)
		if err != nil {
			return QwenFragmentCanonicalization{}, err
		}
		key := string(fragment.SourceKind) + "\x00" + identity
		acc := byIdentity[key]
		if acc == nil {
			acc = &qwenTranscriptAccumulator{
				transcript: QwenCanonicalTranscript{SourceKind: fragment.SourceKind, Identity: identity},
				messages:   make(map[string]QwenCanonicalUsage),
			}
			byIdentity[key] = acc
		}
		acc.transcript.Provenance = append(acc.transcript.Provenance, QwenFragmentProvenance{
			SourcePath: fragment.SourcePath,
			FragmentID: fragment.FragmentID,
		})
		for _, message := range fragment.Messages {
			if message.ID == "" {
				return QwenFragmentCanonicalization{}, &QwenFragmentRefusal{Code: QwenFragmentIdentityMissing, FragmentIndex: i, Detail: "message ID is required"}
			}
			if previous, ok := acc.messages[message.ID]; ok {
				if previous != message.Usage {
					return QwenFragmentCanonicalization{}, &QwenFragmentRefusal{Code: QwenFragmentIdentityAmbiguous, FragmentIndex: i, MessageID: message.ID, Detail: "duplicate message ID has conflicting usage"}
				}
				continue
			}
			acc.messages[message.ID] = message.Usage
			addQwenCanonicalUsage(&acc.transcript.Usage, message.Usage)
		}
	}

	result.Transcripts = make([]QwenCanonicalTranscript, 0, len(byIdentity))
	for _, acc := range byIdentity {
		sort.SliceStable(acc.transcript.Provenance, func(i, j int) bool {
			left, right := acc.transcript.Provenance[i], acc.transcript.Provenance[j]
			if left.SourcePath != right.SourcePath {
				return left.SourcePath < right.SourcePath
			}
			return left.FragmentID < right.FragmentID
		})
		result.Transcripts = append(result.Transcripts, acc.transcript)
	}
	sort.Slice(result.Transcripts, func(i, j int) bool {
		if result.Transcripts[i].SourceKind != result.Transcripts[j].SourceKind {
			return result.Transcripts[i].SourceKind < result.Transcripts[j].SourceKind
		}
		return result.Transcripts[i].Identity < result.Transcripts[j].Identity
	})
	result.CanonicalTranscriptCount = len(result.Transcripts)
	return result, nil
}

func qwenFragmentIdentity(fragment QwenFragment, index int) (string, error) {
	switch fragment.SourceKind {
	case QwenFragmentClaude:
		if fragment.TranscriptID == "" {
			return "", &QwenFragmentRefusal{Code: QwenFragmentIdentityMissing, FragmentIndex: index, Detail: "Claude transcript ID is required"}
		}
		if fragment.SessionID != "" {
			return "", &QwenFragmentRefusal{Code: QwenFragmentIdentityAmbiguous, FragmentIndex: index, Detail: "Claude fragment also supplies a Codex session ID"}
		}
		return fragment.TranscriptID, nil
	case QwenFragmentCodex:
		if fragment.SessionID == "" {
			return "", &QwenFragmentRefusal{Code: QwenFragmentIdentityMissing, FragmentIndex: index, Detail: "Codex session ID is required"}
		}
		if fragment.TranscriptID != "" {
			return "", &QwenFragmentRefusal{Code: QwenFragmentIdentityAmbiguous, FragmentIndex: index, Detail: "Codex fragment also supplies a Claude transcript ID"}
		}
		return fragment.SessionID, nil
	default:
		return "", &QwenFragmentRefusal{Code: QwenFragmentIdentityMissing, FragmentIndex: index, Detail: "source kind must be Claude or Codex"}
	}
}

func addQwenCanonicalUsage(total *QwenCanonicalUsage, usage QwenCanonicalUsage) {
	total.InputTokens += usage.InputTokens
	total.CacheReadTokens += usage.CacheReadTokens
	total.OutputTokens += usage.OutputTokens
	total.UsefulWitnesses += usage.UsefulWitnesses
}
