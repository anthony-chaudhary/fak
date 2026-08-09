package gateway

import (
	"encoding/json"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/toolshape"
)

const reasonResultShapeMismatch = "RESULT_SHAPE_MISMATCH"

// resultContractAdmission validates raw result bytes against the originating
// tool's complete output contract. It is pure and runs before ordinary result
// admission, so a malformed value can only cross the boundary as a quarantine
// stub. The returned receipt contains shape only, never result contents.
func resultContractAdmission(tool string, tools []agent.ToolDef, raw string) (*toolshape.ResultShapeReceipt, *WireVerdict, string) {
	var parameters json.RawMessage
	for i := range tools {
		if tools[i].Function.Name == tool {
			parameters = tools[i].Function.Parameters
			break
		}
	}
	contract, present, err := toolshape.ParseResultContract(parameters)
	if !present {
		return nil, nil, raw
	}
	if err != nil {
		receipt := toolshape.ResultShapeReceipt{
			Schema:  "fak-result-contract/1",
			Failure: toolshape.ResultShapeMalformed,
		}
		return &receipt, resultShapeVerdict(receipt), resultShapeStub(receipt)
	}
	receipt, err := contract.ValidateResult(raw)
	if err == nil {
		return &receipt, nil, raw
	}
	return &receipt, resultShapeVerdict(receipt), resultShapeStub(receipt)
}

func resultShapeVerdict(receipt toolshape.ResultShapeReceipt) *WireVerdict {
	return &WireVerdict{
		Kind:        "QUARANTINE",
		Reason:      reasonResultShapeMismatch,
		By:          "toolshape",
		Disposition: "TERMINAL",
		Detail: map[string]string{
			"failure":        string(receipt.Failure),
			"expected_count": strconv.Itoa(receipt.ExpectedCount),
			"observed_count": strconv.Itoa(receipt.ObservedCount),
		},
	}
}

func resultShapeStub(receipt toolshape.ResultShapeReceipt) string {
	stub := struct {
		Quarantined   bool                       `json:"_quarantined"`
		Boundary      string                     `json:"boundary"`
		Reason        string                     `json:"reason"`
		Failure       toolshape.ResultShapeError `json:"failure"`
		ExpectedCount int                        `json:"expected_count"`
		ObservedCount int                        `json:"observed_count"`
	}{
		Quarantined:   true,
		Boundary:      "result_shape",
		Reason:        reasonResultShapeMismatch,
		Failure:       receipt.Failure,
		ExpectedCount: receipt.ExpectedCount,
		ObservedCount: receipt.ObservedCount,
	}
	b, _ := json.Marshal(stub)
	return string(b)
}
