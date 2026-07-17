package model

// CrystallizationDepth returns the earliest layer at which target is the
// top-logit token and remains top-logit through every later layer. A false
// result means the target is invalid for at least one layer or is not top-1 at
// the final layer.
//
// LayerLogits output includes the embedding projection at index zero, so the
// returned depth uses that same zero-based coordinate.
func CrystallizationDepth(layerLogits [][]float32, target int) (depth int, ok bool) {
	if len(layerLogits) == 0 || target < 0 {
		return 0, false
	}

	stableFrom := len(layerLogits)
	for layer := len(layerLogits) - 1; layer >= 0; layer-- {
		logits := layerLogits[layer]
		if target >= len(logits) || len(logits) == 0 {
			return 0, false
		}
		top := 0
		for token := 1; token < len(logits); token++ {
			if logits[token] > logits[top] {
				top = token
			}
		}
		if top != target {
			break
		}
		stableFrom = layer
	}
	if stableFrom == len(layerLogits) {
		return 0, false
	}
	return stableFrom, true
}
