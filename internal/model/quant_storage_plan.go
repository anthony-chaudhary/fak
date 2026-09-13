package model

// QuantSourceTensorName resolves a canonical source tensor through the same
// name and retention gate used by the quantized model builder. Header-only
// storage planners use the result to classify the weights the builder retains.
// A false keep result means the source tensor is omitted from the model.
func QuantSourceTensorName(cfg Config, name string) (resolved string, keep bool) {
	return quantSourceTensorName(cfg, name)
}
