package compute

const qwen35DecodeBatchScratchEntries = 9

// One request owns these containers. Reset after each GDN layer so released
// tensor and buffer owners do not survive into the next layer.
type qwen35DecodeBatchScratch[B any] struct {
	tensors [qwen35DecodeBatchScratchEntries]Tensor
	buffers [qwen35DecodeBatchScratchEntries]B
}

func (s *qwen35DecodeBatchScratch[B]) prepare(count int) ([]Tensor, []B) {
	if count != len(s.tensors) {
		return make([]Tensor, 0, count), make([]B, 0, count)
	}
	return s.tensors[:0], s.buffers[:0]
}

func (s *qwen35DecodeBatchScratch[B]) reset() {
	clear(s.tensors[:])
	clear(s.buffers[:])
}
