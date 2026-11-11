// graph_backend_assign.go — weight-locality backend assignment for a minimal
// op list: an op whose weight lives in a backend's buffer is assigned to that
// backend, an op without a weight inherits the backend of its first assigned
// input, and an op with no assigned input stays unassigned.
//
// ADAPTED from llama.cpp (MIT; ggml-org/llama.cpp at
// 83078fec0db82d6b5a00d9599062c38c39145755):
//   ggml/src/ggml-backend.cpp:921 ggml_backend_sched_backend_id_from_cur —
//       pass 1 of ggml_backend_sched_split_graph: a tensor whose buffer is
//       pre-allocated (a weight) pins its op to the backend that owns the
//       buffer; an op with no weight source is left unassigned for the
//       input-expansion passes.
// No upstream source is copied. Deliberate deviations from upstream:
//   - the minimal node list carries (id, inputs, weightBackend) instead of
//     ggml_tensor pointers, so the pass is a pure function over ids.
//   - WeightBackend is a pointer so that nil (the zero value) means "no
//     weight", which is the safe default; a weightless op that omits the
//     field cannot be mistaken for a weight on backend 0.
//   - inheritance is "first assigned input in list order" (upstream expands
//     from the adjacent assigned node in topological order, the same rule for
//     a chain); the graph-input-to-last-backend and op-offload exceptions are
//     out of scope for pass 1 here.
//   - malformed lists (empty or duplicate ids, unknown or forward inputs) are
//     refused with sentinel errors instead of aborting.

package compute

import "errors"

// BackendID identifies a compute backend in a weight-locality assignment.
type BackendID int

// BackendUnassigned is the BackendID of a node that no backend owns. It is
// the value AssignByWeightLocality records for unresolved nodes.
const BackendUnassigned BackendID = -1

// WeightLocalityNode is the minimal op descriptor for weight-locality
// assignment: the op id, the ids of the ops that feed it, and a pointer to
// the backend that owns the op's weight buffer. A nil WeightBackend means the
// op carries no weight and must inherit from its inputs.
type WeightLocalityNode struct {
	ID            string     `json:"id"`
	Inputs        []string   `json:"inputs,omitempty"`
	WeightBackend *BackendID `json:"weight_backend,omitempty"`
}

// Sentinel errors for malformed weight-locality node lists.
var (
	// ErrWeightLocalityEmptyID is returned when a node has an empty id.
	ErrWeightLocalityEmptyID = errors.New("compute: weight-locality node has an empty id")
	// ErrWeightLocalityDuplicateID is returned when two nodes share an id.
	ErrWeightLocalityDuplicateID = errors.New("compute: weight-locality node id appears more than once")
	// ErrWeightLocalityUnknownInput is returned when a node names an input
	// that is not in the list.
	ErrWeightLocalityUnknownInput = errors.New("compute: weight-locality node references an input outside the node list")
	// ErrWeightLocalityForwardInput is returned when a node names an input
	// that appears later in the list; the list must be in dependency order.
	ErrWeightLocalityForwardInput = errors.New("compute: weight-locality node references an input that appears later in the list")
)

// AssignByWeightLocality assigns each node to the backend that owns its
// weight buffer; a node without a weight inherits the backend of its first
// assigned input; a node whose inputs are all unassigned stays unassigned.
//
// nodes must be in dependency order: every input id appears earlier in the
// list than the node that uses it, and ids are unique. The result maps every
// node id to its assigned BackendID, with BackendUnassigned for unresolved
// nodes.
func AssignByWeightLocality(nodes []WeightLocalityNode) (map[string]BackendID, error) {
	index := make(map[string]int, len(nodes))
	for i, node := range nodes {
		if node.ID == "" {
			return nil, ErrWeightLocalityEmptyID
		}
		if _, exists := index[node.ID]; exists {
			return nil, ErrWeightLocalityDuplicateID
		}
		index[node.ID] = i
	}
	for i, node := range nodes {
		for _, in := range node.Inputs {
			j, ok := index[in]
			if !ok {
				return nil, ErrWeightLocalityUnknownInput
			}
			if j > i {
				return nil, ErrWeightLocalityForwardInput
			}
		}
	}

	assigned := make(map[string]BackendID, len(nodes))
	for _, node := range nodes {
		backend := BackendUnassigned
		if node.WeightBackend != nil {
			backend = *node.WeightBackend
		}
		if backend == BackendUnassigned {
			for _, in := range node.Inputs {
				if b := assigned[in]; b != BackendUnassigned {
					backend = b
					break
				}
			}
		}
		assigned[node.ID] = backend
	}
	return assigned, nil
}
