// Command seamprobe is the witness that every registration seam fak exports is
// reachable from a foreign module. It imports only pkg/abi and names each
// exported Register* function, so it stops compiling the moment one is removed
// or moved back under internal/.
//
// It registers nothing: the round-trip of a real driver is examples/extdriver. This
// probe answers the narrower question "can an outsider reach the seam at all".
package main

import (
	"fmt"
	"reflect"

	"github.com/anthony-chaudhary/fak/pkg/abi"
)

func main() {
	seams := []any{
		abi.RegisterAdjudicator,
		abi.RegisterResultAdmitter,
		abi.RegisterFastPath,
		abi.RegisterOp,
		abi.RegisterVerdictKind,
		abi.RegisterReason,
		abi.RegisterCapability,
		abi.RegisterEmitter,
		abi.RegisterEngine,
		abi.RegisterRegionBackend,
		abi.RegisterKVBackend,
		abi.RegisterPageOutBackend,
		abi.RegisterWitnessResolver,
		abi.RegisterSteward,
		abi.RegisterProvisionalSink,
		abi.RegisterSemanticScreen,
	}
	for _, s := range seams {
		if reflect.ValueOf(s).Kind() != reflect.Func {
			panic("exported seam is not a function")
		}
	}
	fmt.Printf("seamprobe: %d registration seams reachable from example.com/fakseamprobe\n", len(seams))
}
