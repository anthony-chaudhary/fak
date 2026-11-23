// Package agentdescriptor re-exports pkg/routepolicy/agentdescriptor, the public
// home of the operation descriptor, so in-tree importers keep compiling unchanged.
package agentdescriptor

import pub "github.com/anthony-chaudhary/fak/pkg/routepolicy/agentdescriptor"

const Schema = pub.Schema

type (
	Descriptor       = pub.Descriptor
	AgentCoordinate  = pub.AgentCoordinate
	ModelCoordinate  = pub.ModelCoordinate
	FleetCoordinate  = pub.FleetCoordinate
	OperationReceipt = pub.OperationReceipt
)

func Decode(raw []byte) (Descriptor, error) { return pub.Decode(raw) }

func New(agentID, lifecycle, modelClass, modelID string, cardinality int, topology string) Descriptor {
	return pub.New(agentID, lifecycle, modelClass, modelID, cardinality, topology)
}
