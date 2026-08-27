package agentdescriptor

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const Schema = "fak.agent-operation-descriptor/1"

type Descriptor struct {
	Schema string          `json:"schema"`
	Agent  AgentCoordinate `json:"agent"`
	Model  ModelCoordinate `json:"model"`
	Fleet  FleetCoordinate `json:"fleet"`
}
type AgentCoordinate struct {
	Identity  string `json:"identity"`
	Lifecycle string `json:"lifecycle"`
}
type ModelCoordinate struct {
	Class string `json:"class"`
	ID    string `json:"id,omitempty"`
}
type FleetCoordinate struct {
	Cardinality int    `json:"cardinality"`
	Topology    string `json:"topology"`
}
type OperationReceipt struct {
	Schema      string     `json:"schema"`
	OperationID string     `json:"operation_id"`
	Descriptor  Descriptor `json:"descriptor"`
	RouteRule   string     `json:"route_rule,omitempty"`
}

func (d Descriptor) Validate() error {
	if d.Schema != Schema {
		return fmt.Errorf("descriptor schema must be %q", Schema)
	}
	if strings.TrimSpace(d.Agent.Identity) == "" || strings.TrimSpace(d.Agent.Lifecycle) == "" {
		return errors.New("agent identity and lifecycle are required")
	}
	if strings.TrimSpace(d.Model.Class) == "" {
		return errors.New("model class is required")
	}
	if d.Fleet.Cardinality < 1 || strings.TrimSpace(d.Fleet.Topology) == "" {
		return errors.New("fleet cardinality and topology are required")
	}
	return nil
}
func (d Descriptor) Marshal() ([]byte, error) {
	if err := d.Validate(); err != nil {
		return nil, err
	}
	return json.Marshal(d)
}
func Decode(raw []byte) (Descriptor, error) {
	var d Descriptor
	if err := json.Unmarshal(raw, &d); err != nil {
		return d, err
	}
	return d, d.Validate()
}
func New(agentID, lifecycle, modelClass, modelID string, cardinality int, topology string) Descriptor {
	return Descriptor{Schema: Schema, Agent: AgentCoordinate{agentID, lifecycle}, Model: ModelCoordinate{modelClass, modelID}, Fleet: FleetCoordinate{cardinality, topology}}
}
