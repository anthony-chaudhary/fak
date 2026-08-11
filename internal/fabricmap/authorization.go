package fabricmap

import (
	"errors"
	"fmt"
	"strings"
)

// Operation states what happens to data. It is independent of direction: a
// link from L3 to L1 can copy, move, or write back only when explicitly asked.
type Operation string

const (
	OperationRead      Operation = "read"
	OperationCopy      Operation = "copy"
	OperationMove      Operation = "move"
	OperationWrite     Operation = "write"
	OperationWriteBack Operation = "write-back"
)

func (o Operation) Valid() bool {
	switch o {
	case OperationRead, OperationCopy, OperationMove, OperationWrite, OperationWriteBack:
		return true
	default:
		return false
	}
}

// Access carries caller identity and data properties required end to end.
type Access struct {
	Principal string            `json:"principal"`
	Tenant    string            `json:"tenant"`
	Operation Operation         `json:"operation"`
	Data      map[string]string `json:"data,omitempty"`
}

// AuthorizationRequest is the complete decision input for one directed hop.
type AuthorizationRequest struct {
	Access   Access `json:"access"`
	HopIndex int    `json:"hop_index"`
	Link     Link   `json:"link"`
}

// Authorizer decides one hop. Implementations should return a public reason
// safe for a caller; sensitive policy detail remains behind the seam.
type Authorizer interface {
	Authorize(AuthorizationRequest) (allowed bool, reason string)
}
type AuthorizerFunc func(AuthorizationRequest) (bool, string)

func (f AuthorizerFunc) Authorize(r AuthorizationRequest) (bool, string) { return f(r) }

var ErrUnauthorized = errors.New("fabric route unauthorized")

// AuthorizedRoute binds an already planned route to explicit semantics after
// every hop allows it. Planning and authorization remain separate so topology
// metadata cannot accidentally become security policy.
type AuthorizedRoute struct {
	Route  Route  `json:"route"`
	Access Access `json:"access"`
}

func AuthorizeRoute(route Route, access Access, authorizer Authorizer) (AuthorizedRoute, error) {
	if strings.TrimSpace(access.Principal) == "" {
		return AuthorizedRoute{}, errors.New("access principal is required")
	}
	if strings.TrimSpace(access.Tenant) == "" {
		return AuthorizedRoute{}, errors.New("access tenant is required")
	}
	if !access.Operation.Valid() {
		return AuthorizedRoute{}, fmt.Errorf("invalid transfer operation %q", access.Operation)
	}
	if authorizer == nil {
		return AuthorizedRoute{}, errors.New("fabric authorizer is required")
	}
	for i, link := range route.Links {
		allowed, reason := authorizer.Authorize(AuthorizationRequest{Access: cloneAccess(access), HopIndex: i, Link: cloneLink(link)})
		if !allowed {
			reason = strings.TrimSpace(reason)
			if reason == "" {
				reason = "policy denied"
			}
			return AuthorizedRoute{}, fmt.Errorf("%w at hop %d link %q: %s", ErrUnauthorized, i, link.ID, reason)
		}
	}
	return AuthorizedRoute{Route: cloneRoute(route), Access: cloneAccess(access)}, nil
}

// LabelAuthorizer is a manifest-level default policy. A link may constrain
// tenant, operation, and required data properties using labels prefixed by
// auth.; absent labels impose no constraint.
type LabelAuthorizer struct{}

func (LabelAuthorizer) Authorize(r AuthorizationRequest) (bool, string) {
	labels := r.Link.Labels
	if tenant := labels["auth.tenant"]; tenant != "" && tenant != r.Access.Tenant {
		return false, "tenant not allowed"
	}
	if operations := labels["auth.operations"]; operations != "" && !commaSetHas(operations, string(r.Access.Operation)) {
		return false, "operation not allowed"
	}
	for key, want := range labels {
		const prefix = "auth.data."
		if strings.HasPrefix(key, prefix) && r.Access.Data[strings.TrimPrefix(key, prefix)] != want {
			return false, "required data property missing"
		}
	}
	return true, ""
}
func commaSetHas(set, want string) bool {
	for _, item := range strings.Split(set, ",") {
		if strings.TrimSpace(item) == want {
			return true
		}
	}
	return false
}
func cloneAccess(a Access) Access { a.Data = cloneLabels(a.Data); return a }
func cloneRoute(r Route) Route {
	source := r.Links
	r.Links = make([]Link, len(source))
	for i, l := range source {
		r.Links[i] = cloneLink(l)
	}
	return r
}
