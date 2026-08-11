package fabricmap

import (
	"errors"
	"strings"
	"testing"
)

func TestAuthorizeRouteDeniesAnyIntermediateHop(t *testing.T) {
	route := Route{From: "L3", To: "L1", Links: []Link{{ID: "ssd-nic", From: "L3", To: "nic"}, {ID: "nic-gpu", From: "nic", To: "L1"}}}
	called := 0
	_, err := AuthorizeRoute(route, Access{Principal: "agent", Tenant: "alpha", Operation: OperationCopy}, AuthorizerFunc(func(r AuthorizationRequest) (bool, string) {
		called++
		if r.Link.ID == "nic-gpu" {
			return false, "tenant not allowed"
		}
		return true, ""
	}))
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("error = %v", err)
	}
	if called != 2 {
		t.Fatalf("authorizer calls = %d", called)
	}
	if !strings.Contains(err.Error(), `hop 1 link "nic-gpu"`) || strings.Contains(err.Error(), "secret-policy-row") {
		t.Fatalf("unsafe or imprecise error: %v", err)
	}
}

func TestLabelAuthorizerEnforcesTenantOperationAndDataProperties(t *testing.T) {
	route := Route{From: "storage", To: "gpu", Links: []Link{{ID: "direct", From: "storage", To: "gpu", Labels: map[string]string{"auth.tenant": "alpha", "auth.operations": "read,copy", "auth.data.encryption": "aes-gcm"}}}}
	allowed := Access{Principal: "worker", Tenant: "alpha", Operation: OperationCopy, Data: map[string]string{"encryption": "aes-gcm"}}
	if _, err := AuthorizeRoute(route, allowed, LabelAuthorizer{}); err != nil {
		t.Fatal(err)
	}
	for name, access := range map[string]Access{
		"tenant":    {Principal: "worker", Tenant: "beta", Operation: OperationCopy, Data: allowed.Data},
		"operation": {Principal: "worker", Tenant: "alpha", Operation: OperationMove, Data: allowed.Data},
		"data":      {Principal: "worker", Tenant: "alpha", Operation: OperationCopy, Data: map[string]string{"encryption": "none"}},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := AuthorizeRoute(route, access, LabelAuthorizer{}); !errors.Is(err, ErrUnauthorized) {
				t.Fatalf("error = %v", err)
			}
		})
	}
}

func TestDirectionDoesNotImplyOperation(t *testing.T) {
	forward := Route{From: "L1", To: "L3", Links: []Link{{ID: "down", From: "L1", To: "L3", Labels: map[string]string{"auth.operations": "write-back"}}}}
	reverse := Route{From: "L3", To: "L1", Links: []Link{{ID: "up", From: "L3", To: "L1", Labels: map[string]string{"auth.operations": "read"}}}}
	base := Access{Principal: "p", Tenant: "t"}
	base.Operation = OperationRead
	if _, err := AuthorizeRoute(forward, base, LabelAuthorizer{}); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("L1 -> L3 read should be denied: %v", err)
	}
	if _, err := AuthorizeRoute(reverse, base, LabelAuthorizer{}); err != nil {
		t.Fatalf("L3 -> L1 read: %v", err)
	}
	base.Operation = OperationWriteBack
	if _, err := AuthorizeRoute(forward, base, LabelAuthorizer{}); err != nil {
		t.Fatalf("L1 -> L3 write-back: %v", err)
	}
	if _, err := AuthorizeRoute(reverse, base, LabelAuthorizer{}); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("L3 -> L1 write-back should be denied: %v", err)
	}
}

func TestAuthorizationResultDoesNotAliasCallerData(t *testing.T) {
	access := Access{Principal: "p", Tenant: "t", Operation: OperationCopy, Data: map[string]string{"class": "public"}}
	route := Route{From: "a", To: "b", Links: []Link{{ID: "a-b", From: "a", To: "b", Labels: map[string]string{"x": "y"}}}}
	result, err := AuthorizeRoute(route, access, AuthorizerFunc(func(AuthorizationRequest) (bool, string) { return true, "" }))
	if err != nil {
		t.Fatal(err)
	}
	access.Data["class"] = "secret"
	route.Links[0].Labels["x"] = "changed"
	if result.Access.Data["class"] != "public" || result.Route.Links[0].Labels["x"] != "y" {
		t.Fatalf("result aliased caller: %+v", result)
	}
}
