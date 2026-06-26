package main

import "github.com/anthony-chaudhary/fak/internal/secretload"

const gatewayBearerEnv = "FAK_KEY"

func gatewayBearerLoader(envName string) *secretload.Loader {
	l := secretload.Default()
	l.Require(envName, "gateway bearer token", nil)
	return l
}

func defaultGatewayBearerToken() string {
	v, _ := gatewayBearerLoader(gatewayBearerEnv).Lookup(gatewayBearerEnv)
	return v
}
