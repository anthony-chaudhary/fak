// Package benchloop folds fak's benchmark surfaces into one read-only control loop.
//
// Tier: foundation (1) - see internal/architest. This package imports only
// same-tier benchmark catalog/query leaves plus stdlib, and stays off the live
// request path.
package benchloop
