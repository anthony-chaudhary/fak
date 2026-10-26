//go:build windows

package main

import (
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

// probePlatform is deliberately passive. It queries API version and platform
// authenticator availability only; it never requests an assertion or creates a
// credential, so it cannot display a consent dialog.
func probePlatform() platformProbe {
	// NewLazySystemDLL constrains lookup to the Windows system directory. An
	// unqualified LoadLibrary search would let a working-directory lookalike
	// counterfeit this passive status result.
	dll := windows.NewLazySystemDLL("webauthn.dll")
	versionProc := dll.NewProc("WebAuthNGetApiVersionNumber")
	availabilityProc := dll.NewProc("WebAuthNIsUserVerifyingPlatformAuthenticatorAvailable")
	if err := versionProc.Find(); err != nil {
		return platformProbe{err: fmt.Errorf("WebAuthn API unavailable: %w", err)}
	}
	if err := availabilityProc.Find(); err != nil {
		return platformProbe{err: fmt.Errorf("WebAuthn availability API unavailable: %w", err)}
	}
	version, _, _ := versionProc.Call()
	var available int32
	hresult, _, callErr := availabilityProc.Call(uintptr(unsafe.Pointer(&available)))
	if hresult != 0 {
		return platformProbe{
			apiAvailable: true,
			apiVersion:   uint32(version),
			err:          fmt.Errorf("WebAuthn availability probe HRESULT 0x%08x (%v)", uint32(hresult), callErr),
		}
	}
	return platformProbe{
		apiAvailable:       true,
		apiVersion:         uint32(version),
		authenticatorReady: available != 0,
	}
}
