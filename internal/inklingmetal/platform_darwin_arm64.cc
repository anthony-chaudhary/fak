//go:build cgo

#include "bridge.h"
#include "platform.h"
#if FAK_INKLING_METAL_SDK
#include "metal/MetalBackend.hpp"
int fak_ink_available() {
  if (__builtin_available(macOS 26.4, *)) {
    try {
      return !splash::metal::probeDeviceCapabilities()
                  .validationError()
                  .has_value();
    } catch (...) {
    }
  }
  return 0;
}
#else
#include <cstring>
static char *unavailable() {
  return strdup(
      "Inkling Metal requires macOS 26.4 SDK and Metal 4 shader support");
}
int fak_ink_available() { return 0; }
char *fak_ink_plan(const char *, const char *, uint32_t, FakInkMemory *) {
  return unavailable();
}
char *fak_ink_model(const char *, const char *, const char *, FakInkModel **) {
  return unavailable();
}
void fak_ink_free_model(FakInkModel *) {}
char *fak_ink_session(FakInkModel *, uint32_t, FakInkSession **) {
  return unavailable();
}
char *fak_ink_reset(FakInkSession *, uint32_t) { return unavailable(); }
void fak_ink_free_session(FakInkSession *) {}
char *fak_ink_step(FakInkSession *, const int32_t *, uint32_t, int, float *,
                   int32_t *) {
  return unavailable();
}
char *fak_ink_commit(FakInkSession *, uint32_t) { return unavailable(); }
char *fak_ink_tree(FakInkSession *, const int32_t *, uint32_t, int32_t *) {
  return unavailable();
}
char *fak_ink_propose_verify(FakInkSession *, int32_t, uint32_t, uint32_t,
                             int32_t *, int32_t *, uint32_t *) {
  return unavailable();
}
char *fak_ink_cycle(FakInkSession *, int32_t, uint32_t, uint32_t, uint32_t,
                    int32_t, int32_t *, int32_t *, uint32_t *, uint32_t *) {
  return unavailable();
}
char *fak_ink_commit_tree(FakInkSession *, int32_t) { return unavailable(); }
char *fak_ink_draft_tree(FakInkSession *, int32_t, uint32_t, uint32_t,
                         int32_t *, uint32_t *) {
  return unavailable();
}
char *fak_ink_propose(FakInkSession *, int32_t, uint32_t, int32_t *,
                      uint32_t *) {
  return unavailable();
}
#endif
