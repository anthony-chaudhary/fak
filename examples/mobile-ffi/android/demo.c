/*
 * demo.c — the device-free witness for the Android sample. It links against the
 * cross-compiled archive (libfakmobile_android_arm64.a) and drives the SAME
 * deny/allow round-trip the JNI path (fak_gate.c) carries, printing each
 * Decision and returning non-zero if any leg is wrong. Running it under an
 * arm64 executor (an emulator shell, adb, or qemu-aarch64) is a witness that the
 * FFI archive links and adjudicates correctly — no app, activity, or model
 * needed. See README.md for the compile+run command.
 */
#include <stdio.h>
#include <string.h>
#include "libfakmobile.h"

static int expect(const char *label, const char *call, int want_allow) {
  char *d = FakAdjudicate((char *)call);
  int got_allow = strstr(d, "\"allow\":true") != NULL;
  printf("%s %s\n", label, d);
  FakFree(d);
  if (got_allow != want_allow) {
    fprintf(stderr, "FAIL: %s allow=%d want=%d\n", label, got_allow, want_allow);
    return 1;
  }
  return 0;
}

int main(void) {
  int bad = 0;
  puts("fak Android NDK sample — proposed tool calls through the adjudicator floor");
  bad |= expect("[1] dangerous", "{\"tool\":\"send_sms\",\"args\":{\"to\":\"+1900\"}}", 0);
  bad |= expect("[2] benign   ", "{\"tool\":\"get_battery_level\"}", 1);
  bad |= expect("[3] unknown  ", "{\"tool\":\"transfer_funds\"}", 0);
  if (bad) {
    fputs("\nandroid demo: ROUND-TRIP FAILED\n", stderr);
    return 1;
  }
  puts("\nandroid demo: OK — dangerous denied, benign continued, unknown failed closed");
  return 0;
}
