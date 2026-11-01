# D41-HW-20261028-0310 - four-witness re-run after relative-contract witness fix (receipt v2)

**Tested tree:** parent `6600f3d1c0a906912f7c656d0f611f187af2b462` + single-file patch (sha256 `d035512c8584b0c43bde51b1b3bc5921570ca21d069ed65f12eca43553b16765`), landed as `df410106a54075e9557446aa33bfb4fd7ebf2caa`.

## Result: 4/4 PASS on aegis-fak (RTX 5080, sm_120)

| n | test | result | json sha256 |
|---|------|--------|-------------|
| 1 | TestCUDAV41CompressorRingMatchesPool | PASS | `ffa5fd3b8756ccdc36af68238597dca8dfc85a5f2585e1b6d1968b90a8ac6b4e` |
| 2 | TestCUDAV41CompressorRingRefusesNonFinite | PASS | `8e5e2359eb39d46675de19f4feabc983dc59a9fa01e178625f0dbd2f180d0875` |
| 3 | TestCUDAV41EngramIQ4NLRows | PASS | `9f87a77deeac49777d5d849f92585ae8cef5cf2196c335da59de9c36ad9469af` |
| 4 | TestCUDAQ4KAsyncOwnedPhysical | PASS | `72c7766be42bad3b56af93e034a679b5a9c7ea08696a430733c765fcfaabebef` |

Witness 4 was a deterministic FAIL at the v1 pin `42d81609`; it now PASSES under the arm's relative contract.

**Build:** success, elapsed 64301 ms; libfakcuda.a sha256 `075acfa6bff1f23e625840890d303929338103b9bb498859f51afc8a59159601` (raw-v2/cuda-build-receipt-v2.json).
**Teardown:** owned GPU processes remaining 0; post-run VRAM 5877 MiB used / 10049 MiB free - identical to pre-run baseline.
**Closure:** Witness bound aligned to the arm's documented relative contract (q4kMMVQRelMaxAbs); kernel unchanged; v1 FAIL reproduced pre-fix and cleared post-fix.
