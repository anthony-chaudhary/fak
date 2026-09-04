"""l3_client.rc — CAMA return-code contract.
Applies to CamaClient (TCP) and RDMAClient (RDMA).
"""

# exists(key) -> int
EXISTS_FOUND   = 1   # key is present
EXISTS_MISSING = 0   # key is absent

# set(key, sgl) -> int  |  setstr(key, val) -> int
SET_OK = 0           # write accepted; errors raise RuntimeError

# get(key, sgl) -> int
GET_OK   =  0        # value written to SGL
GET_MISS = -1        # key not found

# delete(key) -> int
DELETE_OK = 0
