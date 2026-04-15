from blockchain_client import get_latest_block, get_block

latest = get_latest_block()
block = get_block(latest["hash"])

print("Height:", block["height"])
print("Hash:", block["hash"])
print("Bits:", block["bits"])
print("Nonce:", block["nonce"])
print("Tx count:", len(block["tx"]))

# Observations:
# The hash has leading zeros → proof-of-work.
# 'bits' encodes the target threshold (compact format).
# Lower target → higher difficulty.
# Nonce is varied to find a valid hash.