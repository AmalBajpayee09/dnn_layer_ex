# utils/constants.py

# Special tokens
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

# Layer tokens (extendable)
LAYER_TOKENS = [
    "bn", "conv", "depthwise", "dropout", "fc", "flatten", "maxpool",
    "relu", "residual", "softmax", "tanh", EOS_TOKEN, PAD_TOKEN
]

# Safety checks
assert PAD_TOKEN in LAYER_TOKENS, "PAD_TOKEN must be in LAYER_TOKENS"
assert EOS_TOKEN in LAYER_TOKENS, "EOS_TOKEN must be in LAYER_TOKENS"

# Index mappings
PAD_IDX = LAYER_TOKENS.index(PAD_TOKEN)
EOS_IDX = LAYER_TOKENS.index(EOS_TOKEN)
VOCAB_SIZE = len(LAYER_TOKENS)

# Lookup dictionaries
LAYER2IDX = {tok: idx for idx, tok in enumerate(LAYER_TOKENS)}
IDX2LAYER = {idx: tok for tok, idx in LAYER2IDX.items()}

# Debug print
if __name__ == "__main__":
    print("📚 LAYER_TOKENS:", LAYER_TOKENS)
    print("🔢 PAD_IDX:", PAD_IDX, "| EOS_IDX:", EOS_IDX, "| VOCAB_SIZE:", VOCAB_SIZE)
    print("🔁 Test Mapping: conv →", LAYER2IDX["conv"], "←", IDX2LAYER[LAYER2IDX["conv"]])
