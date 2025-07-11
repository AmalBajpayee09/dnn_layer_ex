PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

LAYER_TOKENS = [
    "conv", "relu", "fc", "softmax", "flatten", "bn", "dropout",
    "residual", "depthwise", "maxpool", "<eos>", "<pad>"
]

PAD_IDX = LAYER_TOKENS.index("<pad>")
EOS_IDX = LAYER_TOKENS.index("<eos>")
VOCAB_SIZE = len(LAYER_TOKENS)

LAYER2IDX = {tok: idx for idx, tok in enumerate(LAYER_TOKENS)}
IDX2LAYER = {idx: tok for tok, idx in LAYER2IDX.items()}

if __name__ == "__main__":
    print("VOCAB:", LAYER_TOKENS)
    print("PAD_IDX:", PAD_IDX, "EOS_IDX:", EOS_IDX)
    print("conv →", LAYER2IDX["conv"], "←", IDX2LAYER[LAYER2IDX["conv"]])
