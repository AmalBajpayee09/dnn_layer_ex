from phase2.utils import encode_sequence, decode_sequence

seq = ['conv', 'relu', 'pool']
ids = encode_sequence(seq, max_len=10)
print("Encoded:", ids.tolist())
print("Decoded:", decode_sequence(ids))
# Test with EOS token
seq_with_eos = ['conv', 'relu', 'pool', '<EOS>']
ids_with_eos = encode_sequence(seq_with_eos, max_len=10)
print("Encoded with EOS:", ids_with_eos.tolist())
print("Decoded with EOS:", decode_sequence(ids_with_eos))