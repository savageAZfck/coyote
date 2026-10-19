# coyote — real weights from safetensors
# a Qwen3-0.6B self-attention q projection with the actual checkpoint
# tensor, compiled to ANE. file order is [out, in] = [N, K] — the blob
# storage layout exactly.
#
#   coyote run examples/qwenproj.cy -o out/qwenproj

param q: frozen f16[1024, 2048] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight"

io x: in f16[1024]
io y: out f16[2048]

fn main(x: f16[1024]) -> f16[2048] {
    x @ q
}
