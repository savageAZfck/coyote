# coyote — a real Qwen3-0.6B MLP block, layer 0
# input_layernorm -> gate/up projections -> silu*mul -> down -> residual.
# every weight is the actual checkpoint tensor.
#
#   coyote run examples/qwenmlp.cy -o out/qwenmlp

param g0:  frozen f16[1024]        @ "models/qwen3-0.6b/model.safetensors::model.layers.0.input_layernorm.weight"
param wg:  frozen f16[1024, 3072]  @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.gate_proj.weight"
param wu:  frozen f16[1024, 3072]  @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.up_proj.weight"
param wd:  frozen f16[3072, 1024]  @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.down_proj.weight"

io x: in f16[1024]
io y: out f16[1024]

fn main(x: f16[1024]) -> f16[1024] {
    let h = rmsnorm(x, g0);
    x + (silu(h @ wg) * (h @ wu)) @ wd
}
