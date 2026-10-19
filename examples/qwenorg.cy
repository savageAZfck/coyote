// qwenorg.cy — a real autoregressive decode, the organism way.
// the decoder layer is a compiled ANE program; the decode loop is
// host code: gather the token's embedding row, call silicon, argmax,
// witness the token, feed it back. every step is witnessed — the
// token stream is the organism's own chain of thought.
//
//   coyote run examples/qwenorg.cy -o out/qwenorg
//
// needs models/qwen3-0.6b/model.safetensors (HF checkpoint).
param ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.input_layernorm.weight"
param ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.post_attention_layernorm.weight"
param qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_norm.weight[0:64]"
param qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_norm.weight[64:128]"
param knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_norm.weight[0:64]"
param knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_norm.weight[64:128]"
param q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[0:64]"
param o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,0:64]"
param q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[64:128]"
param o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,64:128]"
param q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[128:192]"
param o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,128:192]"
param q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[192:256]"
param o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,192:256]"
param q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[256:320]"
param o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,256:320]"
param q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[320:384]"
param o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,320:384]"
param q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[384:448]"
param o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,384:448]"
param q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[448:512]"
param o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,448:512]"
param q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[512:576]"
param o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,512:576]"
param q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[576:640]"
param o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,576:640]"
param q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[640:704]"
param o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,640:704]"
param q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[704:768]"
param o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,704:768]"
param q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[768:832]"
param o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,768:832]"
param q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[832:896]"
param o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,832:896]"
param q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[896:960]"
param o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,896:960]"
param q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[960:1024]"
param o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,960:1024]"
param q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1024:1088]"
param o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1024:1088]"
param q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1088:1152]"
param o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1088:1152]"
param q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1152:1216]"
param o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1152:1216]"
param q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1216:1280]"
param o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1216:1280]"
param q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1280:1344]"
param o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1280:1344]"
param q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1344:1408]"
param o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1344:1408]"
param q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1408:1472]"
param o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1408:1472]"
param q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1472:1536]"
param o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1472:1536]"
param q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1536:1600]"
param o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1536:1600]"
param q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1600:1664]"
param o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1600:1664]"
param q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1664:1728]"
param o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1664:1728]"
param q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1728:1792]"
param o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1728:1792]"
param q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1792:1856]"
param o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1792:1856]"
param q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1856:1920]"
param o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1856:1920]"
param q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1920:1984]"
param o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1920:1984]"
param q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1984:2048]"
param o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1984:2048]"
param k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[0:64]"
param v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[0:64]"
param k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[64:128]"
param v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[64:128]"
param k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[128:192]"
param v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[128:192]"
param k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[192:256]"
param v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[192:256]"
param k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[256:320]"
param v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[256:320]"
param k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[320:384]"
param v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[320:384]"
param k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[384:448]"
param v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[384:448]"
param k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[448:512]"
param v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[448:512]"
param k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[512:576]"
param v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[512:576]"
param k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[576:640]"
param v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[576:640]"
param k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[640:704]"
param v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[640:704]"
param k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[704:768]"
param v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[704:768]"
param k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[768:832]"
param v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[768:832]"
param k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[832:896]"
param v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[832:896]"
param k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[896:960]"
param v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[896:960]"
param k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[960:1024]"
param v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[960:1024]"
param wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.gate_proj.weight"
param wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.up_proj.weight"
param wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.down_proj.weight"


state kv {
    k0lo: f16[64, 64]
    k0hi: f16[64, 64]
    v0lo: f16[64, 64]
    v0hi: f16[64, 64]
    k1lo: f16[64, 64]
    k1hi: f16[64, 64]
    v1lo: f16[64, 64]
    v1hi: f16[64, 64]
    k2lo: f16[64, 64]
    k2hi: f16[64, 64]
    v2lo: f16[64, 64]
    v2hi: f16[64, 64]
    k3lo: f16[64, 64]
    k3hi: f16[64, 64]
    v3lo: f16[64, 64]
    v3hi: f16[64, 64]
    k4lo: f16[64, 64]
    k4hi: f16[64, 64]
    v4lo: f16[64, 64]
    v4hi: f16[64, 64]
    k5lo: f16[64, 64]
    k5hi: f16[64, 64]
    v5lo: f16[64, 64]
    v5hi: f16[64, 64]
    k6lo: f16[64, 64]
    k6hi: f16[64, 64]
    v6lo: f16[64, 64]
    v6hi: f16[64, 64]
    k7lo: f16[64, 64]
    k7hi: f16[64, 64]
    v7lo: f16[64, 64]
    v7hi: f16[64, 64]
}

// the embedding table is host data — it never touches silicon.
// rows are gathered by index at runtime.
table emb: f16[151936, 1024] @ "models/qwen3-0.6b/model.safetensors::model.embed_tokens.weight"

fn layer(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[2048] {
    let h = rmsnorm(x, ln1);
    let q0l = rmsnorm(h @ q0lo, qnl);
    let q0h = rmsnorm(h @ q0hi, qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ q1lo, qnl);
    let q1h = rmsnorm(h @ q1hi, qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ q2lo, qnl);
    let q2h = rmsnorm(h @ q2hi, qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ q3lo, qnl);
    let q3h = rmsnorm(h @ q3hi, qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ q4lo, qnl);
    let q4h = rmsnorm(h @ q4hi, qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ q5lo, qnl);
    let q5h = rmsnorm(h @ q5hi, qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ q6lo, qnl);
    let q6h = rmsnorm(h @ q6hi, qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ q7lo, qnl);
    let q7h = rmsnorm(h @ q7hi, qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ q8lo, qnl);
    let q8h = rmsnorm(h @ q8hi, qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ q9lo, qnl);
    let q9h = rmsnorm(h @ q9hi, qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ q10lo, qnl);
    let q10h = rmsnorm(h @ q10hi, qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ q11lo, qnl);
    let q11h = rmsnorm(h @ q11hi, qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ q12lo, qnl);
    let q12h = rmsnorm(h @ q12hi, qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ q13lo, qnl);
    let q13h = rmsnorm(h @ q13hi, qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ q14lo, qnl);
    let q14h = rmsnorm(h @ q14hi, qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ q15lo, qnl);
    let q15h = rmsnorm(h @ q15hi, qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ k0lo, knl);
    let k0h = rmsnorm(h @ k0hi, knh);
    let v0l = h @ v0lo;
    let v0h = h @ v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ k1lo, knl);
    let k1h = rmsnorm(h @ k1hi, knh);
    let v1l = h @ v1lo;
    let v1h = h @ v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ k2lo, knl);
    let k2h = rmsnorm(h @ k2hi, knh);
    let v2l = h @ v2lo;
    let v2h = h @ v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ k3lo, knl);
    let k3h = rmsnorm(h @ k3hi, knh);
    let v3l = h @ v3lo;
    let v3h = h @ v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ k4lo, knl);
    let k4h = rmsnorm(h @ k4hi, knh);
    let v4l = h @ v4lo;
    let v4h = h @ v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ k5lo, knl);
    let k5h = rmsnorm(h @ k5hi, knh);
    let v5l = h @ v5lo;
    let v5h = h @ v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ k6lo, knl);
    let k6h = rmsnorm(h @ k6hi, knh);
    let v6l = h @ v6lo;
    let v6h = h @ v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ k7lo, knl);
    let k7h = rmsnorm(h @ k7hi, knh);
    let v7l = h @ v7lo;
    let v7h = h @ v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ o0lo + c0h @ o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ o1lo + c1h @ o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ o2lo + c2h @ o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ o3lo + c3h @ o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ o4lo + c4h @ o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ o5lo + c5h @ o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ o6lo + c6h @ o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ o7lo + c7h @ o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ o8lo + c8h @ o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ o9lo + c9h @ o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ o10lo + c10h @ o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ o11lo + c11h @ o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ o12lo + c12h @ o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ o13lo + c13h @ o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ o14lo + c14h @ o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ o15lo + c15h @ o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, ln2);
    let f = r + (silu(n @ wg) * (n @ wu)) @ wd;
    rmsnorm(f, lnf) @ lmh
}
param lnf: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.norm.weight"
param lmh: frozen f16[1024, 2048] @ "models/qwen3-0.6b/model.safetensors::model.embed_tokens.weight[0:2048]"

host fn main() -> i64 {
    let me = self_token();
    witness("born", me);
    print("decode organism", me.id);
    var tok = 151643;                       // BOS
    var x = emb[tok];                       // f16[1024] — host gather
    var step = 0;
    while step < 8 {
        let lg = layer(
            x,
            onehot(step % 64, 64),
            causal(step % 64, 64),
            rope_cos(step, 64),
            rope_sin(step, 64));
        tok = argmax(lg);
        print("step", step, "tok", tok);
        witness("tok", tok);
        x = emb[tok];
        step += 1;
    }
    witness("done", step);
    return tok;
}
