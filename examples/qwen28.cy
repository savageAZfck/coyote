// qwen28.cy — a full 28-layer Qwen3-0.6B decode, the organism way.
// 28 programs on the chip, each with its own instanced KV cache;
// the decode loop — gather, dispatch, argmax, witness — is all
// host code. every token generated is witnessed by the organism.
//
//   coyote run examples/qwen28.cy -o out/qwen28
//
// needs models/qwen3-0.6b/model.safetensors (HF checkpoint).
// generated from qwenorg.cy — per-layer params/fns, shared state spec.

param lnf: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.norm.weight"
param lmh: frozen f16[1024, 2048] @ "models/qwen3-0.6b/model.safetensors::model.embed_tokens.weight[0:2048]"
table emb: f16[151936, 1024] @ "models/qwen3-0.6b/model.safetensors::model.embed_tokens.weight"

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

param l0_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.input_layernorm.weight"
param l0_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.post_attention_layernorm.weight"
param l0_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_norm.weight[0:64]"
param l0_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_norm.weight[64:128]"
param l0_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_norm.weight[0:64]"
param l0_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_norm.weight[64:128]"
param l0_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[0:64]"
param l0_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,0:64]"
param l0_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[64:128]"
param l0_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,64:128]"
param l0_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[128:192]"
param l0_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,128:192]"
param l0_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[192:256]"
param l0_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,192:256]"
param l0_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[256:320]"
param l0_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,256:320]"
param l0_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[320:384]"
param l0_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,320:384]"
param l0_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[384:448]"
param l0_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,384:448]"
param l0_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[448:512]"
param l0_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,448:512]"
param l0_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[512:576]"
param l0_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,512:576]"
param l0_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[576:640]"
param l0_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,576:640]"
param l0_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[640:704]"
param l0_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,640:704]"
param l0_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[704:768]"
param l0_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,704:768]"
param l0_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[768:832]"
param l0_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,768:832]"
param l0_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[832:896]"
param l0_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,832:896]"
param l0_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[896:960]"
param l0_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,896:960]"
param l0_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[960:1024]"
param l0_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,960:1024]"
param l0_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1024:1088]"
param l0_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1024:1088]"
param l0_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1088:1152]"
param l0_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1088:1152]"
param l0_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1152:1216]"
param l0_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1152:1216]"
param l0_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1216:1280]"
param l0_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1216:1280]"
param l0_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1280:1344]"
param l0_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1280:1344]"
param l0_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1344:1408]"
param l0_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1344:1408]"
param l0_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1408:1472]"
param l0_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1408:1472]"
param l0_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1472:1536]"
param l0_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1472:1536]"
param l0_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1536:1600]"
param l0_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1536:1600]"
param l0_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1600:1664]"
param l0_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1600:1664]"
param l0_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1664:1728]"
param l0_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1664:1728]"
param l0_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1728:1792]"
param l0_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1728:1792]"
param l0_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1792:1856]"
param l0_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1792:1856]"
param l0_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1856:1920]"
param l0_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1856:1920]"
param l0_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1920:1984]"
param l0_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1920:1984]"
param l0_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.q_proj.weight[1984:2048]"
param l0_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.o_proj.weight[0:1024,1984:2048]"
param l0_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[0:64]"
param l0_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[0:64]"
param l0_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[64:128]"
param l0_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[64:128]"
param l0_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[128:192]"
param l0_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[128:192]"
param l0_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[192:256]"
param l0_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[192:256]"
param l0_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[256:320]"
param l0_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[256:320]"
param l0_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[320:384]"
param l0_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[320:384]"
param l0_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[384:448]"
param l0_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[384:448]"
param l0_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[448:512]"
param l0_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[448:512]"
param l0_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[512:576]"
param l0_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[512:576]"
param l0_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[576:640]"
param l0_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[576:640]"
param l0_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[640:704]"
param l0_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[640:704]"
param l0_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[704:768]"
param l0_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[704:768]"
param l0_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[768:832]"
param l0_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[768:832]"
param l0_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[832:896]"
param l0_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[832:896]"
param l0_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[896:960]"
param l0_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[896:960]"
param l0_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.k_proj.weight[960:1024]"
param l0_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.self_attn.v_proj.weight[960:1024]"
param l0_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.gate_proj.weight"
param l0_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.up_proj.weight"
param l0_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.0.mlp.down_proj.weight"

param l1_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.input_layernorm.weight"
param l1_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.post_attention_layernorm.weight"
param l1_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_norm.weight[0:64]"
param l1_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_norm.weight[64:128]"
param l1_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_norm.weight[0:64]"
param l1_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_norm.weight[64:128]"
param l1_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[0:64]"
param l1_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,0:64]"
param l1_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[64:128]"
param l1_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,64:128]"
param l1_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[128:192]"
param l1_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,128:192]"
param l1_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[192:256]"
param l1_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,192:256]"
param l1_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[256:320]"
param l1_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,256:320]"
param l1_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[320:384]"
param l1_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,320:384]"
param l1_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[384:448]"
param l1_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,384:448]"
param l1_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[448:512]"
param l1_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,448:512]"
param l1_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[512:576]"
param l1_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,512:576]"
param l1_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[576:640]"
param l1_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,576:640]"
param l1_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[640:704]"
param l1_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,640:704]"
param l1_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[704:768]"
param l1_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,704:768]"
param l1_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[768:832]"
param l1_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,768:832]"
param l1_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[832:896]"
param l1_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,832:896]"
param l1_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[896:960]"
param l1_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,896:960]"
param l1_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[960:1024]"
param l1_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,960:1024]"
param l1_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1024:1088]"
param l1_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1024:1088]"
param l1_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1088:1152]"
param l1_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1088:1152]"
param l1_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1152:1216]"
param l1_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1152:1216]"
param l1_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1216:1280]"
param l1_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1216:1280]"
param l1_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1280:1344]"
param l1_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1280:1344]"
param l1_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1344:1408]"
param l1_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1344:1408]"
param l1_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1408:1472]"
param l1_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1408:1472]"
param l1_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1472:1536]"
param l1_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1472:1536]"
param l1_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1536:1600]"
param l1_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1536:1600]"
param l1_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1600:1664]"
param l1_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1600:1664]"
param l1_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1664:1728]"
param l1_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1664:1728]"
param l1_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1728:1792]"
param l1_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1728:1792]"
param l1_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1792:1856]"
param l1_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1792:1856]"
param l1_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1856:1920]"
param l1_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1856:1920]"
param l1_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1920:1984]"
param l1_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1920:1984]"
param l1_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.q_proj.weight[1984:2048]"
param l1_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.o_proj.weight[0:1024,1984:2048]"
param l1_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[0:64]"
param l1_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[0:64]"
param l1_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[64:128]"
param l1_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[64:128]"
param l1_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[128:192]"
param l1_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[128:192]"
param l1_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[192:256]"
param l1_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[192:256]"
param l1_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[256:320]"
param l1_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[256:320]"
param l1_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[320:384]"
param l1_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[320:384]"
param l1_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[384:448]"
param l1_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[384:448]"
param l1_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[448:512]"
param l1_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[448:512]"
param l1_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[512:576]"
param l1_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[512:576]"
param l1_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[576:640]"
param l1_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[576:640]"
param l1_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[640:704]"
param l1_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[640:704]"
param l1_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[704:768]"
param l1_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[704:768]"
param l1_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[768:832]"
param l1_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[768:832]"
param l1_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[832:896]"
param l1_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[832:896]"
param l1_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[896:960]"
param l1_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[896:960]"
param l1_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.k_proj.weight[960:1024]"
param l1_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.self_attn.v_proj.weight[960:1024]"
param l1_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.mlp.gate_proj.weight"
param l1_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.mlp.up_proj.weight"
param l1_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.1.mlp.down_proj.weight"

param l2_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.input_layernorm.weight"
param l2_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.post_attention_layernorm.weight"
param l2_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_norm.weight[0:64]"
param l2_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_norm.weight[64:128]"
param l2_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_norm.weight[0:64]"
param l2_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_norm.weight[64:128]"
param l2_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[0:64]"
param l2_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,0:64]"
param l2_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[64:128]"
param l2_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,64:128]"
param l2_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[128:192]"
param l2_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,128:192]"
param l2_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[192:256]"
param l2_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,192:256]"
param l2_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[256:320]"
param l2_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,256:320]"
param l2_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[320:384]"
param l2_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,320:384]"
param l2_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[384:448]"
param l2_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,384:448]"
param l2_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[448:512]"
param l2_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,448:512]"
param l2_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[512:576]"
param l2_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,512:576]"
param l2_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[576:640]"
param l2_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,576:640]"
param l2_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[640:704]"
param l2_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,640:704]"
param l2_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[704:768]"
param l2_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,704:768]"
param l2_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[768:832]"
param l2_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,768:832]"
param l2_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[832:896]"
param l2_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,832:896]"
param l2_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[896:960]"
param l2_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,896:960]"
param l2_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[960:1024]"
param l2_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,960:1024]"
param l2_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1024:1088]"
param l2_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1024:1088]"
param l2_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1088:1152]"
param l2_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1088:1152]"
param l2_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1152:1216]"
param l2_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1152:1216]"
param l2_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1216:1280]"
param l2_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1216:1280]"
param l2_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1280:1344]"
param l2_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1280:1344]"
param l2_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1344:1408]"
param l2_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1344:1408]"
param l2_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1408:1472]"
param l2_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1408:1472]"
param l2_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1472:1536]"
param l2_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1472:1536]"
param l2_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1536:1600]"
param l2_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1536:1600]"
param l2_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1600:1664]"
param l2_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1600:1664]"
param l2_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1664:1728]"
param l2_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1664:1728]"
param l2_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1728:1792]"
param l2_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1728:1792]"
param l2_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1792:1856]"
param l2_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1792:1856]"
param l2_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1856:1920]"
param l2_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1856:1920]"
param l2_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1920:1984]"
param l2_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1920:1984]"
param l2_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.q_proj.weight[1984:2048]"
param l2_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.o_proj.weight[0:1024,1984:2048]"
param l2_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[0:64]"
param l2_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[0:64]"
param l2_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[64:128]"
param l2_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[64:128]"
param l2_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[128:192]"
param l2_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[128:192]"
param l2_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[192:256]"
param l2_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[192:256]"
param l2_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[256:320]"
param l2_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[256:320]"
param l2_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[320:384]"
param l2_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[320:384]"
param l2_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[384:448]"
param l2_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[384:448]"
param l2_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[448:512]"
param l2_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[448:512]"
param l2_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[512:576]"
param l2_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[512:576]"
param l2_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[576:640]"
param l2_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[576:640]"
param l2_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[640:704]"
param l2_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[640:704]"
param l2_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[704:768]"
param l2_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[704:768]"
param l2_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[768:832]"
param l2_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[768:832]"
param l2_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[832:896]"
param l2_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[832:896]"
param l2_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[896:960]"
param l2_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[896:960]"
param l2_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.k_proj.weight[960:1024]"
param l2_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.self_attn.v_proj.weight[960:1024]"
param l2_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.mlp.gate_proj.weight"
param l2_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.mlp.up_proj.weight"
param l2_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.2.mlp.down_proj.weight"

param l3_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.input_layernorm.weight"
param l3_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.post_attention_layernorm.weight"
param l3_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_norm.weight[0:64]"
param l3_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_norm.weight[64:128]"
param l3_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_norm.weight[0:64]"
param l3_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_norm.weight[64:128]"
param l3_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[0:64]"
param l3_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,0:64]"
param l3_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[64:128]"
param l3_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,64:128]"
param l3_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[128:192]"
param l3_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,128:192]"
param l3_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[192:256]"
param l3_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,192:256]"
param l3_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[256:320]"
param l3_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,256:320]"
param l3_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[320:384]"
param l3_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,320:384]"
param l3_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[384:448]"
param l3_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,384:448]"
param l3_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[448:512]"
param l3_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,448:512]"
param l3_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[512:576]"
param l3_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,512:576]"
param l3_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[576:640]"
param l3_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,576:640]"
param l3_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[640:704]"
param l3_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,640:704]"
param l3_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[704:768]"
param l3_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,704:768]"
param l3_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[768:832]"
param l3_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,768:832]"
param l3_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[832:896]"
param l3_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,832:896]"
param l3_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[896:960]"
param l3_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,896:960]"
param l3_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[960:1024]"
param l3_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,960:1024]"
param l3_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1024:1088]"
param l3_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1024:1088]"
param l3_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1088:1152]"
param l3_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1088:1152]"
param l3_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1152:1216]"
param l3_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1152:1216]"
param l3_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1216:1280]"
param l3_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1216:1280]"
param l3_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1280:1344]"
param l3_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1280:1344]"
param l3_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1344:1408]"
param l3_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1344:1408]"
param l3_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1408:1472]"
param l3_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1408:1472]"
param l3_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1472:1536]"
param l3_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1472:1536]"
param l3_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1536:1600]"
param l3_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1536:1600]"
param l3_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1600:1664]"
param l3_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1600:1664]"
param l3_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1664:1728]"
param l3_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1664:1728]"
param l3_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1728:1792]"
param l3_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1728:1792]"
param l3_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1792:1856]"
param l3_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1792:1856]"
param l3_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1856:1920]"
param l3_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1856:1920]"
param l3_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1920:1984]"
param l3_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1920:1984]"
param l3_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.q_proj.weight[1984:2048]"
param l3_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.o_proj.weight[0:1024,1984:2048]"
param l3_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[0:64]"
param l3_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[0:64]"
param l3_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[64:128]"
param l3_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[64:128]"
param l3_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[128:192]"
param l3_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[128:192]"
param l3_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[192:256]"
param l3_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[192:256]"
param l3_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[256:320]"
param l3_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[256:320]"
param l3_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[320:384]"
param l3_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[320:384]"
param l3_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[384:448]"
param l3_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[384:448]"
param l3_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[448:512]"
param l3_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[448:512]"
param l3_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[512:576]"
param l3_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[512:576]"
param l3_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[576:640]"
param l3_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[576:640]"
param l3_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[640:704]"
param l3_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[640:704]"
param l3_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[704:768]"
param l3_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[704:768]"
param l3_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[768:832]"
param l3_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[768:832]"
param l3_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[832:896]"
param l3_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[832:896]"
param l3_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[896:960]"
param l3_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[896:960]"
param l3_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.k_proj.weight[960:1024]"
param l3_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.self_attn.v_proj.weight[960:1024]"
param l3_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.mlp.gate_proj.weight"
param l3_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.mlp.up_proj.weight"
param l3_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.3.mlp.down_proj.weight"

param l4_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.input_layernorm.weight"
param l4_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.post_attention_layernorm.weight"
param l4_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_norm.weight[0:64]"
param l4_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_norm.weight[64:128]"
param l4_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_norm.weight[0:64]"
param l4_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_norm.weight[64:128]"
param l4_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[0:64]"
param l4_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,0:64]"
param l4_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[64:128]"
param l4_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,64:128]"
param l4_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[128:192]"
param l4_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,128:192]"
param l4_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[192:256]"
param l4_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,192:256]"
param l4_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[256:320]"
param l4_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,256:320]"
param l4_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[320:384]"
param l4_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,320:384]"
param l4_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[384:448]"
param l4_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,384:448]"
param l4_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[448:512]"
param l4_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,448:512]"
param l4_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[512:576]"
param l4_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,512:576]"
param l4_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[576:640]"
param l4_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,576:640]"
param l4_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[640:704]"
param l4_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,640:704]"
param l4_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[704:768]"
param l4_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,704:768]"
param l4_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[768:832]"
param l4_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,768:832]"
param l4_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[832:896]"
param l4_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,832:896]"
param l4_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[896:960]"
param l4_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,896:960]"
param l4_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[960:1024]"
param l4_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,960:1024]"
param l4_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1024:1088]"
param l4_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1024:1088]"
param l4_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1088:1152]"
param l4_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1088:1152]"
param l4_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1152:1216]"
param l4_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1152:1216]"
param l4_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1216:1280]"
param l4_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1216:1280]"
param l4_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1280:1344]"
param l4_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1280:1344]"
param l4_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1344:1408]"
param l4_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1344:1408]"
param l4_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1408:1472]"
param l4_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1408:1472]"
param l4_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1472:1536]"
param l4_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1472:1536]"
param l4_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1536:1600]"
param l4_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1536:1600]"
param l4_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1600:1664]"
param l4_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1600:1664]"
param l4_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1664:1728]"
param l4_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1664:1728]"
param l4_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1728:1792]"
param l4_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1728:1792]"
param l4_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1792:1856]"
param l4_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1792:1856]"
param l4_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1856:1920]"
param l4_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1856:1920]"
param l4_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1920:1984]"
param l4_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1920:1984]"
param l4_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.q_proj.weight[1984:2048]"
param l4_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.o_proj.weight[0:1024,1984:2048]"
param l4_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[0:64]"
param l4_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[0:64]"
param l4_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[64:128]"
param l4_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[64:128]"
param l4_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[128:192]"
param l4_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[128:192]"
param l4_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[192:256]"
param l4_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[192:256]"
param l4_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[256:320]"
param l4_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[256:320]"
param l4_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[320:384]"
param l4_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[320:384]"
param l4_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[384:448]"
param l4_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[384:448]"
param l4_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[448:512]"
param l4_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[448:512]"
param l4_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[512:576]"
param l4_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[512:576]"
param l4_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[576:640]"
param l4_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[576:640]"
param l4_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[640:704]"
param l4_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[640:704]"
param l4_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[704:768]"
param l4_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[704:768]"
param l4_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[768:832]"
param l4_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[768:832]"
param l4_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[832:896]"
param l4_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[832:896]"
param l4_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[896:960]"
param l4_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[896:960]"
param l4_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.k_proj.weight[960:1024]"
param l4_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.self_attn.v_proj.weight[960:1024]"
param l4_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.mlp.gate_proj.weight"
param l4_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.mlp.up_proj.weight"
param l4_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.4.mlp.down_proj.weight"

param l5_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.input_layernorm.weight"
param l5_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.post_attention_layernorm.weight"
param l5_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_norm.weight[0:64]"
param l5_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_norm.weight[64:128]"
param l5_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_norm.weight[0:64]"
param l5_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_norm.weight[64:128]"
param l5_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[0:64]"
param l5_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,0:64]"
param l5_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[64:128]"
param l5_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,64:128]"
param l5_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[128:192]"
param l5_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,128:192]"
param l5_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[192:256]"
param l5_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,192:256]"
param l5_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[256:320]"
param l5_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,256:320]"
param l5_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[320:384]"
param l5_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,320:384]"
param l5_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[384:448]"
param l5_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,384:448]"
param l5_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[448:512]"
param l5_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,448:512]"
param l5_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[512:576]"
param l5_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,512:576]"
param l5_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[576:640]"
param l5_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,576:640]"
param l5_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[640:704]"
param l5_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,640:704]"
param l5_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[704:768]"
param l5_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,704:768]"
param l5_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[768:832]"
param l5_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,768:832]"
param l5_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[832:896]"
param l5_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,832:896]"
param l5_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[896:960]"
param l5_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,896:960]"
param l5_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[960:1024]"
param l5_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,960:1024]"
param l5_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1024:1088]"
param l5_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1024:1088]"
param l5_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1088:1152]"
param l5_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1088:1152]"
param l5_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1152:1216]"
param l5_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1152:1216]"
param l5_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1216:1280]"
param l5_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1216:1280]"
param l5_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1280:1344]"
param l5_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1280:1344]"
param l5_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1344:1408]"
param l5_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1344:1408]"
param l5_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1408:1472]"
param l5_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1408:1472]"
param l5_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1472:1536]"
param l5_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1472:1536]"
param l5_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1536:1600]"
param l5_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1536:1600]"
param l5_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1600:1664]"
param l5_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1600:1664]"
param l5_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1664:1728]"
param l5_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1664:1728]"
param l5_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1728:1792]"
param l5_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1728:1792]"
param l5_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1792:1856]"
param l5_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1792:1856]"
param l5_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1856:1920]"
param l5_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1856:1920]"
param l5_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1920:1984]"
param l5_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1920:1984]"
param l5_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.q_proj.weight[1984:2048]"
param l5_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.o_proj.weight[0:1024,1984:2048]"
param l5_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[0:64]"
param l5_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[0:64]"
param l5_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[64:128]"
param l5_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[64:128]"
param l5_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[128:192]"
param l5_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[128:192]"
param l5_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[192:256]"
param l5_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[192:256]"
param l5_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[256:320]"
param l5_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[256:320]"
param l5_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[320:384]"
param l5_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[320:384]"
param l5_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[384:448]"
param l5_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[384:448]"
param l5_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[448:512]"
param l5_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[448:512]"
param l5_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[512:576]"
param l5_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[512:576]"
param l5_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[576:640]"
param l5_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[576:640]"
param l5_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[640:704]"
param l5_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[640:704]"
param l5_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[704:768]"
param l5_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[704:768]"
param l5_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[768:832]"
param l5_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[768:832]"
param l5_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[832:896]"
param l5_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[832:896]"
param l5_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[896:960]"
param l5_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[896:960]"
param l5_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.k_proj.weight[960:1024]"
param l5_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.self_attn.v_proj.weight[960:1024]"
param l5_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.mlp.gate_proj.weight"
param l5_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.mlp.up_proj.weight"
param l5_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.5.mlp.down_proj.weight"

param l6_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.input_layernorm.weight"
param l6_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.post_attention_layernorm.weight"
param l6_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_norm.weight[0:64]"
param l6_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_norm.weight[64:128]"
param l6_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_norm.weight[0:64]"
param l6_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_norm.weight[64:128]"
param l6_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[0:64]"
param l6_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,0:64]"
param l6_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[64:128]"
param l6_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,64:128]"
param l6_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[128:192]"
param l6_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,128:192]"
param l6_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[192:256]"
param l6_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,192:256]"
param l6_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[256:320]"
param l6_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,256:320]"
param l6_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[320:384]"
param l6_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,320:384]"
param l6_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[384:448]"
param l6_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,384:448]"
param l6_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[448:512]"
param l6_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,448:512]"
param l6_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[512:576]"
param l6_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,512:576]"
param l6_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[576:640]"
param l6_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,576:640]"
param l6_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[640:704]"
param l6_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,640:704]"
param l6_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[704:768]"
param l6_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,704:768]"
param l6_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[768:832]"
param l6_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,768:832]"
param l6_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[832:896]"
param l6_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,832:896]"
param l6_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[896:960]"
param l6_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,896:960]"
param l6_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[960:1024]"
param l6_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,960:1024]"
param l6_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1024:1088]"
param l6_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1024:1088]"
param l6_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1088:1152]"
param l6_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1088:1152]"
param l6_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1152:1216]"
param l6_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1152:1216]"
param l6_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1216:1280]"
param l6_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1216:1280]"
param l6_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1280:1344]"
param l6_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1280:1344]"
param l6_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1344:1408]"
param l6_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1344:1408]"
param l6_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1408:1472]"
param l6_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1408:1472]"
param l6_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1472:1536]"
param l6_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1472:1536]"
param l6_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1536:1600]"
param l6_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1536:1600]"
param l6_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1600:1664]"
param l6_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1600:1664]"
param l6_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1664:1728]"
param l6_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1664:1728]"
param l6_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1728:1792]"
param l6_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1728:1792]"
param l6_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1792:1856]"
param l6_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1792:1856]"
param l6_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1856:1920]"
param l6_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1856:1920]"
param l6_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1920:1984]"
param l6_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1920:1984]"
param l6_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.q_proj.weight[1984:2048]"
param l6_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.o_proj.weight[0:1024,1984:2048]"
param l6_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[0:64]"
param l6_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[0:64]"
param l6_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[64:128]"
param l6_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[64:128]"
param l6_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[128:192]"
param l6_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[128:192]"
param l6_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[192:256]"
param l6_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[192:256]"
param l6_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[256:320]"
param l6_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[256:320]"
param l6_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[320:384]"
param l6_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[320:384]"
param l6_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[384:448]"
param l6_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[384:448]"
param l6_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[448:512]"
param l6_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[448:512]"
param l6_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[512:576]"
param l6_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[512:576]"
param l6_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[576:640]"
param l6_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[576:640]"
param l6_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[640:704]"
param l6_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[640:704]"
param l6_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[704:768]"
param l6_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[704:768]"
param l6_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[768:832]"
param l6_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[768:832]"
param l6_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[832:896]"
param l6_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[832:896]"
param l6_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[896:960]"
param l6_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[896:960]"
param l6_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.k_proj.weight[960:1024]"
param l6_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.self_attn.v_proj.weight[960:1024]"
param l6_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.mlp.gate_proj.weight"
param l6_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.mlp.up_proj.weight"
param l6_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.6.mlp.down_proj.weight"

param l7_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.input_layernorm.weight"
param l7_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.post_attention_layernorm.weight"
param l7_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_norm.weight[0:64]"
param l7_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_norm.weight[64:128]"
param l7_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_norm.weight[0:64]"
param l7_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_norm.weight[64:128]"
param l7_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[0:64]"
param l7_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,0:64]"
param l7_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[64:128]"
param l7_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,64:128]"
param l7_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[128:192]"
param l7_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,128:192]"
param l7_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[192:256]"
param l7_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,192:256]"
param l7_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[256:320]"
param l7_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,256:320]"
param l7_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[320:384]"
param l7_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,320:384]"
param l7_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[384:448]"
param l7_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,384:448]"
param l7_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[448:512]"
param l7_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,448:512]"
param l7_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[512:576]"
param l7_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,512:576]"
param l7_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[576:640]"
param l7_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,576:640]"
param l7_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[640:704]"
param l7_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,640:704]"
param l7_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[704:768]"
param l7_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,704:768]"
param l7_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[768:832]"
param l7_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,768:832]"
param l7_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[832:896]"
param l7_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,832:896]"
param l7_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[896:960]"
param l7_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,896:960]"
param l7_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[960:1024]"
param l7_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,960:1024]"
param l7_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1024:1088]"
param l7_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1024:1088]"
param l7_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1088:1152]"
param l7_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1088:1152]"
param l7_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1152:1216]"
param l7_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1152:1216]"
param l7_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1216:1280]"
param l7_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1216:1280]"
param l7_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1280:1344]"
param l7_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1280:1344]"
param l7_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1344:1408]"
param l7_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1344:1408]"
param l7_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1408:1472]"
param l7_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1408:1472]"
param l7_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1472:1536]"
param l7_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1472:1536]"
param l7_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1536:1600]"
param l7_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1536:1600]"
param l7_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1600:1664]"
param l7_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1600:1664]"
param l7_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1664:1728]"
param l7_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1664:1728]"
param l7_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1728:1792]"
param l7_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1728:1792]"
param l7_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1792:1856]"
param l7_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1792:1856]"
param l7_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1856:1920]"
param l7_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1856:1920]"
param l7_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1920:1984]"
param l7_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1920:1984]"
param l7_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.q_proj.weight[1984:2048]"
param l7_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.o_proj.weight[0:1024,1984:2048]"
param l7_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[0:64]"
param l7_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[0:64]"
param l7_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[64:128]"
param l7_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[64:128]"
param l7_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[128:192]"
param l7_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[128:192]"
param l7_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[192:256]"
param l7_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[192:256]"
param l7_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[256:320]"
param l7_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[256:320]"
param l7_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[320:384]"
param l7_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[320:384]"
param l7_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[384:448]"
param l7_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[384:448]"
param l7_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[448:512]"
param l7_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[448:512]"
param l7_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[512:576]"
param l7_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[512:576]"
param l7_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[576:640]"
param l7_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[576:640]"
param l7_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[640:704]"
param l7_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[640:704]"
param l7_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[704:768]"
param l7_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[704:768]"
param l7_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[768:832]"
param l7_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[768:832]"
param l7_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[832:896]"
param l7_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[832:896]"
param l7_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[896:960]"
param l7_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[896:960]"
param l7_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.k_proj.weight[960:1024]"
param l7_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.self_attn.v_proj.weight[960:1024]"
param l7_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.mlp.gate_proj.weight"
param l7_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.mlp.up_proj.weight"
param l7_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.7.mlp.down_proj.weight"

param l8_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.input_layernorm.weight"
param l8_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.post_attention_layernorm.weight"
param l8_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_norm.weight[0:64]"
param l8_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_norm.weight[64:128]"
param l8_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_norm.weight[0:64]"
param l8_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_norm.weight[64:128]"
param l8_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[0:64]"
param l8_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,0:64]"
param l8_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[64:128]"
param l8_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,64:128]"
param l8_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[128:192]"
param l8_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,128:192]"
param l8_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[192:256]"
param l8_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,192:256]"
param l8_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[256:320]"
param l8_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,256:320]"
param l8_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[320:384]"
param l8_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,320:384]"
param l8_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[384:448]"
param l8_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,384:448]"
param l8_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[448:512]"
param l8_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,448:512]"
param l8_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[512:576]"
param l8_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,512:576]"
param l8_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[576:640]"
param l8_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,576:640]"
param l8_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[640:704]"
param l8_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,640:704]"
param l8_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[704:768]"
param l8_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,704:768]"
param l8_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[768:832]"
param l8_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,768:832]"
param l8_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[832:896]"
param l8_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,832:896]"
param l8_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[896:960]"
param l8_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,896:960]"
param l8_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[960:1024]"
param l8_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,960:1024]"
param l8_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1024:1088]"
param l8_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1024:1088]"
param l8_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1088:1152]"
param l8_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1088:1152]"
param l8_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1152:1216]"
param l8_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1152:1216]"
param l8_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1216:1280]"
param l8_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1216:1280]"
param l8_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1280:1344]"
param l8_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1280:1344]"
param l8_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1344:1408]"
param l8_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1344:1408]"
param l8_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1408:1472]"
param l8_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1408:1472]"
param l8_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1472:1536]"
param l8_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1472:1536]"
param l8_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1536:1600]"
param l8_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1536:1600]"
param l8_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1600:1664]"
param l8_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1600:1664]"
param l8_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1664:1728]"
param l8_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1664:1728]"
param l8_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1728:1792]"
param l8_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1728:1792]"
param l8_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1792:1856]"
param l8_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1792:1856]"
param l8_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1856:1920]"
param l8_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1856:1920]"
param l8_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1920:1984]"
param l8_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1920:1984]"
param l8_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.q_proj.weight[1984:2048]"
param l8_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.o_proj.weight[0:1024,1984:2048]"
param l8_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[0:64]"
param l8_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[0:64]"
param l8_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[64:128]"
param l8_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[64:128]"
param l8_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[128:192]"
param l8_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[128:192]"
param l8_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[192:256]"
param l8_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[192:256]"
param l8_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[256:320]"
param l8_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[256:320]"
param l8_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[320:384]"
param l8_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[320:384]"
param l8_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[384:448]"
param l8_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[384:448]"
param l8_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[448:512]"
param l8_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[448:512]"
param l8_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[512:576]"
param l8_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[512:576]"
param l8_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[576:640]"
param l8_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[576:640]"
param l8_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[640:704]"
param l8_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[640:704]"
param l8_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[704:768]"
param l8_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[704:768]"
param l8_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[768:832]"
param l8_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[768:832]"
param l8_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[832:896]"
param l8_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[832:896]"
param l8_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[896:960]"
param l8_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[896:960]"
param l8_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.k_proj.weight[960:1024]"
param l8_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.self_attn.v_proj.weight[960:1024]"
param l8_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.mlp.gate_proj.weight"
param l8_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.mlp.up_proj.weight"
param l8_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.8.mlp.down_proj.weight"

param l9_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.input_layernorm.weight"
param l9_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.post_attention_layernorm.weight"
param l9_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_norm.weight[0:64]"
param l9_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_norm.weight[64:128]"
param l9_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_norm.weight[0:64]"
param l9_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_norm.weight[64:128]"
param l9_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[0:64]"
param l9_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,0:64]"
param l9_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[64:128]"
param l9_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,64:128]"
param l9_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[128:192]"
param l9_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,128:192]"
param l9_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[192:256]"
param l9_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,192:256]"
param l9_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[256:320]"
param l9_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,256:320]"
param l9_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[320:384]"
param l9_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,320:384]"
param l9_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[384:448]"
param l9_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,384:448]"
param l9_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[448:512]"
param l9_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,448:512]"
param l9_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[512:576]"
param l9_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,512:576]"
param l9_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[576:640]"
param l9_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,576:640]"
param l9_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[640:704]"
param l9_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,640:704]"
param l9_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[704:768]"
param l9_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,704:768]"
param l9_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[768:832]"
param l9_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,768:832]"
param l9_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[832:896]"
param l9_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,832:896]"
param l9_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[896:960]"
param l9_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,896:960]"
param l9_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[960:1024]"
param l9_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,960:1024]"
param l9_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1024:1088]"
param l9_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1024:1088]"
param l9_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1088:1152]"
param l9_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1088:1152]"
param l9_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1152:1216]"
param l9_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1152:1216]"
param l9_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1216:1280]"
param l9_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1216:1280]"
param l9_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1280:1344]"
param l9_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1280:1344]"
param l9_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1344:1408]"
param l9_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1344:1408]"
param l9_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1408:1472]"
param l9_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1408:1472]"
param l9_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1472:1536]"
param l9_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1472:1536]"
param l9_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1536:1600]"
param l9_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1536:1600]"
param l9_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1600:1664]"
param l9_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1600:1664]"
param l9_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1664:1728]"
param l9_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1664:1728]"
param l9_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1728:1792]"
param l9_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1728:1792]"
param l9_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1792:1856]"
param l9_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1792:1856]"
param l9_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1856:1920]"
param l9_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1856:1920]"
param l9_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1920:1984]"
param l9_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1920:1984]"
param l9_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.q_proj.weight[1984:2048]"
param l9_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.o_proj.weight[0:1024,1984:2048]"
param l9_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[0:64]"
param l9_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[0:64]"
param l9_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[64:128]"
param l9_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[64:128]"
param l9_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[128:192]"
param l9_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[128:192]"
param l9_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[192:256]"
param l9_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[192:256]"
param l9_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[256:320]"
param l9_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[256:320]"
param l9_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[320:384]"
param l9_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[320:384]"
param l9_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[384:448]"
param l9_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[384:448]"
param l9_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[448:512]"
param l9_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[448:512]"
param l9_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[512:576]"
param l9_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[512:576]"
param l9_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[576:640]"
param l9_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[576:640]"
param l9_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[640:704]"
param l9_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[640:704]"
param l9_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[704:768]"
param l9_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[704:768]"
param l9_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[768:832]"
param l9_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[768:832]"
param l9_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[832:896]"
param l9_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[832:896]"
param l9_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[896:960]"
param l9_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[896:960]"
param l9_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.k_proj.weight[960:1024]"
param l9_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.self_attn.v_proj.weight[960:1024]"
param l9_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.mlp.gate_proj.weight"
param l9_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.mlp.up_proj.weight"
param l9_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.9.mlp.down_proj.weight"

param l10_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.input_layernorm.weight"
param l10_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.post_attention_layernorm.weight"
param l10_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_norm.weight[0:64]"
param l10_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_norm.weight[64:128]"
param l10_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_norm.weight[0:64]"
param l10_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_norm.weight[64:128]"
param l10_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[0:64]"
param l10_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,0:64]"
param l10_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[64:128]"
param l10_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,64:128]"
param l10_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[128:192]"
param l10_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,128:192]"
param l10_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[192:256]"
param l10_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,192:256]"
param l10_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[256:320]"
param l10_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,256:320]"
param l10_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[320:384]"
param l10_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,320:384]"
param l10_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[384:448]"
param l10_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,384:448]"
param l10_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[448:512]"
param l10_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,448:512]"
param l10_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[512:576]"
param l10_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,512:576]"
param l10_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[576:640]"
param l10_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,576:640]"
param l10_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[640:704]"
param l10_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,640:704]"
param l10_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[704:768]"
param l10_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,704:768]"
param l10_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[768:832]"
param l10_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,768:832]"
param l10_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[832:896]"
param l10_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,832:896]"
param l10_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[896:960]"
param l10_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,896:960]"
param l10_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[960:1024]"
param l10_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,960:1024]"
param l10_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1024:1088]"
param l10_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1024:1088]"
param l10_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1088:1152]"
param l10_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1088:1152]"
param l10_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1152:1216]"
param l10_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1152:1216]"
param l10_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1216:1280]"
param l10_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1216:1280]"
param l10_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1280:1344]"
param l10_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1280:1344]"
param l10_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1344:1408]"
param l10_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1344:1408]"
param l10_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1408:1472]"
param l10_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1408:1472]"
param l10_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1472:1536]"
param l10_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1472:1536]"
param l10_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1536:1600]"
param l10_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1536:1600]"
param l10_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1600:1664]"
param l10_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1600:1664]"
param l10_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1664:1728]"
param l10_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1664:1728]"
param l10_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1728:1792]"
param l10_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1728:1792]"
param l10_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1792:1856]"
param l10_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1792:1856]"
param l10_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1856:1920]"
param l10_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1856:1920]"
param l10_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1920:1984]"
param l10_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1920:1984]"
param l10_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.q_proj.weight[1984:2048]"
param l10_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.o_proj.weight[0:1024,1984:2048]"
param l10_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[0:64]"
param l10_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[0:64]"
param l10_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[64:128]"
param l10_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[64:128]"
param l10_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[128:192]"
param l10_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[128:192]"
param l10_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[192:256]"
param l10_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[192:256]"
param l10_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[256:320]"
param l10_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[256:320]"
param l10_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[320:384]"
param l10_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[320:384]"
param l10_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[384:448]"
param l10_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[384:448]"
param l10_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[448:512]"
param l10_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[448:512]"
param l10_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[512:576]"
param l10_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[512:576]"
param l10_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[576:640]"
param l10_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[576:640]"
param l10_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[640:704]"
param l10_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[640:704]"
param l10_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[704:768]"
param l10_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[704:768]"
param l10_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[768:832]"
param l10_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[768:832]"
param l10_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[832:896]"
param l10_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[832:896]"
param l10_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[896:960]"
param l10_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[896:960]"
param l10_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.k_proj.weight[960:1024]"
param l10_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.self_attn.v_proj.weight[960:1024]"
param l10_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.mlp.gate_proj.weight"
param l10_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.mlp.up_proj.weight"
param l10_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.10.mlp.down_proj.weight"

param l11_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.input_layernorm.weight"
param l11_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.post_attention_layernorm.weight"
param l11_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_norm.weight[0:64]"
param l11_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_norm.weight[64:128]"
param l11_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_norm.weight[0:64]"
param l11_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_norm.weight[64:128]"
param l11_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[0:64]"
param l11_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,0:64]"
param l11_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[64:128]"
param l11_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,64:128]"
param l11_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[128:192]"
param l11_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,128:192]"
param l11_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[192:256]"
param l11_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,192:256]"
param l11_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[256:320]"
param l11_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,256:320]"
param l11_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[320:384]"
param l11_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,320:384]"
param l11_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[384:448]"
param l11_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,384:448]"
param l11_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[448:512]"
param l11_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,448:512]"
param l11_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[512:576]"
param l11_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,512:576]"
param l11_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[576:640]"
param l11_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,576:640]"
param l11_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[640:704]"
param l11_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,640:704]"
param l11_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[704:768]"
param l11_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,704:768]"
param l11_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[768:832]"
param l11_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,768:832]"
param l11_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[832:896]"
param l11_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,832:896]"
param l11_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[896:960]"
param l11_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,896:960]"
param l11_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[960:1024]"
param l11_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,960:1024]"
param l11_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1024:1088]"
param l11_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1024:1088]"
param l11_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1088:1152]"
param l11_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1088:1152]"
param l11_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1152:1216]"
param l11_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1152:1216]"
param l11_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1216:1280]"
param l11_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1216:1280]"
param l11_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1280:1344]"
param l11_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1280:1344]"
param l11_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1344:1408]"
param l11_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1344:1408]"
param l11_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1408:1472]"
param l11_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1408:1472]"
param l11_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1472:1536]"
param l11_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1472:1536]"
param l11_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1536:1600]"
param l11_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1536:1600]"
param l11_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1600:1664]"
param l11_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1600:1664]"
param l11_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1664:1728]"
param l11_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1664:1728]"
param l11_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1728:1792]"
param l11_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1728:1792]"
param l11_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1792:1856]"
param l11_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1792:1856]"
param l11_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1856:1920]"
param l11_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1856:1920]"
param l11_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1920:1984]"
param l11_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1920:1984]"
param l11_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.q_proj.weight[1984:2048]"
param l11_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.o_proj.weight[0:1024,1984:2048]"
param l11_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[0:64]"
param l11_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[0:64]"
param l11_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[64:128]"
param l11_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[64:128]"
param l11_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[128:192]"
param l11_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[128:192]"
param l11_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[192:256]"
param l11_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[192:256]"
param l11_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[256:320]"
param l11_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[256:320]"
param l11_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[320:384]"
param l11_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[320:384]"
param l11_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[384:448]"
param l11_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[384:448]"
param l11_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[448:512]"
param l11_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[448:512]"
param l11_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[512:576]"
param l11_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[512:576]"
param l11_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[576:640]"
param l11_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[576:640]"
param l11_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[640:704]"
param l11_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[640:704]"
param l11_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[704:768]"
param l11_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[704:768]"
param l11_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[768:832]"
param l11_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[768:832]"
param l11_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[832:896]"
param l11_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[832:896]"
param l11_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[896:960]"
param l11_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[896:960]"
param l11_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.k_proj.weight[960:1024]"
param l11_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.self_attn.v_proj.weight[960:1024]"
param l11_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.mlp.gate_proj.weight"
param l11_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.mlp.up_proj.weight"
param l11_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.11.mlp.down_proj.weight"

param l12_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.input_layernorm.weight"
param l12_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.post_attention_layernorm.weight"
param l12_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_norm.weight[0:64]"
param l12_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_norm.weight[64:128]"
param l12_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_norm.weight[0:64]"
param l12_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_norm.weight[64:128]"
param l12_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[0:64]"
param l12_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,0:64]"
param l12_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[64:128]"
param l12_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,64:128]"
param l12_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[128:192]"
param l12_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,128:192]"
param l12_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[192:256]"
param l12_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,192:256]"
param l12_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[256:320]"
param l12_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,256:320]"
param l12_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[320:384]"
param l12_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,320:384]"
param l12_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[384:448]"
param l12_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,384:448]"
param l12_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[448:512]"
param l12_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,448:512]"
param l12_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[512:576]"
param l12_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,512:576]"
param l12_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[576:640]"
param l12_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,576:640]"
param l12_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[640:704]"
param l12_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,640:704]"
param l12_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[704:768]"
param l12_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,704:768]"
param l12_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[768:832]"
param l12_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,768:832]"
param l12_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[832:896]"
param l12_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,832:896]"
param l12_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[896:960]"
param l12_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,896:960]"
param l12_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[960:1024]"
param l12_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,960:1024]"
param l12_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1024:1088]"
param l12_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1024:1088]"
param l12_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1088:1152]"
param l12_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1088:1152]"
param l12_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1152:1216]"
param l12_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1152:1216]"
param l12_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1216:1280]"
param l12_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1216:1280]"
param l12_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1280:1344]"
param l12_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1280:1344]"
param l12_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1344:1408]"
param l12_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1344:1408]"
param l12_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1408:1472]"
param l12_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1408:1472]"
param l12_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1472:1536]"
param l12_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1472:1536]"
param l12_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1536:1600]"
param l12_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1536:1600]"
param l12_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1600:1664]"
param l12_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1600:1664]"
param l12_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1664:1728]"
param l12_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1664:1728]"
param l12_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1728:1792]"
param l12_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1728:1792]"
param l12_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1792:1856]"
param l12_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1792:1856]"
param l12_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1856:1920]"
param l12_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1856:1920]"
param l12_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1920:1984]"
param l12_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1920:1984]"
param l12_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.q_proj.weight[1984:2048]"
param l12_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.o_proj.weight[0:1024,1984:2048]"
param l12_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[0:64]"
param l12_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[0:64]"
param l12_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[64:128]"
param l12_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[64:128]"
param l12_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[128:192]"
param l12_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[128:192]"
param l12_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[192:256]"
param l12_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[192:256]"
param l12_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[256:320]"
param l12_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[256:320]"
param l12_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[320:384]"
param l12_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[320:384]"
param l12_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[384:448]"
param l12_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[384:448]"
param l12_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[448:512]"
param l12_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[448:512]"
param l12_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[512:576]"
param l12_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[512:576]"
param l12_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[576:640]"
param l12_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[576:640]"
param l12_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[640:704]"
param l12_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[640:704]"
param l12_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[704:768]"
param l12_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[704:768]"
param l12_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[768:832]"
param l12_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[768:832]"
param l12_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[832:896]"
param l12_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[832:896]"
param l12_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[896:960]"
param l12_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[896:960]"
param l12_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.k_proj.weight[960:1024]"
param l12_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.self_attn.v_proj.weight[960:1024]"
param l12_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.mlp.gate_proj.weight"
param l12_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.mlp.up_proj.weight"
param l12_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.12.mlp.down_proj.weight"

param l13_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.input_layernorm.weight"
param l13_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.post_attention_layernorm.weight"
param l13_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_norm.weight[0:64]"
param l13_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_norm.weight[64:128]"
param l13_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_norm.weight[0:64]"
param l13_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_norm.weight[64:128]"
param l13_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[0:64]"
param l13_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,0:64]"
param l13_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[64:128]"
param l13_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,64:128]"
param l13_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[128:192]"
param l13_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,128:192]"
param l13_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[192:256]"
param l13_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,192:256]"
param l13_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[256:320]"
param l13_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,256:320]"
param l13_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[320:384]"
param l13_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,320:384]"
param l13_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[384:448]"
param l13_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,384:448]"
param l13_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[448:512]"
param l13_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,448:512]"
param l13_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[512:576]"
param l13_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,512:576]"
param l13_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[576:640]"
param l13_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,576:640]"
param l13_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[640:704]"
param l13_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,640:704]"
param l13_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[704:768]"
param l13_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,704:768]"
param l13_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[768:832]"
param l13_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,768:832]"
param l13_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[832:896]"
param l13_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,832:896]"
param l13_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[896:960]"
param l13_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,896:960]"
param l13_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[960:1024]"
param l13_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,960:1024]"
param l13_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1024:1088]"
param l13_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1024:1088]"
param l13_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1088:1152]"
param l13_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1088:1152]"
param l13_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1152:1216]"
param l13_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1152:1216]"
param l13_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1216:1280]"
param l13_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1216:1280]"
param l13_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1280:1344]"
param l13_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1280:1344]"
param l13_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1344:1408]"
param l13_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1344:1408]"
param l13_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1408:1472]"
param l13_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1408:1472]"
param l13_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1472:1536]"
param l13_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1472:1536]"
param l13_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1536:1600]"
param l13_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1536:1600]"
param l13_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1600:1664]"
param l13_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1600:1664]"
param l13_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1664:1728]"
param l13_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1664:1728]"
param l13_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1728:1792]"
param l13_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1728:1792]"
param l13_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1792:1856]"
param l13_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1792:1856]"
param l13_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1856:1920]"
param l13_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1856:1920]"
param l13_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1920:1984]"
param l13_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1920:1984]"
param l13_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.q_proj.weight[1984:2048]"
param l13_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.o_proj.weight[0:1024,1984:2048]"
param l13_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[0:64]"
param l13_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[0:64]"
param l13_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[64:128]"
param l13_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[64:128]"
param l13_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[128:192]"
param l13_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[128:192]"
param l13_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[192:256]"
param l13_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[192:256]"
param l13_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[256:320]"
param l13_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[256:320]"
param l13_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[320:384]"
param l13_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[320:384]"
param l13_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[384:448]"
param l13_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[384:448]"
param l13_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[448:512]"
param l13_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[448:512]"
param l13_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[512:576]"
param l13_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[512:576]"
param l13_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[576:640]"
param l13_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[576:640]"
param l13_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[640:704]"
param l13_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[640:704]"
param l13_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[704:768]"
param l13_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[704:768]"
param l13_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[768:832]"
param l13_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[768:832]"
param l13_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[832:896]"
param l13_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[832:896]"
param l13_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[896:960]"
param l13_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[896:960]"
param l13_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.k_proj.weight[960:1024]"
param l13_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.self_attn.v_proj.weight[960:1024]"
param l13_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.mlp.gate_proj.weight"
param l13_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.mlp.up_proj.weight"
param l13_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.13.mlp.down_proj.weight"

param l14_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.input_layernorm.weight"
param l14_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.post_attention_layernorm.weight"
param l14_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_norm.weight[0:64]"
param l14_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_norm.weight[64:128]"
param l14_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_norm.weight[0:64]"
param l14_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_norm.weight[64:128]"
param l14_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[0:64]"
param l14_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,0:64]"
param l14_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[64:128]"
param l14_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,64:128]"
param l14_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[128:192]"
param l14_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,128:192]"
param l14_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[192:256]"
param l14_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,192:256]"
param l14_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[256:320]"
param l14_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,256:320]"
param l14_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[320:384]"
param l14_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,320:384]"
param l14_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[384:448]"
param l14_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,384:448]"
param l14_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[448:512]"
param l14_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,448:512]"
param l14_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[512:576]"
param l14_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,512:576]"
param l14_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[576:640]"
param l14_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,576:640]"
param l14_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[640:704]"
param l14_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,640:704]"
param l14_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[704:768]"
param l14_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,704:768]"
param l14_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[768:832]"
param l14_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,768:832]"
param l14_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[832:896]"
param l14_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,832:896]"
param l14_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[896:960]"
param l14_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,896:960]"
param l14_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[960:1024]"
param l14_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,960:1024]"
param l14_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1024:1088]"
param l14_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1024:1088]"
param l14_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1088:1152]"
param l14_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1088:1152]"
param l14_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1152:1216]"
param l14_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1152:1216]"
param l14_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1216:1280]"
param l14_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1216:1280]"
param l14_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1280:1344]"
param l14_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1280:1344]"
param l14_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1344:1408]"
param l14_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1344:1408]"
param l14_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1408:1472]"
param l14_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1408:1472]"
param l14_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1472:1536]"
param l14_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1472:1536]"
param l14_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1536:1600]"
param l14_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1536:1600]"
param l14_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1600:1664]"
param l14_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1600:1664]"
param l14_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1664:1728]"
param l14_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1664:1728]"
param l14_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1728:1792]"
param l14_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1728:1792]"
param l14_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1792:1856]"
param l14_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1792:1856]"
param l14_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1856:1920]"
param l14_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1856:1920]"
param l14_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1920:1984]"
param l14_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1920:1984]"
param l14_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.q_proj.weight[1984:2048]"
param l14_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.o_proj.weight[0:1024,1984:2048]"
param l14_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[0:64]"
param l14_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[0:64]"
param l14_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[64:128]"
param l14_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[64:128]"
param l14_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[128:192]"
param l14_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[128:192]"
param l14_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[192:256]"
param l14_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[192:256]"
param l14_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[256:320]"
param l14_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[256:320]"
param l14_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[320:384]"
param l14_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[320:384]"
param l14_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[384:448]"
param l14_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[384:448]"
param l14_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[448:512]"
param l14_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[448:512]"
param l14_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[512:576]"
param l14_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[512:576]"
param l14_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[576:640]"
param l14_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[576:640]"
param l14_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[640:704]"
param l14_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[640:704]"
param l14_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[704:768]"
param l14_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[704:768]"
param l14_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[768:832]"
param l14_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[768:832]"
param l14_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[832:896]"
param l14_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[832:896]"
param l14_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[896:960]"
param l14_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[896:960]"
param l14_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.k_proj.weight[960:1024]"
param l14_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.self_attn.v_proj.weight[960:1024]"
param l14_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.mlp.gate_proj.weight"
param l14_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.mlp.up_proj.weight"
param l14_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.14.mlp.down_proj.weight"

param l15_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.input_layernorm.weight"
param l15_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.post_attention_layernorm.weight"
param l15_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_norm.weight[0:64]"
param l15_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_norm.weight[64:128]"
param l15_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_norm.weight[0:64]"
param l15_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_norm.weight[64:128]"
param l15_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[0:64]"
param l15_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,0:64]"
param l15_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[64:128]"
param l15_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,64:128]"
param l15_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[128:192]"
param l15_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,128:192]"
param l15_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[192:256]"
param l15_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,192:256]"
param l15_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[256:320]"
param l15_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,256:320]"
param l15_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[320:384]"
param l15_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,320:384]"
param l15_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[384:448]"
param l15_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,384:448]"
param l15_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[448:512]"
param l15_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,448:512]"
param l15_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[512:576]"
param l15_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,512:576]"
param l15_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[576:640]"
param l15_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,576:640]"
param l15_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[640:704]"
param l15_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,640:704]"
param l15_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[704:768]"
param l15_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,704:768]"
param l15_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[768:832]"
param l15_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,768:832]"
param l15_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[832:896]"
param l15_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,832:896]"
param l15_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[896:960]"
param l15_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,896:960]"
param l15_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[960:1024]"
param l15_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,960:1024]"
param l15_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1024:1088]"
param l15_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1024:1088]"
param l15_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1088:1152]"
param l15_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1088:1152]"
param l15_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1152:1216]"
param l15_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1152:1216]"
param l15_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1216:1280]"
param l15_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1216:1280]"
param l15_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1280:1344]"
param l15_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1280:1344]"
param l15_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1344:1408]"
param l15_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1344:1408]"
param l15_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1408:1472]"
param l15_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1408:1472]"
param l15_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1472:1536]"
param l15_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1472:1536]"
param l15_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1536:1600]"
param l15_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1536:1600]"
param l15_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1600:1664]"
param l15_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1600:1664]"
param l15_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1664:1728]"
param l15_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1664:1728]"
param l15_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1728:1792]"
param l15_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1728:1792]"
param l15_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1792:1856]"
param l15_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1792:1856]"
param l15_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1856:1920]"
param l15_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1856:1920]"
param l15_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1920:1984]"
param l15_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1920:1984]"
param l15_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.q_proj.weight[1984:2048]"
param l15_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.o_proj.weight[0:1024,1984:2048]"
param l15_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[0:64]"
param l15_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[0:64]"
param l15_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[64:128]"
param l15_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[64:128]"
param l15_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[128:192]"
param l15_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[128:192]"
param l15_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[192:256]"
param l15_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[192:256]"
param l15_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[256:320]"
param l15_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[256:320]"
param l15_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[320:384]"
param l15_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[320:384]"
param l15_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[384:448]"
param l15_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[384:448]"
param l15_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[448:512]"
param l15_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[448:512]"
param l15_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[512:576]"
param l15_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[512:576]"
param l15_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[576:640]"
param l15_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[576:640]"
param l15_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[640:704]"
param l15_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[640:704]"
param l15_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[704:768]"
param l15_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[704:768]"
param l15_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[768:832]"
param l15_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[768:832]"
param l15_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[832:896]"
param l15_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[832:896]"
param l15_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[896:960]"
param l15_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[896:960]"
param l15_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.k_proj.weight[960:1024]"
param l15_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.self_attn.v_proj.weight[960:1024]"
param l15_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.mlp.gate_proj.weight"
param l15_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.mlp.up_proj.weight"
param l15_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.15.mlp.down_proj.weight"

param l16_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.input_layernorm.weight"
param l16_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.post_attention_layernorm.weight"
param l16_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_norm.weight[0:64]"
param l16_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_norm.weight[64:128]"
param l16_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_norm.weight[0:64]"
param l16_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_norm.weight[64:128]"
param l16_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[0:64]"
param l16_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,0:64]"
param l16_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[64:128]"
param l16_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,64:128]"
param l16_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[128:192]"
param l16_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,128:192]"
param l16_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[192:256]"
param l16_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,192:256]"
param l16_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[256:320]"
param l16_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,256:320]"
param l16_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[320:384]"
param l16_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,320:384]"
param l16_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[384:448]"
param l16_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,384:448]"
param l16_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[448:512]"
param l16_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,448:512]"
param l16_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[512:576]"
param l16_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,512:576]"
param l16_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[576:640]"
param l16_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,576:640]"
param l16_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[640:704]"
param l16_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,640:704]"
param l16_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[704:768]"
param l16_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,704:768]"
param l16_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[768:832]"
param l16_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,768:832]"
param l16_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[832:896]"
param l16_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,832:896]"
param l16_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[896:960]"
param l16_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,896:960]"
param l16_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[960:1024]"
param l16_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,960:1024]"
param l16_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1024:1088]"
param l16_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1024:1088]"
param l16_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1088:1152]"
param l16_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1088:1152]"
param l16_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1152:1216]"
param l16_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1152:1216]"
param l16_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1216:1280]"
param l16_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1216:1280]"
param l16_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1280:1344]"
param l16_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1280:1344]"
param l16_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1344:1408]"
param l16_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1344:1408]"
param l16_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1408:1472]"
param l16_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1408:1472]"
param l16_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1472:1536]"
param l16_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1472:1536]"
param l16_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1536:1600]"
param l16_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1536:1600]"
param l16_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1600:1664]"
param l16_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1600:1664]"
param l16_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1664:1728]"
param l16_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1664:1728]"
param l16_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1728:1792]"
param l16_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1728:1792]"
param l16_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1792:1856]"
param l16_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1792:1856]"
param l16_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1856:1920]"
param l16_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1856:1920]"
param l16_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1920:1984]"
param l16_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1920:1984]"
param l16_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.q_proj.weight[1984:2048]"
param l16_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.o_proj.weight[0:1024,1984:2048]"
param l16_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[0:64]"
param l16_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[0:64]"
param l16_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[64:128]"
param l16_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[64:128]"
param l16_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[128:192]"
param l16_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[128:192]"
param l16_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[192:256]"
param l16_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[192:256]"
param l16_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[256:320]"
param l16_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[256:320]"
param l16_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[320:384]"
param l16_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[320:384]"
param l16_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[384:448]"
param l16_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[384:448]"
param l16_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[448:512]"
param l16_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[448:512]"
param l16_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[512:576]"
param l16_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[512:576]"
param l16_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[576:640]"
param l16_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[576:640]"
param l16_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[640:704]"
param l16_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[640:704]"
param l16_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[704:768]"
param l16_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[704:768]"
param l16_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[768:832]"
param l16_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[768:832]"
param l16_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[832:896]"
param l16_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[832:896]"
param l16_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[896:960]"
param l16_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[896:960]"
param l16_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.k_proj.weight[960:1024]"
param l16_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.self_attn.v_proj.weight[960:1024]"
param l16_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.mlp.gate_proj.weight"
param l16_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.mlp.up_proj.weight"
param l16_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.16.mlp.down_proj.weight"

param l17_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.input_layernorm.weight"
param l17_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.post_attention_layernorm.weight"
param l17_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_norm.weight[0:64]"
param l17_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_norm.weight[64:128]"
param l17_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_norm.weight[0:64]"
param l17_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_norm.weight[64:128]"
param l17_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[0:64]"
param l17_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,0:64]"
param l17_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[64:128]"
param l17_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,64:128]"
param l17_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[128:192]"
param l17_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,128:192]"
param l17_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[192:256]"
param l17_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,192:256]"
param l17_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[256:320]"
param l17_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,256:320]"
param l17_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[320:384]"
param l17_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,320:384]"
param l17_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[384:448]"
param l17_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,384:448]"
param l17_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[448:512]"
param l17_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,448:512]"
param l17_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[512:576]"
param l17_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,512:576]"
param l17_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[576:640]"
param l17_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,576:640]"
param l17_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[640:704]"
param l17_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,640:704]"
param l17_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[704:768]"
param l17_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,704:768]"
param l17_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[768:832]"
param l17_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,768:832]"
param l17_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[832:896]"
param l17_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,832:896]"
param l17_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[896:960]"
param l17_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,896:960]"
param l17_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[960:1024]"
param l17_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,960:1024]"
param l17_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1024:1088]"
param l17_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1024:1088]"
param l17_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1088:1152]"
param l17_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1088:1152]"
param l17_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1152:1216]"
param l17_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1152:1216]"
param l17_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1216:1280]"
param l17_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1216:1280]"
param l17_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1280:1344]"
param l17_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1280:1344]"
param l17_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1344:1408]"
param l17_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1344:1408]"
param l17_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1408:1472]"
param l17_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1408:1472]"
param l17_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1472:1536]"
param l17_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1472:1536]"
param l17_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1536:1600]"
param l17_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1536:1600]"
param l17_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1600:1664]"
param l17_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1600:1664]"
param l17_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1664:1728]"
param l17_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1664:1728]"
param l17_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1728:1792]"
param l17_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1728:1792]"
param l17_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1792:1856]"
param l17_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1792:1856]"
param l17_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1856:1920]"
param l17_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1856:1920]"
param l17_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1920:1984]"
param l17_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1920:1984]"
param l17_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.q_proj.weight[1984:2048]"
param l17_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.o_proj.weight[0:1024,1984:2048]"
param l17_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[0:64]"
param l17_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[0:64]"
param l17_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[64:128]"
param l17_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[64:128]"
param l17_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[128:192]"
param l17_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[128:192]"
param l17_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[192:256]"
param l17_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[192:256]"
param l17_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[256:320]"
param l17_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[256:320]"
param l17_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[320:384]"
param l17_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[320:384]"
param l17_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[384:448]"
param l17_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[384:448]"
param l17_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[448:512]"
param l17_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[448:512]"
param l17_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[512:576]"
param l17_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[512:576]"
param l17_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[576:640]"
param l17_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[576:640]"
param l17_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[640:704]"
param l17_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[640:704]"
param l17_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[704:768]"
param l17_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[704:768]"
param l17_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[768:832]"
param l17_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[768:832]"
param l17_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[832:896]"
param l17_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[832:896]"
param l17_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[896:960]"
param l17_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[896:960]"
param l17_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.k_proj.weight[960:1024]"
param l17_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.self_attn.v_proj.weight[960:1024]"
param l17_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.mlp.gate_proj.weight"
param l17_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.mlp.up_proj.weight"
param l17_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.17.mlp.down_proj.weight"

param l18_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.input_layernorm.weight"
param l18_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.post_attention_layernorm.weight"
param l18_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_norm.weight[0:64]"
param l18_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_norm.weight[64:128]"
param l18_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_norm.weight[0:64]"
param l18_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_norm.weight[64:128]"
param l18_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[0:64]"
param l18_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,0:64]"
param l18_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[64:128]"
param l18_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,64:128]"
param l18_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[128:192]"
param l18_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,128:192]"
param l18_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[192:256]"
param l18_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,192:256]"
param l18_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[256:320]"
param l18_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,256:320]"
param l18_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[320:384]"
param l18_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,320:384]"
param l18_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[384:448]"
param l18_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,384:448]"
param l18_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[448:512]"
param l18_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,448:512]"
param l18_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[512:576]"
param l18_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,512:576]"
param l18_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[576:640]"
param l18_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,576:640]"
param l18_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[640:704]"
param l18_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,640:704]"
param l18_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[704:768]"
param l18_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,704:768]"
param l18_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[768:832]"
param l18_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,768:832]"
param l18_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[832:896]"
param l18_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,832:896]"
param l18_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[896:960]"
param l18_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,896:960]"
param l18_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[960:1024]"
param l18_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,960:1024]"
param l18_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1024:1088]"
param l18_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1024:1088]"
param l18_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1088:1152]"
param l18_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1088:1152]"
param l18_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1152:1216]"
param l18_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1152:1216]"
param l18_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1216:1280]"
param l18_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1216:1280]"
param l18_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1280:1344]"
param l18_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1280:1344]"
param l18_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1344:1408]"
param l18_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1344:1408]"
param l18_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1408:1472]"
param l18_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1408:1472]"
param l18_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1472:1536]"
param l18_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1472:1536]"
param l18_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1536:1600]"
param l18_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1536:1600]"
param l18_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1600:1664]"
param l18_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1600:1664]"
param l18_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1664:1728]"
param l18_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1664:1728]"
param l18_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1728:1792]"
param l18_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1728:1792]"
param l18_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1792:1856]"
param l18_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1792:1856]"
param l18_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1856:1920]"
param l18_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1856:1920]"
param l18_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1920:1984]"
param l18_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1920:1984]"
param l18_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.q_proj.weight[1984:2048]"
param l18_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.o_proj.weight[0:1024,1984:2048]"
param l18_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[0:64]"
param l18_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[0:64]"
param l18_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[64:128]"
param l18_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[64:128]"
param l18_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[128:192]"
param l18_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[128:192]"
param l18_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[192:256]"
param l18_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[192:256]"
param l18_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[256:320]"
param l18_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[256:320]"
param l18_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[320:384]"
param l18_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[320:384]"
param l18_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[384:448]"
param l18_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[384:448]"
param l18_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[448:512]"
param l18_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[448:512]"
param l18_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[512:576]"
param l18_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[512:576]"
param l18_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[576:640]"
param l18_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[576:640]"
param l18_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[640:704]"
param l18_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[640:704]"
param l18_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[704:768]"
param l18_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[704:768]"
param l18_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[768:832]"
param l18_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[768:832]"
param l18_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[832:896]"
param l18_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[832:896]"
param l18_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[896:960]"
param l18_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[896:960]"
param l18_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.k_proj.weight[960:1024]"
param l18_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.self_attn.v_proj.weight[960:1024]"
param l18_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.mlp.gate_proj.weight"
param l18_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.mlp.up_proj.weight"
param l18_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.18.mlp.down_proj.weight"

param l19_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.input_layernorm.weight"
param l19_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.post_attention_layernorm.weight"
param l19_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_norm.weight[0:64]"
param l19_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_norm.weight[64:128]"
param l19_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_norm.weight[0:64]"
param l19_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_norm.weight[64:128]"
param l19_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[0:64]"
param l19_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,0:64]"
param l19_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[64:128]"
param l19_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,64:128]"
param l19_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[128:192]"
param l19_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,128:192]"
param l19_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[192:256]"
param l19_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,192:256]"
param l19_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[256:320]"
param l19_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,256:320]"
param l19_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[320:384]"
param l19_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,320:384]"
param l19_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[384:448]"
param l19_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,384:448]"
param l19_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[448:512]"
param l19_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,448:512]"
param l19_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[512:576]"
param l19_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,512:576]"
param l19_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[576:640]"
param l19_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,576:640]"
param l19_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[640:704]"
param l19_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,640:704]"
param l19_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[704:768]"
param l19_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,704:768]"
param l19_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[768:832]"
param l19_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,768:832]"
param l19_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[832:896]"
param l19_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,832:896]"
param l19_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[896:960]"
param l19_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,896:960]"
param l19_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[960:1024]"
param l19_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,960:1024]"
param l19_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1024:1088]"
param l19_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1024:1088]"
param l19_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1088:1152]"
param l19_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1088:1152]"
param l19_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1152:1216]"
param l19_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1152:1216]"
param l19_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1216:1280]"
param l19_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1216:1280]"
param l19_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1280:1344]"
param l19_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1280:1344]"
param l19_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1344:1408]"
param l19_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1344:1408]"
param l19_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1408:1472]"
param l19_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1408:1472]"
param l19_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1472:1536]"
param l19_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1472:1536]"
param l19_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1536:1600]"
param l19_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1536:1600]"
param l19_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1600:1664]"
param l19_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1600:1664]"
param l19_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1664:1728]"
param l19_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1664:1728]"
param l19_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1728:1792]"
param l19_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1728:1792]"
param l19_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1792:1856]"
param l19_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1792:1856]"
param l19_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1856:1920]"
param l19_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1856:1920]"
param l19_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1920:1984]"
param l19_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1920:1984]"
param l19_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.q_proj.weight[1984:2048]"
param l19_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.o_proj.weight[0:1024,1984:2048]"
param l19_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[0:64]"
param l19_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[0:64]"
param l19_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[64:128]"
param l19_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[64:128]"
param l19_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[128:192]"
param l19_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[128:192]"
param l19_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[192:256]"
param l19_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[192:256]"
param l19_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[256:320]"
param l19_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[256:320]"
param l19_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[320:384]"
param l19_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[320:384]"
param l19_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[384:448]"
param l19_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[384:448]"
param l19_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[448:512]"
param l19_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[448:512]"
param l19_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[512:576]"
param l19_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[512:576]"
param l19_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[576:640]"
param l19_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[576:640]"
param l19_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[640:704]"
param l19_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[640:704]"
param l19_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[704:768]"
param l19_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[704:768]"
param l19_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[768:832]"
param l19_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[768:832]"
param l19_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[832:896]"
param l19_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[832:896]"
param l19_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[896:960]"
param l19_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[896:960]"
param l19_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.k_proj.weight[960:1024]"
param l19_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.self_attn.v_proj.weight[960:1024]"
param l19_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.mlp.gate_proj.weight"
param l19_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.mlp.up_proj.weight"
param l19_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.19.mlp.down_proj.weight"

param l20_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.input_layernorm.weight"
param l20_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.post_attention_layernorm.weight"
param l20_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_norm.weight[0:64]"
param l20_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_norm.weight[64:128]"
param l20_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_norm.weight[0:64]"
param l20_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_norm.weight[64:128]"
param l20_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[0:64]"
param l20_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,0:64]"
param l20_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[64:128]"
param l20_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,64:128]"
param l20_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[128:192]"
param l20_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,128:192]"
param l20_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[192:256]"
param l20_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,192:256]"
param l20_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[256:320]"
param l20_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,256:320]"
param l20_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[320:384]"
param l20_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,320:384]"
param l20_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[384:448]"
param l20_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,384:448]"
param l20_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[448:512]"
param l20_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,448:512]"
param l20_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[512:576]"
param l20_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,512:576]"
param l20_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[576:640]"
param l20_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,576:640]"
param l20_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[640:704]"
param l20_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,640:704]"
param l20_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[704:768]"
param l20_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,704:768]"
param l20_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[768:832]"
param l20_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,768:832]"
param l20_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[832:896]"
param l20_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,832:896]"
param l20_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[896:960]"
param l20_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,896:960]"
param l20_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[960:1024]"
param l20_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,960:1024]"
param l20_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1024:1088]"
param l20_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1024:1088]"
param l20_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1088:1152]"
param l20_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1088:1152]"
param l20_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1152:1216]"
param l20_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1152:1216]"
param l20_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1216:1280]"
param l20_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1216:1280]"
param l20_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1280:1344]"
param l20_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1280:1344]"
param l20_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1344:1408]"
param l20_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1344:1408]"
param l20_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1408:1472]"
param l20_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1408:1472]"
param l20_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1472:1536]"
param l20_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1472:1536]"
param l20_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1536:1600]"
param l20_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1536:1600]"
param l20_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1600:1664]"
param l20_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1600:1664]"
param l20_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1664:1728]"
param l20_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1664:1728]"
param l20_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1728:1792]"
param l20_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1728:1792]"
param l20_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1792:1856]"
param l20_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1792:1856]"
param l20_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1856:1920]"
param l20_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1856:1920]"
param l20_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1920:1984]"
param l20_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1920:1984]"
param l20_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.q_proj.weight[1984:2048]"
param l20_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.o_proj.weight[0:1024,1984:2048]"
param l20_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[0:64]"
param l20_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[0:64]"
param l20_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[64:128]"
param l20_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[64:128]"
param l20_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[128:192]"
param l20_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[128:192]"
param l20_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[192:256]"
param l20_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[192:256]"
param l20_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[256:320]"
param l20_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[256:320]"
param l20_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[320:384]"
param l20_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[320:384]"
param l20_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[384:448]"
param l20_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[384:448]"
param l20_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[448:512]"
param l20_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[448:512]"
param l20_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[512:576]"
param l20_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[512:576]"
param l20_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[576:640]"
param l20_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[576:640]"
param l20_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[640:704]"
param l20_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[640:704]"
param l20_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[704:768]"
param l20_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[704:768]"
param l20_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[768:832]"
param l20_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[768:832]"
param l20_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[832:896]"
param l20_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[832:896]"
param l20_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[896:960]"
param l20_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[896:960]"
param l20_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.k_proj.weight[960:1024]"
param l20_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.self_attn.v_proj.weight[960:1024]"
param l20_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.mlp.gate_proj.weight"
param l20_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.mlp.up_proj.weight"
param l20_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.20.mlp.down_proj.weight"

param l21_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.input_layernorm.weight"
param l21_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.post_attention_layernorm.weight"
param l21_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_norm.weight[0:64]"
param l21_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_norm.weight[64:128]"
param l21_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_norm.weight[0:64]"
param l21_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_norm.weight[64:128]"
param l21_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[0:64]"
param l21_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,0:64]"
param l21_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[64:128]"
param l21_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,64:128]"
param l21_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[128:192]"
param l21_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,128:192]"
param l21_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[192:256]"
param l21_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,192:256]"
param l21_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[256:320]"
param l21_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,256:320]"
param l21_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[320:384]"
param l21_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,320:384]"
param l21_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[384:448]"
param l21_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,384:448]"
param l21_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[448:512]"
param l21_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,448:512]"
param l21_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[512:576]"
param l21_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,512:576]"
param l21_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[576:640]"
param l21_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,576:640]"
param l21_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[640:704]"
param l21_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,640:704]"
param l21_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[704:768]"
param l21_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,704:768]"
param l21_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[768:832]"
param l21_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,768:832]"
param l21_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[832:896]"
param l21_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,832:896]"
param l21_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[896:960]"
param l21_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,896:960]"
param l21_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[960:1024]"
param l21_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,960:1024]"
param l21_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1024:1088]"
param l21_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1024:1088]"
param l21_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1088:1152]"
param l21_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1088:1152]"
param l21_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1152:1216]"
param l21_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1152:1216]"
param l21_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1216:1280]"
param l21_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1216:1280]"
param l21_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1280:1344]"
param l21_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1280:1344]"
param l21_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1344:1408]"
param l21_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1344:1408]"
param l21_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1408:1472]"
param l21_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1408:1472]"
param l21_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1472:1536]"
param l21_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1472:1536]"
param l21_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1536:1600]"
param l21_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1536:1600]"
param l21_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1600:1664]"
param l21_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1600:1664]"
param l21_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1664:1728]"
param l21_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1664:1728]"
param l21_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1728:1792]"
param l21_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1728:1792]"
param l21_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1792:1856]"
param l21_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1792:1856]"
param l21_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1856:1920]"
param l21_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1856:1920]"
param l21_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1920:1984]"
param l21_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1920:1984]"
param l21_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.q_proj.weight[1984:2048]"
param l21_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.o_proj.weight[0:1024,1984:2048]"
param l21_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[0:64]"
param l21_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[0:64]"
param l21_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[64:128]"
param l21_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[64:128]"
param l21_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[128:192]"
param l21_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[128:192]"
param l21_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[192:256]"
param l21_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[192:256]"
param l21_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[256:320]"
param l21_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[256:320]"
param l21_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[320:384]"
param l21_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[320:384]"
param l21_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[384:448]"
param l21_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[384:448]"
param l21_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[448:512]"
param l21_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[448:512]"
param l21_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[512:576]"
param l21_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[512:576]"
param l21_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[576:640]"
param l21_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[576:640]"
param l21_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[640:704]"
param l21_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[640:704]"
param l21_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[704:768]"
param l21_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[704:768]"
param l21_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[768:832]"
param l21_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[768:832]"
param l21_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[832:896]"
param l21_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[832:896]"
param l21_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[896:960]"
param l21_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[896:960]"
param l21_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.k_proj.weight[960:1024]"
param l21_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.self_attn.v_proj.weight[960:1024]"
param l21_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.mlp.gate_proj.weight"
param l21_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.mlp.up_proj.weight"
param l21_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.21.mlp.down_proj.weight"

param l22_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.input_layernorm.weight"
param l22_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.post_attention_layernorm.weight"
param l22_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_norm.weight[0:64]"
param l22_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_norm.weight[64:128]"
param l22_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_norm.weight[0:64]"
param l22_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_norm.weight[64:128]"
param l22_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[0:64]"
param l22_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,0:64]"
param l22_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[64:128]"
param l22_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,64:128]"
param l22_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[128:192]"
param l22_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,128:192]"
param l22_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[192:256]"
param l22_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,192:256]"
param l22_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[256:320]"
param l22_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,256:320]"
param l22_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[320:384]"
param l22_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,320:384]"
param l22_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[384:448]"
param l22_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,384:448]"
param l22_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[448:512]"
param l22_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,448:512]"
param l22_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[512:576]"
param l22_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,512:576]"
param l22_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[576:640]"
param l22_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,576:640]"
param l22_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[640:704]"
param l22_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,640:704]"
param l22_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[704:768]"
param l22_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,704:768]"
param l22_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[768:832]"
param l22_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,768:832]"
param l22_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[832:896]"
param l22_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,832:896]"
param l22_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[896:960]"
param l22_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,896:960]"
param l22_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[960:1024]"
param l22_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,960:1024]"
param l22_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1024:1088]"
param l22_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1024:1088]"
param l22_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1088:1152]"
param l22_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1088:1152]"
param l22_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1152:1216]"
param l22_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1152:1216]"
param l22_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1216:1280]"
param l22_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1216:1280]"
param l22_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1280:1344]"
param l22_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1280:1344]"
param l22_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1344:1408]"
param l22_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1344:1408]"
param l22_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1408:1472]"
param l22_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1408:1472]"
param l22_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1472:1536]"
param l22_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1472:1536]"
param l22_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1536:1600]"
param l22_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1536:1600]"
param l22_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1600:1664]"
param l22_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1600:1664]"
param l22_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1664:1728]"
param l22_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1664:1728]"
param l22_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1728:1792]"
param l22_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1728:1792]"
param l22_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1792:1856]"
param l22_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1792:1856]"
param l22_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1856:1920]"
param l22_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1856:1920]"
param l22_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1920:1984]"
param l22_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1920:1984]"
param l22_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.q_proj.weight[1984:2048]"
param l22_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.o_proj.weight[0:1024,1984:2048]"
param l22_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[0:64]"
param l22_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[0:64]"
param l22_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[64:128]"
param l22_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[64:128]"
param l22_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[128:192]"
param l22_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[128:192]"
param l22_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[192:256]"
param l22_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[192:256]"
param l22_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[256:320]"
param l22_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[256:320]"
param l22_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[320:384]"
param l22_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[320:384]"
param l22_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[384:448]"
param l22_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[384:448]"
param l22_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[448:512]"
param l22_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[448:512]"
param l22_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[512:576]"
param l22_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[512:576]"
param l22_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[576:640]"
param l22_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[576:640]"
param l22_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[640:704]"
param l22_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[640:704]"
param l22_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[704:768]"
param l22_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[704:768]"
param l22_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[768:832]"
param l22_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[768:832]"
param l22_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[832:896]"
param l22_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[832:896]"
param l22_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[896:960]"
param l22_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[896:960]"
param l22_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.k_proj.weight[960:1024]"
param l22_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.self_attn.v_proj.weight[960:1024]"
param l22_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.mlp.gate_proj.weight"
param l22_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.mlp.up_proj.weight"
param l22_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.22.mlp.down_proj.weight"

param l23_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.input_layernorm.weight"
param l23_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.post_attention_layernorm.weight"
param l23_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_norm.weight[0:64]"
param l23_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_norm.weight[64:128]"
param l23_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_norm.weight[0:64]"
param l23_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_norm.weight[64:128]"
param l23_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[0:64]"
param l23_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,0:64]"
param l23_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[64:128]"
param l23_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,64:128]"
param l23_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[128:192]"
param l23_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,128:192]"
param l23_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[192:256]"
param l23_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,192:256]"
param l23_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[256:320]"
param l23_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,256:320]"
param l23_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[320:384]"
param l23_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,320:384]"
param l23_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[384:448]"
param l23_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,384:448]"
param l23_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[448:512]"
param l23_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,448:512]"
param l23_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[512:576]"
param l23_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,512:576]"
param l23_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[576:640]"
param l23_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,576:640]"
param l23_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[640:704]"
param l23_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,640:704]"
param l23_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[704:768]"
param l23_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,704:768]"
param l23_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[768:832]"
param l23_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,768:832]"
param l23_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[832:896]"
param l23_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,832:896]"
param l23_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[896:960]"
param l23_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,896:960]"
param l23_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[960:1024]"
param l23_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,960:1024]"
param l23_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1024:1088]"
param l23_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1024:1088]"
param l23_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1088:1152]"
param l23_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1088:1152]"
param l23_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1152:1216]"
param l23_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1152:1216]"
param l23_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1216:1280]"
param l23_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1216:1280]"
param l23_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1280:1344]"
param l23_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1280:1344]"
param l23_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1344:1408]"
param l23_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1344:1408]"
param l23_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1408:1472]"
param l23_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1408:1472]"
param l23_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1472:1536]"
param l23_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1472:1536]"
param l23_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1536:1600]"
param l23_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1536:1600]"
param l23_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1600:1664]"
param l23_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1600:1664]"
param l23_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1664:1728]"
param l23_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1664:1728]"
param l23_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1728:1792]"
param l23_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1728:1792]"
param l23_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1792:1856]"
param l23_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1792:1856]"
param l23_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1856:1920]"
param l23_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1856:1920]"
param l23_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1920:1984]"
param l23_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1920:1984]"
param l23_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.q_proj.weight[1984:2048]"
param l23_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.o_proj.weight[0:1024,1984:2048]"
param l23_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[0:64]"
param l23_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[0:64]"
param l23_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[64:128]"
param l23_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[64:128]"
param l23_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[128:192]"
param l23_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[128:192]"
param l23_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[192:256]"
param l23_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[192:256]"
param l23_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[256:320]"
param l23_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[256:320]"
param l23_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[320:384]"
param l23_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[320:384]"
param l23_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[384:448]"
param l23_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[384:448]"
param l23_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[448:512]"
param l23_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[448:512]"
param l23_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[512:576]"
param l23_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[512:576]"
param l23_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[576:640]"
param l23_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[576:640]"
param l23_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[640:704]"
param l23_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[640:704]"
param l23_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[704:768]"
param l23_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[704:768]"
param l23_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[768:832]"
param l23_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[768:832]"
param l23_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[832:896]"
param l23_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[832:896]"
param l23_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[896:960]"
param l23_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[896:960]"
param l23_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.k_proj.weight[960:1024]"
param l23_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.self_attn.v_proj.weight[960:1024]"
param l23_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.mlp.gate_proj.weight"
param l23_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.mlp.up_proj.weight"
param l23_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.23.mlp.down_proj.weight"

param l24_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.input_layernorm.weight"
param l24_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.post_attention_layernorm.weight"
param l24_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_norm.weight[0:64]"
param l24_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_norm.weight[64:128]"
param l24_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_norm.weight[0:64]"
param l24_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_norm.weight[64:128]"
param l24_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[0:64]"
param l24_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,0:64]"
param l24_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[64:128]"
param l24_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,64:128]"
param l24_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[128:192]"
param l24_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,128:192]"
param l24_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[192:256]"
param l24_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,192:256]"
param l24_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[256:320]"
param l24_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,256:320]"
param l24_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[320:384]"
param l24_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,320:384]"
param l24_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[384:448]"
param l24_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,384:448]"
param l24_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[448:512]"
param l24_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,448:512]"
param l24_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[512:576]"
param l24_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,512:576]"
param l24_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[576:640]"
param l24_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,576:640]"
param l24_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[640:704]"
param l24_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,640:704]"
param l24_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[704:768]"
param l24_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,704:768]"
param l24_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[768:832]"
param l24_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,768:832]"
param l24_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[832:896]"
param l24_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,832:896]"
param l24_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[896:960]"
param l24_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,896:960]"
param l24_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[960:1024]"
param l24_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,960:1024]"
param l24_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1024:1088]"
param l24_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1024:1088]"
param l24_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1088:1152]"
param l24_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1088:1152]"
param l24_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1152:1216]"
param l24_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1152:1216]"
param l24_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1216:1280]"
param l24_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1216:1280]"
param l24_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1280:1344]"
param l24_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1280:1344]"
param l24_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1344:1408]"
param l24_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1344:1408]"
param l24_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1408:1472]"
param l24_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1408:1472]"
param l24_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1472:1536]"
param l24_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1472:1536]"
param l24_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1536:1600]"
param l24_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1536:1600]"
param l24_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1600:1664]"
param l24_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1600:1664]"
param l24_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1664:1728]"
param l24_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1664:1728]"
param l24_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1728:1792]"
param l24_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1728:1792]"
param l24_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1792:1856]"
param l24_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1792:1856]"
param l24_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1856:1920]"
param l24_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1856:1920]"
param l24_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1920:1984]"
param l24_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1920:1984]"
param l24_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.q_proj.weight[1984:2048]"
param l24_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.o_proj.weight[0:1024,1984:2048]"
param l24_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[0:64]"
param l24_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[0:64]"
param l24_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[64:128]"
param l24_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[64:128]"
param l24_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[128:192]"
param l24_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[128:192]"
param l24_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[192:256]"
param l24_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[192:256]"
param l24_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[256:320]"
param l24_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[256:320]"
param l24_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[320:384]"
param l24_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[320:384]"
param l24_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[384:448]"
param l24_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[384:448]"
param l24_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[448:512]"
param l24_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[448:512]"
param l24_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[512:576]"
param l24_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[512:576]"
param l24_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[576:640]"
param l24_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[576:640]"
param l24_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[640:704]"
param l24_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[640:704]"
param l24_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[704:768]"
param l24_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[704:768]"
param l24_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[768:832]"
param l24_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[768:832]"
param l24_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[832:896]"
param l24_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[832:896]"
param l24_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[896:960]"
param l24_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[896:960]"
param l24_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.k_proj.weight[960:1024]"
param l24_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.self_attn.v_proj.weight[960:1024]"
param l24_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.mlp.gate_proj.weight"
param l24_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.mlp.up_proj.weight"
param l24_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.24.mlp.down_proj.weight"

param l25_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.input_layernorm.weight"
param l25_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.post_attention_layernorm.weight"
param l25_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_norm.weight[0:64]"
param l25_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_norm.weight[64:128]"
param l25_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_norm.weight[0:64]"
param l25_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_norm.weight[64:128]"
param l25_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[0:64]"
param l25_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,0:64]"
param l25_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[64:128]"
param l25_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,64:128]"
param l25_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[128:192]"
param l25_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,128:192]"
param l25_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[192:256]"
param l25_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,192:256]"
param l25_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[256:320]"
param l25_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,256:320]"
param l25_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[320:384]"
param l25_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,320:384]"
param l25_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[384:448]"
param l25_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,384:448]"
param l25_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[448:512]"
param l25_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,448:512]"
param l25_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[512:576]"
param l25_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,512:576]"
param l25_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[576:640]"
param l25_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,576:640]"
param l25_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[640:704]"
param l25_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,640:704]"
param l25_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[704:768]"
param l25_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,704:768]"
param l25_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[768:832]"
param l25_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,768:832]"
param l25_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[832:896]"
param l25_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,832:896]"
param l25_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[896:960]"
param l25_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,896:960]"
param l25_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[960:1024]"
param l25_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,960:1024]"
param l25_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1024:1088]"
param l25_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1024:1088]"
param l25_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1088:1152]"
param l25_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1088:1152]"
param l25_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1152:1216]"
param l25_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1152:1216]"
param l25_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1216:1280]"
param l25_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1216:1280]"
param l25_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1280:1344]"
param l25_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1280:1344]"
param l25_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1344:1408]"
param l25_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1344:1408]"
param l25_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1408:1472]"
param l25_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1408:1472]"
param l25_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1472:1536]"
param l25_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1472:1536]"
param l25_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1536:1600]"
param l25_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1536:1600]"
param l25_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1600:1664]"
param l25_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1600:1664]"
param l25_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1664:1728]"
param l25_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1664:1728]"
param l25_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1728:1792]"
param l25_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1728:1792]"
param l25_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1792:1856]"
param l25_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1792:1856]"
param l25_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1856:1920]"
param l25_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1856:1920]"
param l25_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1920:1984]"
param l25_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1920:1984]"
param l25_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.q_proj.weight[1984:2048]"
param l25_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.o_proj.weight[0:1024,1984:2048]"
param l25_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[0:64]"
param l25_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[0:64]"
param l25_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[64:128]"
param l25_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[64:128]"
param l25_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[128:192]"
param l25_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[128:192]"
param l25_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[192:256]"
param l25_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[192:256]"
param l25_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[256:320]"
param l25_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[256:320]"
param l25_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[320:384]"
param l25_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[320:384]"
param l25_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[384:448]"
param l25_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[384:448]"
param l25_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[448:512]"
param l25_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[448:512]"
param l25_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[512:576]"
param l25_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[512:576]"
param l25_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[576:640]"
param l25_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[576:640]"
param l25_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[640:704]"
param l25_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[640:704]"
param l25_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[704:768]"
param l25_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[704:768]"
param l25_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[768:832]"
param l25_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[768:832]"
param l25_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[832:896]"
param l25_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[832:896]"
param l25_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[896:960]"
param l25_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[896:960]"
param l25_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.k_proj.weight[960:1024]"
param l25_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.self_attn.v_proj.weight[960:1024]"
param l25_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.mlp.gate_proj.weight"
param l25_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.mlp.up_proj.weight"
param l25_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.25.mlp.down_proj.weight"

param l26_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.input_layernorm.weight"
param l26_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.post_attention_layernorm.weight"
param l26_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_norm.weight[0:64]"
param l26_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_norm.weight[64:128]"
param l26_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_norm.weight[0:64]"
param l26_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_norm.weight[64:128]"
param l26_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[0:64]"
param l26_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,0:64]"
param l26_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[64:128]"
param l26_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,64:128]"
param l26_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[128:192]"
param l26_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,128:192]"
param l26_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[192:256]"
param l26_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,192:256]"
param l26_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[256:320]"
param l26_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,256:320]"
param l26_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[320:384]"
param l26_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,320:384]"
param l26_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[384:448]"
param l26_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,384:448]"
param l26_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[448:512]"
param l26_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,448:512]"
param l26_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[512:576]"
param l26_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,512:576]"
param l26_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[576:640]"
param l26_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,576:640]"
param l26_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[640:704]"
param l26_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,640:704]"
param l26_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[704:768]"
param l26_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,704:768]"
param l26_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[768:832]"
param l26_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,768:832]"
param l26_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[832:896]"
param l26_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,832:896]"
param l26_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[896:960]"
param l26_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,896:960]"
param l26_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[960:1024]"
param l26_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,960:1024]"
param l26_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1024:1088]"
param l26_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1024:1088]"
param l26_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1088:1152]"
param l26_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1088:1152]"
param l26_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1152:1216]"
param l26_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1152:1216]"
param l26_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1216:1280]"
param l26_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1216:1280]"
param l26_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1280:1344]"
param l26_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1280:1344]"
param l26_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1344:1408]"
param l26_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1344:1408]"
param l26_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1408:1472]"
param l26_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1408:1472]"
param l26_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1472:1536]"
param l26_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1472:1536]"
param l26_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1536:1600]"
param l26_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1536:1600]"
param l26_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1600:1664]"
param l26_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1600:1664]"
param l26_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1664:1728]"
param l26_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1664:1728]"
param l26_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1728:1792]"
param l26_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1728:1792]"
param l26_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1792:1856]"
param l26_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1792:1856]"
param l26_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1856:1920]"
param l26_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1856:1920]"
param l26_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1920:1984]"
param l26_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1920:1984]"
param l26_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.q_proj.weight[1984:2048]"
param l26_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.o_proj.weight[0:1024,1984:2048]"
param l26_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[0:64]"
param l26_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[0:64]"
param l26_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[64:128]"
param l26_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[64:128]"
param l26_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[128:192]"
param l26_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[128:192]"
param l26_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[192:256]"
param l26_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[192:256]"
param l26_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[256:320]"
param l26_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[256:320]"
param l26_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[320:384]"
param l26_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[320:384]"
param l26_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[384:448]"
param l26_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[384:448]"
param l26_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[448:512]"
param l26_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[448:512]"
param l26_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[512:576]"
param l26_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[512:576]"
param l26_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[576:640]"
param l26_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[576:640]"
param l26_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[640:704]"
param l26_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[640:704]"
param l26_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[704:768]"
param l26_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[704:768]"
param l26_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[768:832]"
param l26_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[768:832]"
param l26_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[832:896]"
param l26_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[832:896]"
param l26_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[896:960]"
param l26_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[896:960]"
param l26_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.k_proj.weight[960:1024]"
param l26_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.self_attn.v_proj.weight[960:1024]"
param l26_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.mlp.gate_proj.weight"
param l26_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.mlp.up_proj.weight"
param l26_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.26.mlp.down_proj.weight"

param l27_ln1: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.input_layernorm.weight"
param l27_ln2: frozen f16[1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.post_attention_layernorm.weight"
param l27_qnl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_norm.weight[0:64]"
param l27_qnh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_norm.weight[64:128]"
param l27_knl: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_norm.weight[0:64]"
param l27_knh: frozen f16[64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_norm.weight[64:128]"
param l27_q0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[0:64]"
param l27_o0lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,0:64]"
param l27_q0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[64:128]"
param l27_o0hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,64:128]"
param l27_q1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[128:192]"
param l27_o1lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,128:192]"
param l27_q1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[192:256]"
param l27_o1hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,192:256]"
param l27_q2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[256:320]"
param l27_o2lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,256:320]"
param l27_q2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[320:384]"
param l27_o2hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,320:384]"
param l27_q3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[384:448]"
param l27_o3lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,384:448]"
param l27_q3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[448:512]"
param l27_o3hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,448:512]"
param l27_q4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[512:576]"
param l27_o4lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,512:576]"
param l27_q4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[576:640]"
param l27_o4hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,576:640]"
param l27_q5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[640:704]"
param l27_o5lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,640:704]"
param l27_q5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[704:768]"
param l27_o5hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,704:768]"
param l27_q6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[768:832]"
param l27_o6lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,768:832]"
param l27_q6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[832:896]"
param l27_o6hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,832:896]"
param l27_q7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[896:960]"
param l27_o7lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,896:960]"
param l27_q7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[960:1024]"
param l27_o7hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,960:1024]"
param l27_q8lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1024:1088]"
param l27_o8lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1024:1088]"
param l27_q8hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1088:1152]"
param l27_o8hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1088:1152]"
param l27_q9lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1152:1216]"
param l27_o9lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1152:1216]"
param l27_q9hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1216:1280]"
param l27_o9hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1216:1280]"
param l27_q10lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1280:1344]"
param l27_o10lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1280:1344]"
param l27_q10hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1344:1408]"
param l27_o10hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1344:1408]"
param l27_q11lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1408:1472]"
param l27_o11lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1408:1472]"
param l27_q11hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1472:1536]"
param l27_o11hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1472:1536]"
param l27_q12lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1536:1600]"
param l27_o12lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1536:1600]"
param l27_q12hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1600:1664]"
param l27_o12hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1600:1664]"
param l27_q13lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1664:1728]"
param l27_o13lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1664:1728]"
param l27_q13hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1728:1792]"
param l27_o13hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1728:1792]"
param l27_q14lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1792:1856]"
param l27_o14lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1792:1856]"
param l27_q14hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1856:1920]"
param l27_o14hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1856:1920]"
param l27_q15lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1920:1984]"
param l27_o15lo: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1920:1984]"
param l27_q15hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.q_proj.weight[1984:2048]"
param l27_o15hi: frozen f16[64, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.o_proj.weight[0:1024,1984:2048]"
param l27_k0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[0:64]"
param l27_v0lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[0:64]"
param l27_k0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[64:128]"
param l27_v0hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[64:128]"
param l27_k1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[128:192]"
param l27_v1lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[128:192]"
param l27_k1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[192:256]"
param l27_v1hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[192:256]"
param l27_k2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[256:320]"
param l27_v2lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[256:320]"
param l27_k2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[320:384]"
param l27_v2hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[320:384]"
param l27_k3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[384:448]"
param l27_v3lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[384:448]"
param l27_k3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[448:512]"
param l27_v3hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[448:512]"
param l27_k4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[512:576]"
param l27_v4lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[512:576]"
param l27_k4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[576:640]"
param l27_v4hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[576:640]"
param l27_k5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[640:704]"
param l27_v5lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[640:704]"
param l27_k5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[704:768]"
param l27_v5hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[704:768]"
param l27_k6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[768:832]"
param l27_v6lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[768:832]"
param l27_k6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[832:896]"
param l27_v6hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[832:896]"
param l27_k7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[896:960]"
param l27_v7lo: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[896:960]"
param l27_k7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.k_proj.weight[960:1024]"
param l27_v7hi: frozen f16[1024, 64] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.self_attn.v_proj.weight[960:1024]"
param l27_wg: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.mlp.gate_proj.weight"
param l27_wu: frozen f16[1024, 3072] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.mlp.up_proj.weight"
param l27_wd: frozen f16[3072, 1024] @ "models/qwen3-0.6b/model.safetensors::model.layers.27.mlp.down_proj.weight"

fn l0(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l0_ln1);
    let q0l = rmsnorm(h @ l0_q0lo, l0_qnl);
    let q0h = rmsnorm(h @ l0_q0hi, l0_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l0_q1lo, l0_qnl);
    let q1h = rmsnorm(h @ l0_q1hi, l0_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l0_q2lo, l0_qnl);
    let q2h = rmsnorm(h @ l0_q2hi, l0_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l0_q3lo, l0_qnl);
    let q3h = rmsnorm(h @ l0_q3hi, l0_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l0_q4lo, l0_qnl);
    let q4h = rmsnorm(h @ l0_q4hi, l0_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l0_q5lo, l0_qnl);
    let q5h = rmsnorm(h @ l0_q5hi, l0_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l0_q6lo, l0_qnl);
    let q6h = rmsnorm(h @ l0_q6hi, l0_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l0_q7lo, l0_qnl);
    let q7h = rmsnorm(h @ l0_q7hi, l0_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l0_q8lo, l0_qnl);
    let q8h = rmsnorm(h @ l0_q8hi, l0_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l0_q9lo, l0_qnl);
    let q9h = rmsnorm(h @ l0_q9hi, l0_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l0_q10lo, l0_qnl);
    let q10h = rmsnorm(h @ l0_q10hi, l0_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l0_q11lo, l0_qnl);
    let q11h = rmsnorm(h @ l0_q11hi, l0_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l0_q12lo, l0_qnl);
    let q12h = rmsnorm(h @ l0_q12hi, l0_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l0_q13lo, l0_qnl);
    let q13h = rmsnorm(h @ l0_q13hi, l0_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l0_q14lo, l0_qnl);
    let q14h = rmsnorm(h @ l0_q14hi, l0_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l0_q15lo, l0_qnl);
    let q15h = rmsnorm(h @ l0_q15hi, l0_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l0_k0lo, l0_knl);
    let k0h = rmsnorm(h @ l0_k0hi, l0_knh);
    let v0l = h @ l0_v0lo;
    let v0h = h @ l0_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l0_k1lo, l0_knl);
    let k1h = rmsnorm(h @ l0_k1hi, l0_knh);
    let v1l = h @ l0_v1lo;
    let v1h = h @ l0_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l0_k2lo, l0_knl);
    let k2h = rmsnorm(h @ l0_k2hi, l0_knh);
    let v2l = h @ l0_v2lo;
    let v2h = h @ l0_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l0_k3lo, l0_knl);
    let k3h = rmsnorm(h @ l0_k3hi, l0_knh);
    let v3l = h @ l0_v3lo;
    let v3h = h @ l0_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l0_k4lo, l0_knl);
    let k4h = rmsnorm(h @ l0_k4hi, l0_knh);
    let v4l = h @ l0_v4lo;
    let v4h = h @ l0_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l0_k5lo, l0_knl);
    let k5h = rmsnorm(h @ l0_k5hi, l0_knh);
    let v5l = h @ l0_v5lo;
    let v5h = h @ l0_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l0_k6lo, l0_knl);
    let k6h = rmsnorm(h @ l0_k6hi, l0_knh);
    let v6l = h @ l0_v6lo;
    let v6h = h @ l0_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l0_k7lo, l0_knl);
    let k7h = rmsnorm(h @ l0_k7hi, l0_knh);
    let v7l = h @ l0_v7lo;
    let v7h = h @ l0_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l0_o0lo + c0h @ l0_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l0_o1lo + c1h @ l0_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l0_o2lo + c2h @ l0_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l0_o3lo + c3h @ l0_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l0_o4lo + c4h @ l0_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l0_o5lo + c5h @ l0_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l0_o6lo + c6h @ l0_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l0_o7lo + c7h @ l0_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l0_o8lo + c8h @ l0_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l0_o9lo + c9h @ l0_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l0_o10lo + c10h @ l0_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l0_o11lo + c11h @ l0_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l0_o12lo + c12h @ l0_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l0_o13lo + c13h @ l0_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l0_o14lo + c14h @ l0_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l0_o15lo + c15h @ l0_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l0_ln2);
    let f = r + (silu(n @ l0_wg) * (n @ l0_wu)) @ l0_wd;
    f
}

fn l1(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l1_ln1);
    let q0l = rmsnorm(h @ l1_q0lo, l1_qnl);
    let q0h = rmsnorm(h @ l1_q0hi, l1_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l1_q1lo, l1_qnl);
    let q1h = rmsnorm(h @ l1_q1hi, l1_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l1_q2lo, l1_qnl);
    let q2h = rmsnorm(h @ l1_q2hi, l1_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l1_q3lo, l1_qnl);
    let q3h = rmsnorm(h @ l1_q3hi, l1_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l1_q4lo, l1_qnl);
    let q4h = rmsnorm(h @ l1_q4hi, l1_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l1_q5lo, l1_qnl);
    let q5h = rmsnorm(h @ l1_q5hi, l1_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l1_q6lo, l1_qnl);
    let q6h = rmsnorm(h @ l1_q6hi, l1_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l1_q7lo, l1_qnl);
    let q7h = rmsnorm(h @ l1_q7hi, l1_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l1_q8lo, l1_qnl);
    let q8h = rmsnorm(h @ l1_q8hi, l1_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l1_q9lo, l1_qnl);
    let q9h = rmsnorm(h @ l1_q9hi, l1_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l1_q10lo, l1_qnl);
    let q10h = rmsnorm(h @ l1_q10hi, l1_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l1_q11lo, l1_qnl);
    let q11h = rmsnorm(h @ l1_q11hi, l1_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l1_q12lo, l1_qnl);
    let q12h = rmsnorm(h @ l1_q12hi, l1_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l1_q13lo, l1_qnl);
    let q13h = rmsnorm(h @ l1_q13hi, l1_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l1_q14lo, l1_qnl);
    let q14h = rmsnorm(h @ l1_q14hi, l1_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l1_q15lo, l1_qnl);
    let q15h = rmsnorm(h @ l1_q15hi, l1_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l1_k0lo, l1_knl);
    let k0h = rmsnorm(h @ l1_k0hi, l1_knh);
    let v0l = h @ l1_v0lo;
    let v0h = h @ l1_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l1_k1lo, l1_knl);
    let k1h = rmsnorm(h @ l1_k1hi, l1_knh);
    let v1l = h @ l1_v1lo;
    let v1h = h @ l1_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l1_k2lo, l1_knl);
    let k2h = rmsnorm(h @ l1_k2hi, l1_knh);
    let v2l = h @ l1_v2lo;
    let v2h = h @ l1_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l1_k3lo, l1_knl);
    let k3h = rmsnorm(h @ l1_k3hi, l1_knh);
    let v3l = h @ l1_v3lo;
    let v3h = h @ l1_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l1_k4lo, l1_knl);
    let k4h = rmsnorm(h @ l1_k4hi, l1_knh);
    let v4l = h @ l1_v4lo;
    let v4h = h @ l1_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l1_k5lo, l1_knl);
    let k5h = rmsnorm(h @ l1_k5hi, l1_knh);
    let v5l = h @ l1_v5lo;
    let v5h = h @ l1_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l1_k6lo, l1_knl);
    let k6h = rmsnorm(h @ l1_k6hi, l1_knh);
    let v6l = h @ l1_v6lo;
    let v6h = h @ l1_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l1_k7lo, l1_knl);
    let k7h = rmsnorm(h @ l1_k7hi, l1_knh);
    let v7l = h @ l1_v7lo;
    let v7h = h @ l1_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l1_o0lo + c0h @ l1_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l1_o1lo + c1h @ l1_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l1_o2lo + c2h @ l1_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l1_o3lo + c3h @ l1_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l1_o4lo + c4h @ l1_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l1_o5lo + c5h @ l1_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l1_o6lo + c6h @ l1_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l1_o7lo + c7h @ l1_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l1_o8lo + c8h @ l1_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l1_o9lo + c9h @ l1_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l1_o10lo + c10h @ l1_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l1_o11lo + c11h @ l1_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l1_o12lo + c12h @ l1_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l1_o13lo + c13h @ l1_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l1_o14lo + c14h @ l1_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l1_o15lo + c15h @ l1_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l1_ln2);
    let f = r + (silu(n @ l1_wg) * (n @ l1_wu)) @ l1_wd;
    f
}

fn l2(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l2_ln1);
    let q0l = rmsnorm(h @ l2_q0lo, l2_qnl);
    let q0h = rmsnorm(h @ l2_q0hi, l2_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l2_q1lo, l2_qnl);
    let q1h = rmsnorm(h @ l2_q1hi, l2_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l2_q2lo, l2_qnl);
    let q2h = rmsnorm(h @ l2_q2hi, l2_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l2_q3lo, l2_qnl);
    let q3h = rmsnorm(h @ l2_q3hi, l2_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l2_q4lo, l2_qnl);
    let q4h = rmsnorm(h @ l2_q4hi, l2_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l2_q5lo, l2_qnl);
    let q5h = rmsnorm(h @ l2_q5hi, l2_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l2_q6lo, l2_qnl);
    let q6h = rmsnorm(h @ l2_q6hi, l2_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l2_q7lo, l2_qnl);
    let q7h = rmsnorm(h @ l2_q7hi, l2_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l2_q8lo, l2_qnl);
    let q8h = rmsnorm(h @ l2_q8hi, l2_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l2_q9lo, l2_qnl);
    let q9h = rmsnorm(h @ l2_q9hi, l2_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l2_q10lo, l2_qnl);
    let q10h = rmsnorm(h @ l2_q10hi, l2_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l2_q11lo, l2_qnl);
    let q11h = rmsnorm(h @ l2_q11hi, l2_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l2_q12lo, l2_qnl);
    let q12h = rmsnorm(h @ l2_q12hi, l2_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l2_q13lo, l2_qnl);
    let q13h = rmsnorm(h @ l2_q13hi, l2_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l2_q14lo, l2_qnl);
    let q14h = rmsnorm(h @ l2_q14hi, l2_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l2_q15lo, l2_qnl);
    let q15h = rmsnorm(h @ l2_q15hi, l2_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l2_k0lo, l2_knl);
    let k0h = rmsnorm(h @ l2_k0hi, l2_knh);
    let v0l = h @ l2_v0lo;
    let v0h = h @ l2_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l2_k1lo, l2_knl);
    let k1h = rmsnorm(h @ l2_k1hi, l2_knh);
    let v1l = h @ l2_v1lo;
    let v1h = h @ l2_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l2_k2lo, l2_knl);
    let k2h = rmsnorm(h @ l2_k2hi, l2_knh);
    let v2l = h @ l2_v2lo;
    let v2h = h @ l2_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l2_k3lo, l2_knl);
    let k3h = rmsnorm(h @ l2_k3hi, l2_knh);
    let v3l = h @ l2_v3lo;
    let v3h = h @ l2_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l2_k4lo, l2_knl);
    let k4h = rmsnorm(h @ l2_k4hi, l2_knh);
    let v4l = h @ l2_v4lo;
    let v4h = h @ l2_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l2_k5lo, l2_knl);
    let k5h = rmsnorm(h @ l2_k5hi, l2_knh);
    let v5l = h @ l2_v5lo;
    let v5h = h @ l2_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l2_k6lo, l2_knl);
    let k6h = rmsnorm(h @ l2_k6hi, l2_knh);
    let v6l = h @ l2_v6lo;
    let v6h = h @ l2_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l2_k7lo, l2_knl);
    let k7h = rmsnorm(h @ l2_k7hi, l2_knh);
    let v7l = h @ l2_v7lo;
    let v7h = h @ l2_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l2_o0lo + c0h @ l2_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l2_o1lo + c1h @ l2_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l2_o2lo + c2h @ l2_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l2_o3lo + c3h @ l2_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l2_o4lo + c4h @ l2_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l2_o5lo + c5h @ l2_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l2_o6lo + c6h @ l2_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l2_o7lo + c7h @ l2_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l2_o8lo + c8h @ l2_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l2_o9lo + c9h @ l2_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l2_o10lo + c10h @ l2_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l2_o11lo + c11h @ l2_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l2_o12lo + c12h @ l2_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l2_o13lo + c13h @ l2_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l2_o14lo + c14h @ l2_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l2_o15lo + c15h @ l2_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l2_ln2);
    let f = r + (silu(n @ l2_wg) * (n @ l2_wu)) @ l2_wd;
    f
}

fn l3(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l3_ln1);
    let q0l = rmsnorm(h @ l3_q0lo, l3_qnl);
    let q0h = rmsnorm(h @ l3_q0hi, l3_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l3_q1lo, l3_qnl);
    let q1h = rmsnorm(h @ l3_q1hi, l3_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l3_q2lo, l3_qnl);
    let q2h = rmsnorm(h @ l3_q2hi, l3_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l3_q3lo, l3_qnl);
    let q3h = rmsnorm(h @ l3_q3hi, l3_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l3_q4lo, l3_qnl);
    let q4h = rmsnorm(h @ l3_q4hi, l3_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l3_q5lo, l3_qnl);
    let q5h = rmsnorm(h @ l3_q5hi, l3_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l3_q6lo, l3_qnl);
    let q6h = rmsnorm(h @ l3_q6hi, l3_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l3_q7lo, l3_qnl);
    let q7h = rmsnorm(h @ l3_q7hi, l3_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l3_q8lo, l3_qnl);
    let q8h = rmsnorm(h @ l3_q8hi, l3_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l3_q9lo, l3_qnl);
    let q9h = rmsnorm(h @ l3_q9hi, l3_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l3_q10lo, l3_qnl);
    let q10h = rmsnorm(h @ l3_q10hi, l3_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l3_q11lo, l3_qnl);
    let q11h = rmsnorm(h @ l3_q11hi, l3_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l3_q12lo, l3_qnl);
    let q12h = rmsnorm(h @ l3_q12hi, l3_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l3_q13lo, l3_qnl);
    let q13h = rmsnorm(h @ l3_q13hi, l3_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l3_q14lo, l3_qnl);
    let q14h = rmsnorm(h @ l3_q14hi, l3_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l3_q15lo, l3_qnl);
    let q15h = rmsnorm(h @ l3_q15hi, l3_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l3_k0lo, l3_knl);
    let k0h = rmsnorm(h @ l3_k0hi, l3_knh);
    let v0l = h @ l3_v0lo;
    let v0h = h @ l3_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l3_k1lo, l3_knl);
    let k1h = rmsnorm(h @ l3_k1hi, l3_knh);
    let v1l = h @ l3_v1lo;
    let v1h = h @ l3_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l3_k2lo, l3_knl);
    let k2h = rmsnorm(h @ l3_k2hi, l3_knh);
    let v2l = h @ l3_v2lo;
    let v2h = h @ l3_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l3_k3lo, l3_knl);
    let k3h = rmsnorm(h @ l3_k3hi, l3_knh);
    let v3l = h @ l3_v3lo;
    let v3h = h @ l3_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l3_k4lo, l3_knl);
    let k4h = rmsnorm(h @ l3_k4hi, l3_knh);
    let v4l = h @ l3_v4lo;
    let v4h = h @ l3_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l3_k5lo, l3_knl);
    let k5h = rmsnorm(h @ l3_k5hi, l3_knh);
    let v5l = h @ l3_v5lo;
    let v5h = h @ l3_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l3_k6lo, l3_knl);
    let k6h = rmsnorm(h @ l3_k6hi, l3_knh);
    let v6l = h @ l3_v6lo;
    let v6h = h @ l3_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l3_k7lo, l3_knl);
    let k7h = rmsnorm(h @ l3_k7hi, l3_knh);
    let v7l = h @ l3_v7lo;
    let v7h = h @ l3_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l3_o0lo + c0h @ l3_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l3_o1lo + c1h @ l3_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l3_o2lo + c2h @ l3_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l3_o3lo + c3h @ l3_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l3_o4lo + c4h @ l3_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l3_o5lo + c5h @ l3_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l3_o6lo + c6h @ l3_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l3_o7lo + c7h @ l3_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l3_o8lo + c8h @ l3_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l3_o9lo + c9h @ l3_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l3_o10lo + c10h @ l3_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l3_o11lo + c11h @ l3_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l3_o12lo + c12h @ l3_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l3_o13lo + c13h @ l3_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l3_o14lo + c14h @ l3_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l3_o15lo + c15h @ l3_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l3_ln2);
    let f = r + (silu(n @ l3_wg) * (n @ l3_wu)) @ l3_wd;
    f
}

fn l4(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l4_ln1);
    let q0l = rmsnorm(h @ l4_q0lo, l4_qnl);
    let q0h = rmsnorm(h @ l4_q0hi, l4_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l4_q1lo, l4_qnl);
    let q1h = rmsnorm(h @ l4_q1hi, l4_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l4_q2lo, l4_qnl);
    let q2h = rmsnorm(h @ l4_q2hi, l4_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l4_q3lo, l4_qnl);
    let q3h = rmsnorm(h @ l4_q3hi, l4_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l4_q4lo, l4_qnl);
    let q4h = rmsnorm(h @ l4_q4hi, l4_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l4_q5lo, l4_qnl);
    let q5h = rmsnorm(h @ l4_q5hi, l4_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l4_q6lo, l4_qnl);
    let q6h = rmsnorm(h @ l4_q6hi, l4_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l4_q7lo, l4_qnl);
    let q7h = rmsnorm(h @ l4_q7hi, l4_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l4_q8lo, l4_qnl);
    let q8h = rmsnorm(h @ l4_q8hi, l4_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l4_q9lo, l4_qnl);
    let q9h = rmsnorm(h @ l4_q9hi, l4_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l4_q10lo, l4_qnl);
    let q10h = rmsnorm(h @ l4_q10hi, l4_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l4_q11lo, l4_qnl);
    let q11h = rmsnorm(h @ l4_q11hi, l4_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l4_q12lo, l4_qnl);
    let q12h = rmsnorm(h @ l4_q12hi, l4_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l4_q13lo, l4_qnl);
    let q13h = rmsnorm(h @ l4_q13hi, l4_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l4_q14lo, l4_qnl);
    let q14h = rmsnorm(h @ l4_q14hi, l4_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l4_q15lo, l4_qnl);
    let q15h = rmsnorm(h @ l4_q15hi, l4_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l4_k0lo, l4_knl);
    let k0h = rmsnorm(h @ l4_k0hi, l4_knh);
    let v0l = h @ l4_v0lo;
    let v0h = h @ l4_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l4_k1lo, l4_knl);
    let k1h = rmsnorm(h @ l4_k1hi, l4_knh);
    let v1l = h @ l4_v1lo;
    let v1h = h @ l4_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l4_k2lo, l4_knl);
    let k2h = rmsnorm(h @ l4_k2hi, l4_knh);
    let v2l = h @ l4_v2lo;
    let v2h = h @ l4_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l4_k3lo, l4_knl);
    let k3h = rmsnorm(h @ l4_k3hi, l4_knh);
    let v3l = h @ l4_v3lo;
    let v3h = h @ l4_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l4_k4lo, l4_knl);
    let k4h = rmsnorm(h @ l4_k4hi, l4_knh);
    let v4l = h @ l4_v4lo;
    let v4h = h @ l4_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l4_k5lo, l4_knl);
    let k5h = rmsnorm(h @ l4_k5hi, l4_knh);
    let v5l = h @ l4_v5lo;
    let v5h = h @ l4_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l4_k6lo, l4_knl);
    let k6h = rmsnorm(h @ l4_k6hi, l4_knh);
    let v6l = h @ l4_v6lo;
    let v6h = h @ l4_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l4_k7lo, l4_knl);
    let k7h = rmsnorm(h @ l4_k7hi, l4_knh);
    let v7l = h @ l4_v7lo;
    let v7h = h @ l4_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l4_o0lo + c0h @ l4_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l4_o1lo + c1h @ l4_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l4_o2lo + c2h @ l4_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l4_o3lo + c3h @ l4_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l4_o4lo + c4h @ l4_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l4_o5lo + c5h @ l4_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l4_o6lo + c6h @ l4_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l4_o7lo + c7h @ l4_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l4_o8lo + c8h @ l4_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l4_o9lo + c9h @ l4_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l4_o10lo + c10h @ l4_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l4_o11lo + c11h @ l4_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l4_o12lo + c12h @ l4_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l4_o13lo + c13h @ l4_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l4_o14lo + c14h @ l4_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l4_o15lo + c15h @ l4_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l4_ln2);
    let f = r + (silu(n @ l4_wg) * (n @ l4_wu)) @ l4_wd;
    f
}

fn l5(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l5_ln1);
    let q0l = rmsnorm(h @ l5_q0lo, l5_qnl);
    let q0h = rmsnorm(h @ l5_q0hi, l5_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l5_q1lo, l5_qnl);
    let q1h = rmsnorm(h @ l5_q1hi, l5_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l5_q2lo, l5_qnl);
    let q2h = rmsnorm(h @ l5_q2hi, l5_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l5_q3lo, l5_qnl);
    let q3h = rmsnorm(h @ l5_q3hi, l5_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l5_q4lo, l5_qnl);
    let q4h = rmsnorm(h @ l5_q4hi, l5_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l5_q5lo, l5_qnl);
    let q5h = rmsnorm(h @ l5_q5hi, l5_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l5_q6lo, l5_qnl);
    let q6h = rmsnorm(h @ l5_q6hi, l5_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l5_q7lo, l5_qnl);
    let q7h = rmsnorm(h @ l5_q7hi, l5_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l5_q8lo, l5_qnl);
    let q8h = rmsnorm(h @ l5_q8hi, l5_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l5_q9lo, l5_qnl);
    let q9h = rmsnorm(h @ l5_q9hi, l5_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l5_q10lo, l5_qnl);
    let q10h = rmsnorm(h @ l5_q10hi, l5_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l5_q11lo, l5_qnl);
    let q11h = rmsnorm(h @ l5_q11hi, l5_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l5_q12lo, l5_qnl);
    let q12h = rmsnorm(h @ l5_q12hi, l5_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l5_q13lo, l5_qnl);
    let q13h = rmsnorm(h @ l5_q13hi, l5_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l5_q14lo, l5_qnl);
    let q14h = rmsnorm(h @ l5_q14hi, l5_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l5_q15lo, l5_qnl);
    let q15h = rmsnorm(h @ l5_q15hi, l5_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l5_k0lo, l5_knl);
    let k0h = rmsnorm(h @ l5_k0hi, l5_knh);
    let v0l = h @ l5_v0lo;
    let v0h = h @ l5_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l5_k1lo, l5_knl);
    let k1h = rmsnorm(h @ l5_k1hi, l5_knh);
    let v1l = h @ l5_v1lo;
    let v1h = h @ l5_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l5_k2lo, l5_knl);
    let k2h = rmsnorm(h @ l5_k2hi, l5_knh);
    let v2l = h @ l5_v2lo;
    let v2h = h @ l5_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l5_k3lo, l5_knl);
    let k3h = rmsnorm(h @ l5_k3hi, l5_knh);
    let v3l = h @ l5_v3lo;
    let v3h = h @ l5_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l5_k4lo, l5_knl);
    let k4h = rmsnorm(h @ l5_k4hi, l5_knh);
    let v4l = h @ l5_v4lo;
    let v4h = h @ l5_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l5_k5lo, l5_knl);
    let k5h = rmsnorm(h @ l5_k5hi, l5_knh);
    let v5l = h @ l5_v5lo;
    let v5h = h @ l5_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l5_k6lo, l5_knl);
    let k6h = rmsnorm(h @ l5_k6hi, l5_knh);
    let v6l = h @ l5_v6lo;
    let v6h = h @ l5_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l5_k7lo, l5_knl);
    let k7h = rmsnorm(h @ l5_k7hi, l5_knh);
    let v7l = h @ l5_v7lo;
    let v7h = h @ l5_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l5_o0lo + c0h @ l5_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l5_o1lo + c1h @ l5_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l5_o2lo + c2h @ l5_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l5_o3lo + c3h @ l5_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l5_o4lo + c4h @ l5_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l5_o5lo + c5h @ l5_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l5_o6lo + c6h @ l5_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l5_o7lo + c7h @ l5_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l5_o8lo + c8h @ l5_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l5_o9lo + c9h @ l5_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l5_o10lo + c10h @ l5_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l5_o11lo + c11h @ l5_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l5_o12lo + c12h @ l5_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l5_o13lo + c13h @ l5_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l5_o14lo + c14h @ l5_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l5_o15lo + c15h @ l5_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l5_ln2);
    let f = r + (silu(n @ l5_wg) * (n @ l5_wu)) @ l5_wd;
    f
}

fn l6(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l6_ln1);
    let q0l = rmsnorm(h @ l6_q0lo, l6_qnl);
    let q0h = rmsnorm(h @ l6_q0hi, l6_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l6_q1lo, l6_qnl);
    let q1h = rmsnorm(h @ l6_q1hi, l6_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l6_q2lo, l6_qnl);
    let q2h = rmsnorm(h @ l6_q2hi, l6_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l6_q3lo, l6_qnl);
    let q3h = rmsnorm(h @ l6_q3hi, l6_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l6_q4lo, l6_qnl);
    let q4h = rmsnorm(h @ l6_q4hi, l6_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l6_q5lo, l6_qnl);
    let q5h = rmsnorm(h @ l6_q5hi, l6_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l6_q6lo, l6_qnl);
    let q6h = rmsnorm(h @ l6_q6hi, l6_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l6_q7lo, l6_qnl);
    let q7h = rmsnorm(h @ l6_q7hi, l6_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l6_q8lo, l6_qnl);
    let q8h = rmsnorm(h @ l6_q8hi, l6_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l6_q9lo, l6_qnl);
    let q9h = rmsnorm(h @ l6_q9hi, l6_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l6_q10lo, l6_qnl);
    let q10h = rmsnorm(h @ l6_q10hi, l6_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l6_q11lo, l6_qnl);
    let q11h = rmsnorm(h @ l6_q11hi, l6_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l6_q12lo, l6_qnl);
    let q12h = rmsnorm(h @ l6_q12hi, l6_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l6_q13lo, l6_qnl);
    let q13h = rmsnorm(h @ l6_q13hi, l6_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l6_q14lo, l6_qnl);
    let q14h = rmsnorm(h @ l6_q14hi, l6_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l6_q15lo, l6_qnl);
    let q15h = rmsnorm(h @ l6_q15hi, l6_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l6_k0lo, l6_knl);
    let k0h = rmsnorm(h @ l6_k0hi, l6_knh);
    let v0l = h @ l6_v0lo;
    let v0h = h @ l6_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l6_k1lo, l6_knl);
    let k1h = rmsnorm(h @ l6_k1hi, l6_knh);
    let v1l = h @ l6_v1lo;
    let v1h = h @ l6_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l6_k2lo, l6_knl);
    let k2h = rmsnorm(h @ l6_k2hi, l6_knh);
    let v2l = h @ l6_v2lo;
    let v2h = h @ l6_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l6_k3lo, l6_knl);
    let k3h = rmsnorm(h @ l6_k3hi, l6_knh);
    let v3l = h @ l6_v3lo;
    let v3h = h @ l6_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l6_k4lo, l6_knl);
    let k4h = rmsnorm(h @ l6_k4hi, l6_knh);
    let v4l = h @ l6_v4lo;
    let v4h = h @ l6_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l6_k5lo, l6_knl);
    let k5h = rmsnorm(h @ l6_k5hi, l6_knh);
    let v5l = h @ l6_v5lo;
    let v5h = h @ l6_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l6_k6lo, l6_knl);
    let k6h = rmsnorm(h @ l6_k6hi, l6_knh);
    let v6l = h @ l6_v6lo;
    let v6h = h @ l6_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l6_k7lo, l6_knl);
    let k7h = rmsnorm(h @ l6_k7hi, l6_knh);
    let v7l = h @ l6_v7lo;
    let v7h = h @ l6_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l6_o0lo + c0h @ l6_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l6_o1lo + c1h @ l6_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l6_o2lo + c2h @ l6_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l6_o3lo + c3h @ l6_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l6_o4lo + c4h @ l6_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l6_o5lo + c5h @ l6_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l6_o6lo + c6h @ l6_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l6_o7lo + c7h @ l6_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l6_o8lo + c8h @ l6_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l6_o9lo + c9h @ l6_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l6_o10lo + c10h @ l6_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l6_o11lo + c11h @ l6_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l6_o12lo + c12h @ l6_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l6_o13lo + c13h @ l6_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l6_o14lo + c14h @ l6_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l6_o15lo + c15h @ l6_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l6_ln2);
    let f = r + (silu(n @ l6_wg) * (n @ l6_wu)) @ l6_wd;
    f
}

fn l7(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l7_ln1);
    let q0l = rmsnorm(h @ l7_q0lo, l7_qnl);
    let q0h = rmsnorm(h @ l7_q0hi, l7_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l7_q1lo, l7_qnl);
    let q1h = rmsnorm(h @ l7_q1hi, l7_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l7_q2lo, l7_qnl);
    let q2h = rmsnorm(h @ l7_q2hi, l7_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l7_q3lo, l7_qnl);
    let q3h = rmsnorm(h @ l7_q3hi, l7_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l7_q4lo, l7_qnl);
    let q4h = rmsnorm(h @ l7_q4hi, l7_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l7_q5lo, l7_qnl);
    let q5h = rmsnorm(h @ l7_q5hi, l7_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l7_q6lo, l7_qnl);
    let q6h = rmsnorm(h @ l7_q6hi, l7_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l7_q7lo, l7_qnl);
    let q7h = rmsnorm(h @ l7_q7hi, l7_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l7_q8lo, l7_qnl);
    let q8h = rmsnorm(h @ l7_q8hi, l7_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l7_q9lo, l7_qnl);
    let q9h = rmsnorm(h @ l7_q9hi, l7_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l7_q10lo, l7_qnl);
    let q10h = rmsnorm(h @ l7_q10hi, l7_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l7_q11lo, l7_qnl);
    let q11h = rmsnorm(h @ l7_q11hi, l7_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l7_q12lo, l7_qnl);
    let q12h = rmsnorm(h @ l7_q12hi, l7_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l7_q13lo, l7_qnl);
    let q13h = rmsnorm(h @ l7_q13hi, l7_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l7_q14lo, l7_qnl);
    let q14h = rmsnorm(h @ l7_q14hi, l7_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l7_q15lo, l7_qnl);
    let q15h = rmsnorm(h @ l7_q15hi, l7_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l7_k0lo, l7_knl);
    let k0h = rmsnorm(h @ l7_k0hi, l7_knh);
    let v0l = h @ l7_v0lo;
    let v0h = h @ l7_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l7_k1lo, l7_knl);
    let k1h = rmsnorm(h @ l7_k1hi, l7_knh);
    let v1l = h @ l7_v1lo;
    let v1h = h @ l7_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l7_k2lo, l7_knl);
    let k2h = rmsnorm(h @ l7_k2hi, l7_knh);
    let v2l = h @ l7_v2lo;
    let v2h = h @ l7_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l7_k3lo, l7_knl);
    let k3h = rmsnorm(h @ l7_k3hi, l7_knh);
    let v3l = h @ l7_v3lo;
    let v3h = h @ l7_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l7_k4lo, l7_knl);
    let k4h = rmsnorm(h @ l7_k4hi, l7_knh);
    let v4l = h @ l7_v4lo;
    let v4h = h @ l7_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l7_k5lo, l7_knl);
    let k5h = rmsnorm(h @ l7_k5hi, l7_knh);
    let v5l = h @ l7_v5lo;
    let v5h = h @ l7_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l7_k6lo, l7_knl);
    let k6h = rmsnorm(h @ l7_k6hi, l7_knh);
    let v6l = h @ l7_v6lo;
    let v6h = h @ l7_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l7_k7lo, l7_knl);
    let k7h = rmsnorm(h @ l7_k7hi, l7_knh);
    let v7l = h @ l7_v7lo;
    let v7h = h @ l7_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l7_o0lo + c0h @ l7_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l7_o1lo + c1h @ l7_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l7_o2lo + c2h @ l7_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l7_o3lo + c3h @ l7_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l7_o4lo + c4h @ l7_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l7_o5lo + c5h @ l7_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l7_o6lo + c6h @ l7_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l7_o7lo + c7h @ l7_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l7_o8lo + c8h @ l7_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l7_o9lo + c9h @ l7_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l7_o10lo + c10h @ l7_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l7_o11lo + c11h @ l7_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l7_o12lo + c12h @ l7_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l7_o13lo + c13h @ l7_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l7_o14lo + c14h @ l7_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l7_o15lo + c15h @ l7_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l7_ln2);
    let f = r + (silu(n @ l7_wg) * (n @ l7_wu)) @ l7_wd;
    f
}

fn l8(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l8_ln1);
    let q0l = rmsnorm(h @ l8_q0lo, l8_qnl);
    let q0h = rmsnorm(h @ l8_q0hi, l8_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l8_q1lo, l8_qnl);
    let q1h = rmsnorm(h @ l8_q1hi, l8_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l8_q2lo, l8_qnl);
    let q2h = rmsnorm(h @ l8_q2hi, l8_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l8_q3lo, l8_qnl);
    let q3h = rmsnorm(h @ l8_q3hi, l8_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l8_q4lo, l8_qnl);
    let q4h = rmsnorm(h @ l8_q4hi, l8_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l8_q5lo, l8_qnl);
    let q5h = rmsnorm(h @ l8_q5hi, l8_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l8_q6lo, l8_qnl);
    let q6h = rmsnorm(h @ l8_q6hi, l8_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l8_q7lo, l8_qnl);
    let q7h = rmsnorm(h @ l8_q7hi, l8_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l8_q8lo, l8_qnl);
    let q8h = rmsnorm(h @ l8_q8hi, l8_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l8_q9lo, l8_qnl);
    let q9h = rmsnorm(h @ l8_q9hi, l8_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l8_q10lo, l8_qnl);
    let q10h = rmsnorm(h @ l8_q10hi, l8_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l8_q11lo, l8_qnl);
    let q11h = rmsnorm(h @ l8_q11hi, l8_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l8_q12lo, l8_qnl);
    let q12h = rmsnorm(h @ l8_q12hi, l8_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l8_q13lo, l8_qnl);
    let q13h = rmsnorm(h @ l8_q13hi, l8_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l8_q14lo, l8_qnl);
    let q14h = rmsnorm(h @ l8_q14hi, l8_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l8_q15lo, l8_qnl);
    let q15h = rmsnorm(h @ l8_q15hi, l8_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l8_k0lo, l8_knl);
    let k0h = rmsnorm(h @ l8_k0hi, l8_knh);
    let v0l = h @ l8_v0lo;
    let v0h = h @ l8_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l8_k1lo, l8_knl);
    let k1h = rmsnorm(h @ l8_k1hi, l8_knh);
    let v1l = h @ l8_v1lo;
    let v1h = h @ l8_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l8_k2lo, l8_knl);
    let k2h = rmsnorm(h @ l8_k2hi, l8_knh);
    let v2l = h @ l8_v2lo;
    let v2h = h @ l8_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l8_k3lo, l8_knl);
    let k3h = rmsnorm(h @ l8_k3hi, l8_knh);
    let v3l = h @ l8_v3lo;
    let v3h = h @ l8_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l8_k4lo, l8_knl);
    let k4h = rmsnorm(h @ l8_k4hi, l8_knh);
    let v4l = h @ l8_v4lo;
    let v4h = h @ l8_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l8_k5lo, l8_knl);
    let k5h = rmsnorm(h @ l8_k5hi, l8_knh);
    let v5l = h @ l8_v5lo;
    let v5h = h @ l8_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l8_k6lo, l8_knl);
    let k6h = rmsnorm(h @ l8_k6hi, l8_knh);
    let v6l = h @ l8_v6lo;
    let v6h = h @ l8_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l8_k7lo, l8_knl);
    let k7h = rmsnorm(h @ l8_k7hi, l8_knh);
    let v7l = h @ l8_v7lo;
    let v7h = h @ l8_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l8_o0lo + c0h @ l8_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l8_o1lo + c1h @ l8_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l8_o2lo + c2h @ l8_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l8_o3lo + c3h @ l8_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l8_o4lo + c4h @ l8_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l8_o5lo + c5h @ l8_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l8_o6lo + c6h @ l8_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l8_o7lo + c7h @ l8_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l8_o8lo + c8h @ l8_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l8_o9lo + c9h @ l8_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l8_o10lo + c10h @ l8_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l8_o11lo + c11h @ l8_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l8_o12lo + c12h @ l8_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l8_o13lo + c13h @ l8_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l8_o14lo + c14h @ l8_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l8_o15lo + c15h @ l8_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l8_ln2);
    let f = r + (silu(n @ l8_wg) * (n @ l8_wu)) @ l8_wd;
    f
}

fn l9(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l9_ln1);
    let q0l = rmsnorm(h @ l9_q0lo, l9_qnl);
    let q0h = rmsnorm(h @ l9_q0hi, l9_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l9_q1lo, l9_qnl);
    let q1h = rmsnorm(h @ l9_q1hi, l9_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l9_q2lo, l9_qnl);
    let q2h = rmsnorm(h @ l9_q2hi, l9_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l9_q3lo, l9_qnl);
    let q3h = rmsnorm(h @ l9_q3hi, l9_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l9_q4lo, l9_qnl);
    let q4h = rmsnorm(h @ l9_q4hi, l9_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l9_q5lo, l9_qnl);
    let q5h = rmsnorm(h @ l9_q5hi, l9_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l9_q6lo, l9_qnl);
    let q6h = rmsnorm(h @ l9_q6hi, l9_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l9_q7lo, l9_qnl);
    let q7h = rmsnorm(h @ l9_q7hi, l9_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l9_q8lo, l9_qnl);
    let q8h = rmsnorm(h @ l9_q8hi, l9_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l9_q9lo, l9_qnl);
    let q9h = rmsnorm(h @ l9_q9hi, l9_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l9_q10lo, l9_qnl);
    let q10h = rmsnorm(h @ l9_q10hi, l9_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l9_q11lo, l9_qnl);
    let q11h = rmsnorm(h @ l9_q11hi, l9_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l9_q12lo, l9_qnl);
    let q12h = rmsnorm(h @ l9_q12hi, l9_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l9_q13lo, l9_qnl);
    let q13h = rmsnorm(h @ l9_q13hi, l9_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l9_q14lo, l9_qnl);
    let q14h = rmsnorm(h @ l9_q14hi, l9_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l9_q15lo, l9_qnl);
    let q15h = rmsnorm(h @ l9_q15hi, l9_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l9_k0lo, l9_knl);
    let k0h = rmsnorm(h @ l9_k0hi, l9_knh);
    let v0l = h @ l9_v0lo;
    let v0h = h @ l9_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l9_k1lo, l9_knl);
    let k1h = rmsnorm(h @ l9_k1hi, l9_knh);
    let v1l = h @ l9_v1lo;
    let v1h = h @ l9_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l9_k2lo, l9_knl);
    let k2h = rmsnorm(h @ l9_k2hi, l9_knh);
    let v2l = h @ l9_v2lo;
    let v2h = h @ l9_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l9_k3lo, l9_knl);
    let k3h = rmsnorm(h @ l9_k3hi, l9_knh);
    let v3l = h @ l9_v3lo;
    let v3h = h @ l9_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l9_k4lo, l9_knl);
    let k4h = rmsnorm(h @ l9_k4hi, l9_knh);
    let v4l = h @ l9_v4lo;
    let v4h = h @ l9_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l9_k5lo, l9_knl);
    let k5h = rmsnorm(h @ l9_k5hi, l9_knh);
    let v5l = h @ l9_v5lo;
    let v5h = h @ l9_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l9_k6lo, l9_knl);
    let k6h = rmsnorm(h @ l9_k6hi, l9_knh);
    let v6l = h @ l9_v6lo;
    let v6h = h @ l9_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l9_k7lo, l9_knl);
    let k7h = rmsnorm(h @ l9_k7hi, l9_knh);
    let v7l = h @ l9_v7lo;
    let v7h = h @ l9_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l9_o0lo + c0h @ l9_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l9_o1lo + c1h @ l9_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l9_o2lo + c2h @ l9_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l9_o3lo + c3h @ l9_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l9_o4lo + c4h @ l9_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l9_o5lo + c5h @ l9_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l9_o6lo + c6h @ l9_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l9_o7lo + c7h @ l9_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l9_o8lo + c8h @ l9_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l9_o9lo + c9h @ l9_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l9_o10lo + c10h @ l9_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l9_o11lo + c11h @ l9_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l9_o12lo + c12h @ l9_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l9_o13lo + c13h @ l9_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l9_o14lo + c14h @ l9_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l9_o15lo + c15h @ l9_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l9_ln2);
    let f = r + (silu(n @ l9_wg) * (n @ l9_wu)) @ l9_wd;
    f
}

fn l10(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l10_ln1);
    let q0l = rmsnorm(h @ l10_q0lo, l10_qnl);
    let q0h = rmsnorm(h @ l10_q0hi, l10_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l10_q1lo, l10_qnl);
    let q1h = rmsnorm(h @ l10_q1hi, l10_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l10_q2lo, l10_qnl);
    let q2h = rmsnorm(h @ l10_q2hi, l10_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l10_q3lo, l10_qnl);
    let q3h = rmsnorm(h @ l10_q3hi, l10_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l10_q4lo, l10_qnl);
    let q4h = rmsnorm(h @ l10_q4hi, l10_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l10_q5lo, l10_qnl);
    let q5h = rmsnorm(h @ l10_q5hi, l10_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l10_q6lo, l10_qnl);
    let q6h = rmsnorm(h @ l10_q6hi, l10_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l10_q7lo, l10_qnl);
    let q7h = rmsnorm(h @ l10_q7hi, l10_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l10_q8lo, l10_qnl);
    let q8h = rmsnorm(h @ l10_q8hi, l10_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l10_q9lo, l10_qnl);
    let q9h = rmsnorm(h @ l10_q9hi, l10_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l10_q10lo, l10_qnl);
    let q10h = rmsnorm(h @ l10_q10hi, l10_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l10_q11lo, l10_qnl);
    let q11h = rmsnorm(h @ l10_q11hi, l10_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l10_q12lo, l10_qnl);
    let q12h = rmsnorm(h @ l10_q12hi, l10_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l10_q13lo, l10_qnl);
    let q13h = rmsnorm(h @ l10_q13hi, l10_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l10_q14lo, l10_qnl);
    let q14h = rmsnorm(h @ l10_q14hi, l10_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l10_q15lo, l10_qnl);
    let q15h = rmsnorm(h @ l10_q15hi, l10_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l10_k0lo, l10_knl);
    let k0h = rmsnorm(h @ l10_k0hi, l10_knh);
    let v0l = h @ l10_v0lo;
    let v0h = h @ l10_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l10_k1lo, l10_knl);
    let k1h = rmsnorm(h @ l10_k1hi, l10_knh);
    let v1l = h @ l10_v1lo;
    let v1h = h @ l10_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l10_k2lo, l10_knl);
    let k2h = rmsnorm(h @ l10_k2hi, l10_knh);
    let v2l = h @ l10_v2lo;
    let v2h = h @ l10_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l10_k3lo, l10_knl);
    let k3h = rmsnorm(h @ l10_k3hi, l10_knh);
    let v3l = h @ l10_v3lo;
    let v3h = h @ l10_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l10_k4lo, l10_knl);
    let k4h = rmsnorm(h @ l10_k4hi, l10_knh);
    let v4l = h @ l10_v4lo;
    let v4h = h @ l10_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l10_k5lo, l10_knl);
    let k5h = rmsnorm(h @ l10_k5hi, l10_knh);
    let v5l = h @ l10_v5lo;
    let v5h = h @ l10_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l10_k6lo, l10_knl);
    let k6h = rmsnorm(h @ l10_k6hi, l10_knh);
    let v6l = h @ l10_v6lo;
    let v6h = h @ l10_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l10_k7lo, l10_knl);
    let k7h = rmsnorm(h @ l10_k7hi, l10_knh);
    let v7l = h @ l10_v7lo;
    let v7h = h @ l10_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l10_o0lo + c0h @ l10_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l10_o1lo + c1h @ l10_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l10_o2lo + c2h @ l10_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l10_o3lo + c3h @ l10_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l10_o4lo + c4h @ l10_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l10_o5lo + c5h @ l10_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l10_o6lo + c6h @ l10_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l10_o7lo + c7h @ l10_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l10_o8lo + c8h @ l10_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l10_o9lo + c9h @ l10_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l10_o10lo + c10h @ l10_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l10_o11lo + c11h @ l10_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l10_o12lo + c12h @ l10_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l10_o13lo + c13h @ l10_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l10_o14lo + c14h @ l10_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l10_o15lo + c15h @ l10_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l10_ln2);
    let f = r + (silu(n @ l10_wg) * (n @ l10_wu)) @ l10_wd;
    f
}

fn l11(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l11_ln1);
    let q0l = rmsnorm(h @ l11_q0lo, l11_qnl);
    let q0h = rmsnorm(h @ l11_q0hi, l11_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l11_q1lo, l11_qnl);
    let q1h = rmsnorm(h @ l11_q1hi, l11_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l11_q2lo, l11_qnl);
    let q2h = rmsnorm(h @ l11_q2hi, l11_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l11_q3lo, l11_qnl);
    let q3h = rmsnorm(h @ l11_q3hi, l11_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l11_q4lo, l11_qnl);
    let q4h = rmsnorm(h @ l11_q4hi, l11_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l11_q5lo, l11_qnl);
    let q5h = rmsnorm(h @ l11_q5hi, l11_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l11_q6lo, l11_qnl);
    let q6h = rmsnorm(h @ l11_q6hi, l11_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l11_q7lo, l11_qnl);
    let q7h = rmsnorm(h @ l11_q7hi, l11_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l11_q8lo, l11_qnl);
    let q8h = rmsnorm(h @ l11_q8hi, l11_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l11_q9lo, l11_qnl);
    let q9h = rmsnorm(h @ l11_q9hi, l11_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l11_q10lo, l11_qnl);
    let q10h = rmsnorm(h @ l11_q10hi, l11_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l11_q11lo, l11_qnl);
    let q11h = rmsnorm(h @ l11_q11hi, l11_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l11_q12lo, l11_qnl);
    let q12h = rmsnorm(h @ l11_q12hi, l11_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l11_q13lo, l11_qnl);
    let q13h = rmsnorm(h @ l11_q13hi, l11_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l11_q14lo, l11_qnl);
    let q14h = rmsnorm(h @ l11_q14hi, l11_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l11_q15lo, l11_qnl);
    let q15h = rmsnorm(h @ l11_q15hi, l11_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l11_k0lo, l11_knl);
    let k0h = rmsnorm(h @ l11_k0hi, l11_knh);
    let v0l = h @ l11_v0lo;
    let v0h = h @ l11_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l11_k1lo, l11_knl);
    let k1h = rmsnorm(h @ l11_k1hi, l11_knh);
    let v1l = h @ l11_v1lo;
    let v1h = h @ l11_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l11_k2lo, l11_knl);
    let k2h = rmsnorm(h @ l11_k2hi, l11_knh);
    let v2l = h @ l11_v2lo;
    let v2h = h @ l11_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l11_k3lo, l11_knl);
    let k3h = rmsnorm(h @ l11_k3hi, l11_knh);
    let v3l = h @ l11_v3lo;
    let v3h = h @ l11_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l11_k4lo, l11_knl);
    let k4h = rmsnorm(h @ l11_k4hi, l11_knh);
    let v4l = h @ l11_v4lo;
    let v4h = h @ l11_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l11_k5lo, l11_knl);
    let k5h = rmsnorm(h @ l11_k5hi, l11_knh);
    let v5l = h @ l11_v5lo;
    let v5h = h @ l11_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l11_k6lo, l11_knl);
    let k6h = rmsnorm(h @ l11_k6hi, l11_knh);
    let v6l = h @ l11_v6lo;
    let v6h = h @ l11_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l11_k7lo, l11_knl);
    let k7h = rmsnorm(h @ l11_k7hi, l11_knh);
    let v7l = h @ l11_v7lo;
    let v7h = h @ l11_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l11_o0lo + c0h @ l11_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l11_o1lo + c1h @ l11_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l11_o2lo + c2h @ l11_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l11_o3lo + c3h @ l11_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l11_o4lo + c4h @ l11_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l11_o5lo + c5h @ l11_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l11_o6lo + c6h @ l11_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l11_o7lo + c7h @ l11_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l11_o8lo + c8h @ l11_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l11_o9lo + c9h @ l11_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l11_o10lo + c10h @ l11_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l11_o11lo + c11h @ l11_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l11_o12lo + c12h @ l11_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l11_o13lo + c13h @ l11_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l11_o14lo + c14h @ l11_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l11_o15lo + c15h @ l11_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l11_ln2);
    let f = r + (silu(n @ l11_wg) * (n @ l11_wu)) @ l11_wd;
    f
}

fn l12(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l12_ln1);
    let q0l = rmsnorm(h @ l12_q0lo, l12_qnl);
    let q0h = rmsnorm(h @ l12_q0hi, l12_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l12_q1lo, l12_qnl);
    let q1h = rmsnorm(h @ l12_q1hi, l12_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l12_q2lo, l12_qnl);
    let q2h = rmsnorm(h @ l12_q2hi, l12_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l12_q3lo, l12_qnl);
    let q3h = rmsnorm(h @ l12_q3hi, l12_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l12_q4lo, l12_qnl);
    let q4h = rmsnorm(h @ l12_q4hi, l12_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l12_q5lo, l12_qnl);
    let q5h = rmsnorm(h @ l12_q5hi, l12_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l12_q6lo, l12_qnl);
    let q6h = rmsnorm(h @ l12_q6hi, l12_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l12_q7lo, l12_qnl);
    let q7h = rmsnorm(h @ l12_q7hi, l12_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l12_q8lo, l12_qnl);
    let q8h = rmsnorm(h @ l12_q8hi, l12_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l12_q9lo, l12_qnl);
    let q9h = rmsnorm(h @ l12_q9hi, l12_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l12_q10lo, l12_qnl);
    let q10h = rmsnorm(h @ l12_q10hi, l12_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l12_q11lo, l12_qnl);
    let q11h = rmsnorm(h @ l12_q11hi, l12_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l12_q12lo, l12_qnl);
    let q12h = rmsnorm(h @ l12_q12hi, l12_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l12_q13lo, l12_qnl);
    let q13h = rmsnorm(h @ l12_q13hi, l12_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l12_q14lo, l12_qnl);
    let q14h = rmsnorm(h @ l12_q14hi, l12_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l12_q15lo, l12_qnl);
    let q15h = rmsnorm(h @ l12_q15hi, l12_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l12_k0lo, l12_knl);
    let k0h = rmsnorm(h @ l12_k0hi, l12_knh);
    let v0l = h @ l12_v0lo;
    let v0h = h @ l12_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l12_k1lo, l12_knl);
    let k1h = rmsnorm(h @ l12_k1hi, l12_knh);
    let v1l = h @ l12_v1lo;
    let v1h = h @ l12_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l12_k2lo, l12_knl);
    let k2h = rmsnorm(h @ l12_k2hi, l12_knh);
    let v2l = h @ l12_v2lo;
    let v2h = h @ l12_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l12_k3lo, l12_knl);
    let k3h = rmsnorm(h @ l12_k3hi, l12_knh);
    let v3l = h @ l12_v3lo;
    let v3h = h @ l12_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l12_k4lo, l12_knl);
    let k4h = rmsnorm(h @ l12_k4hi, l12_knh);
    let v4l = h @ l12_v4lo;
    let v4h = h @ l12_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l12_k5lo, l12_knl);
    let k5h = rmsnorm(h @ l12_k5hi, l12_knh);
    let v5l = h @ l12_v5lo;
    let v5h = h @ l12_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l12_k6lo, l12_knl);
    let k6h = rmsnorm(h @ l12_k6hi, l12_knh);
    let v6l = h @ l12_v6lo;
    let v6h = h @ l12_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l12_k7lo, l12_knl);
    let k7h = rmsnorm(h @ l12_k7hi, l12_knh);
    let v7l = h @ l12_v7lo;
    let v7h = h @ l12_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l12_o0lo + c0h @ l12_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l12_o1lo + c1h @ l12_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l12_o2lo + c2h @ l12_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l12_o3lo + c3h @ l12_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l12_o4lo + c4h @ l12_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l12_o5lo + c5h @ l12_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l12_o6lo + c6h @ l12_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l12_o7lo + c7h @ l12_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l12_o8lo + c8h @ l12_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l12_o9lo + c9h @ l12_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l12_o10lo + c10h @ l12_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l12_o11lo + c11h @ l12_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l12_o12lo + c12h @ l12_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l12_o13lo + c13h @ l12_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l12_o14lo + c14h @ l12_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l12_o15lo + c15h @ l12_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l12_ln2);
    let f = r + (silu(n @ l12_wg) * (n @ l12_wu)) @ l12_wd;
    f
}

fn l13(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l13_ln1);
    let q0l = rmsnorm(h @ l13_q0lo, l13_qnl);
    let q0h = rmsnorm(h @ l13_q0hi, l13_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l13_q1lo, l13_qnl);
    let q1h = rmsnorm(h @ l13_q1hi, l13_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l13_q2lo, l13_qnl);
    let q2h = rmsnorm(h @ l13_q2hi, l13_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l13_q3lo, l13_qnl);
    let q3h = rmsnorm(h @ l13_q3hi, l13_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l13_q4lo, l13_qnl);
    let q4h = rmsnorm(h @ l13_q4hi, l13_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l13_q5lo, l13_qnl);
    let q5h = rmsnorm(h @ l13_q5hi, l13_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l13_q6lo, l13_qnl);
    let q6h = rmsnorm(h @ l13_q6hi, l13_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l13_q7lo, l13_qnl);
    let q7h = rmsnorm(h @ l13_q7hi, l13_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l13_q8lo, l13_qnl);
    let q8h = rmsnorm(h @ l13_q8hi, l13_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l13_q9lo, l13_qnl);
    let q9h = rmsnorm(h @ l13_q9hi, l13_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l13_q10lo, l13_qnl);
    let q10h = rmsnorm(h @ l13_q10hi, l13_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l13_q11lo, l13_qnl);
    let q11h = rmsnorm(h @ l13_q11hi, l13_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l13_q12lo, l13_qnl);
    let q12h = rmsnorm(h @ l13_q12hi, l13_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l13_q13lo, l13_qnl);
    let q13h = rmsnorm(h @ l13_q13hi, l13_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l13_q14lo, l13_qnl);
    let q14h = rmsnorm(h @ l13_q14hi, l13_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l13_q15lo, l13_qnl);
    let q15h = rmsnorm(h @ l13_q15hi, l13_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l13_k0lo, l13_knl);
    let k0h = rmsnorm(h @ l13_k0hi, l13_knh);
    let v0l = h @ l13_v0lo;
    let v0h = h @ l13_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l13_k1lo, l13_knl);
    let k1h = rmsnorm(h @ l13_k1hi, l13_knh);
    let v1l = h @ l13_v1lo;
    let v1h = h @ l13_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l13_k2lo, l13_knl);
    let k2h = rmsnorm(h @ l13_k2hi, l13_knh);
    let v2l = h @ l13_v2lo;
    let v2h = h @ l13_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l13_k3lo, l13_knl);
    let k3h = rmsnorm(h @ l13_k3hi, l13_knh);
    let v3l = h @ l13_v3lo;
    let v3h = h @ l13_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l13_k4lo, l13_knl);
    let k4h = rmsnorm(h @ l13_k4hi, l13_knh);
    let v4l = h @ l13_v4lo;
    let v4h = h @ l13_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l13_k5lo, l13_knl);
    let k5h = rmsnorm(h @ l13_k5hi, l13_knh);
    let v5l = h @ l13_v5lo;
    let v5h = h @ l13_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l13_k6lo, l13_knl);
    let k6h = rmsnorm(h @ l13_k6hi, l13_knh);
    let v6l = h @ l13_v6lo;
    let v6h = h @ l13_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l13_k7lo, l13_knl);
    let k7h = rmsnorm(h @ l13_k7hi, l13_knh);
    let v7l = h @ l13_v7lo;
    let v7h = h @ l13_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l13_o0lo + c0h @ l13_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l13_o1lo + c1h @ l13_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l13_o2lo + c2h @ l13_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l13_o3lo + c3h @ l13_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l13_o4lo + c4h @ l13_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l13_o5lo + c5h @ l13_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l13_o6lo + c6h @ l13_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l13_o7lo + c7h @ l13_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l13_o8lo + c8h @ l13_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l13_o9lo + c9h @ l13_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l13_o10lo + c10h @ l13_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l13_o11lo + c11h @ l13_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l13_o12lo + c12h @ l13_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l13_o13lo + c13h @ l13_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l13_o14lo + c14h @ l13_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l13_o15lo + c15h @ l13_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l13_ln2);
    let f = r + (silu(n @ l13_wg) * (n @ l13_wu)) @ l13_wd;
    f
}

fn l14(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l14_ln1);
    let q0l = rmsnorm(h @ l14_q0lo, l14_qnl);
    let q0h = rmsnorm(h @ l14_q0hi, l14_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l14_q1lo, l14_qnl);
    let q1h = rmsnorm(h @ l14_q1hi, l14_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l14_q2lo, l14_qnl);
    let q2h = rmsnorm(h @ l14_q2hi, l14_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l14_q3lo, l14_qnl);
    let q3h = rmsnorm(h @ l14_q3hi, l14_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l14_q4lo, l14_qnl);
    let q4h = rmsnorm(h @ l14_q4hi, l14_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l14_q5lo, l14_qnl);
    let q5h = rmsnorm(h @ l14_q5hi, l14_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l14_q6lo, l14_qnl);
    let q6h = rmsnorm(h @ l14_q6hi, l14_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l14_q7lo, l14_qnl);
    let q7h = rmsnorm(h @ l14_q7hi, l14_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l14_q8lo, l14_qnl);
    let q8h = rmsnorm(h @ l14_q8hi, l14_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l14_q9lo, l14_qnl);
    let q9h = rmsnorm(h @ l14_q9hi, l14_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l14_q10lo, l14_qnl);
    let q10h = rmsnorm(h @ l14_q10hi, l14_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l14_q11lo, l14_qnl);
    let q11h = rmsnorm(h @ l14_q11hi, l14_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l14_q12lo, l14_qnl);
    let q12h = rmsnorm(h @ l14_q12hi, l14_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l14_q13lo, l14_qnl);
    let q13h = rmsnorm(h @ l14_q13hi, l14_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l14_q14lo, l14_qnl);
    let q14h = rmsnorm(h @ l14_q14hi, l14_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l14_q15lo, l14_qnl);
    let q15h = rmsnorm(h @ l14_q15hi, l14_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l14_k0lo, l14_knl);
    let k0h = rmsnorm(h @ l14_k0hi, l14_knh);
    let v0l = h @ l14_v0lo;
    let v0h = h @ l14_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l14_k1lo, l14_knl);
    let k1h = rmsnorm(h @ l14_k1hi, l14_knh);
    let v1l = h @ l14_v1lo;
    let v1h = h @ l14_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l14_k2lo, l14_knl);
    let k2h = rmsnorm(h @ l14_k2hi, l14_knh);
    let v2l = h @ l14_v2lo;
    let v2h = h @ l14_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l14_k3lo, l14_knl);
    let k3h = rmsnorm(h @ l14_k3hi, l14_knh);
    let v3l = h @ l14_v3lo;
    let v3h = h @ l14_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l14_k4lo, l14_knl);
    let k4h = rmsnorm(h @ l14_k4hi, l14_knh);
    let v4l = h @ l14_v4lo;
    let v4h = h @ l14_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l14_k5lo, l14_knl);
    let k5h = rmsnorm(h @ l14_k5hi, l14_knh);
    let v5l = h @ l14_v5lo;
    let v5h = h @ l14_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l14_k6lo, l14_knl);
    let k6h = rmsnorm(h @ l14_k6hi, l14_knh);
    let v6l = h @ l14_v6lo;
    let v6h = h @ l14_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l14_k7lo, l14_knl);
    let k7h = rmsnorm(h @ l14_k7hi, l14_knh);
    let v7l = h @ l14_v7lo;
    let v7h = h @ l14_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l14_o0lo + c0h @ l14_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l14_o1lo + c1h @ l14_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l14_o2lo + c2h @ l14_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l14_o3lo + c3h @ l14_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l14_o4lo + c4h @ l14_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l14_o5lo + c5h @ l14_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l14_o6lo + c6h @ l14_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l14_o7lo + c7h @ l14_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l14_o8lo + c8h @ l14_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l14_o9lo + c9h @ l14_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l14_o10lo + c10h @ l14_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l14_o11lo + c11h @ l14_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l14_o12lo + c12h @ l14_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l14_o13lo + c13h @ l14_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l14_o14lo + c14h @ l14_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l14_o15lo + c15h @ l14_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l14_ln2);
    let f = r + (silu(n @ l14_wg) * (n @ l14_wu)) @ l14_wd;
    f
}

fn l15(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l15_ln1);
    let q0l = rmsnorm(h @ l15_q0lo, l15_qnl);
    let q0h = rmsnorm(h @ l15_q0hi, l15_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l15_q1lo, l15_qnl);
    let q1h = rmsnorm(h @ l15_q1hi, l15_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l15_q2lo, l15_qnl);
    let q2h = rmsnorm(h @ l15_q2hi, l15_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l15_q3lo, l15_qnl);
    let q3h = rmsnorm(h @ l15_q3hi, l15_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l15_q4lo, l15_qnl);
    let q4h = rmsnorm(h @ l15_q4hi, l15_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l15_q5lo, l15_qnl);
    let q5h = rmsnorm(h @ l15_q5hi, l15_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l15_q6lo, l15_qnl);
    let q6h = rmsnorm(h @ l15_q6hi, l15_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l15_q7lo, l15_qnl);
    let q7h = rmsnorm(h @ l15_q7hi, l15_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l15_q8lo, l15_qnl);
    let q8h = rmsnorm(h @ l15_q8hi, l15_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l15_q9lo, l15_qnl);
    let q9h = rmsnorm(h @ l15_q9hi, l15_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l15_q10lo, l15_qnl);
    let q10h = rmsnorm(h @ l15_q10hi, l15_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l15_q11lo, l15_qnl);
    let q11h = rmsnorm(h @ l15_q11hi, l15_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l15_q12lo, l15_qnl);
    let q12h = rmsnorm(h @ l15_q12hi, l15_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l15_q13lo, l15_qnl);
    let q13h = rmsnorm(h @ l15_q13hi, l15_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l15_q14lo, l15_qnl);
    let q14h = rmsnorm(h @ l15_q14hi, l15_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l15_q15lo, l15_qnl);
    let q15h = rmsnorm(h @ l15_q15hi, l15_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l15_k0lo, l15_knl);
    let k0h = rmsnorm(h @ l15_k0hi, l15_knh);
    let v0l = h @ l15_v0lo;
    let v0h = h @ l15_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l15_k1lo, l15_knl);
    let k1h = rmsnorm(h @ l15_k1hi, l15_knh);
    let v1l = h @ l15_v1lo;
    let v1h = h @ l15_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l15_k2lo, l15_knl);
    let k2h = rmsnorm(h @ l15_k2hi, l15_knh);
    let v2l = h @ l15_v2lo;
    let v2h = h @ l15_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l15_k3lo, l15_knl);
    let k3h = rmsnorm(h @ l15_k3hi, l15_knh);
    let v3l = h @ l15_v3lo;
    let v3h = h @ l15_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l15_k4lo, l15_knl);
    let k4h = rmsnorm(h @ l15_k4hi, l15_knh);
    let v4l = h @ l15_v4lo;
    let v4h = h @ l15_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l15_k5lo, l15_knl);
    let k5h = rmsnorm(h @ l15_k5hi, l15_knh);
    let v5l = h @ l15_v5lo;
    let v5h = h @ l15_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l15_k6lo, l15_knl);
    let k6h = rmsnorm(h @ l15_k6hi, l15_knh);
    let v6l = h @ l15_v6lo;
    let v6h = h @ l15_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l15_k7lo, l15_knl);
    let k7h = rmsnorm(h @ l15_k7hi, l15_knh);
    let v7l = h @ l15_v7lo;
    let v7h = h @ l15_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l15_o0lo + c0h @ l15_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l15_o1lo + c1h @ l15_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l15_o2lo + c2h @ l15_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l15_o3lo + c3h @ l15_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l15_o4lo + c4h @ l15_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l15_o5lo + c5h @ l15_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l15_o6lo + c6h @ l15_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l15_o7lo + c7h @ l15_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l15_o8lo + c8h @ l15_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l15_o9lo + c9h @ l15_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l15_o10lo + c10h @ l15_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l15_o11lo + c11h @ l15_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l15_o12lo + c12h @ l15_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l15_o13lo + c13h @ l15_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l15_o14lo + c14h @ l15_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l15_o15lo + c15h @ l15_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l15_ln2);
    let f = r + (silu(n @ l15_wg) * (n @ l15_wu)) @ l15_wd;
    f
}

fn l16(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l16_ln1);
    let q0l = rmsnorm(h @ l16_q0lo, l16_qnl);
    let q0h = rmsnorm(h @ l16_q0hi, l16_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l16_q1lo, l16_qnl);
    let q1h = rmsnorm(h @ l16_q1hi, l16_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l16_q2lo, l16_qnl);
    let q2h = rmsnorm(h @ l16_q2hi, l16_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l16_q3lo, l16_qnl);
    let q3h = rmsnorm(h @ l16_q3hi, l16_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l16_q4lo, l16_qnl);
    let q4h = rmsnorm(h @ l16_q4hi, l16_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l16_q5lo, l16_qnl);
    let q5h = rmsnorm(h @ l16_q5hi, l16_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l16_q6lo, l16_qnl);
    let q6h = rmsnorm(h @ l16_q6hi, l16_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l16_q7lo, l16_qnl);
    let q7h = rmsnorm(h @ l16_q7hi, l16_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l16_q8lo, l16_qnl);
    let q8h = rmsnorm(h @ l16_q8hi, l16_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l16_q9lo, l16_qnl);
    let q9h = rmsnorm(h @ l16_q9hi, l16_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l16_q10lo, l16_qnl);
    let q10h = rmsnorm(h @ l16_q10hi, l16_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l16_q11lo, l16_qnl);
    let q11h = rmsnorm(h @ l16_q11hi, l16_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l16_q12lo, l16_qnl);
    let q12h = rmsnorm(h @ l16_q12hi, l16_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l16_q13lo, l16_qnl);
    let q13h = rmsnorm(h @ l16_q13hi, l16_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l16_q14lo, l16_qnl);
    let q14h = rmsnorm(h @ l16_q14hi, l16_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l16_q15lo, l16_qnl);
    let q15h = rmsnorm(h @ l16_q15hi, l16_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l16_k0lo, l16_knl);
    let k0h = rmsnorm(h @ l16_k0hi, l16_knh);
    let v0l = h @ l16_v0lo;
    let v0h = h @ l16_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l16_k1lo, l16_knl);
    let k1h = rmsnorm(h @ l16_k1hi, l16_knh);
    let v1l = h @ l16_v1lo;
    let v1h = h @ l16_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l16_k2lo, l16_knl);
    let k2h = rmsnorm(h @ l16_k2hi, l16_knh);
    let v2l = h @ l16_v2lo;
    let v2h = h @ l16_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l16_k3lo, l16_knl);
    let k3h = rmsnorm(h @ l16_k3hi, l16_knh);
    let v3l = h @ l16_v3lo;
    let v3h = h @ l16_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l16_k4lo, l16_knl);
    let k4h = rmsnorm(h @ l16_k4hi, l16_knh);
    let v4l = h @ l16_v4lo;
    let v4h = h @ l16_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l16_k5lo, l16_knl);
    let k5h = rmsnorm(h @ l16_k5hi, l16_knh);
    let v5l = h @ l16_v5lo;
    let v5h = h @ l16_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l16_k6lo, l16_knl);
    let k6h = rmsnorm(h @ l16_k6hi, l16_knh);
    let v6l = h @ l16_v6lo;
    let v6h = h @ l16_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l16_k7lo, l16_knl);
    let k7h = rmsnorm(h @ l16_k7hi, l16_knh);
    let v7l = h @ l16_v7lo;
    let v7h = h @ l16_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l16_o0lo + c0h @ l16_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l16_o1lo + c1h @ l16_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l16_o2lo + c2h @ l16_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l16_o3lo + c3h @ l16_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l16_o4lo + c4h @ l16_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l16_o5lo + c5h @ l16_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l16_o6lo + c6h @ l16_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l16_o7lo + c7h @ l16_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l16_o8lo + c8h @ l16_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l16_o9lo + c9h @ l16_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l16_o10lo + c10h @ l16_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l16_o11lo + c11h @ l16_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l16_o12lo + c12h @ l16_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l16_o13lo + c13h @ l16_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l16_o14lo + c14h @ l16_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l16_o15lo + c15h @ l16_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l16_ln2);
    let f = r + (silu(n @ l16_wg) * (n @ l16_wu)) @ l16_wd;
    f
}

fn l17(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l17_ln1);
    let q0l = rmsnorm(h @ l17_q0lo, l17_qnl);
    let q0h = rmsnorm(h @ l17_q0hi, l17_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l17_q1lo, l17_qnl);
    let q1h = rmsnorm(h @ l17_q1hi, l17_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l17_q2lo, l17_qnl);
    let q2h = rmsnorm(h @ l17_q2hi, l17_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l17_q3lo, l17_qnl);
    let q3h = rmsnorm(h @ l17_q3hi, l17_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l17_q4lo, l17_qnl);
    let q4h = rmsnorm(h @ l17_q4hi, l17_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l17_q5lo, l17_qnl);
    let q5h = rmsnorm(h @ l17_q5hi, l17_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l17_q6lo, l17_qnl);
    let q6h = rmsnorm(h @ l17_q6hi, l17_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l17_q7lo, l17_qnl);
    let q7h = rmsnorm(h @ l17_q7hi, l17_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l17_q8lo, l17_qnl);
    let q8h = rmsnorm(h @ l17_q8hi, l17_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l17_q9lo, l17_qnl);
    let q9h = rmsnorm(h @ l17_q9hi, l17_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l17_q10lo, l17_qnl);
    let q10h = rmsnorm(h @ l17_q10hi, l17_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l17_q11lo, l17_qnl);
    let q11h = rmsnorm(h @ l17_q11hi, l17_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l17_q12lo, l17_qnl);
    let q12h = rmsnorm(h @ l17_q12hi, l17_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l17_q13lo, l17_qnl);
    let q13h = rmsnorm(h @ l17_q13hi, l17_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l17_q14lo, l17_qnl);
    let q14h = rmsnorm(h @ l17_q14hi, l17_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l17_q15lo, l17_qnl);
    let q15h = rmsnorm(h @ l17_q15hi, l17_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l17_k0lo, l17_knl);
    let k0h = rmsnorm(h @ l17_k0hi, l17_knh);
    let v0l = h @ l17_v0lo;
    let v0h = h @ l17_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l17_k1lo, l17_knl);
    let k1h = rmsnorm(h @ l17_k1hi, l17_knh);
    let v1l = h @ l17_v1lo;
    let v1h = h @ l17_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l17_k2lo, l17_knl);
    let k2h = rmsnorm(h @ l17_k2hi, l17_knh);
    let v2l = h @ l17_v2lo;
    let v2h = h @ l17_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l17_k3lo, l17_knl);
    let k3h = rmsnorm(h @ l17_k3hi, l17_knh);
    let v3l = h @ l17_v3lo;
    let v3h = h @ l17_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l17_k4lo, l17_knl);
    let k4h = rmsnorm(h @ l17_k4hi, l17_knh);
    let v4l = h @ l17_v4lo;
    let v4h = h @ l17_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l17_k5lo, l17_knl);
    let k5h = rmsnorm(h @ l17_k5hi, l17_knh);
    let v5l = h @ l17_v5lo;
    let v5h = h @ l17_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l17_k6lo, l17_knl);
    let k6h = rmsnorm(h @ l17_k6hi, l17_knh);
    let v6l = h @ l17_v6lo;
    let v6h = h @ l17_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l17_k7lo, l17_knl);
    let k7h = rmsnorm(h @ l17_k7hi, l17_knh);
    let v7l = h @ l17_v7lo;
    let v7h = h @ l17_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l17_o0lo + c0h @ l17_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l17_o1lo + c1h @ l17_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l17_o2lo + c2h @ l17_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l17_o3lo + c3h @ l17_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l17_o4lo + c4h @ l17_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l17_o5lo + c5h @ l17_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l17_o6lo + c6h @ l17_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l17_o7lo + c7h @ l17_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l17_o8lo + c8h @ l17_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l17_o9lo + c9h @ l17_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l17_o10lo + c10h @ l17_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l17_o11lo + c11h @ l17_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l17_o12lo + c12h @ l17_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l17_o13lo + c13h @ l17_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l17_o14lo + c14h @ l17_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l17_o15lo + c15h @ l17_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l17_ln2);
    let f = r + (silu(n @ l17_wg) * (n @ l17_wu)) @ l17_wd;
    f
}

fn l18(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l18_ln1);
    let q0l = rmsnorm(h @ l18_q0lo, l18_qnl);
    let q0h = rmsnorm(h @ l18_q0hi, l18_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l18_q1lo, l18_qnl);
    let q1h = rmsnorm(h @ l18_q1hi, l18_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l18_q2lo, l18_qnl);
    let q2h = rmsnorm(h @ l18_q2hi, l18_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l18_q3lo, l18_qnl);
    let q3h = rmsnorm(h @ l18_q3hi, l18_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l18_q4lo, l18_qnl);
    let q4h = rmsnorm(h @ l18_q4hi, l18_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l18_q5lo, l18_qnl);
    let q5h = rmsnorm(h @ l18_q5hi, l18_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l18_q6lo, l18_qnl);
    let q6h = rmsnorm(h @ l18_q6hi, l18_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l18_q7lo, l18_qnl);
    let q7h = rmsnorm(h @ l18_q7hi, l18_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l18_q8lo, l18_qnl);
    let q8h = rmsnorm(h @ l18_q8hi, l18_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l18_q9lo, l18_qnl);
    let q9h = rmsnorm(h @ l18_q9hi, l18_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l18_q10lo, l18_qnl);
    let q10h = rmsnorm(h @ l18_q10hi, l18_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l18_q11lo, l18_qnl);
    let q11h = rmsnorm(h @ l18_q11hi, l18_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l18_q12lo, l18_qnl);
    let q12h = rmsnorm(h @ l18_q12hi, l18_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l18_q13lo, l18_qnl);
    let q13h = rmsnorm(h @ l18_q13hi, l18_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l18_q14lo, l18_qnl);
    let q14h = rmsnorm(h @ l18_q14hi, l18_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l18_q15lo, l18_qnl);
    let q15h = rmsnorm(h @ l18_q15hi, l18_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l18_k0lo, l18_knl);
    let k0h = rmsnorm(h @ l18_k0hi, l18_knh);
    let v0l = h @ l18_v0lo;
    let v0h = h @ l18_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l18_k1lo, l18_knl);
    let k1h = rmsnorm(h @ l18_k1hi, l18_knh);
    let v1l = h @ l18_v1lo;
    let v1h = h @ l18_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l18_k2lo, l18_knl);
    let k2h = rmsnorm(h @ l18_k2hi, l18_knh);
    let v2l = h @ l18_v2lo;
    let v2h = h @ l18_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l18_k3lo, l18_knl);
    let k3h = rmsnorm(h @ l18_k3hi, l18_knh);
    let v3l = h @ l18_v3lo;
    let v3h = h @ l18_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l18_k4lo, l18_knl);
    let k4h = rmsnorm(h @ l18_k4hi, l18_knh);
    let v4l = h @ l18_v4lo;
    let v4h = h @ l18_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l18_k5lo, l18_knl);
    let k5h = rmsnorm(h @ l18_k5hi, l18_knh);
    let v5l = h @ l18_v5lo;
    let v5h = h @ l18_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l18_k6lo, l18_knl);
    let k6h = rmsnorm(h @ l18_k6hi, l18_knh);
    let v6l = h @ l18_v6lo;
    let v6h = h @ l18_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l18_k7lo, l18_knl);
    let k7h = rmsnorm(h @ l18_k7hi, l18_knh);
    let v7l = h @ l18_v7lo;
    let v7h = h @ l18_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l18_o0lo + c0h @ l18_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l18_o1lo + c1h @ l18_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l18_o2lo + c2h @ l18_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l18_o3lo + c3h @ l18_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l18_o4lo + c4h @ l18_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l18_o5lo + c5h @ l18_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l18_o6lo + c6h @ l18_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l18_o7lo + c7h @ l18_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l18_o8lo + c8h @ l18_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l18_o9lo + c9h @ l18_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l18_o10lo + c10h @ l18_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l18_o11lo + c11h @ l18_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l18_o12lo + c12h @ l18_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l18_o13lo + c13h @ l18_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l18_o14lo + c14h @ l18_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l18_o15lo + c15h @ l18_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l18_ln2);
    let f = r + (silu(n @ l18_wg) * (n @ l18_wu)) @ l18_wd;
    f
}

fn l19(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l19_ln1);
    let q0l = rmsnorm(h @ l19_q0lo, l19_qnl);
    let q0h = rmsnorm(h @ l19_q0hi, l19_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l19_q1lo, l19_qnl);
    let q1h = rmsnorm(h @ l19_q1hi, l19_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l19_q2lo, l19_qnl);
    let q2h = rmsnorm(h @ l19_q2hi, l19_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l19_q3lo, l19_qnl);
    let q3h = rmsnorm(h @ l19_q3hi, l19_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l19_q4lo, l19_qnl);
    let q4h = rmsnorm(h @ l19_q4hi, l19_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l19_q5lo, l19_qnl);
    let q5h = rmsnorm(h @ l19_q5hi, l19_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l19_q6lo, l19_qnl);
    let q6h = rmsnorm(h @ l19_q6hi, l19_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l19_q7lo, l19_qnl);
    let q7h = rmsnorm(h @ l19_q7hi, l19_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l19_q8lo, l19_qnl);
    let q8h = rmsnorm(h @ l19_q8hi, l19_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l19_q9lo, l19_qnl);
    let q9h = rmsnorm(h @ l19_q9hi, l19_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l19_q10lo, l19_qnl);
    let q10h = rmsnorm(h @ l19_q10hi, l19_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l19_q11lo, l19_qnl);
    let q11h = rmsnorm(h @ l19_q11hi, l19_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l19_q12lo, l19_qnl);
    let q12h = rmsnorm(h @ l19_q12hi, l19_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l19_q13lo, l19_qnl);
    let q13h = rmsnorm(h @ l19_q13hi, l19_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l19_q14lo, l19_qnl);
    let q14h = rmsnorm(h @ l19_q14hi, l19_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l19_q15lo, l19_qnl);
    let q15h = rmsnorm(h @ l19_q15hi, l19_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l19_k0lo, l19_knl);
    let k0h = rmsnorm(h @ l19_k0hi, l19_knh);
    let v0l = h @ l19_v0lo;
    let v0h = h @ l19_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l19_k1lo, l19_knl);
    let k1h = rmsnorm(h @ l19_k1hi, l19_knh);
    let v1l = h @ l19_v1lo;
    let v1h = h @ l19_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l19_k2lo, l19_knl);
    let k2h = rmsnorm(h @ l19_k2hi, l19_knh);
    let v2l = h @ l19_v2lo;
    let v2h = h @ l19_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l19_k3lo, l19_knl);
    let k3h = rmsnorm(h @ l19_k3hi, l19_knh);
    let v3l = h @ l19_v3lo;
    let v3h = h @ l19_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l19_k4lo, l19_knl);
    let k4h = rmsnorm(h @ l19_k4hi, l19_knh);
    let v4l = h @ l19_v4lo;
    let v4h = h @ l19_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l19_k5lo, l19_knl);
    let k5h = rmsnorm(h @ l19_k5hi, l19_knh);
    let v5l = h @ l19_v5lo;
    let v5h = h @ l19_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l19_k6lo, l19_knl);
    let k6h = rmsnorm(h @ l19_k6hi, l19_knh);
    let v6l = h @ l19_v6lo;
    let v6h = h @ l19_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l19_k7lo, l19_knl);
    let k7h = rmsnorm(h @ l19_k7hi, l19_knh);
    let v7l = h @ l19_v7lo;
    let v7h = h @ l19_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l19_o0lo + c0h @ l19_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l19_o1lo + c1h @ l19_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l19_o2lo + c2h @ l19_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l19_o3lo + c3h @ l19_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l19_o4lo + c4h @ l19_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l19_o5lo + c5h @ l19_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l19_o6lo + c6h @ l19_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l19_o7lo + c7h @ l19_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l19_o8lo + c8h @ l19_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l19_o9lo + c9h @ l19_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l19_o10lo + c10h @ l19_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l19_o11lo + c11h @ l19_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l19_o12lo + c12h @ l19_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l19_o13lo + c13h @ l19_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l19_o14lo + c14h @ l19_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l19_o15lo + c15h @ l19_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l19_ln2);
    let f = r + (silu(n @ l19_wg) * (n @ l19_wu)) @ l19_wd;
    f
}

fn l20(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l20_ln1);
    let q0l = rmsnorm(h @ l20_q0lo, l20_qnl);
    let q0h = rmsnorm(h @ l20_q0hi, l20_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l20_q1lo, l20_qnl);
    let q1h = rmsnorm(h @ l20_q1hi, l20_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l20_q2lo, l20_qnl);
    let q2h = rmsnorm(h @ l20_q2hi, l20_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l20_q3lo, l20_qnl);
    let q3h = rmsnorm(h @ l20_q3hi, l20_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l20_q4lo, l20_qnl);
    let q4h = rmsnorm(h @ l20_q4hi, l20_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l20_q5lo, l20_qnl);
    let q5h = rmsnorm(h @ l20_q5hi, l20_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l20_q6lo, l20_qnl);
    let q6h = rmsnorm(h @ l20_q6hi, l20_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l20_q7lo, l20_qnl);
    let q7h = rmsnorm(h @ l20_q7hi, l20_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l20_q8lo, l20_qnl);
    let q8h = rmsnorm(h @ l20_q8hi, l20_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l20_q9lo, l20_qnl);
    let q9h = rmsnorm(h @ l20_q9hi, l20_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l20_q10lo, l20_qnl);
    let q10h = rmsnorm(h @ l20_q10hi, l20_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l20_q11lo, l20_qnl);
    let q11h = rmsnorm(h @ l20_q11hi, l20_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l20_q12lo, l20_qnl);
    let q12h = rmsnorm(h @ l20_q12hi, l20_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l20_q13lo, l20_qnl);
    let q13h = rmsnorm(h @ l20_q13hi, l20_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l20_q14lo, l20_qnl);
    let q14h = rmsnorm(h @ l20_q14hi, l20_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l20_q15lo, l20_qnl);
    let q15h = rmsnorm(h @ l20_q15hi, l20_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l20_k0lo, l20_knl);
    let k0h = rmsnorm(h @ l20_k0hi, l20_knh);
    let v0l = h @ l20_v0lo;
    let v0h = h @ l20_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l20_k1lo, l20_knl);
    let k1h = rmsnorm(h @ l20_k1hi, l20_knh);
    let v1l = h @ l20_v1lo;
    let v1h = h @ l20_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l20_k2lo, l20_knl);
    let k2h = rmsnorm(h @ l20_k2hi, l20_knh);
    let v2l = h @ l20_v2lo;
    let v2h = h @ l20_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l20_k3lo, l20_knl);
    let k3h = rmsnorm(h @ l20_k3hi, l20_knh);
    let v3l = h @ l20_v3lo;
    let v3h = h @ l20_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l20_k4lo, l20_knl);
    let k4h = rmsnorm(h @ l20_k4hi, l20_knh);
    let v4l = h @ l20_v4lo;
    let v4h = h @ l20_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l20_k5lo, l20_knl);
    let k5h = rmsnorm(h @ l20_k5hi, l20_knh);
    let v5l = h @ l20_v5lo;
    let v5h = h @ l20_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l20_k6lo, l20_knl);
    let k6h = rmsnorm(h @ l20_k6hi, l20_knh);
    let v6l = h @ l20_v6lo;
    let v6h = h @ l20_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l20_k7lo, l20_knl);
    let k7h = rmsnorm(h @ l20_k7hi, l20_knh);
    let v7l = h @ l20_v7lo;
    let v7h = h @ l20_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l20_o0lo + c0h @ l20_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l20_o1lo + c1h @ l20_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l20_o2lo + c2h @ l20_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l20_o3lo + c3h @ l20_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l20_o4lo + c4h @ l20_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l20_o5lo + c5h @ l20_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l20_o6lo + c6h @ l20_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l20_o7lo + c7h @ l20_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l20_o8lo + c8h @ l20_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l20_o9lo + c9h @ l20_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l20_o10lo + c10h @ l20_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l20_o11lo + c11h @ l20_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l20_o12lo + c12h @ l20_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l20_o13lo + c13h @ l20_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l20_o14lo + c14h @ l20_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l20_o15lo + c15h @ l20_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l20_ln2);
    let f = r + (silu(n @ l20_wg) * (n @ l20_wu)) @ l20_wd;
    f
}

fn l21(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l21_ln1);
    let q0l = rmsnorm(h @ l21_q0lo, l21_qnl);
    let q0h = rmsnorm(h @ l21_q0hi, l21_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l21_q1lo, l21_qnl);
    let q1h = rmsnorm(h @ l21_q1hi, l21_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l21_q2lo, l21_qnl);
    let q2h = rmsnorm(h @ l21_q2hi, l21_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l21_q3lo, l21_qnl);
    let q3h = rmsnorm(h @ l21_q3hi, l21_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l21_q4lo, l21_qnl);
    let q4h = rmsnorm(h @ l21_q4hi, l21_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l21_q5lo, l21_qnl);
    let q5h = rmsnorm(h @ l21_q5hi, l21_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l21_q6lo, l21_qnl);
    let q6h = rmsnorm(h @ l21_q6hi, l21_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l21_q7lo, l21_qnl);
    let q7h = rmsnorm(h @ l21_q7hi, l21_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l21_q8lo, l21_qnl);
    let q8h = rmsnorm(h @ l21_q8hi, l21_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l21_q9lo, l21_qnl);
    let q9h = rmsnorm(h @ l21_q9hi, l21_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l21_q10lo, l21_qnl);
    let q10h = rmsnorm(h @ l21_q10hi, l21_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l21_q11lo, l21_qnl);
    let q11h = rmsnorm(h @ l21_q11hi, l21_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l21_q12lo, l21_qnl);
    let q12h = rmsnorm(h @ l21_q12hi, l21_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l21_q13lo, l21_qnl);
    let q13h = rmsnorm(h @ l21_q13hi, l21_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l21_q14lo, l21_qnl);
    let q14h = rmsnorm(h @ l21_q14hi, l21_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l21_q15lo, l21_qnl);
    let q15h = rmsnorm(h @ l21_q15hi, l21_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l21_k0lo, l21_knl);
    let k0h = rmsnorm(h @ l21_k0hi, l21_knh);
    let v0l = h @ l21_v0lo;
    let v0h = h @ l21_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l21_k1lo, l21_knl);
    let k1h = rmsnorm(h @ l21_k1hi, l21_knh);
    let v1l = h @ l21_v1lo;
    let v1h = h @ l21_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l21_k2lo, l21_knl);
    let k2h = rmsnorm(h @ l21_k2hi, l21_knh);
    let v2l = h @ l21_v2lo;
    let v2h = h @ l21_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l21_k3lo, l21_knl);
    let k3h = rmsnorm(h @ l21_k3hi, l21_knh);
    let v3l = h @ l21_v3lo;
    let v3h = h @ l21_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l21_k4lo, l21_knl);
    let k4h = rmsnorm(h @ l21_k4hi, l21_knh);
    let v4l = h @ l21_v4lo;
    let v4h = h @ l21_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l21_k5lo, l21_knl);
    let k5h = rmsnorm(h @ l21_k5hi, l21_knh);
    let v5l = h @ l21_v5lo;
    let v5h = h @ l21_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l21_k6lo, l21_knl);
    let k6h = rmsnorm(h @ l21_k6hi, l21_knh);
    let v6l = h @ l21_v6lo;
    let v6h = h @ l21_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l21_k7lo, l21_knl);
    let k7h = rmsnorm(h @ l21_k7hi, l21_knh);
    let v7l = h @ l21_v7lo;
    let v7h = h @ l21_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l21_o0lo + c0h @ l21_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l21_o1lo + c1h @ l21_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l21_o2lo + c2h @ l21_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l21_o3lo + c3h @ l21_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l21_o4lo + c4h @ l21_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l21_o5lo + c5h @ l21_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l21_o6lo + c6h @ l21_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l21_o7lo + c7h @ l21_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l21_o8lo + c8h @ l21_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l21_o9lo + c9h @ l21_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l21_o10lo + c10h @ l21_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l21_o11lo + c11h @ l21_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l21_o12lo + c12h @ l21_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l21_o13lo + c13h @ l21_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l21_o14lo + c14h @ l21_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l21_o15lo + c15h @ l21_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l21_ln2);
    let f = r + (silu(n @ l21_wg) * (n @ l21_wu)) @ l21_wd;
    f
}

fn l22(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l22_ln1);
    let q0l = rmsnorm(h @ l22_q0lo, l22_qnl);
    let q0h = rmsnorm(h @ l22_q0hi, l22_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l22_q1lo, l22_qnl);
    let q1h = rmsnorm(h @ l22_q1hi, l22_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l22_q2lo, l22_qnl);
    let q2h = rmsnorm(h @ l22_q2hi, l22_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l22_q3lo, l22_qnl);
    let q3h = rmsnorm(h @ l22_q3hi, l22_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l22_q4lo, l22_qnl);
    let q4h = rmsnorm(h @ l22_q4hi, l22_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l22_q5lo, l22_qnl);
    let q5h = rmsnorm(h @ l22_q5hi, l22_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l22_q6lo, l22_qnl);
    let q6h = rmsnorm(h @ l22_q6hi, l22_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l22_q7lo, l22_qnl);
    let q7h = rmsnorm(h @ l22_q7hi, l22_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l22_q8lo, l22_qnl);
    let q8h = rmsnorm(h @ l22_q8hi, l22_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l22_q9lo, l22_qnl);
    let q9h = rmsnorm(h @ l22_q9hi, l22_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l22_q10lo, l22_qnl);
    let q10h = rmsnorm(h @ l22_q10hi, l22_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l22_q11lo, l22_qnl);
    let q11h = rmsnorm(h @ l22_q11hi, l22_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l22_q12lo, l22_qnl);
    let q12h = rmsnorm(h @ l22_q12hi, l22_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l22_q13lo, l22_qnl);
    let q13h = rmsnorm(h @ l22_q13hi, l22_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l22_q14lo, l22_qnl);
    let q14h = rmsnorm(h @ l22_q14hi, l22_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l22_q15lo, l22_qnl);
    let q15h = rmsnorm(h @ l22_q15hi, l22_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l22_k0lo, l22_knl);
    let k0h = rmsnorm(h @ l22_k0hi, l22_knh);
    let v0l = h @ l22_v0lo;
    let v0h = h @ l22_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l22_k1lo, l22_knl);
    let k1h = rmsnorm(h @ l22_k1hi, l22_knh);
    let v1l = h @ l22_v1lo;
    let v1h = h @ l22_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l22_k2lo, l22_knl);
    let k2h = rmsnorm(h @ l22_k2hi, l22_knh);
    let v2l = h @ l22_v2lo;
    let v2h = h @ l22_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l22_k3lo, l22_knl);
    let k3h = rmsnorm(h @ l22_k3hi, l22_knh);
    let v3l = h @ l22_v3lo;
    let v3h = h @ l22_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l22_k4lo, l22_knl);
    let k4h = rmsnorm(h @ l22_k4hi, l22_knh);
    let v4l = h @ l22_v4lo;
    let v4h = h @ l22_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l22_k5lo, l22_knl);
    let k5h = rmsnorm(h @ l22_k5hi, l22_knh);
    let v5l = h @ l22_v5lo;
    let v5h = h @ l22_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l22_k6lo, l22_knl);
    let k6h = rmsnorm(h @ l22_k6hi, l22_knh);
    let v6l = h @ l22_v6lo;
    let v6h = h @ l22_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l22_k7lo, l22_knl);
    let k7h = rmsnorm(h @ l22_k7hi, l22_knh);
    let v7l = h @ l22_v7lo;
    let v7h = h @ l22_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l22_o0lo + c0h @ l22_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l22_o1lo + c1h @ l22_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l22_o2lo + c2h @ l22_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l22_o3lo + c3h @ l22_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l22_o4lo + c4h @ l22_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l22_o5lo + c5h @ l22_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l22_o6lo + c6h @ l22_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l22_o7lo + c7h @ l22_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l22_o8lo + c8h @ l22_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l22_o9lo + c9h @ l22_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l22_o10lo + c10h @ l22_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l22_o11lo + c11h @ l22_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l22_o12lo + c12h @ l22_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l22_o13lo + c13h @ l22_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l22_o14lo + c14h @ l22_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l22_o15lo + c15h @ l22_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l22_ln2);
    let f = r + (silu(n @ l22_wg) * (n @ l22_wu)) @ l22_wd;
    f
}

fn l23(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l23_ln1);
    let q0l = rmsnorm(h @ l23_q0lo, l23_qnl);
    let q0h = rmsnorm(h @ l23_q0hi, l23_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l23_q1lo, l23_qnl);
    let q1h = rmsnorm(h @ l23_q1hi, l23_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l23_q2lo, l23_qnl);
    let q2h = rmsnorm(h @ l23_q2hi, l23_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l23_q3lo, l23_qnl);
    let q3h = rmsnorm(h @ l23_q3hi, l23_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l23_q4lo, l23_qnl);
    let q4h = rmsnorm(h @ l23_q4hi, l23_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l23_q5lo, l23_qnl);
    let q5h = rmsnorm(h @ l23_q5hi, l23_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l23_q6lo, l23_qnl);
    let q6h = rmsnorm(h @ l23_q6hi, l23_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l23_q7lo, l23_qnl);
    let q7h = rmsnorm(h @ l23_q7hi, l23_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l23_q8lo, l23_qnl);
    let q8h = rmsnorm(h @ l23_q8hi, l23_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l23_q9lo, l23_qnl);
    let q9h = rmsnorm(h @ l23_q9hi, l23_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l23_q10lo, l23_qnl);
    let q10h = rmsnorm(h @ l23_q10hi, l23_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l23_q11lo, l23_qnl);
    let q11h = rmsnorm(h @ l23_q11hi, l23_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l23_q12lo, l23_qnl);
    let q12h = rmsnorm(h @ l23_q12hi, l23_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l23_q13lo, l23_qnl);
    let q13h = rmsnorm(h @ l23_q13hi, l23_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l23_q14lo, l23_qnl);
    let q14h = rmsnorm(h @ l23_q14hi, l23_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l23_q15lo, l23_qnl);
    let q15h = rmsnorm(h @ l23_q15hi, l23_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l23_k0lo, l23_knl);
    let k0h = rmsnorm(h @ l23_k0hi, l23_knh);
    let v0l = h @ l23_v0lo;
    let v0h = h @ l23_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l23_k1lo, l23_knl);
    let k1h = rmsnorm(h @ l23_k1hi, l23_knh);
    let v1l = h @ l23_v1lo;
    let v1h = h @ l23_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l23_k2lo, l23_knl);
    let k2h = rmsnorm(h @ l23_k2hi, l23_knh);
    let v2l = h @ l23_v2lo;
    let v2h = h @ l23_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l23_k3lo, l23_knl);
    let k3h = rmsnorm(h @ l23_k3hi, l23_knh);
    let v3l = h @ l23_v3lo;
    let v3h = h @ l23_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l23_k4lo, l23_knl);
    let k4h = rmsnorm(h @ l23_k4hi, l23_knh);
    let v4l = h @ l23_v4lo;
    let v4h = h @ l23_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l23_k5lo, l23_knl);
    let k5h = rmsnorm(h @ l23_k5hi, l23_knh);
    let v5l = h @ l23_v5lo;
    let v5h = h @ l23_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l23_k6lo, l23_knl);
    let k6h = rmsnorm(h @ l23_k6hi, l23_knh);
    let v6l = h @ l23_v6lo;
    let v6h = h @ l23_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l23_k7lo, l23_knl);
    let k7h = rmsnorm(h @ l23_k7hi, l23_knh);
    let v7l = h @ l23_v7lo;
    let v7h = h @ l23_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l23_o0lo + c0h @ l23_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l23_o1lo + c1h @ l23_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l23_o2lo + c2h @ l23_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l23_o3lo + c3h @ l23_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l23_o4lo + c4h @ l23_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l23_o5lo + c5h @ l23_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l23_o6lo + c6h @ l23_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l23_o7lo + c7h @ l23_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l23_o8lo + c8h @ l23_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l23_o9lo + c9h @ l23_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l23_o10lo + c10h @ l23_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l23_o11lo + c11h @ l23_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l23_o12lo + c12h @ l23_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l23_o13lo + c13h @ l23_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l23_o14lo + c14h @ l23_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l23_o15lo + c15h @ l23_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l23_ln2);
    let f = r + (silu(n @ l23_wg) * (n @ l23_wu)) @ l23_wd;
    f
}

fn l24(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l24_ln1);
    let q0l = rmsnorm(h @ l24_q0lo, l24_qnl);
    let q0h = rmsnorm(h @ l24_q0hi, l24_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l24_q1lo, l24_qnl);
    let q1h = rmsnorm(h @ l24_q1hi, l24_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l24_q2lo, l24_qnl);
    let q2h = rmsnorm(h @ l24_q2hi, l24_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l24_q3lo, l24_qnl);
    let q3h = rmsnorm(h @ l24_q3hi, l24_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l24_q4lo, l24_qnl);
    let q4h = rmsnorm(h @ l24_q4hi, l24_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l24_q5lo, l24_qnl);
    let q5h = rmsnorm(h @ l24_q5hi, l24_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l24_q6lo, l24_qnl);
    let q6h = rmsnorm(h @ l24_q6hi, l24_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l24_q7lo, l24_qnl);
    let q7h = rmsnorm(h @ l24_q7hi, l24_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l24_q8lo, l24_qnl);
    let q8h = rmsnorm(h @ l24_q8hi, l24_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l24_q9lo, l24_qnl);
    let q9h = rmsnorm(h @ l24_q9hi, l24_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l24_q10lo, l24_qnl);
    let q10h = rmsnorm(h @ l24_q10hi, l24_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l24_q11lo, l24_qnl);
    let q11h = rmsnorm(h @ l24_q11hi, l24_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l24_q12lo, l24_qnl);
    let q12h = rmsnorm(h @ l24_q12hi, l24_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l24_q13lo, l24_qnl);
    let q13h = rmsnorm(h @ l24_q13hi, l24_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l24_q14lo, l24_qnl);
    let q14h = rmsnorm(h @ l24_q14hi, l24_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l24_q15lo, l24_qnl);
    let q15h = rmsnorm(h @ l24_q15hi, l24_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l24_k0lo, l24_knl);
    let k0h = rmsnorm(h @ l24_k0hi, l24_knh);
    let v0l = h @ l24_v0lo;
    let v0h = h @ l24_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l24_k1lo, l24_knl);
    let k1h = rmsnorm(h @ l24_k1hi, l24_knh);
    let v1l = h @ l24_v1lo;
    let v1h = h @ l24_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l24_k2lo, l24_knl);
    let k2h = rmsnorm(h @ l24_k2hi, l24_knh);
    let v2l = h @ l24_v2lo;
    let v2h = h @ l24_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l24_k3lo, l24_knl);
    let k3h = rmsnorm(h @ l24_k3hi, l24_knh);
    let v3l = h @ l24_v3lo;
    let v3h = h @ l24_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l24_k4lo, l24_knl);
    let k4h = rmsnorm(h @ l24_k4hi, l24_knh);
    let v4l = h @ l24_v4lo;
    let v4h = h @ l24_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l24_k5lo, l24_knl);
    let k5h = rmsnorm(h @ l24_k5hi, l24_knh);
    let v5l = h @ l24_v5lo;
    let v5h = h @ l24_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l24_k6lo, l24_knl);
    let k6h = rmsnorm(h @ l24_k6hi, l24_knh);
    let v6l = h @ l24_v6lo;
    let v6h = h @ l24_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l24_k7lo, l24_knl);
    let k7h = rmsnorm(h @ l24_k7hi, l24_knh);
    let v7l = h @ l24_v7lo;
    let v7h = h @ l24_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l24_o0lo + c0h @ l24_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l24_o1lo + c1h @ l24_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l24_o2lo + c2h @ l24_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l24_o3lo + c3h @ l24_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l24_o4lo + c4h @ l24_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l24_o5lo + c5h @ l24_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l24_o6lo + c6h @ l24_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l24_o7lo + c7h @ l24_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l24_o8lo + c8h @ l24_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l24_o9lo + c9h @ l24_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l24_o10lo + c10h @ l24_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l24_o11lo + c11h @ l24_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l24_o12lo + c12h @ l24_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l24_o13lo + c13h @ l24_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l24_o14lo + c14h @ l24_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l24_o15lo + c15h @ l24_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l24_ln2);
    let f = r + (silu(n @ l24_wg) * (n @ l24_wu)) @ l24_wd;
    f
}

fn l25(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l25_ln1);
    let q0l = rmsnorm(h @ l25_q0lo, l25_qnl);
    let q0h = rmsnorm(h @ l25_q0hi, l25_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l25_q1lo, l25_qnl);
    let q1h = rmsnorm(h @ l25_q1hi, l25_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l25_q2lo, l25_qnl);
    let q2h = rmsnorm(h @ l25_q2hi, l25_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l25_q3lo, l25_qnl);
    let q3h = rmsnorm(h @ l25_q3hi, l25_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l25_q4lo, l25_qnl);
    let q4h = rmsnorm(h @ l25_q4hi, l25_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l25_q5lo, l25_qnl);
    let q5h = rmsnorm(h @ l25_q5hi, l25_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l25_q6lo, l25_qnl);
    let q6h = rmsnorm(h @ l25_q6hi, l25_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l25_q7lo, l25_qnl);
    let q7h = rmsnorm(h @ l25_q7hi, l25_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l25_q8lo, l25_qnl);
    let q8h = rmsnorm(h @ l25_q8hi, l25_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l25_q9lo, l25_qnl);
    let q9h = rmsnorm(h @ l25_q9hi, l25_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l25_q10lo, l25_qnl);
    let q10h = rmsnorm(h @ l25_q10hi, l25_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l25_q11lo, l25_qnl);
    let q11h = rmsnorm(h @ l25_q11hi, l25_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l25_q12lo, l25_qnl);
    let q12h = rmsnorm(h @ l25_q12hi, l25_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l25_q13lo, l25_qnl);
    let q13h = rmsnorm(h @ l25_q13hi, l25_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l25_q14lo, l25_qnl);
    let q14h = rmsnorm(h @ l25_q14hi, l25_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l25_q15lo, l25_qnl);
    let q15h = rmsnorm(h @ l25_q15hi, l25_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l25_k0lo, l25_knl);
    let k0h = rmsnorm(h @ l25_k0hi, l25_knh);
    let v0l = h @ l25_v0lo;
    let v0h = h @ l25_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l25_k1lo, l25_knl);
    let k1h = rmsnorm(h @ l25_k1hi, l25_knh);
    let v1l = h @ l25_v1lo;
    let v1h = h @ l25_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l25_k2lo, l25_knl);
    let k2h = rmsnorm(h @ l25_k2hi, l25_knh);
    let v2l = h @ l25_v2lo;
    let v2h = h @ l25_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l25_k3lo, l25_knl);
    let k3h = rmsnorm(h @ l25_k3hi, l25_knh);
    let v3l = h @ l25_v3lo;
    let v3h = h @ l25_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l25_k4lo, l25_knl);
    let k4h = rmsnorm(h @ l25_k4hi, l25_knh);
    let v4l = h @ l25_v4lo;
    let v4h = h @ l25_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l25_k5lo, l25_knl);
    let k5h = rmsnorm(h @ l25_k5hi, l25_knh);
    let v5l = h @ l25_v5lo;
    let v5h = h @ l25_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l25_k6lo, l25_knl);
    let k6h = rmsnorm(h @ l25_k6hi, l25_knh);
    let v6l = h @ l25_v6lo;
    let v6h = h @ l25_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l25_k7lo, l25_knl);
    let k7h = rmsnorm(h @ l25_k7hi, l25_knh);
    let v7l = h @ l25_v7lo;
    let v7h = h @ l25_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l25_o0lo + c0h @ l25_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l25_o1lo + c1h @ l25_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l25_o2lo + c2h @ l25_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l25_o3lo + c3h @ l25_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l25_o4lo + c4h @ l25_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l25_o5lo + c5h @ l25_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l25_o6lo + c6h @ l25_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l25_o7lo + c7h @ l25_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l25_o8lo + c8h @ l25_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l25_o9lo + c9h @ l25_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l25_o10lo + c10h @ l25_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l25_o11lo + c11h @ l25_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l25_o12lo + c12h @ l25_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l25_o13lo + c13h @ l25_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l25_o14lo + c14h @ l25_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l25_o15lo + c15h @ l25_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l25_ln2);
    let f = r + (silu(n @ l25_wg) * (n @ l25_wu)) @ l25_wd;
    f
}

fn l26(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l26_ln1);
    let q0l = rmsnorm(h @ l26_q0lo, l26_qnl);
    let q0h = rmsnorm(h @ l26_q0hi, l26_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l26_q1lo, l26_qnl);
    let q1h = rmsnorm(h @ l26_q1hi, l26_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l26_q2lo, l26_qnl);
    let q2h = rmsnorm(h @ l26_q2hi, l26_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l26_q3lo, l26_qnl);
    let q3h = rmsnorm(h @ l26_q3hi, l26_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l26_q4lo, l26_qnl);
    let q4h = rmsnorm(h @ l26_q4hi, l26_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l26_q5lo, l26_qnl);
    let q5h = rmsnorm(h @ l26_q5hi, l26_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l26_q6lo, l26_qnl);
    let q6h = rmsnorm(h @ l26_q6hi, l26_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l26_q7lo, l26_qnl);
    let q7h = rmsnorm(h @ l26_q7hi, l26_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l26_q8lo, l26_qnl);
    let q8h = rmsnorm(h @ l26_q8hi, l26_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l26_q9lo, l26_qnl);
    let q9h = rmsnorm(h @ l26_q9hi, l26_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l26_q10lo, l26_qnl);
    let q10h = rmsnorm(h @ l26_q10hi, l26_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l26_q11lo, l26_qnl);
    let q11h = rmsnorm(h @ l26_q11hi, l26_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l26_q12lo, l26_qnl);
    let q12h = rmsnorm(h @ l26_q12hi, l26_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l26_q13lo, l26_qnl);
    let q13h = rmsnorm(h @ l26_q13hi, l26_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l26_q14lo, l26_qnl);
    let q14h = rmsnorm(h @ l26_q14hi, l26_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l26_q15lo, l26_qnl);
    let q15h = rmsnorm(h @ l26_q15hi, l26_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l26_k0lo, l26_knl);
    let k0h = rmsnorm(h @ l26_k0hi, l26_knh);
    let v0l = h @ l26_v0lo;
    let v0h = h @ l26_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l26_k1lo, l26_knl);
    let k1h = rmsnorm(h @ l26_k1hi, l26_knh);
    let v1l = h @ l26_v1lo;
    let v1h = h @ l26_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l26_k2lo, l26_knl);
    let k2h = rmsnorm(h @ l26_k2hi, l26_knh);
    let v2l = h @ l26_v2lo;
    let v2h = h @ l26_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l26_k3lo, l26_knl);
    let k3h = rmsnorm(h @ l26_k3hi, l26_knh);
    let v3l = h @ l26_v3lo;
    let v3h = h @ l26_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l26_k4lo, l26_knl);
    let k4h = rmsnorm(h @ l26_k4hi, l26_knh);
    let v4l = h @ l26_v4lo;
    let v4h = h @ l26_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l26_k5lo, l26_knl);
    let k5h = rmsnorm(h @ l26_k5hi, l26_knh);
    let v5l = h @ l26_v5lo;
    let v5h = h @ l26_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l26_k6lo, l26_knl);
    let k6h = rmsnorm(h @ l26_k6hi, l26_knh);
    let v6l = h @ l26_v6lo;
    let v6h = h @ l26_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l26_k7lo, l26_knl);
    let k7h = rmsnorm(h @ l26_k7hi, l26_knh);
    let v7l = h @ l26_v7lo;
    let v7h = h @ l26_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l26_o0lo + c0h @ l26_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l26_o1lo + c1h @ l26_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l26_o2lo + c2h @ l26_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l26_o3lo + c3h @ l26_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l26_o4lo + c4h @ l26_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l26_o5lo + c5h @ l26_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l26_o6lo + c6h @ l26_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l26_o7lo + c7h @ l26_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l26_o8lo + c8h @ l26_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l26_o9lo + c9h @ l26_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l26_o10lo + c10h @ l26_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l26_o11lo + c11h @ l26_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l26_o12lo + c12h @ l26_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l26_o13lo + c13h @ l26_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l26_o14lo + c14h @ l26_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l26_o15lo + c15h @ l26_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l26_ln2);
    let f = r + (silu(n @ l26_wg) * (n @ l26_wu)) @ l26_wd;
    f
}

fn l27(x: f16[1024], pos: f16[64], mask: f16[64], cos: f16[64], sin: f16[64]) -> f16[1024] {
    let h = rmsnorm(x, l27_ln1);
    let q0l = rmsnorm(h @ l27_q0lo, l27_qnl);
    let q0h = rmsnorm(h @ l27_q0hi, l27_qnh);
    let rq0l = q0l * cos - q0h * sin;
    let rq0h = q0h * cos + q0l * sin;
    let q1l = rmsnorm(h @ l27_q1lo, l27_qnl);
    let q1h = rmsnorm(h @ l27_q1hi, l27_qnh);
    let rq1l = q1l * cos - q1h * sin;
    let rq1h = q1h * cos + q1l * sin;
    let q2l = rmsnorm(h @ l27_q2lo, l27_qnl);
    let q2h = rmsnorm(h @ l27_q2hi, l27_qnh);
    let rq2l = q2l * cos - q2h * sin;
    let rq2h = q2h * cos + q2l * sin;
    let q3l = rmsnorm(h @ l27_q3lo, l27_qnl);
    let q3h = rmsnorm(h @ l27_q3hi, l27_qnh);
    let rq3l = q3l * cos - q3h * sin;
    let rq3h = q3h * cos + q3l * sin;
    let q4l = rmsnorm(h @ l27_q4lo, l27_qnl);
    let q4h = rmsnorm(h @ l27_q4hi, l27_qnh);
    let rq4l = q4l * cos - q4h * sin;
    let rq4h = q4h * cos + q4l * sin;
    let q5l = rmsnorm(h @ l27_q5lo, l27_qnl);
    let q5h = rmsnorm(h @ l27_q5hi, l27_qnh);
    let rq5l = q5l * cos - q5h * sin;
    let rq5h = q5h * cos + q5l * sin;
    let q6l = rmsnorm(h @ l27_q6lo, l27_qnl);
    let q6h = rmsnorm(h @ l27_q6hi, l27_qnh);
    let rq6l = q6l * cos - q6h * sin;
    let rq6h = q6h * cos + q6l * sin;
    let q7l = rmsnorm(h @ l27_q7lo, l27_qnl);
    let q7h = rmsnorm(h @ l27_q7hi, l27_qnh);
    let rq7l = q7l * cos - q7h * sin;
    let rq7h = q7h * cos + q7l * sin;
    let q8l = rmsnorm(h @ l27_q8lo, l27_qnl);
    let q8h = rmsnorm(h @ l27_q8hi, l27_qnh);
    let rq8l = q8l * cos - q8h * sin;
    let rq8h = q8h * cos + q8l * sin;
    let q9l = rmsnorm(h @ l27_q9lo, l27_qnl);
    let q9h = rmsnorm(h @ l27_q9hi, l27_qnh);
    let rq9l = q9l * cos - q9h * sin;
    let rq9h = q9h * cos + q9l * sin;
    let q10l = rmsnorm(h @ l27_q10lo, l27_qnl);
    let q10h = rmsnorm(h @ l27_q10hi, l27_qnh);
    let rq10l = q10l * cos - q10h * sin;
    let rq10h = q10h * cos + q10l * sin;
    let q11l = rmsnorm(h @ l27_q11lo, l27_qnl);
    let q11h = rmsnorm(h @ l27_q11hi, l27_qnh);
    let rq11l = q11l * cos - q11h * sin;
    let rq11h = q11h * cos + q11l * sin;
    let q12l = rmsnorm(h @ l27_q12lo, l27_qnl);
    let q12h = rmsnorm(h @ l27_q12hi, l27_qnh);
    let rq12l = q12l * cos - q12h * sin;
    let rq12h = q12h * cos + q12l * sin;
    let q13l = rmsnorm(h @ l27_q13lo, l27_qnl);
    let q13h = rmsnorm(h @ l27_q13hi, l27_qnh);
    let rq13l = q13l * cos - q13h * sin;
    let rq13h = q13h * cos + q13l * sin;
    let q14l = rmsnorm(h @ l27_q14lo, l27_qnl);
    let q14h = rmsnorm(h @ l27_q14hi, l27_qnh);
    let rq14l = q14l * cos - q14h * sin;
    let rq14h = q14h * cos + q14l * sin;
    let q15l = rmsnorm(h @ l27_q15lo, l27_qnl);
    let q15h = rmsnorm(h @ l27_q15hi, l27_qnh);
    let rq15l = q15l * cos - q15h * sin;
    let rq15h = q15h * cos + q15l * sin;
    let k0l = rmsnorm(h @ l27_k0lo, l27_knl);
    let k0h = rmsnorm(h @ l27_k0hi, l27_knh);
    let v0l = h @ l27_v0lo;
    let v0h = h @ l27_v0hi;
    kv.k0lo[pos] = k0l;
    kv.k0hi[pos] = k0h;
    kv.v0lo[pos] = v0l;
    kv.v0hi[pos] = v0h;
    let k1l = rmsnorm(h @ l27_k1lo, l27_knl);
    let k1h = rmsnorm(h @ l27_k1hi, l27_knh);
    let v1l = h @ l27_v1lo;
    let v1h = h @ l27_v1hi;
    kv.k1lo[pos] = k1l;
    kv.k1hi[pos] = k1h;
    kv.v1lo[pos] = v1l;
    kv.v1hi[pos] = v1h;
    let k2l = rmsnorm(h @ l27_k2lo, l27_knl);
    let k2h = rmsnorm(h @ l27_k2hi, l27_knh);
    let v2l = h @ l27_v2lo;
    let v2h = h @ l27_v2hi;
    kv.k2lo[pos] = k2l;
    kv.k2hi[pos] = k2h;
    kv.v2lo[pos] = v2l;
    kv.v2hi[pos] = v2h;
    let k3l = rmsnorm(h @ l27_k3lo, l27_knl);
    let k3h = rmsnorm(h @ l27_k3hi, l27_knh);
    let v3l = h @ l27_v3lo;
    let v3h = h @ l27_v3hi;
    kv.k3lo[pos] = k3l;
    kv.k3hi[pos] = k3h;
    kv.v3lo[pos] = v3l;
    kv.v3hi[pos] = v3h;
    let k4l = rmsnorm(h @ l27_k4lo, l27_knl);
    let k4h = rmsnorm(h @ l27_k4hi, l27_knh);
    let v4l = h @ l27_v4lo;
    let v4h = h @ l27_v4hi;
    kv.k4lo[pos] = k4l;
    kv.k4hi[pos] = k4h;
    kv.v4lo[pos] = v4l;
    kv.v4hi[pos] = v4h;
    let k5l = rmsnorm(h @ l27_k5lo, l27_knl);
    let k5h = rmsnorm(h @ l27_k5hi, l27_knh);
    let v5l = h @ l27_v5lo;
    let v5h = h @ l27_v5hi;
    kv.k5lo[pos] = k5l;
    kv.k5hi[pos] = k5h;
    kv.v5lo[pos] = v5l;
    kv.v5hi[pos] = v5h;
    let k6l = rmsnorm(h @ l27_k6lo, l27_knl);
    let k6h = rmsnorm(h @ l27_k6hi, l27_knh);
    let v6l = h @ l27_v6lo;
    let v6h = h @ l27_v6hi;
    kv.k6lo[pos] = k6l;
    kv.k6hi[pos] = k6h;
    kv.v6lo[pos] = v6l;
    kv.v6hi[pos] = v6h;
    let k7l = rmsnorm(h @ l27_k7lo, l27_knl);
    let k7h = rmsnorm(h @ l27_k7hi, l27_knh);
    let v7l = h @ l27_v7lo;
    let v7h = h @ l27_v7hi;
    kv.k7lo[pos] = k7l;
    kv.k7hi[pos] = k7h;
    kv.v7lo[pos] = v7l;
    kv.v7hi[pos] = v7h;
    let s0 = softmax((rq0l @ kv.k0lo.T + rq0h @ kv.k0hi.T) * 0.125 + mask);
    let c0l = s0 @ kv.v0lo;
    let c0h = s0 @ kv.v0hi;
    let a0 = c0l @ l27_o0lo + c0h @ l27_o0hi;
    let s1 = softmax((rq1l @ kv.k0lo.T + rq1h @ kv.k0hi.T) * 0.125 + mask);
    let c1l = s1 @ kv.v0lo;
    let c1h = s1 @ kv.v0hi;
    let a1 = c1l @ l27_o1lo + c1h @ l27_o1hi;
    let s2 = softmax((rq2l @ kv.k1lo.T + rq2h @ kv.k1hi.T) * 0.125 + mask);
    let c2l = s2 @ kv.v1lo;
    let c2h = s2 @ kv.v1hi;
    let a2 = c2l @ l27_o2lo + c2h @ l27_o2hi;
    let s3 = softmax((rq3l @ kv.k1lo.T + rq3h @ kv.k1hi.T) * 0.125 + mask);
    let c3l = s3 @ kv.v1lo;
    let c3h = s3 @ kv.v1hi;
    let a3 = c3l @ l27_o3lo + c3h @ l27_o3hi;
    let s4 = softmax((rq4l @ kv.k2lo.T + rq4h @ kv.k2hi.T) * 0.125 + mask);
    let c4l = s4 @ kv.v2lo;
    let c4h = s4 @ kv.v2hi;
    let a4 = c4l @ l27_o4lo + c4h @ l27_o4hi;
    let s5 = softmax((rq5l @ kv.k2lo.T + rq5h @ kv.k2hi.T) * 0.125 + mask);
    let c5l = s5 @ kv.v2lo;
    let c5h = s5 @ kv.v2hi;
    let a5 = c5l @ l27_o5lo + c5h @ l27_o5hi;
    let s6 = softmax((rq6l @ kv.k3lo.T + rq6h @ kv.k3hi.T) * 0.125 + mask);
    let c6l = s6 @ kv.v3lo;
    let c6h = s6 @ kv.v3hi;
    let a6 = c6l @ l27_o6lo + c6h @ l27_o6hi;
    let s7 = softmax((rq7l @ kv.k3lo.T + rq7h @ kv.k3hi.T) * 0.125 + mask);
    let c7l = s7 @ kv.v3lo;
    let c7h = s7 @ kv.v3hi;
    let a7 = c7l @ l27_o7lo + c7h @ l27_o7hi;
    let s8 = softmax((rq8l @ kv.k4lo.T + rq8h @ kv.k4hi.T) * 0.125 + mask);
    let c8l = s8 @ kv.v4lo;
    let c8h = s8 @ kv.v4hi;
    let a8 = c8l @ l27_o8lo + c8h @ l27_o8hi;
    let s9 = softmax((rq9l @ kv.k4lo.T + rq9h @ kv.k4hi.T) * 0.125 + mask);
    let c9l = s9 @ kv.v4lo;
    let c9h = s9 @ kv.v4hi;
    let a9 = c9l @ l27_o9lo + c9h @ l27_o9hi;
    let s10 = softmax((rq10l @ kv.k5lo.T + rq10h @ kv.k5hi.T) * 0.125 + mask);
    let c10l = s10 @ kv.v5lo;
    let c10h = s10 @ kv.v5hi;
    let a10 = c10l @ l27_o10lo + c10h @ l27_o10hi;
    let s11 = softmax((rq11l @ kv.k5lo.T + rq11h @ kv.k5hi.T) * 0.125 + mask);
    let c11l = s11 @ kv.v5lo;
    let c11h = s11 @ kv.v5hi;
    let a11 = c11l @ l27_o11lo + c11h @ l27_o11hi;
    let s12 = softmax((rq12l @ kv.k6lo.T + rq12h @ kv.k6hi.T) * 0.125 + mask);
    let c12l = s12 @ kv.v6lo;
    let c12h = s12 @ kv.v6hi;
    let a12 = c12l @ l27_o12lo + c12h @ l27_o12hi;
    let s13 = softmax((rq13l @ kv.k6lo.T + rq13h @ kv.k6hi.T) * 0.125 + mask);
    let c13l = s13 @ kv.v6lo;
    let c13h = s13 @ kv.v6hi;
    let a13 = c13l @ l27_o13lo + c13h @ l27_o13hi;
    let s14 = softmax((rq14l @ kv.k7lo.T + rq14h @ kv.k7hi.T) * 0.125 + mask);
    let c14l = s14 @ kv.v7lo;
    let c14h = s14 @ kv.v7hi;
    let a14 = c14l @ l27_o14lo + c14h @ l27_o14hi;
    let s15 = softmax((rq15l @ kv.k7lo.T + rq15h @ kv.k7hi.T) * 0.125 + mask);
    let c15l = s15 @ kv.v7lo;
    let c15h = s15 @ kv.v7hi;
    let a15 = c15l @ l27_o15lo + c15h @ l27_o15hi;
    let att = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
    let r = x + att;
    let n = rmsnorm(r, l27_ln2);
    let f = r + (silu(n @ l27_wg) * (n @ l27_wu)) @ l27_wd;
    f
}

fn head(h: f16[1024]) -> f16[2048] {
    rmsnorm(h, lnf) @ lmh
}

host fn main() -> i64 {
    let me = self_token();
    witness("born", me);
    print("qwen3-0.6b organism", me.id);
    var tok = 151643;                       // BOS
    var x = emb[tok];                       // f16[1024] — host gather
    var step = 0;
    while step < 8 {
        let p = onehot(step % 64, 64);
        let m = causal(step % 64, 64);
        let c = rope_cos(step, 64);
        let s = rope_sin(step, 64);
        x = l0(x, p, m, c, s);
        x = l1(x, p, m, c, s);
        x = l2(x, p, m, c, s);
        x = l3(x, p, m, c, s);
        x = l4(x, p, m, c, s);
        x = l5(x, p, m, c, s);
        x = l6(x, p, m, c, s);
        x = l7(x, p, m, c, s);
        x = l8(x, p, m, c, s);
        x = l9(x, p, m, c, s);
        x = l10(x, p, m, c, s);
        x = l11(x, p, m, c, s);
        x = l12(x, p, m, c, s);
        x = l13(x, p, m, c, s);
        x = l14(x, p, m, c, s);
        x = l15(x, p, m, c, s);
        x = l16(x, p, m, c, s);
        x = l17(x, p, m, c, s);
        x = l18(x, p, m, c, s);
        x = l19(x, p, m, c, s);
        x = l20(x, p, m, c, s);
        x = l21(x, p, m, c, s);
        x = l22(x, p, m, c, s);
        x = l23(x, p, m, c, s);
        x = l24(x, p, m, c, s);
        x = l25(x, p, m, c, s);
        x = l26(x, p, m, c, s);
        x = l27(x, p, m, c, s);
        let lg = head(x);
        tok = argmax(lg);
        print("step", step, "tok", tok);
        witness("tok", tok);
        x = emb[tok];
        step += 1;
    }
    witness("done", step);
    return tok;
}
