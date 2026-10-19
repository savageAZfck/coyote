// sovereign.cy — the full sovereign stack in one common language.
//
// Exercises:
// - `import "organs.cy"` multi-file module composition
// - `enum` (unit + payload variants) & exhaustive `match`
// - `{K: V}` maps (`has`, `keys`, indexing, mutation) + string ops
// - `reflex` (`axon`): deterministic, zero-cap, auto-witnessed on firing
// - `dfa_compile` / `dfa_scan` (`macuahuitl` / `mil_machines` Aho-Corasick)
// - `quorum` (`tokala`): multi-seat vote with preserved minority dissent
// - `propose_patch` (`akicita`/`xipe`): protected-surface Hold vs Permit
// - `erase` (`camazotz`): certified erasure with decrypt-failure proof
// - `read_file` / `write_file` / `cortex_ask`: capability-gated (`cap`)
// - `det` vs `stoch` static determinism enforcement
// - ANE silicon dispatch (and `--cpu` universal fallback)

import "organs.cy"

cap patch, erase, fs, cortex
invariant reflexes_brake_never_steer
invariant minority_dissent_preserved
invariant sealed_keys_unnameable

param w_encoder: frozen f16[64, 64]
param enclave_root: sealed f16[64]

fn encode(x: f16[64]) -> f16[64] {
    silu(x @ w_encoder) + x
}

// Explicit capability restriction: this function may ONLY use `patch`.
host det fn try_self_patch() -> i64 cap patch {
    // 1. Attempting to patch a protected surface (`hostck.rs`) -> Hold
    let d1 = propose_patch(
        "src/hostck.rs",
        "sealed.contains(s)",
        "false",
        "weaken sealed boundary"
    );
    var score = 0;
    match d1 {
        Decision::Permit => {
            score += 100;
        }
        Decision::Hold(why) => {
            print("marshal held protected surface:", why);
            score += 10;
        }
        Decision::Deny(why) => {
            print("marshal denied:", why);
        }
    }

    // 2. Bounded in-policy patch on an unprotected surface -> Permit
    let d2 = propose_patch(
        "src/feature.cy",
        "let step = 1;",
        "let step = 2;",
        "double decode stride"
    );
    match d2 {
        Decision::Permit => {
            print("marshal permitted bounded patch");
            score += 20;
        }
        Decision::Hold(why) => {
            print("unexpected hold:", why);
        }
        Decision::Deny(why) => {
            print("unexpected deny:", why);
        }
    }
    return score;
}

host stoch fn sample_thought(prompt: str) -> str cap cortex {
    let jitter = rand();
    let reply = cortex_ask(prompt);
    return slice_str(reply, 0, 14) + ":" + str(i64(jitter * 1000.0));
}

host fn main() -> i64 {
    let me = self_token();
    witness("born", me);

    // 1. Streaming Aho-Corasick firewall (`macuahuitl`) + Reflex arc (`axon`)
    let fw = dfa_compile(["exfil://", "DROP TABLE", "PRIVATE_KEY"]);
    let r_clean = perimeter_guard(fw, "status: nominal heartbeat", 42.0);
    let r_block = perimeter_guard(fw, "attempt exfil://secret", 42.0);
    let p1 = classify_reflex(r_clean);
    let p2 = classify_reflex(r_block);
    var posture_code = 0;
    match p1 {
        Posture::Nominal => { posture_code += 1; }
        _ => {}
    }
    match p2 {
        Posture::Nominal => {}
        Posture::Elevated(m) => { print("elevated:", m); }
        Posture::Locked(m, level) => {
            print("reflex locked:", m, "level", level);
            posture_code += level * 10;
        }
    }

    // 2. Dissenting council quorum (`tokala`)
    let votes = [
        Vote { seat: "Aurelius", approve: true, reason: "within constitutional bounds" },
        Vote { seat: "SunTzu", approve: true, reason: "terrain verified local" },
        Vote { seat: "Buffett", approve: false, reason: "prefer wider margin of safety" }
    ];
    let qv = quorum("deploy_adapter_v2", votes);
    print("quorum:", qv.approved, "yes:", qv.yes, "no:", qv.no, "dissent:", qv.dissent[0]);

    // 3. Bounded autonomy marshal (`akicita` / `xipe`)
    let marshal_score = try_self_patch();

    // 4. Certified erasure (`camazotz`)
    let cert = erase("subject_42_ephemeral", "sensitive session transcript");
    print("camazotz cert:", slice_str(cert, 0, 16));

    // 5. Maps `{K: V}` + string operations
    var metrics: {str: i64} = {};
    metrics["posture"] = posture_code;
    metrics["marshal"] = marshal_score;
    metrics["quorum_yes"] = qv.yes;
    if has(metrics, "marshal") {
        metrics["total"] = metrics["posture"] + metrics["marshal"] + metrics["quorum_yes"];
    }
    let ks = keys(metrics);
    let parts = split("my:data:my:silicon:my:lang", ":");
    print("map keys:", len(ks), "total:", metrics["total"], " motto:", parts[5]);

    // 6. Capability-gated filesystem + stochastic cortex call
    write_file("/tmp/coyote_sovereign_receipt.txt", "cert=" + slice_str(cert, 0, 16));
    let disk_check = read_file("/tmp/coyote_sovereign_receipt.txt");
    let thought = sample_thought("status check");
    if contains(disk_check, "cert=") {
        print("fs receipt verified,", thought);
    }

    // 7. Silicon dispatch (`encode` on ANE or `--cpu`)
    let x = onehot(0, 64);
    let y = encode(x);
    let top = argmax(y);
    witness("complete", metrics["total"] + top);
    return metrics["total"];
}
