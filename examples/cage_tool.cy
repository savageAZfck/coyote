// cage_tool.cy — Coyote organism compiled to `.wasm` (`coyote build --wasm`)
// for execution inside Bad Apple's `WasmCage`.

invariant wasm_cage_compatible

reflex inspect_payload(input: str) -> Reflex {
    if contains(input, "exfil") {
        return Reflex::Quarantine("blocked exfil payload");
    }
    return Reflex::Pass;
}

host det fn fib(n: i64) -> i64 {
    if n < 2 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

host det fn main(input: str) -> str {
    let verdict = inspect_payload(input);
    match verdict {
        Reflex::Pass => {
            let f = fib(10);
            witness("fib10", f);
            return "CLEAN:" + input + ":fib=" + str(f);
        }
        Reflex::Quarantine(reason) => {
            witness("quarantine", reason);
            return "QUARANTINED:" + reason;
        }
        _ => {
            return "HALT";
        }
    }
}
