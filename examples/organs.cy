// organs.cy — shared sovereign organ types & deterministic helpers,
// imported by sovereign.cy via `import "organs.cy"`.

enum Posture {
    Nominal,
    Elevated(str),
    Locked(str, i64),
}

// `reflex` (`axon`) — statically verified by the compiler to be
// deterministic (`det`), capability-free (`cap` = none), and
// non-steering (cannot call ANE programs, migrate, or die).
// Every non-Pass firing is automatically hash-chained into witness.jsonl.
reflex perimeter_guard(fw: dfa, packet: str, temp: f64) -> Reflex {
    let hit = dfa_scan(fw, packet);
    if hit >= 0 {
        return Reflex::Quarantine("firewall hit pattern #" + str(hit));
    }
    if temp > 95.0 {
        return Reflex::Brake("thermal ceiling exceeded");
    }
    if temp > 80.0 {
        return Reflex::Alert("thermal elevated");
    }
    return Reflex::Pass;
}

host det fn classify_reflex(r: Reflex) -> Posture {
    match r {
        Reflex::Pass => {
            return Posture::Nominal;
        }
        Reflex::Alert(msg) => {
            return Posture::Elevated(msg);
        }
        Reflex::Quarantine(msg) => {
            return Posture::Locked(msg, 1);
        }
        Reflex::Brake(msg) => {
            return Posture::Locked(msg, 2);
        }
    }
}
