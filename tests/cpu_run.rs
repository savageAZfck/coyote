//! `coyote run --cpu` must run on the CPU reference evaluator alone: no ANE
//! compile and no ANE load, so it works on Macs whose Neural Engine can't
//! load coyote's (M4/h16) binaries. Before the fix, the single-program
//! `run` path ignored `--cpu` and failed there with
//! `Program load failure (0x170004)`.

use std::path::Path;

#[test]
fn run_cpu_skips_ane() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let tmp = Path::new(env!("CARGO_TARGET_TMPDIR"));
    // inference example + a trainable one (exercises the param feedback loop)
    let cases: [(&str, &[&str]); 2] = [("ffn", &[]), ("train", &["--steps", "2"])];
    let mut failed = Vec::new();
    for (name, extra) in cases {
        let out = tmp.join(format!("cpu_run_{name}"));
        let _ = std::fs::remove_dir_all(&out);
        let src = root.join("examples").join(format!("{name}.cy"));
        let mut argv: Vec<String> = [
            "coyote",
            "run",
            src.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--cpu",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        argv.extend(extra.iter().map(|s| s.to_string()));
        if coyote::cli::run(&argv).is_err() {
            failed.push(format!("{name}: run --cpu returned Err"));
            continue;
        }
        let ane_bins: Vec<_> = std::fs::read_dir(&out)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "macho" || x == "hwx"))
            .collect();
        if !ane_bins.is_empty() {
            failed.push(format!(
                "{name}: ANE binaries emitted under --cpu: {ane_bins:?}"
            ));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}
