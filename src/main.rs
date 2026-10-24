//! coyote — source to silicon. Thin shim over `coyote::cli`.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(code) = coyote::cli::run(&args) {
        std::process::exit(code);
    }
}
