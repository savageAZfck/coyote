//! backend — the stratum seam. A `TProgram` carries a `target`; the
//! core ships the ANE path natively. Other compute targets (`gpu fn`,
//! and anything a distribution adds) resolve through registered
//! backends: the language knows a program *has* a target; what runs it
//! is pluggable.

use crate::typeck::{CTy, TProgram};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// What an emitter hands back: the runtime-facing io plan plus a
/// backend-private payload for `load`.
pub struct BackendEmit {
    /// io in ports: (port, ty, contig)
    pub inputs: Vec<(String, CTy, bool)>,
    /// trainable params: (name, port, ty, src)
    pub trainable: Vec<(String, String, CTy, Option<String>)>,
    /// state fields: (key `s.f`, port `s_f`, ty)
    pub states: Vec<(String, String, CTy)>,
    /// updated params out: (param, port, ty)
    pub param_outs: Vec<(String, String, CTy)>,
    /// updated state out: (port, value-name, ty)
    pub state_outs: Vec<(String, String, CTy)>,
    /// output name + type
    pub output: (String, CTy),
    /// backend-private payload returned to `load` (e.g. emitted source).
    pub opaque: Box<dyn std::any::Any>,
    /// display line for CLI output (e.g. "msl sha abc123…").
    pub note: String,
}

/// A loaded, dispatchable program instance.
pub trait BackendExec {
    /// Stage (port → f32 data) writes, dispatch, return the output
    /// row-major.
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String>;
}

/// A compute stratum: lower a typed program and produce its executor.
pub trait Backend: Send + Sync {
    /// Emit `tp` into `dir` (artifacts + manifest) and return the plan.
    fn emit(&self, src: &str, name: &str, tp: &TProgram, dir: &Path)
        -> Result<BackendEmit, String>;
    /// Build the executor from the emit result.
    fn load(
        &self,
        tp: &TProgram,
        name: &str,
        emit: &BackendEmit,
        dir: &Path,
    ) -> Result<Box<dyn BackendExec>, String>;
}

static BACKENDS: RwLock<Option<HashMap<crate::ast::Target, Arc<dyn Backend>>>> = RwLock::new(None);

/// Register a backend for a non-ANE target. Distributions call this
/// before dispatching programs whose `target` isn't `Ane`.
pub fn register_backend(target: crate::ast::Target, b: Arc<dyn Backend>) {
    let mut g = BACKENDS.write().unwrap();
    g.get_or_insert_with(HashMap::new).insert(target, b);
}

/// Fetch the registered backend for `target`, if any.
pub fn backend_for(target: crate::ast::Target) -> Result<Arc<dyn Backend>, String> {
    let g = BACKENDS.read().unwrap();
    g.as_ref()
        .and_then(|m| m.get(&target).cloned())
        .ok_or_else(|| format!("no backend registered for {} programs", target.name()))
}
