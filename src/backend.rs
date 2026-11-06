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
    /// Device-side duration of the last dispatch in milliseconds, if
    /// the backend can measure it (Metal command-buffer timestamps).
    fn device_ms(&self) -> Option<f64> {
        None
    }
    /// Read a resident buffer (param/state port) back to the host —
    /// backends that mutate weights or state on-device surface their
    /// `param_outs`/`state_outs` this way so the host store can feed
    /// the updated values forward to the next call.
    fn read(&self, _port: &str) -> Option<Vec<f32>> {
        None
    }
    /// Read for checkpoint export — lazy executors may hold no device
    /// state between dispatches, so this variant is allowed to
    /// materialize the runtime on demand. Default: plain `read`.
    fn read_for_export(&mut self, port: &str) -> Option<Vec<f32>> {
        self.read(port)
    }
    /// Remote-witness drain — dispatch backends (mesh/pcc sites) queue
    /// witness entries the remote node produced during `run`; the VM
    /// pulls them after each dispatch and lands them on its own
    /// hash-chained log, so one computation across N nodes produces
    /// one verifiable chain.
    fn drain_remote(&mut self) -> Vec<(String, String)> {
        Vec::new()
    }
    /// Drop resident device state so the executor can be re-materialized
    /// later (bounded GPU residency). Returns false when the executor
    /// holds nothing evictable.
    fn evict(&mut self) -> bool {
        false
    }
    /// Whether the executor currently holds device resources. Lazy
    /// executors report false until first dispatch.
    fn resident(&self) -> bool {
        true
    }
}

/// Executor that defers `load` to first dispatch and supports eviction
/// — large GPU programs share a memory-constrained device by
/// materializing a bounded set at a time. Params survive eviction in
/// the backend's shared arena; scratch/adjoint buffers re-derive on
/// reload and host-mirrored state restages on next dispatch.
pub struct LazyExec<F>
where
    F: FnMut() -> Result<Box<dyn BackendExec>, String>,
{
    load: F,
    inner: Option<Box<dyn BackendExec>>,
}

impl<F> LazyExec<F>
where
    F: FnMut() -> Result<Box<dyn BackendExec>, String>,
{
    pub fn new(load: F) -> Self {
        Self { load, inner: None }
    }

    fn ensure(&mut self) -> Result<&mut Box<dyn BackendExec>, String> {
        if self.inner.is_none() {
            self.inner = Some((self.load)()?);
        }
        Ok(self.inner.as_mut().unwrap())
    }
}

impl<F> BackendExec for LazyExec<F>
where
    F: FnMut() -> Result<Box<dyn BackendExec>, String>,
{
    fn run(&mut self, writes: &[(String, Vec<f32>)]) -> Result<Vec<f32>, String> {
        self.ensure()?.run(writes)
    }
    fn device_ms(&self) -> Option<f64> {
        self.inner.as_ref().and_then(|r| r.device_ms())
    }
    fn read(&self, port: &str) -> Option<Vec<f32>> {
        self.inner.as_ref().and_then(|r| r.read(port))
    }
    fn read_for_export(&mut self, port: &str) -> Option<Vec<f32>> {
        // export must see device-trained values even after eviction —
        // materialize, read, and let the caller evict again.
        self.ensure().ok()?.read(port)
    }
    fn drain_remote(&mut self) -> Vec<(String, String)> {
        self.inner
            .as_mut()
            .map(|r| r.drain_remote())
            .unwrap_or_default()
    }
    fn evict(&mut self) -> bool {
        if self.inner.take().is_some() {
            true
        } else {
            false
        }
    }
    fn resident(&self) -> bool {
        self.inner.is_some()
    }
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

/// Execution-site registry — `mesh fn`/`pcc fn` programs lower through
/// the same Backend seam but their "engine" is a wire dispatch to a
/// remote node. Registered by the distribution, never upstream.
static SITES: RwLock<Option<HashMap<crate::ast::Site, Arc<dyn Backend>>>> = RwLock::new(None);

/// Register a dispatch backend for an execution site.
pub fn register_site(site: crate::ast::Site, b: Arc<dyn Backend>) {
    let mut g = SITES.write().unwrap();
    g.get_or_insert_with(HashMap::new).insert(site, b);
}

/// Fetch the registered site backend, if any.
pub fn site_backend_for(site: crate::ast::Site) -> Result<Arc<dyn Backend>, String> {
    let g = SITES.read().unwrap();
    g.as_ref()
        .and_then(|m| m.get(&site).cloned())
        .ok_or_else(|| format!("no site backend registered for {} programs", site.name()))
}
