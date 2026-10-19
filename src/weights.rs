//! Weight sources — `param w: f16[K, N] @ "file.safetensors::tensor"`.
//!
//! Self-contained safetensors reader: header JSON, BF16/F16/F32 decode,
//! no external crates. A param without a source falls back to the
//! deterministic xorshift init so programs stay reproducible offline.

use std::collections::HashMap;

/// minimal safetensors index — name → (data range, shape, dtype)
pub struct SafeTensors {
    buf: Vec<u8>,
    index: HashMap<String, (usize, usize, Vec<usize>, String)>,
}

impl SafeTensors {
    pub fn load(path: &str) -> Result<Self, String> {
        let buf = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if buf.len() < 8 {
            return Err(format!("{path}: not a safetensors file"));
        }
        let hlen = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
        // hand-rolled JSON scan: `"name":{"dtype":"..","shape":[..],"data_offsets":[a,b]}`
        let head = std::str::from_utf8(&buf[8..8 + hlen])
            .map_err(|_| format!("{path}: bad header utf8"))?;
        let mut index = HashMap::new();
        // naive but robust-enough parse: find each top-level "key":{
        let mut i = 0;
        let bytes = head.as_bytes();
        while i < bytes.len() {
            // find `"` start of a key at depth 1
            if bytes[i] == b'"' {
                let end = head[i + 1..]
                    .find('"')
                    .map(|e| i + 1 + e)
                    .ok_or_else(|| format!("{path}: truncated header"))?;
                let name = &head[i + 1..end];
                i = end + 1;
                // skip ws + ':'
                while i < bytes.len() && (bytes[i] == b':' || bytes[i] == b' ') {
                    i += 1;
                }
                if i >= bytes.len() {
                    break;
                }
                if bytes[i] == b'{' {
                    // grab the object body
                    let start = i;
                    let mut depth = 0;
                    while i < bytes.len() {
                        match bytes[i] {
                            b'{' => depth += 1,
                            b'}' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                    let obj = &head[start..=i];
                    i += 1;
                    if name == "__metadata__" {
                        continue;
                    }
                    let dt = json_str(obj, "dtype").unwrap_or_default();
                    let shape = json_nums(obj, "shape")
                        .iter()
                        .map(|&v| v as usize)
                        .collect::<Vec<_>>();
                    let offs = json_nums(obj, "data_offsets");
                    if offs.len() == 2 {
                        let (a, b) = (
                            offs[0] as usize + 8 + hlen,
                            offs[1] as usize + 8 + hlen,
                        );
                        index.insert(name.to_string(), (a, b, shape, dt));
                    }
                }
            } else {
                i += 1;
            }
        }
        Ok(SafeTensors { buf, index })
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|(_, _, s, _)| s.as_slice())
    }

    /// decode a single row of a rank-2 tensor — an embedding gather
    /// without touching the rest of the file.
    pub fn f32s_slice(&self, name: &str, row: usize, width: usize) -> Result<Vec<f32>, String> {
        let (a, b, shape, dt) = self
            .index
            .get(name)
            .ok_or_else(|| format!("safetensors: missing {name}"))?;
        let cols = shape.get(1).copied().unwrap_or(1);
        if shape.len() != 2 || width != cols || row >= shape[0] {
            return Err(format!("safetensors {name}: row {row}/{width} of {shape:?}"));
        }
        let esz = match dt.as_str() {
            "BF16" | "F16" => 2,
            "F32" => 4,
            other => return Err(format!("safetensors {name}: dtype {other} unsupported")),
        };
        let off = a + row * cols * esz;
        let raw = &self.buf[off..(off + cols * esz).min(*b)];
        Ok(match dt.as_str() {
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            _ => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }

    /// decode the tensor to f32 (BF16/F16/F32 supported)
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>, String> {
        let (a, b, _, dt) = self
            .index
            .get(name)
            .ok_or_else(|| format!("safetensors: missing {name}"))?;
        let raw = &self.buf[*a..*b];
        Ok(match dt.as_str() {
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            other => return Err(format!("safetensors {name}: dtype {other} unsupported")),
        })
    }
}

/// `"dtype":"X"` inside a small object body.
fn json_str(obj: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let p = obj.find(&pat)? + pat.len();
    let rest = &obj[p..];
    let q1 = rest.find('"')?;
    let q2 = rest[q1 + 1..].find('"')? + q1 + 1;
    Some(rest[q1 + 1..q2].to_string())
}

/// `"key":[n0,n1,..]` inside a small object body.
fn json_nums(obj: &str, key: &str) -> Vec<u64> {
    let pat = format!("\"{key}\"");
    let Some(p) = obj.find(&pat) else {
        return Vec::new();
    };
    let rest = &obj[p + pat.len()..];
    let Some(b0) = rest.find('[') else {
        return Vec::new();
    };
    let Some(b1) = rest[b0..].find(']') else {
        return Vec::new();
    };
    rest[b0 + 1..b0 + b1]
        .split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect()
}

/// resolve a param's initial values: file source if declared, else the
/// deterministic name-seeded init the caller supplies as `fallback`.
/// `expect` is the FILE-side shape (storage order: `[N, K]` for matvec
/// weights) — mismatched orientation is a hard error, not a silent
/// transpose.
pub fn resolve<F: Fn(&str, usize) -> Vec<f32>>(
    name: &str,
    expect: Option<&[usize]>,
    src: Option<&str>,
    fallback: F,
) -> Result<Vec<f32>, String> {
    let numel = expect
        .map(|s| s.iter().product())
        .unwrap_or(0);
    let Some(src) = src else {
        return Ok(fallback(name, numel));
    };
    let (path, spec) = src
        .split_once("::")
        .ok_or_else(|| format!("param source {src:?} must be \"file::tensor\""))?;
    // `tensor` or `tensor[r0:r1]` or `tensor[r0:r1,c0:c1]` — a row-major
    // range slice taken at load time (head-split weights, vocab slices).
    let (tensor, ranges) = match spec.split_once('[') {
        Some((t, rest)) => {
            let inner = rest
                .strip_suffix(']')
                .ok_or_else(|| format!("param source {src:?}: unterminated range"))?;
            let mut rs = Vec::new();
            for part in inner.split(',') {
                let (a, b) = part
                    .split_once(':')
                    .ok_or_else(|| format!("param source {src:?}: bad range {part:?}"))?;
                rs.push((
                    a.parse::<usize>()
                        .map_err(|_| format!("param source {src:?}: bad range {part:?}"))?,
                    b.parse::<usize>()
                        .map_err(|_| format!("param source {src:?}: bad range {part:?}"))?,
                ));
            }
            (t, Some(rs))
        }
        None => (spec, None),
    };
    let st = SafeTensors::load(path)?;
    let shape = st
        .shape(tensor)
        .ok_or_else(|| format!("param {name}: missing tensor {tensor}"))?
        .to_vec();
    let vals = st.f32s(tensor)?;
    let vals = match &ranges {
        None => {
            if let Some(want) = expect {
                if shape != want {
                    return Err(format!(
                        "param {name}: source {tensor} is {shape:?}, storage wants {want:?} ([N, K] = [out, in])"
                    ));
                }
            }
            vals
        }
        Some(rs) => slice(&vals, &shape, rs).map_err(|e| format!("param {name}: {e}"))?,
    };
    if let Some(want) = expect {
        let got = vals.len();
        let want_n: usize = want.iter().product();
        if got != want_n {
            return Err(format!("param {name}: source {spec} gives {got} elems, declared {want_n}"));
        }
    }
    Ok(vals)
}

/// row-major range slice of a decoded tensor; each dim's `[a,b)` must
/// fit its extent. Omitted-tail dims take the full extent.
fn slice(vals: &[f32], shape: &[usize], ranges: &[(usize, usize)]) -> Result<Vec<f32>, String> {
    let nd = shape.len().max(ranges.len());
    let mut lo = vec![0usize; nd];
    let mut hi = vec![1usize; nd];
    for d in 0..nd {
        hi[d] = shape.get(d).copied().unwrap_or(1);
        if let Some(&(a, b)) = ranges.get(d) {
            if b > hi[d] || a >= b {
                return Err(format!("range [{a}:{b}] out of dim-{d} extent {}", hi[d]));
            }
            lo[d] = a;
            hi[d] = b;
        }
    }
    // row-major walk over the strided box
    let mut stride = vec![1usize; nd];
    for d in (0..nd.saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * shape.get(d + 1).copied().unwrap_or(1);
    }
    let mut out = Vec::new();
    let mut idx = lo.clone();
    let total: usize = (0..nd).map(|d| hi[d] - lo[d]).product();
    out.reserve(total);
    for _ in 0..total {
        out.push(vals[(0..nd).map(|d| idx[d] * stride[d]).sum::<usize>()]);
        for d in (0..nd).rev() {
            idx[d] += 1;
            if idx[d] < hi[d] {
                break;
            }
            idx[d] = lo[d];
        }
    }
    Ok(out)
}
