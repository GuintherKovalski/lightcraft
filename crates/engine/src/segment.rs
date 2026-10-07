//! AI masks: Object (clicks) and Describe (text) selections computed with SAM 3
//! (`lightcraft-segment`, cargo feature `sam`; the desktop app enables it).
//!
//! The model sees the photo as developed, uncropped and without masks, at 1008 px — the frame
//! mask coordinates live in — so a stored segmentation lines up with the photo at any size.
//! Encoding the image is the slow part (a few seconds on a laptop GPU); it runs once per photo
//! and look, on a worker thread when [`Session::segment_prepare`] is called (the Masking panel
//! does this when an AI mask is started), and each click then takes milliseconds.
//!
//! Without the feature (web, CLI) every request reports that this build has no model; stored
//! segmentations still render (they are plain data in the develop settings).

use std::path::PathBuf;

use lightcraft_catalog::PhotoId;
use lightcraft_develop::{MaskShape, SegMask};
use lightcraft_geom::Point;

use crate::Session;

/// Long edge of the image the model sees.
pub const INPUT_EDGE: usize = 1008;

/// One click of an Object selection (normalized image coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Click {
    pub at: Point,
    pub include: bool,
}

/// What a request runs on: the photo, its look (settings without masks), the model input.
#[cfg_attr(not(feature = "sam"), allow(dead_code))]
struct Input {
    key: u64,
    rgb: Vec<u8>,
    w: usize,
    h: usize,
}

#[cfg(feature = "sam")]
mod model {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, PoisonError};

    use lightcraft_segment::{Encoded, Sam3};

    #[derive(Default)]
    pub struct Inner {
        pub model: Option<Sam3>,
        /// The last encoded image and its key.
        pub cache: Option<(u64, Encoded)>,
    }

    #[derive(Clone, Default)]
    pub struct Shared {
        pub inner: Arc<Mutex<Inner>>,
        pub busy: Arc<AtomicBool>,
        /// A zoomed-in detail pass is running.
        pub detail_busy: Arc<AtomicBool>,
        /// Finished detail passes, waiting for [`crate::Session::segment_poll`].
        pub results: Arc<Mutex<Vec<super::DetailResult>>>,
    }

    impl Shared {
        pub fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
            self.inner.lock().unwrap_or_else(PoisonError::into_inner)
        }

        pub fn busy(&self) -> bool {
            self.busy.load(Ordering::Relaxed)
        }

        pub fn set_busy(&self, v: bool) {
            self.busy.store(v, Ordering::Relaxed);
        }
    }
}

/// The model (loaded on first use) and the last encoded photo.
#[derive(Default)]
pub struct Segmenter {
    /// Where the `facebook/sam3` checkpoint lives (set by the app).
    pub dir: Option<PathBuf>,
    #[cfg(feature = "sam")]
    shared: model::Shared,
}

impl Segmenter {
    /// Whether this build can compute AI masks at all.
    pub const AVAILABLE: bool = cfg!(feature = "sam");

    /// Whether a worker is preparing a photo right now.
    pub fn busy(&self) -> bool {
        #[cfg(feature = "sam")]
        return self.shared.busy();
        #[cfg(not(feature = "sam"))]
        false
    }

    /// Whether a zoomed-in detail pass is running.
    pub fn detail_busy(&self) -> bool {
        #[cfg(feature = "sam")]
        return self.shared.detail_busy.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(feature = "sam"))]
        false
    }

    /// The model directory, if its files are there.
    pub fn model_dir(&self) -> Result<PathBuf, String> {
        if !Self::AVAILABLE {
            return Err("AI masks are not available in this build".into());
        }
        let dir = self.dir.clone().ok_or("no folder is set for the SAM 3 model")?;
        #[cfg(feature = "sam")]
        if !lightcraft_segment::is_model_dir(&dir) {
            return Err(format!(
                "the SAM 3 model is not installed: put the facebook/sam3 files (model.safetensors, vocab.json, merges.txt) in {}",
                dir.display()
            ));
        }
        Ok(dir)
    }
}

/// Mix `v` into a 64-bit hash.
fn mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x0100_0000_01b3).rotate_left(29)
}

/// A finished zoomed-in pass: for which component (and the shape it was computed from, so a
/// component changed meanwhile is left alone), and the patch (`None`: nothing found there).
#[cfg_attr(not(feature = "sam"), allow(dead_code))]
pub struct DetailResult {
    photo: PhotoId,
    mask: u32,
    comp: usize,
    shape: MaskShape,
    patch: Option<SegMask>,
}

/// What a detail pass re-runs.
#[cfg_attr(not(feature = "sam"), allow(dead_code))]
enum Prompt {
    Clicks(Vec<Click>),
    Text(String),
}

/// Objects covering more than this much of the photo gain little from a zoomed pass.
const DETAIL_MAX_AREA: f64 = 0.4;

/// The part of the photo to zoom into for `seg`'s selection: its bounding box with a margin of
/// a quarter of its size, at least 12 % of the long edge, at most 2:1, inside the photo
/// (`aspect` = width / height). `None` when nothing is selected or it is most of the photo.
pub fn detail_region(seg: &SegMask, aspect: f64) -> Option<[f64; 4]> {
    let logits = seg.logits()?;
    let r = seg.bounds()?;
    let side = seg.side as usize;
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, l) in logits.iter().enumerate() {
        if *l > 0.0 {
            let (x, y) = (i % side, i / side);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
        }
    }
    if x0 == usize::MAX {
        return None;
    }
    let (rw, rh) = (r[2] - r[0], r[3] - r[1]);
    let nx = |c: usize| r[0] + c as f64 / side as f64 * rw;
    let ny = |c: usize| r[1] + c as f64 / side as f64 * rh;
    let (bx0, by0, bx1, by1) = (nx(x0), ny(y0), nx(x1), ny(y1));
    if (bx1 - bx0) * (by1 - by0) > DETAIL_MAX_AREA {
        return None;
    }
    // in units of the photo's height: x × aspect
    let a = aspect.clamp(0.1, 10.0);
    let (cx, cy) = ((bx0 + bx1) / 2.0 * a, (by0 + by1) / 2.0);
    let size = ((bx1 - bx0) * a).max(by1 - by0);
    let m = 0.25 * size;
    let mut w = ((bx1 - bx0) * a + 2.0 * m).max(0.12 * a.max(1.0));
    let mut h = ((by1 - by0) + 2.0 * m).max(0.12 * a.max(1.0));
    w = w.max(h / 2.0).min(a);
    h = h.max(w / 2.0).min(1.0);
    let x = (cx - w / 2.0).clamp(0.0, (a - w).max(0.0));
    let y = (cy - h / 2.0).clamp(0.0, (1.0 - h).max(0.0));
    let out = [x / a, y, (x + w) / a, y + h];
    // a region nearly the whole photo isn't worth a second pass
    ((out[2] - out[0]) * (out[3] - out[1]) < 0.8).then_some(out)
}

impl Session {
    /// The key and settings of photo `id` as the model sees it: its look without masks, uncropped.
    fn segment_key(&self, id: PhotoId) -> Option<(u64, lightcraft_develop::DevelopSettings)> {
        let p = self.catalog.photo(id)?;
        let mut d = (*p.develop).clone();
        d.masks.clear();
        let content = crate::media::content_key(p);
        let mut h = mix(0xcbf2_9ce4_8422_2325, d.hash64());
        h = mix(h, id.0);
        for b in content.bytes() {
            h = mix(h, u64::from(b));
        }
        Some((h, d))
    }

    /// Render the model input for `id`.
    #[cfg_attr(not(feature = "sam"), allow(dead_code))]
    fn segment_input(&mut self, id: PhotoId) -> Result<Input, String> {
        let (key, d) = self.segment_key(id).ok_or("no such photo")?;
        let job = self.preview_job(id, INPUT_EDGE, INPUT_EDGE, false, &d).ok_or("no such photo")?;
        let r = job.run();
        self.accept(&r);
        let img = match r.rendered {
            Ok(r) => r.image,
            Err(e) => {
                // the usual reason: the file was moved or deleted (and there's no smart preview)
                if let Some(lightcraft_catalog::Source::File { path }) = self.catalog.photo(id).map(|p| &p.source)
                    && !std::path::Path::new(path).exists()
                {
                    return Err(format!(
                        "This photo's file is missing (moved or deleted), so it can't be analyzed for AI masks: {path}. Library ▸ Find Missing Photos can relink it."
                    ));
                }
                return Err(format!("couldn't render the photo for AI masks: {e}"));
            }
        };
        let rgb = img.data.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        Ok(Input { key, rgb, w: img.width, h: img.height })
    }

    /// Start preparing photo `id` for AI masks on a worker thread (load the model, encode the
    /// image). Returns at once; does nothing when it is ready or being prepared.
    pub fn segment_prepare(&mut self, id: PhotoId) -> Result<(), String> {
        let dir = self.segmenter.model_dir()?;
        #[cfg(feature = "sam")]
        {
            let (key, _) = self.segment_key(id).ok_or("no such photo")?;
            let shared = self.segmenter.shared.clone();
            if shared.busy() {
                return Ok(());
            }
            if let Ok(inner) = shared.inner.try_lock()
                && inner.cache.as_ref().is_some_and(|c| c.0 == key)
            {
                return Ok(());
            }
            let input = self.segment_input(id)?;
            shared.set_busy(true);
            let spawned = std::thread::Builder::new().name("sam3-prepare".into()).spawn(move || {
                let mut inner = shared.lock();
                if let Err(e) = prepare(&mut inner, &dir, &input) {
                    log::warn!("SAM 3: {e}");
                }
                shared.set_busy(false);
            });
            if let Err(e) = spawned {
                self.segmenter.shared.set_busy(false);
                return Err(format!("could not start the model: {e}"));
            }
        }
        #[cfg(not(feature = "sam"))]
        let _ = (dir, id);
        Ok(())
    }

    /// Segment the object `clicks` pick on photo `id` (waits for a running prepare).
    pub fn segment_clicks(&mut self, id: PhotoId, clicks: &[Click]) -> Result<SegMask, String> {
        let dir = self.segmenter.model_dir()?;
        if !clicks.iter().any(|c| c.include) {
            return Err("click the object to include it first".into());
        }
        #[cfg(feature = "sam")]
        {
            let input = self.segment_input_if_needed(id)?;
            let shared = self.segmenter.shared.clone();
            let mut inner = shared.lock();
            let enc = encoded(&mut inner, &dir, input)?;
            let clicks: Vec<lightcraft_segment::Click> =
                clicks.iter().map(|c| lightcraft_segment::Click { x: c.at.x as f32, y: c.at.y as f32, positive: c.include }).collect();
            let (model, enc) = enc;
            let p = model.segment_clicks(enc, &clicks).map_err(|e| e.to_string())?;
            Ok(SegMask::from_logits(lightcraft_segment::MASK_SIDE, &p.logits))
        }
        #[cfg(not(feature = "sam"))]
        {
            let _ = (dir, id);
            Err("AI masks are not available in this build".into())
        }
    }

    /// Segment everything `text` describes on photo `id`; `Ok(None)` when nothing matches.
    pub fn segment_text(&mut self, id: PhotoId, text: &str) -> Result<Option<SegMask>, String> {
        let dir = self.segmenter.model_dir()?;
        let text = text.trim();
        if text.is_empty() {
            return Err("describe what to select".into());
        }
        #[cfg(feature = "sam")]
        {
            let input = self.segment_input_if_needed(id)?;
            let shared = self.segmenter.shared.clone();
            let mut inner = shared.lock();
            let (model, enc) = encoded(&mut inner, &dir, input)?;
            // "car, road": each phrase on its own, merged (the per-pixel maximum)
            let mut merged: Option<Vec<f32>> = None;
            for phrase in text.split([',', ';']).map(str::trim).filter(|p| !p.is_empty()) {
                if let Some(p) = model.segment_text(enc, phrase, 0.5).map_err(|e| e.to_string())? {
                    merged = Some(match merged {
                        None => p.logits,
                        Some(m) => m.iter().zip(&p.logits).map(|(a, b)| a.max(*b)).collect(),
                    });
                }
            }
            Ok(merged.map(|l| SegMask::from_logits(lightcraft_segment::MASK_SIDE, &l)))
        }
        #[cfg(not(feature = "sam"))]
        {
            let _ = (dir, id);
            Err("AI masks are not available in this build".into())
        }
    }

    /// Render the part `region` (normalized) of photo `id` for the model, from a source sharp
    /// enough for it.
    #[cfg_attr(not(feature = "sam"), allow(dead_code))]
    fn segment_region_input(&mut self, id: PhotoId, region: [f64; 4]) -> Result<Input, String> {
        let (_, mut d) = self.segment_key(id).ok_or("no such photo")?;
        d.crop.geometry = lightcraft_geom::CropGeometry { rect: lightcraft_geom::Rect::new(region[0], region[1], region[2], region[3]), angle: 0.0 };
        d.crop.flip_h = false;
        d.crop.flip_v = false;
        d.geometry.constrain_crop = false;
        let mut job = self.preview_job(id, INPUT_EDGE, INPUT_EDGE, true, &d).ok_or("no such photo")?;
        let frac = (region[2] - region[0]).max(region[3] - region[1]).max(0.01);
        let p = self.catalog.photo(id).ok_or("no such photo")?.clone();
        job.level = crate::media::SourceLevel::for_size((INPUT_EDGE as f64 / frac) as usize);
        job.source = self.media.source_ref(&p, job.level);
        let r = job.run();
        self.accept(&r);
        let img = r.rendered.map_err(|e| format!("couldn't render the photo for AI masks: {e}"))?.image;
        let rgb = img.data.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
        Ok(Input { key: 0, rgb, w: img.width, h: img.height })
    }

    /// Start a zoomed-in pass for component `comp` of mask `mask` on photo `id` (an Object or
    /// Describe selection): the photo around the selection is rendered at a higher resolution
    /// and segmented again, on a worker thread; [`Session::segment_poll`] applies the result.
    /// `Ok(false)` when there is nothing to refine (no selection yet, or it is most of the photo).
    pub fn segment_detail(&mut self, id: PhotoId, mask: u32, comp: usize) -> Result<bool, String> {
        let dir = self.segmenter.model_dir()?;
        let d = self.develop_of(id).ok_or("no such photo")?;
        let shape = d.masks.iter().find(|m| m.id == mask).and_then(|m| m.components.get(comp)).map(|c| c.shape.clone()).ok_or("no such mask")?;
        let (seg, prompt) = match &shape {
            MaskShape::Object { hint, exclude, seg: Some(seg), .. } if !hint.is_empty() => (
                seg.clone(),
                Prompt::Clicks(
                    hint.iter().map(|p| Click { at: *p, include: true }).chain(exclude.iter().map(|p| Click { at: *p, include: false })).collect(),
                ),
            ),
            MaskShape::Prompt { text, seg: Some(seg), .. } => (seg.clone(), Prompt::Text(text.clone())),
            _ => return Ok(false),
        };
        let p = self.catalog.photo(id).ok_or("no such photo")?;
        let (pw, ph) = (f64::from(p.width.max(1)), f64::from(p.height.max(1)));
        let aspect = if d.orientation.swaps_axes() { ph / pw } else { pw / ph };
        let Some(region) = detail_region(&seg, aspect) else { return Ok(false) };
        #[cfg(feature = "sam")]
        {
            let shared = self.segmenter.shared.clone();
            if shared.detail_busy.swap(true, std::sync::atomic::Ordering::Relaxed) {
                return Ok(false);
            }
            let input = match self.segment_region_input(id, region) {
                Ok(i) => i,
                Err(e) => {
                    shared.detail_busy.store(false, std::sync::atomic::Ordering::Relaxed);
                    return Err(e);
                }
            };
            let shape = strip_detail(shape);
            let spawned = std::thread::Builder::new().name("sam3-detail".into()).spawn({
                let shared = shared.clone();
                move || {
                    let patch = {
                        let mut inner = shared.lock();
                        detail_pass(&mut inner, &dir, &input, &prompt, region)
                    };
                    match patch {
                        Ok(patch) => {
                            let r = DetailResult { photo: id, mask, comp, shape, patch };
                            shared.results.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(r);
                        }
                        Err(e) => log::warn!("SAM 3 detail pass: {e}"),
                    }
                    shared.detail_busy.store(false, std::sync::atomic::Ordering::Relaxed);
                }
            });
            if let Err(e) = spawned {
                shared.detail_busy.store(false, std::sync::atomic::Ordering::Relaxed);
                return Err(format!("could not start the detail pass: {e}"));
            }
            Ok(true)
        }
        #[cfg(not(feature = "sam"))]
        {
            let _ = (dir, prompt, region, mask);
            Ok(false)
        }
    }

    /// Apply finished detail passes (to components that haven't changed since). True when a
    /// mask changed.
    pub fn segment_poll(&mut self) -> bool {
        #[cfg(feature = "sam")]
        {
            let done: Vec<DetailResult> =
                std::mem::take(&mut *self.segmenter.shared.results.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
            let mut changed = false;
            for r in done {
                let Some(patch) = r.patch else { continue };
                let Some(d) = self.develop_of(r.photo) else { continue };
                let mut d = (*d).clone();
                let Some(c) = d.masks.iter_mut().find(|m| m.id == r.mask).and_then(|m| m.components.get_mut(r.comp)) else { continue };
                if strip_detail(c.shape.clone()) != r.shape {
                    continue; // clicked again meanwhile: a newer pass will follow
                }
                match &mut c.shape {
                    MaskShape::Object { detail, .. } | MaskShape::Prompt { detail, .. } => *detail = vec![patch],
                    _ => continue,
                }
                match self.set_develop(r.photo, d, "AI Mask Detail") {
                    Ok(()) => changed = true,
                    Err(e) => log::warn!("AI mask detail: {e}"),
                }
            }
            changed
        }
        #[cfg(not(feature = "sam"))]
        false
    }

    /// The model input for `id`, unless the cache already holds it (or a prepare is running,
    /// which will have it).
    #[cfg(feature = "sam")]
    fn segment_input_if_needed(&mut self, id: PhotoId) -> Result<Either, String> {
        let (key, _) = self.segment_key(id).ok_or("no such photo")?;
        if self.segmenter.shared.busy() {
            return Ok(Either::Key(key));
        }
        let cached = self.segmenter.shared.lock().cache.as_ref().is_some_and(|c| c.0 == key);
        if cached { Ok(Either::Key(key)) } else { Ok(Either::Input(self.segment_input(id)?)) }
    }
}

#[cfg(feature = "sam")]
enum Either {
    /// Already encoded (or being encoded) under this key.
    Key(u64),
    Input(Input),
}

#[cfg(feature = "sam")]
fn load(inner: &mut model::Inner, dir: &std::path::Path) -> Result<(), String> {
    if inner.model.is_none() {
        let t = web_time::Instant::now();
        inner.model = Some(lightcraft_segment::Sam3::load(dir).map_err(|e| e.to_string())?);
        log::info!("SAM 3 loaded in {:?}", t.elapsed());
    }
    Ok(())
}

/// Load the model and encode `input` into the cache (with the click features ready).
#[cfg(feature = "sam")]
fn prepare(inner: &mut model::Inner, dir: &std::path::Path, input: &Input) -> Result<(), String> {
    load(inner, dir)?;
    let model = inner.model.as_mut().ok_or("the model did not load")?;
    let t = web_time::Instant::now();
    let mut enc = model.encode(&input.rgb, input.w, input.h).map_err(|e| e.to_string())?;
    model.prepare_clicks(&mut enc).map_err(|e| e.to_string())?;
    log::info!("SAM 3 encoded {}×{} in {:?}", input.w, input.h, t.elapsed());
    inner.cache = Some((input.key, enc));
    Ok(())
}

/// The model and the encoded image for `want` (encoding it now when needed).
#[cfg(feature = "sam")]
fn encoded<'a>(
    inner: &'a mut model::Inner,
    dir: &std::path::Path,
    want: Either,
) -> Result<(&'a mut lightcraft_segment::Sam3, &'a mut lightcraft_segment::Encoded), String> {
    match want {
        Either::Input(input) => prepare(inner, dir, &input)?,
        Either::Key(key) => {
            if !inner.cache.as_ref().is_some_and(|c| c.0 == key) {
                return Err("the photo changed while it was being prepared; try again".into());
            }
        }
    }
    let model::Inner { model, cache } = inner;
    match (model.as_mut(), cache.as_mut()) {
        (Some(m), Some((_, e))) => Ok((m, e)),
        _ => Err("the model is not ready".into()),
    }
}

/// `shape` without its detail patches (what a detail pass was computed from)
#[cfg_attr(not(feature = "sam"), allow(dead_code))]
fn strip_detail(mut shape: MaskShape) -> MaskShape {
    // (nor its edge setting: moving the slider meanwhile keeps the pass)
    if let MaskShape::Object { detail, edge, .. } | MaskShape::Prompt { detail, edge, .. } = &mut shape {
        detail.clear();
        *edge = 0.0;
    }
    shape
}

/// Segment `input` (the photo's part `region`) with `prompt`, as a patch over `region`.
#[cfg(feature = "sam")]
fn detail_pass(inner: &mut model::Inner, dir: &std::path::Path, input: &Input, prompt: &Prompt, region: [f64; 4]) -> Result<Option<SegMask>, String> {
    load(inner, dir)?;
    let model = inner.model.as_mut().ok_or("the model did not load")?;
    let t = web_time::Instant::now();
    let mut enc = model.encode(&input.rgb, input.w, input.h).map_err(|e| e.to_string())?;
    let (rw, rh) = (region[2] - region[0], region[3] - region[1]);
    let logits = match prompt {
        Prompt::Clicks(clicks) => {
            let inside: Vec<lightcraft_segment::Click> = clicks
                .iter()
                .map(|c| ((c.at.x - region[0]) / rw, (c.at.y - region[1]) / rh, c.include))
                .filter(|(x, y, _)| (0.0..=1.0).contains(x) && (0.0..=1.0).contains(y))
                .map(|(x, y, positive)| lightcraft_segment::Click { x: x as f32, y: y as f32, positive })
                .collect();
            if !inside.iter().any(|c| c.positive) {
                return Ok(None);
            }
            Some(model.segment_clicks(&mut enc, &inside).map_err(|e| e.to_string())?.logits)
        }
        Prompt::Text(text) => {
            let mut merged: Option<Vec<f32>> = None;
            for phrase in text.split([',', ';']).map(str::trim).filter(|p| !p.is_empty()) {
                if let Some(p) = model.segment_text(&mut enc, phrase, 0.5).map_err(|e| e.to_string())? {
                    merged = Some(match merged {
                        None => p.logits,
                        Some(m) => m.iter().zip(&p.logits).map(|(a, b)| a.max(*b)).collect(),
                    });
                }
            }
            merged
        }
    };
    log::info!("SAM 3 detail pass ({}×{}) in {:?}", input.w, input.h, t.elapsed());
    Ok(logits.map(|l| SegMask::from_logits_in(lightcraft_segment::MASK_SIDE, &l, region)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg_box(side: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> SegMask {
        let l: Vec<f32> =
            (0..side * side).map(|i| if (x0..x1).contains(&(i % side)) && (y0..y1).contains(&(i / side)) { 8.0 } else { -8.0 }).collect();
        SegMask::from_logits(side, &l)
    }

    #[test]
    fn detail_region_frames_the_selection_with_a_margin() {
        // a small object in the middle of a square photo
        let r = detail_region(&seg_box(100, 40, 45, 50, 55), 1.0).unwrap();
        assert!(r[0] < 0.4 && r[2] > 0.5 && r[1] < 0.45 && r[3] > 0.55, "contains it: {r:?}");
        assert!(r[2] - r[0] < 0.25 && r[3] - r[1] < 0.25, "zoomed in: {r:?}");
        assert!(r.iter().all(|v| (0.0..=1.0).contains(v)));
        // at the corner: shifted inside the photo, not cut
        let r = detail_region(&seg_box(100, 0, 0, 5, 5), 1.5).unwrap();
        assert!(r[0] >= 0.0 && r[1] >= 0.0 && r[0] < 1e-9 && r[1] < 1e-9);
        // a thin object stays at most 2:1 (in photo pixels, 1.5:1 photo)
        let r = detail_region(&seg_box(100, 10, 50, 90, 52), 1.5).unwrap();
        let (w, h) = ((r[2] - r[0]) * 1.5, r[3] - r[1]);
        assert!(w / h <= 2.0 + 1e-9, "{w} × {h}");
    }

    #[test]
    fn no_region_for_nothing_or_most_of_the_photo() {
        assert!(detail_region(&seg_box(50, 0, 0, 0, 0), 1.0).is_none());
        assert!(detail_region(&seg_box(50, 2, 2, 48, 48), 1.0).is_none());
        let damaged = SegMask { side: 50, data: "?".into(), rect: None };
        assert!(detail_region(&damaged, 1.0).is_none());
    }
}
