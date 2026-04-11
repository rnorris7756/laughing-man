# Proposal: Improving Face Detection Across Viewing Angles

## Implementation status (laughing-man repo)

This section records what is **already implemented** in code versus what remains **future / exploratory** from this document. Last updated to reflect the current tree.

| Theme | In this proposal | Status |
| ----- | ---------------- | ------ |
| Temporal smoothing of the face box | §4.3 (IoU / Kalman / flow discussion) | **Done (partial):** Default **EMA** via `--roi-lambda` / `--size-lambda` on center and size. **`--roi-motion kalman`** — constant-velocity Kalman on `(cx, cy, w, h)`. **`--roi-motion kalman_flow`** — same Kalman plus sparse LK flow to blend the measured center toward frame-to-frame motion (`box_tracking.py`). Noise gains are constants in `constants.py` (`KALMAN_*`, `FLOW_CENTER_MIX`). **Not implemented:** detect-every-N, dense flow, IoU association across multiple faces, learned trackers. |
| Alternate face detector | §3.1 (YuNet row) | **Done:** OpenCV **YuNet** is available as `--face-backend yunet` (optional ONNX download). Empirically: often **less jitter** than BlazeFace; **worse recall** on small / distant faces—kept as an optional illustration of swapping `FaceBoxSource` implementations. |
| CLI-selectable backend | — (product note) | **Done:** `--face-backend` (`blaze` or `yunet`); compare by running the app twice. Active backend is logged at startup. |
| Crop-then-detect cascade | §4.2, §3.4 “Rough ROI then refine” | **Done (narrow):** `--cascade-margin` applies **only to YuNet**—expand the **previous smoothed** ROI, run detection on the crop, fall back to full frame. BlazeFace is not run on variable crops (VIDEO mode). Not implemented: separate head/pose stage, person detector, or union with a dedicated tracker beyond existing ROI state. |
| Stronger detectors (SCRFD, RetinaFace, YOLO-face, …) | §3.1 | **Not done** (still valid options). |
| Landmarks / align / re-detect | §3.6 | **Not done.** |
| Multi-scale, TTA, tiling | §3.3 | **Not done.** |
| TensorRT / INT8 deployment | §3.7 | **Not done.** |
| Training / fine-tuning | §3.2 | **Not done.** |

The sections below remain the **full design reference**; completed items are not removed, but superseded in part by the table above where applicable.

---

## 1. Problem framing

“Different angles” usually mixes two phenomena:


| Phenomenon            | What changes                                                | Typical mitigation                                                                                                        |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **In-plane rotation** | Roll: the image is rotated; the face still faces the camera | Rotation augmentation, multi-orientation inference, or rotation-calibration cascades                                      |
| **Out-of-plane pose** | Yaw / pitch: true 3D head turn (profile, looking up/down)   | Richer training data, larger receptive fields, harder-architecture detectors, sometimes a second stage (pose / landmarks) |


Most public face detectors are trained heavily on **WIDER FACE**, whose **Hard** split includes small faces, occlusion, and challenging poses—so “accuracy on Hard” correlates with **non-frontal** robustness, but it is not a dedicated profile benchmark. For profile-heavy workloads you should validate on **your** data or add labeled tilt/profile examples.

---

## 2. Hardware you can leverage

Your platform (**NVIDIA RTX 4080 Super**, **recent Ryzen**, **~90 GB DDR5**) supports several strategies that are impractical on mobile or edge-only setups:

- **GPU throughput**: High FPS at moderate resolutions; room for **TensorRT** (FP16/INT8), **larger backbones**, **batched inference**, and **test-time augmentation (TTA)** copies in parallel.
- **VRAM (4080 Super class)**: Enough for **640×640–1280×1280** inputs, multi-scale pyramids, and moderate batch sizes without constant CPU–GPU ping-pong.
- **Large system RAM**: **Aggressive caching** of decoded frames, **parallel CPU pipelines** (decode, resize, optional OpenCV rotations), and **ensemble** or **multi-model** routing without memory pressure.

Constraints to keep in mind: the **CPU** still matters for decode, some preprocessing, and orchestration; **PCIe** and **copy** costs can dominate if the pipeline is not pipelined.

---

## 3. Approach families (not limited to BlazeFace)

### 3.1 Stronger single-shot detectors (replace or complement BlazeFace)


| Approach                    | Idea                                                                         | Pros                                                                                                    | Cons                                                                   |
| --------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **SCRFD** (InsightFace)     | Efficient anchor design + “sample/computation redistribution” for face stats | Excellent **accuracy vs. compute** on WIDER FACE; many sizes (mobile to heavy); ONNX/TensorRT ecosystem | Not as tiny as BlazeFace; integration effort                           |
| **RetinaFace**              | Single-shot with optional landmarks                                          | Mature; landmarks help **downstream alignment**; good baseline                                          | Often slower / heavier than SCRFD at same AP on modern comparisons     |
| **YOLOv8-Face** (community) | YOLO-style single pass, WIDER FACE–trained weights                           | Very fast on desktop GPUs; simple deployment story                                                      | Landmark / edge-case behavior varies by fork; validate on profile data |
| **YuNet** (OpenCV)          | Tiny CNN                                                                     | Extremely cheap; good when faces are large and frontal-ish                                              | Weaker on **Hard** / extreme pose vs. heavier models                   |

*Repo note:* YuNet is wired as an **optional** `--face-backend` for experimentation; in practice it trades some **distant-face recall** for **often steadier** boxes vs. BlazeFace on this pipeline.

**Trade-off summary:** BlazeFace optimizes for **mobile latency**; on a **4080 Super**, you can usually afford **SCRFD or a YOLO-face variant** if angle robustness and recall matter more than sub-millisecond mobile budgets.

---

### 3.2 Training and data (often the highest ROI)


| Technique                                   | What you do                                            | Pros                                           | Cons                                           |
| ------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------- | ---------------------------------------------- |
| **Rotation / affine augmentation**          | Random roll, mild yaw/pitch via 2D warps, scale jitter | Cheap; improves in-plane and mild out-of-plane | Extreme 3D pose is not fully modeled by 2D aug |
| **Copy-paste / mosaic** (detector-specific) | Face patches pasted into new backgrounds               | Improves clutter + scale                       | Needs care to avoid artifacts                  |
| **Fine-tune on your domain**                | Label or weak-label **your** tilt/profile failures     | Directly fixes **your** error modes            | Labeling cost; overfit risk if dataset tiny    |
| **Synthetic or semi-synthetic heads**       | Rendered faces at controlled poses (when acceptable)   | Controlled yaw/pitch coverage                  | Domain gap; pipeline complexity                |


**Trade-off summary:** If failures are **systematic** (e.g., studio lighting + profile), **data + fine-tuning** beats swapping models alone.

---

### 3.3 Inference-time scaling (no retrain)


| Technique                               | Idea                                               | Pros                                                   | Cons                                        |
| --------------------------------------- | -------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------- |
| **Multi-scale / image pyramid**         | Run detector at 0.8×, 1×, 1.2× (or more) and merge | Helps **small** and **partial** faces (common in Hard) | Linear-ish cost in scales; NMS tuning       |
| **TTA (e.g., flips + small rotations)** | Average or vote over transforms                    | Often +robustness with fixed weights                   | 2–8× compute; latency unless batched on GPU |
| **Tile / stride for 4K**                | Split frame into overlapping tiles                 | Recovers small faces                                   | More windows; duplicate suppression         |
| **Higher input resolution**             | e.g., 640 → 896 or 1024                            | Better for small faces                                 | Slower; may need batch=1 tuning             |


On your GPU, **TTA and multi-scale** are realistic **when latency budgets allow** (e.g., offline analysis, batch video, or non-real-time pipelines).

---

### 3.4 Multi-stage / “search then verify” geometry


| Approach                                                     | Idea                                                                 | Pros                                 | Cons                                                                     |
| ------------------------------------------------------------ | -------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| **PCN-style progressive calibration** (historical reference) | Coarse face candidates → progressively **correct in-plane rotation** | Designed for **360° roll** scenarios | More stages → more engineering; newer single-shot models may reduce need |
| **Discrete angle sweep**                                     | Run a fast detector on copies rotated by e.g. {−30°, 0°, +30°}       | Simple to implement; can rescue roll | 3× cost unless optimized; merge logic                                    |
| **Rough ROI then refine**                                    | Cheap motion / skin / prior box → heavy detector on crop             | Saves compute on full 4K             | False ROI hurts recall; tuning                                           |


**Trade-off summary:** These shine when **single-pass** misses are **structured** (e.g., strong roll) and you can pay **2–4×** compute for recall.

---

### 3.5 Oriented boxes and rotation-specific heads

General object detection has moved toward **rotated (oriented) boxes** for aerial scenes; for faces, **axis-aligned boxes** remain standard, but research explores **rotation-invariant** or **polar-transform** feature spaces and **OBB** formulations.


| Idea                                       | Pros                               | Cons                                                                                        |
| ------------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------------------------- |
| **OBB / angle regression**                 | Box matches tilted head outline    | More complex NMS; angle periodicity / boundary issues; less off-the-shelf tooling for faces |
| **Polar or rotation-equivariant features** | Reduces explicit rotation sampling | Newer / less turnkey in production face stacks                                              |


**Trade-off summary:** Usually **not the first lever** unless you own training and evaluation for **tilted bounding-box** definitions.

---

### 3.6 Landmarks, pose, and “assistive” second models


| Pattern                                        | Flow                                                                                | Pros                                                        | Cons                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------- |
| **Detect → align → re-detect on aligned chip** | Small landmark model or 5-point estimator to **canonicalize** the face patch        | Helps recognition / tracking; sometimes improves downstream | Extra model; can fail on extreme pose |
| **Pose-guided gating**                         | If pose estimator says “profile”, switch to **profile-tuned** detector or threshold | Can optimize per mode                                       | Two models; temporal stability        |


**Trade-off summary:** Strong when the product is **recognition or tracking**, not only a box for UI.

---

### 3.7 Deployment: TensorRT, mixed precision, and batching


| Option                                           | Effect                    | Trade-off                                                                    |
| ------------------------------------------------ | ------------------------- | ---------------------------------------------------------------------------- |
| **TensorRT** (FP16, later INT8 with calibration) | Higher FPS, lower latency | Build step per architecture; INT8 can hurt rare poses if not calibrated well |
| **Dynamic shapes**                               | One engine, variable H×W  | Slightly trickier than fixed                                                 |
| **CUDA streams + async**                         | Overlap copy and compute  | Implementation complexity                                                    |


Your GPU is a good fit for **FP16 TensorRT** as a default; **INT8** is optional after you confirm **profile and small-face** recall.

---

## 4. Application scenario: 1080p–2K webcam, accuracy-first, soft latency

This section narrows the proposal to **live webcam** input (roughly **1920×1080–2560×1440**), **no hard latency cap**, primary goal **detection accuracy**, and a **privacy** goal: behave sensibly when **face detection fails** but a **person’s head/face may still be in frame**.

### 4.1 What 1080p–2K implies

- At typical desk distances, the **face occupies a moderate-to-large** fraction of the frame; **full-frame** face detection at **512–896 px** short side (or letterboxed **640×640**) is usually **well within** RTX 4080 Super capacity, so you are rarely forced into tiny-model trade-offs.
- **2K** increases pixels but not necessarily face size; the main effect is **more background** and slightly higher decode/resize cost—still minor compared to a heavy detector on GPU.
- Accuracy work should prioritize **recall on profile / pitch / partial occlusion** and **temporal consistency**, not raw FPS unless you later optimize.

### 4.2 Head- or pose-guided search (ROI shrink, then face)

Using a **head / upper-body / pose** stage to **propose a crop**, then running a **strong face detector** only inside that crop, is a legitimate way to improve **accuracy per pixel** and sometimes **efficiency**:


| Stage                                                                                                   | Typical role                      | Benefit for angled faces                                                                                                       | Caveats                                                                                                             |
| ------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Person or upper-body detector**                                                                       | Coarse “human in frame”           | Reduces full-frame false face picks on posters/screens                                                                         | Does not localize face; may be wide ROI                                                                             |
| **Head / face landmark–free head box** (e.g., pose mesh, holistic landmarks, dedicated “head” detector) | Tighter ROI around the cranium    | Profile and turned heads often have **clearer silhouette** than a small face box; crop **enlarges** the face in detector input | Back of head, hoods, hair; occasional false head boxes                                                              |
| **Face detector on crop**                                                                               | High-res face in a smaller tensor | Better **effective resolution** on the face; can use **higher input** to SCRFD/YOLO-face within crop                           | Must **expand** crop margins (shoulders, rotation) so you do not clip cheeks; merge boxes back to full-frame coords |


**Practical merge rule:** Expand the head ROI by a **fixed margin** (e.g., 15–35% per side) or from **nose/ear landmarks** if available, run face detection, then **union** with last known face box under tracking (see below) to avoid flicker.

*Repo note:* A **narrow** version of “expand ROI → detect on crop” exists for **YuNet** only (`--cascade-margin`), using the **smoothed** box from the prior frame—not a separate head/pose model. Helpers like `union_boxes` exist for future merge logic.

This cascade is **orthogonal** to BlazeFace vs SCRFD: the second stage should still be a **strong** face model if accuracy is the goal.

### 4.3 Temporal tracking: when it helps latency and stability

Without a hard latency cap, tracking is still valuable: it **reduces how often** you run expensive full-frame or multi-scale detection, and it **smooths** boxes so downstream privacy logic does not thrash.


| Mechanism                                                                           | Role                                    | Privacy / engineering notes                                                     |
| ----------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| **IoU / Kalman association**                                                        | Link detections frame-to-frame          | No appearance embedding; good default                                           |
| **Optical flow** (sparse LK on a grid or corners, or dense flow on a **small ROI**) | Predict box shift when motion is smooth | Cheap on CPU; can **warp** last rectangle or mask; weak on sudden moves or cuts |
| **Discriminative short-term trackers** (CSRT, KCF, Nano, etc.)                      | Track a **rectangle** given init        | **Do not** treat as identity; drift on occlusion—**re-detect** periodically     |
| **Learned trackers** (e.g., transformer or correlation filters on crops)            | Stronger motion modeling                | Heavier; validate on your motion patterns                                       |


**Suggested pattern:** **Detect** every *N* frames or when **track confidence** drops; **track** (flow or lightweight tracker) on intermediate frames inside the **head ROI** or last **expanded face box**. On **detection failure**, keep a **conservative “attention” region** from head/pose or last good box for privacy (next subsection).

*Repo note:* Default mode uses **EMA** on box center (2D) and size when a face is present, plus **no-face debounce** (`--no-face-blur-frames`). **`--roi-motion`** selects **Kalman** or **Kalman + LK flow** instead; detect-every-N and richer trackers are still open.

### 4.4 Privacy-preserving behavior when face detection fails

The product requirement: **maximize accurate face localization when possible**, but when the detector is **uncertain or silent**, still **protect** regions where a **face plausibly remains**—without relying on **identity** or long-term **biometric** storage.

Design principles:

1. **Hierarchy of signals (coarse → fine):** Maintain parallel hypotheses: **person present** → **head/upper body** → **face**. Face detector high confidence refines the **inner** mask; when it fails, **fall back** to a **larger, coarser** region derived from head pose, body keypoints, or **stabilized** last-known geometry.
2. **Conservative default:** If **tracking** suggests a stable head region but **face score** drops (profile, motion blur), treat the **interior** of the head ROI as **sensitive**: apply blur, block, or redact **precautionarily**. Prefer **over-covering** over leaking identifiable texture at low face confidence.
3. **Avoid embedding-based Re-ID for “privacy” pipelines** unless you explicitly need re-identification: **appearance embeddings** (often used in multi-object trackers) increase **identifiability** of stored or leaked state. Prefer **geometric** association (IoU, centroid distance capped by motion, flow consistency).
4. **Volatile buffers:** Keep **raw frames** and detailed crops in **short-lived** memory; persist only **non-identifying** metadata (e.g., “privacy zone active”, timestamps, coarse box if policy allows) if persistence is required.
5. **Explicit degradation levels:** e.g., **Level A** — high-confidence face box, tight mask; **Level B** — head/pose-only region, expanded ellipse; **Level C** — person-level bbox only. Transitions should **hysteresis** (stick to B briefly after A drops) to reduce flicker.

Optional aids when faces are hard (use only if acceptable to your threat model): **skin-tone–free** saliency or **motion-only** masks for “something changed here”—crude, but can flag **review** or **temporary** wide redaction.

### 4.5 Summary for this scenario

- **1080p–2K webcam + 4080 Super:** favor **accurate** face models and **head-guided crops** for hard poses; latency is usually **not** the binding constraint.
- **Head → face** cascades can **improve** effective resolution and reduce full-frame confusion; they need **margin** and **fallback** when head models fail.
- **Optical flow or lightweight tracking** can **lower average** compute and stabilize boxes; **re-detect** on drift or schedule.
- **Privacy:** combine **hierarchical** regions, **conservative** coverage when face confidence drops, **non-biometric** tracking, and **hysteresis** so “detection failed” does not mean “privacy off.”

---

## 5. Suggested decision paths (practical)

1. **Baseline upgrade path (most common):** Move from BlazeFace-class to **SCRFD** or a **well-tested YOLOv8-face** fork → validate on **your** angled clips → enable **multi-scale** or light **TTA** if recall is still short.
2. **Data path (if errors are domain-specific):** Curate failure cases → **fine-tune** with strong augmentation (roll, scale, blur) → re-check false positives on background classes.
3. **Latency-sensitive path:** Single **SCRFD** TensorRT FP16 engine, fixed resolution, minimal scales; add **angle sweep** only on **tracks** that are unstable.
4. **Throughput path (offline / batch):** Larger input, **multi-scale + TTA**, optional **ensemble** (two architectures) using your **RAM** to buffer frames.
5. **Webcam + accuracy + privacy (this use case):** Run a **head or pose ROI** (with generous margin) → **strong face detector** inside crop; **track** with IoU/Kalman plus **sparse optical flow** or a lightweight rectangle tracker between full detections; define **A/B/C degradation** (face vs head vs person) with **hysteresis** when face scores drop; avoid **Re-ID embeddings** unless you explicitly accept higher identifiability.

---

## 6. Risks and evaluation pitfalls

- **Metric mismatch:** WIDER FACE **Hard** AP is useful but not a substitute for **your** profile and motion-blur distribution.
- **NMS / duplicate boxes:** Multi-scale and TTA require careful merging to avoid **double detections** or **suppressed true positives**.
- **Temporal jitter:** Video may need **tracking** (IoU / Kalman) to stabilize boxes across angles.
- **Privacy vs. utility:** Over-wide **precautionary** redaction may obscure non-face content; tune margins with **user-visible** or **internal** QA. **Head/body** models can miss or mis-localize; combine with **decay timers** so privacy zones do not disappear instantly on a single bad frame.
- **Tracker drift:** Short-term trackers and optical flow **drift** on occlusion; **periodic re-detection** is required to avoid **persistently wrong** privacy regions.

---

## 7. References and starting points (non-exhaustive)

- SCRFD (ICLR 2022) — InsightFace project and `detection/scrfd` documentation.
- RetinaFace (CVPR 2020) — widely used single-shot face localization baseline.
- WIDER FACE — standard benchmark; **Easy / Medium / Hard** protocol.
- PCN and later rotation-invariant cascades — useful conceptual background for **in-plane** rotation handling.
- Community YOLOv8-face repositories — verify license, maintenance, and reported WIDER FACE numbers before standardizing.

---

## 8. Summary

Improving detection “at different angles” is usually a **stacked** problem: **stronger modern detectors** (SCRFD / YOLO-face / RetinaFace-class), **better training data or fine-tuning** for your domain, and **inference scaling** (multi-scale, TTA, occasional rotation sweeps). Your **RTX 4080 Super** and **large system memory** favor **TensorRT FP16**, **parallel augmentations**, and **multi-model or multi-scale** strategies where a phone-class model would be starved for compute. BlazeFace remains excellent for **ultra-light** targets but is rarely the ceiling on **desktop-class** hardware when angle robustness is the priority.

For **1080p–2K webcam** workloads with **accuracy** and **privacy** as top concerns, combine **head-guided crops** (for harder poses), **geometric tracking** with **scheduled re-detection**, and **tiered privacy coverage** when face confidence drops—so “no face box” does not imply “no sensitive region,” without defaulting to **embedding-heavy** multi-object tracking unless you explicitly want that identification surface.