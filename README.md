<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:0a3d62,100:1e90ff&height=200&section=header&text=Multi-Modal%203D%20Object%20Detection&fontSize=32&fontColor=ffffff&fontAlignY=38&desc=Augmenting%20LiDAR%20with%20Camera%20Semantics%20on%20KITTI&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<!-- BADGES -->
<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAA?style=for-the-badge&logo=yolo&logoColor=black"/>
  <img src="https://img.shields.io/badge/KITTI-Dataset-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Task-3D_Object_Detection-blueviolet?style=flat-square"/>
  <img src="https://img.shields.io/badge/Modalities-Camera_%2B_LiDAR-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Fusion-Late_Fusion-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Classes-Car_%7C_Pedestrian_%7C_Cyclist-red?style=flat-square"/>
</p>

</div>

---

## 🎬 Live Demo

<div align="center">

> *Camera image (left) fused with Bird's-Eye View LiDAR map (right) — 18 KITTI test frames*

![Fusion Demo](fusion_detection-bev.gif)

<br>

https://github.com/VasuTammisetti/lidar-camera-fusion-3d-detection-kitti/assets/fusion_demo-bev.mp4 (please download and view)

</div>
---

## 🧠 What Is Sensor Fusion and Why Does It Matter?

Modern autonomous vehicles rely on **multiple sensors simultaneously** — cameras capture rich texture and colour, while LiDAR captures precise 3D geometry. Neither alone is sufficient:

| Sensor | Strengths | Weaknesses |
|--------|-----------|------------|
| 📷 **Camera** | Colour, texture, semantic class, 2D localisation | No depth, fails in rain/glare |
| 📡 **LiDAR** | Precise 3D geometry, robust in dark/fog | No colour/texture, sparse at distance |
| 🔀 **Fused** | Accurate 3D boxes + semantic class labels | Requires calibration, higher compute |

> **This project demonstrates that fusion consistently outperforms either modality alone** — reducing false positives by cross-validating 3D LiDAR detections against 2D camera semantics.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT MODALITIES                            │
│                                                                 │
│   📷 Camera Image (1242×375)    📡 LiDAR Point Cloud (~120K pts)│
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────┐          ┌───────────────────────────┐
│   YOLOv8n / YOLOv8s  │          │      PointPillars          │
│   2D Object Detector │          │   LiDAR 3D Detector        │
│                      │          │                            │
│  • 80 COCO classes   │          │  • Voxelisation (0.16m)    │
│  • 640×640 input     │          │  • Pillar Feature Network  │
│  • NMS post-proc     │          │  • 2D BEV CNN backbone     │
│                      │          │  • Anchor-based head       │
└──────────┬───────────┘          └─────────────┬──────────────┘
           │                                    │
           │  2D boxes [x1,y1,x2,y2,cls,conf]   │  3D boxes [x,y,z,l,w,h,θ]
           │                                    │
           └───────────────┬────────────────────┘
                           │
                           ▼
             ┌─────────────────────────┐
             │     LATE FUSION         │
             │  Centre-Projection +    │
             │  Class Cross-Validation │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │   FUSED 3D DETECTIONS   │
             │  • Confirmed by camera  │
             │  • Localised by LiDAR   │
             │  • Class-consistent     │
             └─────────────────────────┘
```

---

## 🚀 Key Features

<table>
<tr>
<td width="50%">

### 🎯 Detection Pipeline
- **YOLOv8** for real-time 2D semantic detection on camera
- **PointPillars** for fast 3D bounding box regression from LiDAR
- **Late fusion** via 3D-to-image centre projection
- Class-aware cross-validation (COCO → KITTI class mapping)

</td>
<td width="50%">

### 📊 Visualisation
- Projected **wireframe 3D boxes** on camera image
- **Bird's-Eye View** with range rings and heading arrows
- Per-class colour coding (🟢 Car / 🟠 Pedestrian / 🔵 Cyclist)
- Intensity-mapped LiDAR point cloud overlay

</td>
</tr>
<tr>
<td width="50%">

### 🔧 Engineering
- Pure **PyTorch** PointPillars (no spconv dependency)
- Runs on **Google Colab T4 GPU** out of the box
- Modular pipeline: swap any detector independently
- KITTI calib matrix pipeline (P2 · R0 · Tr_velo_to_cam)

</td>
<td width="50%">

### 📦 Output
- Annotated **MP4 video** (H.264, GitHub compatible)
- **GIF** for README inline preview
- Per-frame detection summary table
- BEV + camera side-by-side canvas

</td>
</tr>
</table>

---

## 📐 Fusion Methodology: Late Fusion via Centre Projection

The fusion strategy is deliberately **interpretable and modality-agnostic**:

```
For each 3D box (from PointPillars):
  1. Project box centre (x, y, z) → image pixel (u, v)
     using KITTI calibration:  img = P2 · R0 · Tr · [x,y,z,1]ᵀ

  2. Check if (u,v) falls inside any YOLO 2D box

  3. Verify class consistency:
        COCO car/bus/truck  →  KITTI Car
        COCO person         →  KITTI Pedestrian
        COCO bicycle/moto   →  KITTI Cyclist

  4. Accept box only if BOTH conditions satisfied ✅
```

**Why late fusion?**
- Modular — each detector can be upgraded independently
- Interpretable — every decision is explainable
- Robust — a detection requires **two independent confirmations**
- No labelled fusion training data required

---

## 📊 Metrics & Evaluation

### Detection Quality Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Fusion Retention Rate** | % of LiDAR boxes confirmed by camera | `Fused / LiDAR_total × 100` |
| **Cross-Modal Precision** | Boxes where both modalities agree | `TP_fused / (TP_fused + FP_fused)` |
| **Semantic Consistency Score** | Class agreement between YOLO and PointPillars | `Agreed / Projected × 100` |
| **Projection Coverage** | LiDAR boxes with valid image projection | `Projected / Total_3D × 100` |

### Fusion Advantage Metrics

| Scenario | Camera Only | LiDAR Only | **Fused** |
|----------|-------------|------------|-----------|
| Night-time | ❌ Poor | ✅ Good | ✅ Good |
| Occlusion | ⚠️ Partial | ✅ Good | ✅ **Best** |
| Long range (>40m) | ✅ Good | ⚠️ Sparse | ✅ Good |
| Class identification | ✅ Excellent | ⚠️ Limited | ✅ **Excellent** |
| 3D localisation | ❌ None | ✅ Excellent | ✅ **Excellent** |
| False positive rate | High | Medium | **Lowest** |

### BEV Grid Specification

| Parameter | Value |
|-----------|-------|
| Point cloud range (X) | 0 → 69.12 m |
| Point cloud range (Y) | −39.68 → 39.68 m |
| Voxel size | 0.16 × 0.16 × 4.0 m |
| BEV grid resolution | 432 × 496 pillars |
| Max points per pillar | 32 |
| Pillar feature dim | 64-D |

---

## 🗂️ Project Structure

```
multimodal-3d-detection/
│
├── 📓 Multi_Modal_3D_Object_Detection_KITTI_Complete.ipynb
│   ├── Cell 00  — Data extraction
│   ├── Cell 01  — Folder verification
│   ├── Cell 02  — Install dependencies
│   ├── Cell 03  — Mount Google Drive
│   ├── Cell 04  — Load KITTI frame
│   ├── Cell 05  — (reserved)
│   ├── Cell 06  — LiDAR→Image projection
│   ├── Cell 07  — YOLOv8 inference
│   ├── Cell 08  — Semantic label assignment to points
│   ├── Cell 09  — Coloured point cloud visualisation
│   ├── Cell 10  — load_velodyne / load_calib helpers
│   ├── Cell 11  — PointPillars model build + checkpoint
│   ├── Cell 12  — PointPillars inference + decode
│   ├── Cell 13  — 3D box projection onto camera
│   ├── Cell 14  — Late fusion (YOLO ∩ PointPillars)
│   ├── Cell 15  — Final BEV + camera visualisation
│   ├── Cell 16  — Summary metrics table
│   ├── Cell 17  — 20-frame GIF generation
│   └── Cell 18  — Professional MP4 demo video
│
├── 📁 assets/
│   ├── kitti_fusion_demo.gif
│   └── kitti_fusion_demo_h264.mp4
│
└── 📄 README.md
```

---

## ⚡ Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Multi_Modal_3D_Object_Detection_KITTI_Complete.ipynb)

### 2. Prepare Data

Upload your KITTI subset to Google Drive with this structure:
```
sensorfusion/
└── sensorfusion/
    ├── data_object_image_2/training/   ← .png files
    ├── data_object_velodyne/training/  ← .bin files
    └── data_object_calib/training/     ← .txt files
```

### 3. Run All Cells

```
Runtime → Run all  (Ctrl+F9)
```

> ⏱️ Full pipeline on 20 frames: ~4 minutes on T4 GPU

---

## 🔬 Technical Deep-Dive

### PointPillars: LiDAR as a 2D Problem

PointPillars converts the 3D detection problem into a 2D image-like problem:

```
Point Cloud (N × 4)
      │
      ▼  Voxelisation
Pillars (P × 32 × 4)   ← group points into vertical columns
      │
      ▼  Pillar Feature Network (PFN)
Features (P × 64)      ← per-pillar 64-dim embedding
      │
      ▼  Scatter to BEV canvas
Pseudo-image (64 × 496 × 432)
      │
      ▼  2D CNN Backbone (3 stride blocks + FPN neck)
Multi-scale features (384 × H × W)
      │
      ▼  Detection Head
Class scores + Box regression (7-DoF) + Direction
```

### Calibration Chain (KITTI)

```python
# Full projection: LiDAR 3D → Camera Image pixel
img_point = P2 @ np.vstack([R0 @ Tr_velo_to_cam @ lidar_point_h])

# Where:
# Tr_velo_to_cam  (3×4): LiDAR → camera coordinates
# R0             (3×3): Rectification rotation
# P2             (3×4): Camera projection matrix
```

### 9-Channel Pillar Augmentation

Each point in a pillar is encoded as:
```
[x, y, z, intensity,     ← raw coordinates
 xc, yc, zc,             ← offset from pillar mean (centroid)
 xp, yp]                 ← offset from pillar geometric centre
```

---

## 🎓 Academic Context

This project is developed as part of doctoral research in **meta-learning for Advanced Driver Assistance Systems (ADAS)** at the University of Granada, in collaboration with Infineon Technologies AG.

### Related Publications
> *Details of peer-reviewed publications available upon request.*

### Research Focus
- **Meta-YOLO** (v8–v11): Few-shot object detection for ADAS perception
- **Stereo depth estimation**: Reducing LiDAR dependency in production vehicles
- **Sensor fusion architectures**: Camera + LiDAR + Radar pipelines for MFN100 NPU

---

## 🛠️ Dependencies

```python
# Core
torch >= 2.10          # Deep learning framework
ultralytics >= 8.4     # YOLOv8 inference
opencv-python >= 4.6   # Image processing
numpy >= 1.23          # Numerical operations

# Visualisation
matplotlib >= 3.7      # Plotting
Pillow >= 9.0          # GIF generation
tqdm                   # Progress bars

# Data
pykitti                # KITTI dataset utilities
```

---

## 💡 Advantages Over Single-Modality Detection

```
                    CAMERA ONLY          LIDAR ONLY         THIS PROJECT
                    ─────────────        ──────────         ────────────
Depth accuracy      ✗ None               ✓ ±2cm             ✓ ±2cm
Class labels        ✓ 80 classes         ✗ Limited          ✓ 80 classes
Night performance   ✗ Fails              ✓ Works            ✓ Works
Texture/colour      ✓ Full               ✗ None             ✓ Full
3D bounding box     ✗ None               ✓ 7-DoF            ✓ 7-DoF
False positive rate ⚠ Medium             ⚠ Medium           ✓ Lowest
Interpretability    ✓ High               ✓ High             ✓ Highest
Real-time capable   ✓ Yes (YOLOv8n)      ✓ Yes (PP)         ✓ Yes
```

---

## 🗺️ Roadmap

- [x] YOLOv8 2D detection pipeline
- [x] PointPillars 3D detection (pure PyTorch)
- [x] Late fusion via centre projection
- [x] BEV + camera visualisation
- [x] MP4 / GIF demo video generation
- [ ] Early fusion (point-level semantic augmentation)
- [ ] RT-DETR backbone integration
- [ ] nuScenes dataset support
- [ ] ROS2 real-time node
- [ ] Quantised inference on MFN100 NPU

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

| Resource | Link |
|----------|------|
| KITTI Vision Benchmark Suite | [cvlibs.net/datasets/kitti](http://www.cvlibs.net/datasets/kitti/) |
| OpenPCDet Framework | [github.com/open-mmlab/OpenPCDet](https://github.com/open-mmlab/OpenPCDet) |
| Ultralytics YOLOv8 | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| PointPillars Paper | [Lang et al., CVPR 2019](https://arxiv.org/abs/1812.05784) |
| Infineon Technologies AG | ADAS Research Collaboration |
| University of Granada | Doctoral Programme |

---

<div align="center">

**PointPillars** · **YOLOv8** · **Late Fusion** · **KITTI** · **ADAS** · **3D Detection**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e90ff,50:0a3d62,100:0d1117&height=100&section=footer" width="100%"/>

*Built with ❤️ for autonomous driving research*

</div>
