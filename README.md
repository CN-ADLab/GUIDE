<p align="center">
  <h1>GUIDE: Gaussian Unified Instance Detection for Enhanced Obstacle Perception in Autonomous Driving</h1>
</p>

<p align="center">
  <sup>AAAI 2026</sup>
</p>

<p align="center">
  <img src="https://github.com/CN-ADLab/GUIDE/new/main/images/teaser.png" alt="GUIDE Teaser Figure" width="80%" />
</p>

---

## 📌 Overview

**GUIDE** (Gaussian Unified Instance Detection) is a novel framework for autonomous driving perception that unifies **3D instance detection** and **occupancy tracking** using **sparse 3D Gaussian splatting**. Built upon the GaussianFormer architecture, GUIDE enables efficient, memory-friendly, and instance-aware 3D scene understanding — even under high-density, long-tail obstacle scenarios.

Unlike traditional voxel-based methods, GUIDE represents obstacles as learnable 3D Gaussians, allowing for:
- ✅ **Instance-level discrimination** without dense voxelization  
- ✅ **Computational efficiency** via sparse representation  
- ✅ **Seamless integration** of detection and tracking  

This work was accepted at **AAAI 2026**.
