# Pancreatic Tumor Classification & XAI Trustworthiness via 3D Foundation Model (CT-FM)

This repository outlines the technical roadmap, data preprocessing guidelines, and experimental setup for evaluating **pancreatic tumor benign vs. malignant classification** and **explainable AI (XAI) disagreement as a proxy for trustworthiness**, powered by the 3D CT vision foundation model **CT-FM**.

---

## 1. Selected Pre-trained Foundation Model
* **Model Name**: **CT-FM** (*A 3D Image-Based Foundation Model for Computed Tomography*)
* **Official Repository**: [project-lighter/CT-FM](https://github.com/project-lighter/CT-FM) (Pre-trained weights hosted on Hugging Face)
* **Background & Highlights**:
  * An open-source, native 3D CT vision foundation model developed by Harvard/MGH and collaborators.
  * Pre-trained on **148,000 unannotated 3D CT scans** using self-supervised contrastive learning (modified 3D SimCLR).
  * Backbone architecture: 3D **`SegResEncoder`** (~77M parameters).
  * Encodes rich, general-purpose anatomical and pathological representations directly suitable as a feature extractor for downstream tasks.

---

## 2. External Dataset & Leakage-Free Sampling: PanTS
To train and validate our downstream classification and XAI pipelines, we leverage the **PanTS** (*Pancreatic Tumor Segmentation Dataset*, [arXiv:2507.01291](https://arxiv.org/abs/2507.01291), [MrGiovanni/PanTS](https://github.com/MrGiovanni/PanTS)):

* **Hugging Face Dataset Access Link**:
  * Official Hugging Face repository: [huggingface.co/datasets/BodyMaps/PanTSMini](https://huggingface.co/datasets/BodyMaps/PanTSMini) (and full training partition releases indexed under the `BodyMaps` organization on Hugging Face).
  * Official download script available in the GitHub repo (`bash download_PanTS_data.sh`).
* **Zero-Leakage & Anti-Contamination Sampling Protocol**:
  * **Strict Exclusion**: To prevent data contamination against CT-FM's pre-training corpus (sourced from NCI IDC / TCIA), we strictly **exclude** legacy cohorts including `TCIA-PDAC`, `MSD-Pancreas (Task07)`, and early `KiTS` mirrors.
  * **Inclusion & Stratified Sampling (~100 Cases)**:
    1. Filter for contrast-enhanced scans in the **portal venous or late arterial phases** with slice thickness $\le 3.0\text{ mm}$.
    2. Curate ~100 cases using stratified multi-center sampling across unseen institutional centers and scanner vendors.
    3. Map clinical/pathological metadata to binary labels: **Benign / Cystic Neoplasms (0)** vs. **Malignant / PDAC (1)**.

---

## 3. Implementation Pipeline (Tailored for Small-Sample Scenarios)
Given our curated ~100 pancreatic tumor CT cohort:

1. **Stage 1: Feature Extraction (Linear Probing / Frozen Backbone)**
   * **Freeze** the pre-trained CT-FM Encoder weights entirely to prevent overfitting on the small cohort.
   * Pass each patient's 3D ROI volume through the encoder in inference mode (`torch.no_grad()`) to extract dense embedding vectors.
2. **Stage 2: Downstream Classifier Training**
   * Attach a lightweight classification head (Adaptive 3D Pooling + Linear / MLP layers, or train a gradient boosting model like LightGBM).
   * Task: Binary classification (**Benign vs. Malignant**).
   * *Feasibility: Runs smoothly in seconds on a standard Apple Silicon Mac (M4, 24GB Unified Memory).*

---

## 4. Input Data Specifications (Pre-processing Guidelines)
Since CT-FM operates strictly on native 3D convolutions, inputs **must be continuous, localized 3D voxel patches**:

| Processing Dimension | Standard / Recommendation | Notes & Rationales |
| :--- | :--- | :--- |
| **Tensor Shape** | `(C, D, H, W)` | Single-channel: typically `(1, 16, 96, 96)` or `(1, 32, 96, 96)` |
| **Slice Depth ($D$, Z-axis)** | **16 or 32 consecutive slices** | **Do NOT skip slices!** Slices must be physically continuous. Must be $\ge 16$ to avoid out-of-bound errors during multi-stage 3D downsampling. |
| **In-plane Size ($H, W$)** | $96 \times 96$ (or $128 \times 128$) | Crop a localized ROI centered around the pancreas/lesion to eliminate non-relevant abdominal organs and air cavities. |
| **Intensity (HU Windowing)** | Abdominal soft-tissue window | Common window: $[ -100, 250 ]\text{ HU}$, then linearly scaled/normalized to $[0, 1]$ or $[-1, 1]$. |
| **Voxel Spacing** | Isotropic resampling | Recommended to resample volumes to a standardized physical spacing (e.g., $1.0 \times 1.0 \times 1.0 \text{ mm}^3$). |

---

## 5. Downstream Objective: XAI Disagreement & Trustworthiness Estimation
The core research contribution focuses on **evaluating the clinical reliability of model explanations**:

* **Two Complementary XAI Approaches**:
  1. **Grad-CAM**: Leverages deep feature maps and gradients to highlight macroscopic, semantic lesion regions (coarse-grained).
  2. **Integrated Gradients (IG)**: Uses axiomatic path-integral gradients back to input voxels to reveal micro-textures, necrotic cores, and invasive margins (fine-grained).
* **Working Hypothesis**:
  * **High Consensus (Low Disagreement)**: Both methods highlight the same spatial area $\rightarrow$ High probability that the explanation aligns with true tumor pathology (**High Trustworthiness**).
  * **High Divergence (High Disagreement)**: Methods point to conflicting anatomical structures (e.g., one on tumor, one on bowel gas, vessel calcification, or motion artifact) $\rightarrow$ Indicates potential "shortcut learning" or model fragility (**Trigger "Untrustworthy / Review Needed" alert**).
* **Validation & Evaluation Metric**:
  * Align spatial resolutions and compute 3D IoU / Dice between the two attribution maps.
  * Correlate the **Disagreement Score** with radiologists' ground-truth lesion masks (Pointing Game / Energy-in-BBox) to quantitatively verify whether XAI divergence can effectively predict actual explanation fidelity.

---

## 6. Next Steps & Action Items (后续具体工作规划)
1. **Metadata Inspection & Cohort Selection**:
   * Inspect the PanTS metadata table, apply the anti-leakage exclusion filter (drop TCIA / MSD cohorts), and select ~100 cases with stratified center distribution and binary benign/malignant labels.
2. **Patch Extraction Pipeline**:
   * Write a Python/MONAI preprocessing script to crop localized 3D patches (`16x96x96` or `32x96x96`) around the tumor centroid, apply soft-tissue HU clipping (`[-100, 250]`), and save normalized tensors.
3. **Feature Extraction & Baseline Classification**:
   * Load CT-FM `SegResEncoder` pre-trained weights, perform forward-only feature extraction, and train a lightweight classifier (Linear Probing / LightGBM) to record benchmark AUROC / F1.
4. **3D XAI Pipeline Implementation**:
   * Implement 3D Grad-CAM (using high-level convolutional layers) and 3D Integrated Gradients (with baseline and step size optimized for M4 24GB memory).
5. **Divergence vs. Fidelity Quantification**:
   * Compute pairwise voxel agreement (3D Dice / IoU) between Grad-CAM and IG, and benchmark against PanTS ground-truth lesion masks to test the core trustworthiness hypothesis.

---

## 7. Bilingual Reference (附中文版本)

<details>
<summary>点击展开中文版讨论便签</summary>

### 1. 采用的核心预训练模型
* **模型名称**：**CT-FM** (A 3D Image-Based Foundation Model for Computed Tomography)
* **开源地址**：`https://github.com/project-lighter/CT-FM`
* **背景与定位**：
  * 哈佛/MGH 等团队开源的原生 3D CT 放射学基础大模型。
  * 基于 14.8 万例 3D CT 扫描无监督对比学习预训练，骨干结构为 3D `SegResEncoder`（约 77M 参数）。
  * 具备通用的 3D 解剖表征与病灶感知能力，直接作为下游分类的特征提取器（Backbone）。

### 2. 外部数据集与零泄漏抽样方案：PanTS
* **Hugging Face 数据集地址**：
  * `https://huggingface.co/datasets/BodyMaps/PanTSMini`（及 Hugging Face `BodyMaps` 机构下的 PanTS 数据分区）。
* **防污染抽样策略（约 100 例）**：
  * **坚决剔除**：TCIA-PDAC、MSD-Pancreas 等老旧数据集（防止与 CT-FM 预训练的 IDC 库重合造成数据泄漏）。
  * **保留多中心子集**：保留门脉期/动脉期高质量增强 CT（层厚 $\le 3\text{mm}$），按医院机构和良恶性（PDAC vs 囊性/良性）分层随机抽取约 100 例。

### 3. 实施方案（适配小样本）
1. **阶段一：特征提取（Linear Probing / 冻结骨干）**：冻结 Encoder 权重，一次性提取高维特征向量。
2. **阶段二：下游分类头训练**：轻量分类头（Pooling + MLP / LightGBM），二分类目标（良性 vs. 恶性）。在 Mac M4 (24GB) 上可快速完成。

### 4. 输入数据规范
* **张量形状**：`(1, 16, 96, 96)` 或 `(1, 32, 96, 96)`。
* **连续切片**：必须是物理连续切片，不得少于 16 层（防止多级下采样越界）。
* **平面尺寸**：$96 \times 96$ 或 $128 \times 128$（局部 ROI 裁剪）。
* **窗宽窗位**：腹部软组织窗 $[-100, 250]\text{ HU}$，归一化到 $[0, 1]$。
* **重采样**：各向同性重采样至 $1.0 \times 1.0 \times 1.0 \text{ mm}^3$。

### 5. XAI 解释分歧度与可信度
* **方法组合**：Grad-CAM（宏观语义） + Integrated Gradients（微观体素）。
* **核心假设**：高共识 $\rightarrow$ 高可信；高分歧 $\rightarrow$ 触发不可信警报（如捷径学习、伪影干扰）。
* **定量指标**：对齐分辨率后计算 3D Dice / IoU，与真实病灶 Mask 验证命中率。

### 6. 后续具体工作规划
1. **元数据过滤与抽样**：下载元数据表，排除 TCIA/MSD，分层抽取约 100 例多中心良恶性病例。
2. **3D Patch 裁剪脚本开发**：基于 MONAI/NiBabel 裁剪 $16 \times 96 \times 96$ 的局部肿瘤 ROI。
3. **特征提取与分类基线**：加载 CT-FM 提取特征，跑通 LightGBM / MLP 分类基线。
4. **3D Grad-CAM 与 3D IG 部署**：基于 Captum 编写适配 M4 统一内存的串行累加 XAI 计算脚本。
5. **分歧度与真实病灶命中率关联分析**：评估 XAI 分歧度预测可信度的有效性。

</details>
