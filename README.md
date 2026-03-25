# MAViS: A Multi-Agent Approach for Training-Free Referring Video Object Segmentation

🚀 **MAViS** is a training-free multi-agent pipeline for referring video object segmentation.

🎉 To be accepted in **IEEE Transactions on Consumer Electronics (TCE)**

---

## ✨ Highlights

- 🔹 Training-free referring video object segmentation
- 🔹 Multi-agent pipeline (Video Summary → Keyframe Selection → Object Grounding)
- 🔹 Efficient inference with local MLLM (Qwen-VL)
- 🔹 Compatible with multiple benchmarks (Ref-Youtube-VOS, DAVIS17, MeViS, RVOS)

---

## 📦 Installation

### 1️⃣ Create environment

```bash
conda env create -f environment.yml
conda activate mavis
pip install -r requirements.txt
```
### 2️⃣ Install Python dependencies
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2
cd Grounded-SAM-2

pip install -e ./grounding_dino
pip install -e .
```
### ⚠️ Optional (Performance Boost)
```bash
pip install flash-attn
```

### ⚠️Transformers Version
We recommend:
```bash
pip install "transformers>=4.50"
```
### 📁 Project Structure
MAViS/
├── agents/                     # three core agents
├── pipeline/
│   ├── mavis_inference_pipeline.py
│   └── benchmark_inference_pipeline.py
├── datasets/
├── requirements.txt
├── environment.yml
└── README.md

## Data Preparation
Referring Video Object Segmentation
For Ref-Youtube-VOS and Ref-DAVIS17, data preparation follows [ReferFormer](https://github.com/wjn922/ReferFormer). For [MeViS](https://github.com/henghuiding/MeViS), please follow the data preparation instructions provided in MeViS.

## Evaluation
Before jumping into the following commands, you may look into the involved scripts and configure the data paths.

### MeViS

Submit your result to the online evaluation [server](https://www.codabench.org/competitions/11420/).

### Ref-YouTube-VOS

Submit your result to the online evaluation [server](https://codalab.lisn.upsaclay.fr/competitions/3282#participate-submit_results).

### Ref-DAVIS-17

###🚀 Quick Demo (Single Video)
python mavis_inference_pipeline.py \
  --video_dir /path/to/video_frames \
  --description "a man in black shirt" \
  --sam2_checkpoint /path/to/sam2.pt \
  --sam2_config /path/to/sam2.yaml \
  --output_json output.json

###🧪 Benchmark Inference
We provide a unified benchmark script:
python benchmark_inference_pipeline.py \
  --dataset {referformer,davis,rvos,mevis} \
  --dataset_root /path/to/dataset \
  --split train \
  --sam2_checkpoint /path/to/sam2.pt \
  --sam2_config /path/to/sam2.yaml \
  --prediction_root outputs \
  --metadata_root meta

## Release Notes
- **[2026/01/06]** 🔥 Release our training-free Referring Video Object Segmentation GitHub page.
- **[2025/12/25]** 🎉 Our Paper has been accepted by **IEEE Transactions on Consumer Electronics.**!
