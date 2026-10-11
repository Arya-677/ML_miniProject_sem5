# Distracted Driver Detection (UE24CS352A – ML Mini-Project, Problem #24)

**Team:** Arya Devang (PES1UG24AM054), Ananthakrishna (PES1UG24AM036)

Classifies driver images from the State Farm Distracted Driver Detection dataset (10 classes: safe driving + 9 distractions)
with five models:

| Model | Implementation |
|---|---|
| Linear SVM (hinge loss + L2) | NumPy, from scratch |
| Softmax classifier | NumPy, from scratch |
| Two-layer neural net (100 ReLU units) | NumPy, from scratch |
| Gaussian Naive Bayes | scikit-learn |
| Decision Tree | scikit-learn |

Pipeline: image (640x480x3) → resize 64x64x3 → flatten (12,288) → scale to [0,1], subtract train-mean image → classifier.
Based on the CS229 report *Machine Learning Techniques for Distracted Driver Detection* (Feng & Yue, 2019).

## Results (validation accuracy, %)

| Model | Random split | Driver-wise split |
|---|---|---|
| Naive Bayes | 55.22 | 32.61 |
| Decision Tree | 89.12 | 29.66 |
| Linear SVM | 95.34 | 45.39 |
| Softmax | 71.39 | 41.93 |
| Two-layer NN | **98.02** | **51.55** |

Holding out whole drivers (frames of one driver are near-duplicates) drops accuracy sharply, so the random split is optimistic.
Full numbers, per-class accuracy and confusion matrices: `results_random/` and `results_driver/`.

## Repository contents
| File | What it is |
|---|---|
| `distracted_driver_detection.ipynb` | Everything: data download, preprocessing, models, gradient check, tuning, evaluation, live demo, report generation (outputs included) |
| `writeup.pdf` | Two-page write-up |
| `slides.pptx` | Presentation slides |
| `results_random/`, `results_driver/` | `results.json`, `confusion_matrix.png`, `loss_curves.png`, saved models (`model_*.npz`), `preprocess.npz` |
| `requirements.txt` | Python dependencies |

## Run on Google Colab (recommended)
1. Upload the notebook to https://colab.research.google.com (File → Upload notebook).
2. Accept the competition rules at https://www.kaggle.com/c/state-farm-distracted-driver-detection/rules and create a Kaggle API token
   (Kaggle → Settings → API). Either upload `kaggle.json` when asked, or set `KAGGLE_API_TOKEN` (see the commented line in the first cell).
3. Edit `NAMES`, `SRNS`, `REPO_URL` in section 9, then *Runtime → Run all* (about 40 minutes on the free CPU).

## Run locally
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook distracted_driver_detection.ipynb
```
The notebook downloads the data with the Kaggle CLI. Alternatively unzip the Kaggle data to `data/state-farm/` so that it contains
`driver_imgs_list.csv` and `train/c0 … c9/`.

## Live demo
Last notebook section: loads `results_driver/model_nn.npz`, takes any driver image (`IMAGE_PATH`) and prints the predicted activity with probabilities.

## Notes
- Hyper-parameters are chosen on an inner split of the training data, never on the validation set.
- Fixed seed (SVM/softmax are sensitive to weight initialisation); results use a single seed and a single split.
- The decision tree on 12,288 features is the slowest step (about 4–5 minutes per protocol).
# ML_miniProject_sem5
