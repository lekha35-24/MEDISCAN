## Setup

```bash
git clone https://github.com/lekha35-24/MEDISCAN.git
cd MEDISCAN
pip install torch torchvision opencv-python easyocr scikit-learn matplotlib
```

## Usage

**Train:**
```bash
python train_model.py
```

**Evaluate:**
```bash
python evaluatemodel.py
```

**Run inference:**
```bash
python main.py
# Enter 1
# Enter image path when prompted
```

## Results

> Populate this section with real numbers from `evaluatemodel.py` — accuracy, precision, recall, F1, ROC-AUC — before publishing.

## Limitations

- Verifies packaging authenticity and label validity — **not** chemical composition. Not a substitute for laboratory-based drug verification.
- Performance depends on image quality, lighting, and viewing angle.
- OCR reliability decreases on heavily degraded, embossed-only, or very low-contrast printed text.
- Trained on a limited, self-collected dataset — generalization across all manufacturer packaging designs is not yet validated.

## Future Work

- Prescription-equivalence validation via NLP-based drug-name matching
- Lightweight backbone (MobileNetV3) for on-device mobile deployment
- Expanded, more diverse training dataset

## License

[Add your license here]

## Contact

[Your name / email / GitHub]
