# MediScan-Net 💊🔍

**AI-powered counterfeit medicine detection through visual authentication and OCR-based label verification.**

MediScan-Net combines a fine-tuned ResNet-18 classifier with Grad-CAM explainability and a rotation-invariant OCR pipeline to help verify medicine packaging authenticity and check expiry/batch validity — without specialized hardware.

## The Problem

The WHO estimates nearly 10% of medical products in developing countries are substandard or falsified, contributing to treatment failure and rising antibiotic resistance. Existing anti-counterfeiting methods (holograms, QR codes, RFID) protect the supply chain but don't let a consumer verify the actual product in their hand.

## What It Does

MediScan-Net runs two independent verification pipelines on a single photo and fuses the results into one transparent verdict:

- **Visual Authentication** — a fine-tuned ResNet-18 CNN classifies packaging as genuine or counterfeit, with Grad-CAM heatmaps showing exactly which regions influenced the decision.
- **OCR Label Verification** — a rotation-invariant, dual-preprocessing OCR pipeline extracts and validates expiry dates and batch numbers directly from the strip.
- **Decision Fusion** — combines both signals into a confidence-tiered verdict (e.g. `LIKELY GENUINE`, `SUSPICIOUS`, `REJECTED: Expired`, `UNCERTAIN`), and is explicit about which signal it's relying on when one is missing.

## Features

- 🧠 Transfer-learning CNN (ResNet-18, ImageNet-pretrained, `layer4` fine-tuned)
- 🔥 Grad-CAM visual explainability on every prediction
- 🔄 4-way rotation-invariant OCR with a fast orientation pre-scan
- 📅 Label-aware expiry parsing — handles both numeric (`05/2026`) and month-name (`JUL 2026`) formats, distinguishes MFG from EXP dates
- ⚖️ Rule-based decision fusion with confidence tiers, never a silent single-signal guess
- 🧹 Dataset integrity tooling — duplicate-hash checker across train/val/test splits, synthetic-file cleanup utility

## Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch, torchvision (ResNet-18) |
| OCR | EasyOCR, OpenCV (CLAHE, Otsu thresholding) |
| Explainability | Custom Grad-CAM (hook-based) |
| Evaluation | scikit-learn, Matplotlib |
| Language | Python 3 |

## Project Structure
