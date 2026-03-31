# Anatomy-DT: A Cross-Diffusion Digital Twin for Anatomical Evolution

**Status**: Under Review at ICML 2026

---

## Training

```bash
python3 train.py \
--data_root /path/to/UCSF_DT \
--images_csv metadata_images.csv \
--patients_csv metadata_patients.csv \
--grid 96 96 \
--epochs 50 \
--batch_size 8 \
--lr 1e-2 \
--K 5 \

```
