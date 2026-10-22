# MTG Card Corner Detection

## Environnement : WSL Ubuntu 24.04

### 1. Installation des dépendances

```bash
cd mtg
source venvwsl/bin/activate
pip install ultralytics
```

---

## Entraîner le modèle YOLO Pose (model 5)

### 2. Configurer le modèle

Dans `algo_corner_detection.py`, ligne 20 :

```python
selection_model = 5
```

### 3. Lancer l'entraînement

```bash
source venvwsl/bin/activate
python3 algo_corner_detection.py
```

Cela va automatiquement :
- Générer 2000 images synthétiques dans `yolo_pose_dataset/`
- Télécharger `yolov8s-pose.pt` (~22 MB) depuis Ultralytics
- Entraîner le modèle (50 epochs)
- Sauvegarder le meilleur modèle dans `runs/pose/card_corners/weights/best.pt`

---

## Entraîner les modèles Keras (model 0 à 4)

### 2. Configurer le modèle

Dans `algo_corner_detection.py`, ligne 20 :

```python
selection_model = 0  # ou 1, 2, 3, 4
```

### 3. Pré-générer le dataset (optionnel, recommandé)

```bash
source venvwsl/bin/activate
python3 dataset.py
```

### 4. Lancer l'entraînement

```bash
source venvwsl/bin/activate
python3 algo_corner_detection.py
```

---

## Inference webcam (Windows uniquement)

```Powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python3 webcam.py
```

> `selection_model` doit correspondre au modèle entraîné.
