import os
import csv
import numpy as np
import cv2 as cv
from pathlib import Path

# ------------ Config ------------
DATA_ROOT = Path("data")  # répertoire qui contient s1, s2, ..., s40
OUT_DIR   = Path("scripts/orb_out")  # répertoire de sortie pour les empreintes ORB
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ORB: paramètres courants; ajustez au besoin
ORB_NFEATURES = 500
ORB_SCALEFACTOR = 1.2
ORB_NLEVELS = 8
ORB_WTA_K = 2            # 2 -> Hamming distance standard (256 bits)
ORB_PATCHSIZE = 31
ORB_FAST_TH = 20         # threshold pour FAST (détection de coins)

# --------------------------------

def load_gray(img_path: Path) -> np.ndarray:
    """Charge une image PGM en niveaux de gris; taille attendue 92x112."""
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Impossible de lire {img_path}")
    # Optionnel: normalisation légère pour robustesse
    img = cv.equalizeHist(img)
    return img

def orb_descriptors(img: np.ndarray) -> np.ndarray:
    """Retourne un tableau (N, 32) uint8 de descripteurs ORB."""
    orb = cv.ORB_create(
        nfeatures=ORB_NFEATURES,
        scaleFactor=ORB_SCALEFACTOR,
        nlevels=ORB_NLEVELS,
        edgeThreshold=ORB_PATCHSIZE,
        firstLevel=0,
        WTA_K=ORB_WTA_K,
        scoreType=cv.ORB_HARRIS_SCORE,
        patchSize=ORB_PATCHSIZE,
        fastThreshold=ORB_FAST_TH,
    )
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    # des: None si aucun point détecté
    return des

def majority_bit_aggregate(des: np.ndarray) -> np.ndarray:
    """
    Agrège (N,32) -> (32,) uint8 (256 bits) via vote majoritaire bit-à-bit.
    Si N==0, renvoie 32 octets de 0.
    """
    if des is None or len(des) == 0:
        return np.zeros(32, dtype=np.uint8)

    # Unpack en bits (N, 32*8) = (N, 256)
    bits = np.unpackbits(des, axis=1)  # uint8 -> {0,1}
    # Vote majoritaire
    votes = bits.mean(axis=0) > 0.5     # bool, longueur 256
    # Repack en 32 octets
    agg = np.packbits(votes.astype(np.uint8))
    return agg  # shape (32,)

def hamming_256(a_bytes: np.ndarray, b_bytes: np.ndarray) -> int:
    """
    Hamming sur 256 bits (32 octets) côté clair, utile pour valider.
    """
    x = np.bitwise_xor(a_bytes, b_bytes)
    # popcount vectorisée
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return int(lut[x].sum())

def build_all(data_root: Path, out_dir: Path):
    rows = []
    for s in sorted(data_root.glob("s*")):
        if not s.is_dir():
            continue
        subject = s.name  # "s1", ...
        for img_path in sorted(s.glob("*.pgm")):
            img = load_gray(img_path)
            des = orb_descriptors(img)
            fp = majority_bit_aggregate(des)  # 32 octets
            # Sauvegarde binaire .npy (facile à recharger)
            npy_path = out_dir / f"{subject}_{img_path.stem}.npy"
            # fp_le = fp.astype('<u1')   # "little-endian uint8"
            # np.save(npy_path, fp_le)
            np.save(npy_path, fp)
            # Ligne CSV: id, chemin brut, hex (64 hex chars = 32 octets)
            rows.append({
                "id": f"{subject}/{img_path.stem}",
                "path": str(img_path),
                "hex_256": fp.tobytes().hex()
            })
    # CSV récap
    csv_path = out_dir / "orb_fingerprints.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "path", "hex_256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✔ Sauvé {len(rows)} empreintes dans {csv_path}")

if __name__ == "__main__":
    # build_all(DATA_ROOT, OUT_DIR)

    # Petit test: distance intra-sujet vs inter-sujet
    # Charge deux empreintes et affiche distance
    # (adaptez les chemins au besoin)
    a = np.load(OUT_DIR / "s1_1.npy")
    b = np.load(OUT_DIR / "s1_2.npy")
    # c = np.load(OUT_DIR / "s2_1.npy")
    for i in range(32):
        print(f"Byte {i}: {a[i]} {b[i]}")
    print("Hamming(s1/1, s1/2) =", hamming_256(a, b))
    # print("Hamming(s1/1, s2/1) =", hamming_256(a, c))
