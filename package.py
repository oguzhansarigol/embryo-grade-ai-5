"""Build the final delivery archive (doc §7) — `DeepEmbryo_Teslim.zip`.

Bundles the trained model, source code, app, notebook and reports. RAR is not
available on Colab; create a zip here and re-compress to .rar locally if the
course requires that exact extension.
"""
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"
DELIVERY = ROOT / "DeepEmbryo_Teslim"

INCLUDE = [
    ("src", "src"),
    ("app", "app"),
    ("notebooks/DeepEmbryo_Colab.ipynb", "notebooks/DeepEmbryo_Colab.ipynb"),
    ("requirements.txt", "requirements.txt"),
    ("README.md", "README.md"),
    ("Proje Analiz ve İsterler Dokümanı-2.pdf",
     "Proje Analiz ve İsterler Dokümanı-2.pdf"),
    ("outputs/checkpoints/final_model.pth", "model/final_model.pth"),
    ("outputs/figures", "reports/figures"),
    ("outputs/reports", "reports"),
    ("outputs/predictions_test.csv", "reports/predictions_test.csv"),
    ("outputs/logs", "reports/training_logs"),
]


def build():
    if DELIVERY.exists():
        shutil.rmtree(DELIVERY)
    DELIVERY.mkdir(parents=True)

    for src_rel, dst_rel in INCLUDE:
        src = ROOT / src_rel
        dst = DELIVERY / dst_rel
        if not src.exists():
            print(f"  ! atlandı (yok): {src_rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"  + {src_rel} -> {dst_rel}")

    archive = ROOT / "DeepEmbryo_Teslim"
    out = shutil.make_archive(str(archive), "zip", root_dir=DELIVERY)
    print(f"\nPaket hazır: {out}")
    print("Yerelde .rar'a çevirmek için: WinRAR ile bu zip'i aç → 'rar' olarak kaydet.")
    return out


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
