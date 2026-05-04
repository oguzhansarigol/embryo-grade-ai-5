"""Flask demo: upload an embryo image -> prediction + Grad-CAM + warning + history."""
import io
import sys
import uuid
from pathlib import Path

from flask import (
    Flask,
    request,
    url_for,
    Response,
    abort,
    jsonify,
    render_template,
    redirect,
    flash,
)
from flask_cors import CORS

# Ensure `src` is importable when launching `python app/app.py` from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as cfg  # noqa: E402
from src.infer import EmbryoPredictor  # noqa: E402
from app.db import HistoryDB  # noqa: E402

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "static" / "uploads"
GRADCAM_DIR = APP_ROOT / "static" / "gradcam"
DB_PATH = cfg.OUTPUT_DIR / "history.sqlite3"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(APP_ROOT / "static"))
CORS(app)
app.secret_key = "deepembryo-demo"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload cap

ALLOWED_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

predictor: EmbryoPredictor | None = None
db = HistoryDB(DB_PATH)


def get_predictor() -> EmbryoPredictor:
    """Lazy-load: avoids the multi-second model load on app import."""
    global predictor
    if predictor is None:
        try:
            predictor = EmbryoPredictor()
        except FileNotFoundError:
            # Common local path when the checkpoint is kept out of git due to size limits.
            fallback = Path(__file__).resolve().parent.parent / "model" / "final_model.pth"
            predictor = EmbryoPredictor(checkpoint_path=fallback)
    return predictor


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", classes=cfg.CLASSES)


@app.route("/predict", methods=["POST"])
def predict_page():
    files = request.files.getlist("images")
    files = [f for f in files if f and f.filename]
    if not files:
        flash("Lütfen en az bir görüntü yükleyin.")
        return redirect(url_for("index"))

    results = []
    errors: list[str] = []
    for f in files:
        if not _allowed(f.filename):
            errors.append(f"Desteklenmeyen dosya: {f.filename}")
            continue
        uid = uuid.uuid4().hex[:8]
        safe_name = f"{uid}_{Path(f.filename).name}"
        save_path = UPLOAD_DIR / safe_name
        f.save(save_path)

        gradcam_path = GRADCAM_DIR / f"{uid}_gradcam.png"
        try:
            pred = get_predictor().predict_single(save_path, gradcam_save_path=gradcam_path)
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
            continue

        pred_id = db.insert(
            image_filename=safe_name,
            predicted_class=pred.predicted_class,
            confidence=pred.confidence,
            warning_flag=pred.warning is not None,
            gradcam_path=str(gradcam_path.relative_to(APP_ROOT / "static")),
        )

        results.append(
            {
                "id": pred_id,
                "filename": safe_name,
                "image_url": url_for("static", filename=f"uploads/{safe_name}"),
                "gradcam_url": url_for("static", filename=f"gradcam/{gradcam_path.name}"),
                "predicted_class": pred.predicted_class,
                "confidence": pred.confidence,
                "warning": pred.warning,
                "probabilities": pred.probabilities,
            }
        )

    if not results:
        msg = "Analiz başlatılamadı."
        if errors:
            msg += " " + " | ".join(errors[:3])
        flash(msg)
        return redirect(url_for("index"))

    if errors:
        flash("Bazı dosyalar atlandı: " + " | ".join(errors[:3]))

    return render_template("result.html", results=results, threshold=cfg.CONFIDENCE_THRESHOLD)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    files = request.files.getlist("images")
    files = [f for f in files if f and f.filename]
    if not files:
        return jsonify({"error": "Lütfen en az bir görüntü yükleyin."}), 400

    results = []
    errors: list[str] = []
    for f in files:
        if not _allowed(f.filename):
            errors.append(f"Desteklenmeyen dosya: {f.filename}")
            continue
        uid = uuid.uuid4().hex[:8]
        safe_name = f"{uid}_{Path(f.filename).name}"
        save_path = UPLOAD_DIR / safe_name
        f.save(save_path)

        gradcam_path = GRADCAM_DIR / f"{uid}_gradcam.png"
        try:
            pred = get_predictor().predict_single(save_path,
                                                  gradcam_save_path=gradcam_path)
        except Exception as e:
            errors.append(f"{f.filename}: {e}")
            continue

        pred_id = db.insert(
            image_filename=safe_name,
            predicted_class=pred.predicted_class,
            confidence=pred.confidence,
            warning_flag=pred.warning is not None,
            gradcam_path=str(gradcam_path.relative_to(APP_ROOT / "static")),
        )

        results.append({
            "id": pred_id,
            "filename": safe_name,
            "image_url": request.host_url + url_for("static", filename=f"uploads/{safe_name}").lstrip('/'),
            "gradcam_url": request.host_url + url_for("static", filename=f"gradcam/{gradcam_path.name}").lstrip('/'),
            "predicted_class": pred.predicted_class,
            "confidence": pred.confidence,
            "warning": pred.warning,
            "probabilities": pred.probabilities,
        })

    if not results:
        return jsonify({"error": "Hiçbir görüntü analiz edilemedi.", "details": errors}), 500
    return jsonify({"results": results, "threshold": cfg.CONFIDENCE_THRESHOLD, "errors": errors})


@app.route("/history", methods=["GET"])
def history_page():
    rows = db.list_all()
    return render_template("history.html", rows=rows)


@app.route("/api/history")
def api_history():
    rows = db.list_all()
    return jsonify([dict(r) for r in rows])


@app.route("/api/history/<int:pred_id>/followup", methods=["POST"])
def update_followup(pred_id: int):
    # Depending on what React sends: JSON or form data
    if request.is_json:
        data = request.get_json()
        actual = data.get("actual_class")
        outcome = data.get("pregnancy_outcome")
    else:
        actual = request.form.get("actual_class") or None
        outcome = request.form.get("pregnancy_outcome") or None
    db.update_followup(pred_id, actual, outcome)
    if request.is_json:
        return jsonify({"status": "success"})
    flash("Takip bilgisi kaydedildi.")
    return redirect(url_for("history_page"))


@app.route("/api/export.csv")
def export_csv():
    csv_text = db.export_csv()
    return Response(csv_text, mimetype="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=deepembryo_history.csv"})


@app.route("/api/export.pdf")
def export_pdf():
    """PDF report of stored predictions (with images)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph,
            Spacer,
            Image as RLImage,
            PageBreak,
        )
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        abort(500, "reportlab is not installed; run `pip install reportlab`.")

    # ---- Font: ensure Turkish characters render properly (ç, ğ, ı, İ, ö, ş, ü)
    font_name = "Helvetica"
    try:
        candidates = [
            Path("C:/Windows/Fonts/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/dejavusans.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/Arial.ttf"),
            # common in python envs
            Path(sys.prefix) / "Lib" / "site-packages" / "matplotlib" / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf",
        ]
        font_path = next((p for p in candidates if p.exists()), None)
        if font_path is not None:
            pdfmetrics.registerFont(TTFont("DeepEmbryoSans", str(font_path)))
            font_name = "DeepEmbryoSans"
    except Exception:
        # Fall back to Helvetica if font registration fails
        font_name = "Helvetica"

    rows = db.list_all(limit=10_000)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        title="DeepEmbryo Tahmin Geçmişi",
        leftMargin=1.2 * cm,
        rightMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "deepembryo_title",
        parent=styles["Title"],
        fontName=font_name,
    )
    normal_style = ParagraphStyle(
        "deepembryo_normal",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=11,
    )
    small_style = ParagraphStyle(
        "deepembryo_small",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=8,
        leading=10,
    )

    def _img_cell(path: Path, w_cm: float = 3.4, h_cm: float = 3.4):
        """Return a small image for the table cell; empty string if missing."""
        try:
            if not path.exists():
                return ""
            img = RLImage(str(path))
            img.drawWidth = w_cm * cm
            img.drawHeight = h_cm * cm
            return img
        except Exception:
            return ""

    # Summary header
    total = len(rows)
    warn_n = sum(1 for r in rows if int(r["warning_flag"]) == 1) if rows else 0
    elems = [
        Paragraph("DeepEmbryo — Tahmin Geçmişi Raporu", title_style),
        Spacer(1, 6),
        Paragraph(f"Toplam kayıt: <b>{total}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Uyarılı (güven &lt; 0.70): <b>{warn_n}</b>", normal_style),
        Spacer(1, 10),
    ]

    if rows:
        # Show most recent first, with thumbnails.
        cols = ["ID", "Tarih/Saat", "Dosya", "Tahmin", "Güven", "Uyarı", "Orijinal", "Grad-CAM", "Gerçek", "Sonuç"]
        data = [cols]

        static_root = APP_ROOT / "static"
        for r in rows:
            image_path = static_root / "uploads" / r["image_filename"]
            gradcam_rel = r["gradcam_path"]
            gradcam_path = static_root / Path(gradcam_rel) if gradcam_rel else None

            data.append([
                str(r["id"]),
                str(r["timestamp"]),
                str(r["image_filename"]),
                str(r["predicted_class"]),
                f"{float(r['confidence']):.3f}" if r["confidence"] is not None else "",
                "Evet" if int(r["warning_flag"]) == 1 else "Hayır",
                _img_cell(image_path),
                _img_cell(gradcam_path) if gradcam_path else "",
                "" if r["actual_class"] is None else str(r["actual_class"]),
                "" if r["pregnancy_outcome"] is None else str(r["pregnancy_outcome"]),
            ])

        col_widths = [1.0 * cm, 2.3 * cm, 3.2 * cm, 1.4 * cm, 1.4 * cm, 1.2 * cm, 3.6 * cm, 3.6 * cm, 1.4 * cm, 1.6 * cm]
        t = Table(data, repeatRows=1, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, -1), font_name),
            ("FONTSIZE", (0, 0), (-1, 0), 8),
            ("FONTSIZE", (0, 1), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (0, 1), (5, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightcyan]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        elems.append(t)
        elems.append(Spacer(1, 8))
        elems.append(Paragraph("Not: Görseller, ilgili kayıt sırasında saklanan dosyalardan üretilir. Silinmişse raporda boş görünür.", small_style))
    else:
        elems.append(Paragraph("Henüz tahmin kaydı yok.", normal_style))
    doc.build(elems)
    buf.seek(0)
    return Response(buf.read(), mimetype="application/pdf",
                    headers={"Content-Disposition":
                             "attachment; filename=deepembryo_history.pdf"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
