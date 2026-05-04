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
    """Minimal PDF report of all stored predictions."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                        Paragraph, Spacer)
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        abort(500, "reportlab is not installed; run `pip install reportlab`.")

    rows = db.list_all(limit=10_000)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="DeepEmbryo Tahmin Geçmişi")
    styles = getSampleStyleSheet()
    elems = [Paragraph("DeepEmbryo — Tahmin Geçmişi", styles["Title"]),
             Spacer(1, 12)]
    if rows:
        cols = ["id", "timestamp", "image_filename", "predicted_class",
                "confidence", "warning_flag", "actual_class", "pregnancy_outcome"]
        data = [cols] + [[("" if r[c] is None else str(r[c])) for c in cols] for r in rows]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        elems.append(t)
    else:
        elems.append(Paragraph("Henüz tahmin kaydı yok.", styles["Normal"]))
    doc.build(elems)
    buf.seek(0)
    return Response(buf.read(), mimetype="application/pdf",
                    headers={"Content-Disposition":
                             "attachment; filename=deepembryo_history.pdf"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
