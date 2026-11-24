import os
import sys

# Ensure MKL / PyTorch DLLs in venv/Library/bin are discoverable on Windows
lib_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Library", "bin")
if os.path.exists(lib_bin) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(lib_bin)
        os.environ["PATH"] = lib_bin + ";" + os.environ.get("PATH", "")
    except Exception:
        pass

import base64

import cv2
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from backend.denoising import run_all_methods
from backend.metrics import evaluate_method
from backend.recommendation import recommend_best_method

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_bgr):
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/denoise", methods=["POST"])
def denoise():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use png, jpg, jpeg or bmp."}), 400

    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_DIR, filename)
    file.save(upload_path)

    image_bgr = cv2.imread(upload_path)
    if image_bgr is None:
        return jsonify({"error": "Could not read image. File may be corrupted."}), 400

    # Auto-resize images to fit safely within Render's 512MB RAM limit
    max_dim = 500
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    try:
        results = run_all_methods(image_bgr)
    except Exception as e:
        app.logger.error(f"Denoising error: {e}", exc_info=True)
        return jsonify({"error": f"Denoising failed: {str(e)}"}), 500

    response_methods = {}
    evaluations = {}
    session_id = os.path.splitext(filename)[0]

    for name, data in results.items():
        img = data["image"]
        metrics = evaluate_method(img, data["time_sec"])
        evaluations[name] = metrics

        out_filename = f"{session_id}_{name}.png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        cv2.imwrite(out_path, img)

        response_methods[name] = {
            "image_base64": image_to_base64(img),
            "metrics": metrics,
            "download_name": out_filename,
        }

    best_method, reasons, verdict_line = recommend_best_method(evaluations, raw_image=image_bgr)

    # Compute quick highlights
    fastest_method = min(evaluations, key=lambda k: evaluations[k]["time_sec"]) if evaluations else None
    least_noise_method = min(evaluations, key=lambda k: evaluations[k]["noise_estimate"]) if evaluations else None
    sharpest_method = max(evaluations, key=lambda k: evaluations[k]["sharpness"]) if evaluations else None

    return jsonify({
        "original_image_base64": image_to_base64(image_bgr),
        "methods": response_methods,
        "recommendation": {
            "method": best_method,
            "reasons": reasons,
            "verdict_line": verdict_line,
            "highlights": {
                "fastest": {"name": fastest_method, "value": f"{evaluations[fastest_method]['time_sec']}s"},
                "least_noise": {"name": least_noise_method, "value": evaluations[least_noise_method]["noise_estimate"]},
                "sharpest": {"name": sharpest_method, "value": evaluations[sharpest_method]["sharpness"]},
            } if evaluations else {},
        },
    })


@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(OUTPUT_DIR, secure_filename(filename))
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
