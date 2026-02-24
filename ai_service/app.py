from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import os
import json
import cv2 # OpenCV for robust image analysis

app = Flask(__name__)
CORS(app)

# Optional ML model. Prefer full 9-class model; fall back to HAM-only or OpenCV.
TF_AVAILABLE = False
tf = None
MODEL = None
MODEL_LABELS = None
USE_FULL_MODEL = False  # True if 9-class model loaded

HAM_LABELS_FALLBACK = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
HAM_LABEL_TO_NAME = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

def _load_labels(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and all(isinstance(x, str) for x in labels) and labels:
            return labels
    except Exception:
        return None
    return None

def _load_model_if_available():
    global TF_AVAILABLE, tf, MODEL, MODEL_LABELS, USE_FULL_MODEL
    try:
        import tensorflow as _tf
        tf = _tf
        TF_AVAILABLE = True
    except Exception:
        print("AI ENGINE: TensorFlow not available. Falling back to OpenCV heuristics.")
        return

    # Prefer full 9-class model
    full_model = os.environ.get("SKINDL_MODEL_PATH", os.path.join("models", "skin_full.keras"))
    ham_model = os.path.join("models", "ham_efficientnetb0.keras")
    full_labels = os.path.join("models", "skin_label_map.json")
    ham_labels_path = os.path.join("models", "ham_label_map.json")

    for model_path, labels_path, is_full in [
        (full_model, full_labels, True),
        (ham_model, ham_labels_path, False),
    ]:
        if not os.path.exists(model_path):
            continue
        try:
            MODEL = tf.keras.models.load_model(model_path)
            MODEL_LABELS = _load_labels(labels_path) or (CLASSES if is_full else HAM_LABELS_FALLBACK)
            USE_FULL_MODEL = is_full
            print(f"AI ENGINE: Loaded {'9-class' if is_full else 'HAM'} model from {model_path!r} with {len(MODEL_LABELS)} labels.")
            return
        except Exception as e:
            print(f"AI ENGINE: Failed to load {model_path}: {e}")
            continue

    MODEL = None
    MODEL_LABELS = None
    USE_FULL_MODEL = False
    print("AI ENGINE: No trained model found. Using OpenCV heuristics.")

_load_model_if_available()

CLASSES = [
    "Acne Vulgaris", "Atopic Dermatitis (Eczema)", "Psoriasis", 
    "Malignant Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma",
    "Melasma", "Rosacea", "Healthy Skin"
]

def get_progression(condition):
    progression = {
        "Acne Vulgaris": {
            "3_months": "With treatment: 60% reduction in inflammatory lesions. Without treatment: Potential for cystic progression and scarring.",
            "6_months": "With treatment: Near complete clearance, transitioning to post-inflammatory hyperpigmentation management."
        },
        "Atopic Dermatitis (Eczema)": {
            "3_months": "Potential for lichenification (thickening) of skin if scratching persists. Reduced inflammation with consistent corticosteroid use.",
            "6_months": "Maintenance phase; periodic flare-ups possible but severity should decrease with consistent barrier repair."
        },
        "Psoriasis": {
            "3_months": "Increased plaque thickness and scaling without systemic therapy. Potential joint pain (psoriatic arthritis) markers.",
            "6_months": "Plaques may spread to elbows/knees. Biological treatments could show 75-90% improvement by this stage."
        },
        "Malignant Melanoma": {
            "3_months": "CRITICAL: Potential vertical growth phase, increasing risk of depth-based metastasis. Lesion may become asymmetrical.",
            "6_months": "EXTREME RISK: Higher probability of lymph node involvement. Immediate biopsy and surgical intervention are non-negotiable."
        },
        "Basal Cell Carcinoma": {
            "3_months": "Slow enlargement (approx 1-2mm). Potential central ulceration (rodent ulcer) and 'pearly' border development.",
            "6_months": "Likely local tissue destruction. While rarely metastatic, it can cause significant disfigurement if left on the face."
        },
        "Squamous Cell Carcinoma": {
            "3_months": "Rapid growth likely. Lesion may become hard, raised, and painful. Higher risk of local invasion.",
            "6_months": "Moderate risk of metastasis to local lymph nodes. Requires definitive surgical or radiation intervention."
        },
        "Melasma": {
            "3_months": "Pigmentation may darken with UV exposure. Slow fading (10-20%) with strict adherence to triple-combination creams.",
            "6_months": "Noticeable improvement if UV protection is maintained. High recurrence risk upon sun exposure."
        },
        "Rosacea": {
            "3_months": "Increased telangiectasia (visible blood vessels) and persistent redness. Potential for ocular involvement.",
            "6_months": "Potential development of rhinophyma (thickening of nose skin) in severe untreated cases. Periodic flares continue."
        },
        "Healthy Skin": {
            "3_months": "Maintain routine: Continued hydration and broad-spectrum SPF 50+ to prevent photoaging.",
            "6_months": "No significant changes expected. Regular annual dermatological screenings recommended."
        }
    }
    return progression.get(condition, {
        "3_months": "Consult a specialist for specific long-term prognosis.",
        "6_months": "Regular monitoring and follow-up clinical exams required."
    })

def analyze_image_cv(img_array):
    """
    Absolute Clinical Barrier Engine.
    Uses mutual-exclusion thresholds to stop result swapping.
    """
    img_bgr = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 1. Advanced Preprocessing
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    if os.environ.get("SKINDL_DEBUG_IMAGES") == "1":
        cv2.imwrite("debug_input.jpg", img_bgr) # Optional debug output
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    pixel_count = float(hsv.shape[0] * hsv.shape[1]) or 1.0

    # 2. Hard Feature Extraction
    
    # REDNESS (Inflammation)
    red_mask = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])) | \
               cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    red_pct = (np.count_nonzero(red_mask) / pixel_count) * 100

    # DARK PIGMENT (Melanoma)
    # Target deep black/brown: Low Value (V < 60)
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
    dark_pct = (np.count_nonzero(dark_mask) / pixel_count) * 100

    # WHITE SCALES (Psoriasis)
    # Brightness > 220 + Local Roughness (Laplacian)
    white_mask = cv2.inRange(gray, 220, 255)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    scale_pct = (np.count_nonzero(white_mask & cv2.dilate(red_mask, None)) / pixel_count) * 100

    # ACNE BLOBS
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 12, param1=50, param2=25, minRadius=3, maxRadius=15)
    acne_count = len(circles[0]) if circles is not None else 0

    # 3. Decision Matrix (Barrier System)
    # Standard values
    scores = {
        "Acne Vulgaris": 0,
        "Atopic Dermatitis (Eczema)": 0,
        "Psoriasis": 0,
        "Malignant Melanoma": 0,
        "Healthy Skin": 50
    }

    # CRITICAL: Melanoma Logic (Strong Dark Barrier)
    # A mole is dark. A rash is not black.
    if dark_pct > 2.0 and red_pct < 60:
        scores["Malignant Melanoma"] = 500 + (dark_pct * 10)
    
    # CRITICAL: Psoriasis Logic (White Scale Barrier)
    # Must have lots of redness AND genuine white scales
    if red_pct > 15 and scale_pct > 1.5:
        scores["Psoriasis"] = 500 + (scale_pct * 20)
        # Block Melanoma if psoriasis is likely (scales)
        scores["Malignant Melanoma"] = 0

    # CRITICAL: Acne Logic (Blob Barrier)
    if acne_count >= 2:
        scores["Acne Vulgaris"] = 400 + (acne_count * 20)

    # CRITICAL: Eczema Logic (Pure Rash Barrier)
    # Only if it's mostly red without scales or dark pigment
    if red_pct > 20 and scale_pct < 1.0 and dark_pct < 1.5 and acne_count < 2:
        scores["Atopic Dermatitis (Eczema)"] = 450 + (red_pct * 2)

    # 4. Final Ranking
    total = sum(v for v in scores.values() if v > 0) or 1.0
    final = [{"label": k, "prob": round((v/total)*100, 1)} for k, v in scores.items()]
    final.sort(key=lambda x: x['prob'], reverse=True)

    print(f"DIAGNOSTIC: Red:{red_pct:.1f}% | Dark:{dark_pct:.1f}% | Scale:{scale_pct:.1f}% | Acne:{acne_count} | Result:{final[0]['label']}")
    return final

def _predict_model(pil_rgb: Image.Image):
    """
    Returns (condition, confidence, all_scores, method) or None if model unavailable.
    """
    if not TF_AVAILABLE or MODEL is None:
        return None

    img = pil_rgb.resize((224, 224))
    x = np.asarray(img).astype(np.float32)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    probs = MODEL.predict(x, verbose=0)[0]
    probs = np.asarray(probs).astype(np.float32)
    if probs.ndim != 1 or MODEL_LABELS is None or probs.size != len(MODEL_LABELS):
        return None

    top_idx = int(np.argmax(probs))
    top_label = MODEL_LABELS[top_idx]
    top_prob = float(probs[top_idx])

    topk_idx = np.argsort(-probs)[:5]
    all_scores = [
        {"label": MODEL_LABELS[int(i)], "prob": round(float(probs[int(i)]) * 100, 1)}
        for i in topk_idx
    ]

    return {
        "condition": top_label,
        "confidence": f"{round(top_prob * 100, 1)}%",
        "all_scores": all_scores,
        "method": "9-class EfficientNet" if USE_FULL_MODEL else "HAM10000 EfficientNet",
    }

def _map_ham_to_condition(ham_label: str):
    if ham_label == "mel":
        return "Malignant Melanoma"
    if ham_label == "bcc":
        return "Basal Cell Carcinoma"
    if ham_label == "akiec":
        return "Squamous Cell Carcinoma"
    return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        image_string = data['image']
        if "," in image_string:
            image_data = image_string.split(",")[1]
        else:
            image_data = image_string
            
        print("Received image length:", len(image_data))
        
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        print("Image size:", img.size)
        
        img_256 = img.resize((256, 256))
        img_array = np.array(img_256) / 255.0

        h, w = img_array.shape[:2]
        center = img_array[h//4:3*h//4, w//4:3*w//4]

        model_pred = _predict_model(img_256)
        analysis_cv = analyze_image_cv(center)
        top_cv = analysis_cv[0]

        if model_pred is not None and USE_FULL_MODEL:
            chosen = {
                "condition": model_pred["condition"],
                "confidence": model_pred["confidence"],
                "all_scores": model_pred["all_scores"],
                "method": model_pred["method"],
            }
        elif model_pred is not None:
            chosen = {
                "condition": top_cv["label"],
                "confidence": f"{top_cv['prob']}%",
                "all_scores": analysis_cv[:5],
                "method": "OpenCV heuristic engine",
            }
            top_ham = model_pred["all_scores"][0]["label"]
            mapped = _map_ham_to_condition(top_ham)
            threshold = float(os.environ.get("SKINDL_HAM_CANCER_THRESHOLD", "0.55"))
            top_prob = float(model_pred["all_scores"][0]["prob"]) / 100.0
            if mapped and top_prob >= threshold:
                chosen["condition"] = mapped
                chosen["confidence"] = model_pred["confidence"]
                chosen["method"] = model_pred["method"]
        else:
            chosen = {
                "condition": top_cv["label"],
                "confidence": f"{top_cv['prob']}%",
                "all_scores": analysis_cv[:5],
                "method": "OpenCV heuristic engine",
            }

        chosen["progression"] = get_progression(chosen["condition"])
        return jsonify(chosen)

    except Exception as e:
        print(f"Error in predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("AI Service starting on port 5002...")
    app.run(port=5002, debug=True)
