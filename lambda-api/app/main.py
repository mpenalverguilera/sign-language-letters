import os, json, numpy as np, joblib

# ── Load model and encoder once at cold start ────────────────
MODEL_PATH   = os.getenv("MODEL_PATH",   "model.joblib")
ENCODER_PATH = os.getenv("ENCODER_PATH", "encoder.joblib")
model   = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

ENV     = os.getenv("ENV", "prod")
IS_LOCAL = ENV == "local" or os.getenv("AWS_SAM_LOCAL") == "true"

# ── CORS headers utility ─────────────────────────────────────
CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json"
}

if IS_LOCAL:
    print("[INFO] Running in LOCAL mode")

def cors_response(code: int, body: dict | str):
    """Return a JSON response with proper CORS headers."""
    return {
        "statusCode": code,
        "headers":   CORS_HEADERS,
        "body":      json.dumps(body),
    }

# ── AWS Lambda entrypoint ────────────────────────────────────
def lambda_handler(event, context):
    if IS_LOCAL:
        print("[DEBUG] Event received:", event)

    # CORS pre-flight check
    if event.get("httpMethod") == "OPTIONS":
        return cors_response(200, {"message": "CORS pre-flight OK"})

    # Only allow POST method
    if event.get("httpMethod") != "POST":
        return cors_response(405, {"error": "Method not allowed"})

    try:
        body = json.loads(event.get("body", "{}"))
        lm   = body.get("landmarks")
        if not lm or len(lm) != 63:
            return cors_response(400, {"error": f"landmarks must be 63 floats, recived: {len(lm)}"})

        lm = np.asarray(lm, dtype=np.float32).reshape(21, 3)
        lm -= lm[0]  # center on wrist
        scale = np.max(np.ptp(lm, axis=0)) or 1.0
        vec60 = (lm[1:] / scale).flatten().reshape(1, -1)

        if IS_LOCAL:
            print("[DEBUG] Normalized vec:", vec60)

        probs = model.predict_proba(vec60)[0]
        conf  = np.max(probs)
        pred_idx = np.argmax(probs)
        letter = encoder.inverse_transform([pred_idx])[0]
        return cors_response(200, {"prediction": letter, "confidence": conf})

    except Exception as exc:
        if IS_LOCAL:
            print("[ERROR]", exc)
        return cors_response(500, {"error": str(exc)})