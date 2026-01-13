from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import cv2
import numpy as np
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime

# Optional heavy dependency: face_recognition (dlib). Try to import, else fall back.
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
    print("✅ face_recognition available: using encodings matcher")
except Exception as e:
    face_recognition = None
    FACE_REC_AVAILABLE = False
    print(f"⚠️ face_recognition not available: {e}")

# Lightweight perceptual-hash verifier (optional)
try:
    from PIL import Image
    import imagehash
    IMAGEHASH_AVAILABLE = True
    print("✅ imagehash available: using phash fallback")
except Exception as e:
    Image = None
    imagehash = None
    IMAGEHASH_AVAILABLE = False
    print(f"⚠️ imagehash not available: {e}")

app = Flask(__name__, static_folder="static")

# Directories
KNOWN_DIR = "known_faces"
STATIC_DIR = "static"
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Load known faces

# If face_recognition is available we store encodings, otherwise store file paths for phash fallback
known_encodings = []
known_names = []
known_files = []

def load_known_faces():
    known_encodings.clear()
    known_names.clear()

    for fname in os.listdir(KNOWN_DIR):
        # ✅ allow ONLY image files
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(KNOWN_DIR, fname)

        try:
            img = face_recognition.load_image_file(path)
            encs = face_recognition.face_encodings(img)

            if encs:
                known_encodings.append(encs[0])
                known_names.append(os.path.splitext(fname)[0])
                print(f"✅ Loaded face: {fname}")
            else:
                print(f"⚠️ No face found in: {fname}")

        except Exception as e:
            print(f"❌ Skipped file {fname}: {e}")

load_known_faces()

# Email config from Railway env vars
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

# Cooldown
last_alert_time = 0
ALERT_COOLDOWN = 30

def send_email_alert(image_path, person_name):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"🚨 ALERT: {person_name} Detected!"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECEIVER_EMAIL
        msg.set_content(f"{person_name} detected at {datetime.now()}")

        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print(f"📧 Email sent for {person_name}")
        return True
    except Exception as e:
        print("❌ Email error:", e)
        return False

@app.route("/")
def index():
    faces = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return render_template("index.html", faces=faces)

@app.route("/known_faces/<path:filename>")
def known_faces_file(filename):
    return send_from_directory(KNOWN_DIR, filename)

@app.route("/upload", methods=["POST"])
def upload_image():
    global last_alert_time

    try:
        img_bytes = request.data
        if not img_bytes:
            return jsonify({"error": "empty_image"}), 400

        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "invalid_image"}), 400

        # Save latest frame (for UI)
        latest_path = os.path.join(STATIC_DIR, "latest.jpg")
        cv2.imwrite(latest_path, frame)

        if not known_encodings:
            return jsonify({"status": "no_known_faces"}), 200

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb)
        if not locations:
            return jsonify({"status": "no_face_detected"}), 200

        encodings = face_recognition.face_encodings(rgb, locations)
        if not encodings:
            return jsonify({"status": "no_face_detected"}), 200

        for enc in encodings:
            matches = face_recognition.compare_faces(
                known_encodings, enc, tolerance=0.50
            )

            if True in matches:
                idx = matches.index(True)
                name = known_names[idx]

                now = datetime.now().timestamp()
                if now - last_alert_time > ALERT_COOLDOWN:
                    alert_path = os.path.join(
                        STATIC_DIR, f"alert_{name}.jpg"
                    )
                    cv2.imwrite(alert_path, frame)
                    send_email_alert(alert_path, name)
                    last_alert_time = now

                return jsonify({"status": "match", "person": name}), 200

        return jsonify({"status": "no_match"}), 200

    except Exception as e:
        print("UPLOAD ERROR:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/register", methods=["POST"])
def register_face():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"error": "Missing data"}), 400

    name = request.form['name'].strip()
    file = request.files['image']
    filename = f"{name}.jpg"
    save_path = os.path.join(KNOWN_DIR, filename)
    file.save(save_path)

    load_known_faces()
    print(f"🟢 Registered face: {name}")
    return redirect("/")


@app.route("/upload_known", methods=["POST"])
def upload_known():
    # Simple uploader for known face images (accepts raw file upload under 'file')
    if "file" not in request.files:
        return "No file", 400

    file = request.files["file"]
    if file.filename == "":
        return "No filename", 400

    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return "Invalid image format", 400

    save_path = os.path.join(KNOWN_DIR, file.filename)
    file.save(save_path)

    # Refresh in-memory known faces
    load_known_faces()

    return f"Known face {file.filename} uploaded successfully", 200

@app.route("/delete/<filename>")
def delete_face(filename):
    path = os.path.join(KNOWN_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        print("🗑️ Deleted:", filename)
    load_known_faces()
    return redirect("/")

@app.route("/list_faces")
def list_faces():
    faces = []
    for f in os.listdir(KNOWN_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            faces.append(os.path.splitext(f)[0])

    return jsonify({
        "count": len(faces),
        "faces": faces
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
