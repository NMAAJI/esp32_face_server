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
    known_files.clear()
    for fname in os.listdir(KNOWN_DIR):
        path = os.path.join(KNOWN_DIR, fname)
        name = os.path.splitext(fname)[0]
        known_names.append(name)
        known_files.append(path)
        if FACE_REC_AVAILABLE:
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
            except Exception as e:
                print(f"Could not encode known face {fname}: {e}")

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
    img_bytes = request.data
    if not img_bytes:
        return jsonify({"error": "No image"}), 400

    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = os.path.join(STATIC_DIR, f"temp_{timestamp}.jpg")
    cv2.imwrite(temp_path, frame)

    if not (known_encodings or known_files):
        print("⚠️ No known faces yet")
        return jsonify({"status": "no_known_faces"})

    # If face_recognition available, prefer the high-quality encoding matcher
    if FACE_REC_AVAILABLE and known_encodings:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, locations)

            for enc in encodings:
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.50)
                if True in matches:
                    matched_index = matches.index(True)
                    person_name = known_names[matched_index]
                    current_time = datetime.now().timestamp()
                    if current_time - last_alert_time > ALERT_COOLDOWN:
                        alert_path = os.path.join(STATIC_DIR, f"alert_{person_name}_{timestamp}.jpg")
                        cv2.imwrite(alert_path, frame)
                        send_email_alert(alert_path, person_name)
                        last_alert_time = current_time
                        return jsonify({"status": "match", "person": person_name})
                    else:
                        return jsonify({"status": "match", "person": person_name, "cooldown": True})
        except Exception as e:
            print(f"Error during face_recognition flow: {e}")

    # Fallback: imagehash perceptual hash comparison (lightweight)
    if IMAGEHASH_AVAILABLE and known_files:
        try:
            from PIL import Image
            import imagehash
            h_target = imagehash.phash(Image.open(temp_path))
            for idx, kpath in enumerate(known_files):
                try:
                    h_known = imagehash.phash(Image.open(kpath))
                    d = int(h_target - h_known)
                    # small hamming distance => likely same person/image
                    if d <= 10:
                        person_name = known_names[idx]
                        current_time = datetime.now().timestamp()
                        if current_time - last_alert_time > ALERT_COOLDOWN:
                            alert_path = os.path.join(STATIC_DIR, f"alert_{person_name}_{timestamp}.jpg")
                            cv2.imwrite(alert_path, frame)
                            send_email_alert(alert_path, person_name)
                            last_alert_time = current_time
                            return jsonify({"status": "match", "person": person_name})
                        else:
                            return jsonify({"status": "match", "person": person_name, "cooldown": True})
                except Exception as e:
                    print(f"Could not phash compare with {kpath}: {e}")
        except Exception as e:
            print(f"imagehash fallback error: {e}")

    return jsonify({"status": "no_match"})

    return jsonify({"status": "no_match"})

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
    faces = [os.path.splitext(f)[0] for f in os.listdir(KNOWN_DIR)]
    return jsonify({"count": len(faces), "faces": faces})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
