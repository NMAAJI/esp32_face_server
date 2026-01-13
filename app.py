from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import cv2
import numpy as np
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime

# Face recognition
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
    print("✅ face_recognition available")
except Exception as e:
    face_recognition = None
    FACE_REC_AVAILABLE = False
    print("❌ face_recognition not available:", e)

app = Flask(__name__, static_folder="static")

# Directories
KNOWN_DIR = "known_faces"
STATIC_DIR = "static"
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Known faces memory
known_encodings = []
known_names = []

def load_known_faces():
    known_encodings.clear()
    known_names.clear()

    for fname in os.listdir(KNOWN_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(KNOWN_DIR, fname)

        try:
            img = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(img)

            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(fname)[0])
                print("✅ Loaded:", fname)
            else:
                print("❌ No faces found in:", fname)
        except Exception as e:
            print("Error loading face from", fname, ":", e)

# Load known faces at startup
load_known_faces()

# Email alert settings
EMAIL_ADDRESS = "your_email@gmail.com"
EMAIL_PASSWORD = "your_password"
ALERT_COOLDOWN = 10  # seconds
last_alert_time = 0

def send_email_alert(image_path, person_name):
    try:
        msg = EmailMessage()
        msg["Subject"] = f"Alert: {person_name} detected"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_ADDRESS

        msg.set_content(f"Alert: {person_name} was detected by the system.")

        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            img_name = os.path.basename(image_path)
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=img_name)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print("✅ Alert email sent")
    except Exception as e:
        print("Error sending email:", e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    global last_alert_time

    try:
        print("[UPLOAD] Request received")
        print("[UPLOAD] Content-Type:", request.content_type)
        print("[UPLOAD] request.data length:", len(request.data) if request.data else 0)
        print("[UPLOAD] request.files:", list(request.files.keys()))

        # 1️⃣ Read image safely
        if request.data:
            img_bytes = request.data
            print("[UPLOAD] Using request.data")
        elif "file" in request.files:
            img_bytes = request.files["file"].read()
            print("[UPLOAD] Using request.files['file']")
        else:
            print("[UPLOAD] No image found in request")
            return jsonify({"error": "empty_image"}), 400

        # 2️⃣ Decode image
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        print("[UPLOAD] Decoded frame:", "OK" if frame is not None else "None")

        if frame is None:
            print("[UPLOAD] Invalid image data")
            return jsonify({"error": "invalid_image"}), 400

        # 3️⃣ Save for UI
        cv2.imwrite(os.path.join(STATIC_DIR, "latest.jpg"), frame)
        print("[UPLOAD] Saved latest.jpg")

        if not known_encodings:
            print("[UPLOAD] No known faces loaded")
            return jsonify({"status": "no_known_faces"}), 200

        if not FACE_REC_AVAILABLE:
            print("[UPLOAD] face_recognition not available")
            return jsonify({"error": "face_recognition_not_available"}), 500

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        print(f"[UPLOAD] Found {len(locations)} face locations")
        if not locations:
            return jsonify({"status": "no_face_detected"}), 200

        encodings = face_recognition.face_encodings(rgb, locations)
        print(f"[UPLOAD] Found {len(encodings)} encodings")

        for enc in encodings:
            matches = face_recognition.compare_faces(
                known_encodings, enc, tolerance=0.5
            )
            print(f"[UPLOAD] Matches: {matches}")

            if True in matches:
                name = known_names[matches.index(True)]
                now = datetime.now().timestamp()
                if now - last_alert_time > ALERT_COOLDOWN:
                    alert_path = os.path.join(STATIC_DIR, f"alert_{name}.jpg")
                    cv2.imwrite(alert_path, frame)
                    send_email_alert(alert_path, name)
                    last_alert_time = now
                print(f"[UPLOAD] Match found: {name}")
                return jsonify({"status": "match", "person": name}), 200

        print("[UPLOAD] No match found")
        return jsonify({"status": "no_match"}), 200

    except Exception as e:
        print("UPLOAD ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": "server_error", "details": str(e)}), 500

@app.route("/known_faces")
def known_faces():
    try:
        faces = [{"name": name, "image": f"/{KNOWN_DIR}/{name}.jpg"} for name in known_names]
        return jsonify(faces), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/add_face", methods=["POST"])
def add_face():
    try:
        name = request.form.get("name")
        if not name:
            return jsonify({"error": "Name is required"}), 400

        image_file = request.files.get("file")
        if not image_file:
            return jsonify({"error": "Image file is required"}), 400

        # Save the new face
        image_path = os.path.join(KNOWN_DIR, f"{name}.jpg")
        image_file.save(image_path)
        print(f"✅ Face saved for {name}")

        # Reload known faces
        load_known_faces()

        return jsonify({"status": "face_added", "person": name}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/remove_face", methods=["POST"])
def remove_face():
    try:
        name = request.form.get("name")
        if not name:
            return jsonify({"error": "Name is required"}), 400

        # Remove the face file
        image_path = os.path.join(KNOWN_DIR, f"{name}.jpg")
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"✅ Face removed for {name}")
        else:
            print(f"❌ Face file not found for {name}")

        # Reload known faces
        load_known_faces()

        return jsonify({"status": "face_removed", "person": name}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory(STATIC_DIR, path)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
