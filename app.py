from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
from deepface import DeepFace

app = Flask(__name__, static_folder="static")


# Directories
KNOWN_DIR = "known_faces"
STATIC_DIR = "static"
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Email configuration (FILL THESE IN!)
EMAIL_ADDRESS = "n.maajid982010@gmail.com"
EMAIL_PASSWORD = "vjiv jzhm ppgl jlxw"
RECEIVER_EMAIL = "naruto982010@gmail.com"

# Alert tracking (prevent spam)
last_alert_time = 0
ALERT_COOLDOWN = 30  # seconds between alerts

def send_email_alert(image_path, person_name):
    """Send email with detected person image"""
    try:
        msg = EmailMessage()
        msg["Subject"] = f"🚨 ALERT: {person_name} Detected!"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECEIVER_EMAIL
        msg.set_content(f"Wanted person '{person_name}' was detected by ESP32-CAM at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Attach image
        with open(image_path, "rb") as f:
            img_data = f.read()
            msg.add_attachment(img_data, maintype="image", subtype="jpeg", filename=f"{person_name}_detected.jpg")

        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        
        print(f"✅ Email alert sent for {person_name}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False

@app.route('/')
def index():
    """Home page with UI"""
    known_faces = [f for f in os.listdir(KNOWN_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('index.html', faces=known_faces)


@app.route('/health')
def health():
    """Simple health/status endpoint for quick checks"""
    return "ESP32 Face Server Running"

@app.route('/known_faces/<path:filename>')
def known_faces_file(filename):
    return send_from_directory(KNOWN_DIR, filename)


@app.route('/upload', methods=['POST'])
def upload_image():
    """Upload ESP32 image and check for face match"""
    global last_alert_time
    
    try:
        # Get image from ESP32
        img_bytes = request.data
        
        if len(img_bytes) == 0:
            return jsonify({"error": "No image data"}), 400
        
        # Convert to OpenCV format
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Save incoming frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = os.path.join(STATIC_DIR, f"temp_{timestamp}.jpg")
        cv2.imwrite(temp_path, frame)
        
        # Check if there are known faces
        known_faces = [f for f in os.listdir(KNOWN_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not known_faces:
            print("⚠️ No known faces in database")
            return jsonify({"status": "no_known_faces"})
        
        # Check each known face
        for known_face in known_faces:
            known_path = os.path.join(KNOWN_DIR, known_face)
            person_name = os.path.splitext(known_face)[0]
            
            try:
                # Verify face using DeepFace
                result = DeepFace.verify(
                    img1_path=temp_path,
                    img2_path=known_path,
                    model_name='VGG-Face',
                    detector_backend='opencv',
                    enforce_detection=False
                )
                
                # If match found
                if result['verified']:
                    print(f"🚨 MATCH FOUND: {person_name}")
                    
                    # Check cooldown
                    current_time = datetime.now().timestamp()
                    if current_time - last_alert_time > ALERT_COOLDOWN:
                        # Save alert image
                        alert_path = os.path.join(STATIC_DIR, f"alert_{person_name}_{timestamp}.jpg")
                        cv2.imwrite(alert_path, frame)
                        
                        # Send email
                        send_email_alert(alert_path, person_name)
                        last_alert_time = current_time
                        
                        return jsonify({
                            "status": "match",
                            "person": person_name,
                            "confidence": float(1 - result['distance']),
                            "alert_sent": True
                        })
                    else:
                        return jsonify({
                            "status": "match",
                            "person": person_name,
                            "alert_sent": False,
                            "message": "Cooldown active"
                        })
                        
            except Exception as e:
                print(f"Error checking {known_face}: {e}")
                continue
        
        # No match found
        print("✅ No match found")
        return jsonify({"status": "no_match"})
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/register', methods=['POST'])
def register_face():
    """Register new wanted person via web UI"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        if 'name' not in request.form:
            return jsonify({'error': 'No name provided'}), 400
        
        name = request.form['name']
        file = request.files['image']
        
        # Save face
        filename = f"{name}.jpg"
        save_path = os.path.join(KNOWN_DIR, filename)
        file.save(save_path)
        
        print(f"✅ Registered: {name}")
        return redirect('/')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete/<filename>')
def delete_face(filename):
    """Delete a known face"""
    try:
        file_path = os.path.join(KNOWN_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ Deleted: {filename}")
        return redirect('/')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list_faces', methods=['GET'])
def list_faces():
    """List all registered faces"""
    known_faces = [f for f in os.listdir(KNOWN_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    return jsonify({
        'count': len(known_faces),
        'faces': [os.path.splitext(f)[0] for f in known_faces]
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 ESP32 Face Recognition Server")
    print("=" * 50)
    print(f"📁 Known faces directory: {KNOWN_DIR}")
    print(f"📧 Email alerts: {'Configured' if EMAIL_ADDRESS != 'your_email@gmail.com' else 'NOT CONFIGURED'}")
    print("=" * 50)
    print("Starting server on http://0.0.0.0:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)