from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import cv2
import numpy as np
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage
# Detection/verification backends. Try to use DeepFace first, then face_recognition,
# then a lightweight perceptual-hash (imagehash) fallback so the app can run
# in environments (like Railway) without heavy system libraries.
DEEPFACE_AVAILABLE = False
FACE_REC_AVAILABLE = False
IMAGEHASH_AVAILABLE = False
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception as _e:
    DEEPFACE_AVAILABLE = False
    try:
        import face_recognition
        FACE_REC_AVAILABLE = True
    except Exception as _e2:
        FACE_REC_AVAILABLE = False
        try:
            from PIL import Image
            import imagehash
            IMAGEHASH_AVAILABLE = True
        except Exception:
            IMAGEHASH_AVAILABLE = False



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
# Temporary test-mail button expiry (timestamp)
TEST_BUTTON_EXPIRY = 0

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


@app.route('/enable_test_mail', methods=['POST'])
def enable_test_mail():
    """Enable the Send Test Email button for 20 minutes."""
    global TEST_BUTTON_EXPIRY
    expiry = datetime.now().timestamp() + (20 * 60)
    TEST_BUTTON_EXPIRY = expiry
    return jsonify({'expiry': expiry})


@app.route('/test_mail_status', methods=['GET'])
def test_mail_status():
    """Return current expiry timestamp (0 if not enabled)."""
    return jsonify({'expiry': TEST_BUTTON_EXPIRY})


@app.route('/send_test_mail', methods=['POST'])
def send_test_mail():
    """Send a test email with a small generated image if the temporary button is enabled."""
    try:
        if TEST_BUTTON_EXPIRY <= datetime.now().timestamp():
            return jsonify({'error': 'test button expired'}), 403

        # create a small test image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_path = os.path.join(STATIC_DIR, f'test_email_{timestamp}.jpg')
        try:
            import numpy as _np
            import cv2 as _cv2
            img = _np.full((200, 300, 3), 240, dtype=_np.uint8)
            _cv2.putText(img, 'ESP32 Test Email', (8, 100), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
            _cv2.imwrite(test_path, img)
        except Exception as e:
            print(f'Could not create test image: {e}')
            test_path = None

        # Send email (attach image if created)
        success = False
        if test_path and os.path.exists(test_path):
            success = send_email_alert(test_path, 'TestUser')
        else:
            # fallback: send text-only email
            try:
                msg = EmailMessage()
                msg['Subject'] = '🔔 Test email from ESP32 Face Server'
                msg['From'] = EMAIL_ADDRESS
                msg['To'] = RECEIVER_EMAIL
                msg.set_content('This is a test email from your ESP32 Face Server.')
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    smtp.send_message(msg)
                success = True
            except Exception as e:
                print(f'Failed sending fallback test email: {e}')
                success = False

        return jsonify({'sent': bool(success)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
        
        # Helper to verify two image files. Returns dict with 'verified' (bool) and 'distance' (float)
        def verify_images(img1_path, img2_path):
            # Try DeepFace first
            if DEEPFACE_AVAILABLE:
                try:
                    result = DeepFace.verify(
                        img1_path=img1_path,
                        img2_path=img2_path,
                        model_name='VGG-Face',
                        detector_backend='opencv',
                        enforce_detection=False
                    )
                    dist = result.get('distance', 1.0)
                    # normalize/clip distance to 0..1 for consistency
                    try:
                        dist = float(dist)
                        if dist > 1.0:
                            dist = 1.0
                    except Exception:
                        dist = 1.0
                    return {'verified': result.get('verified', False), 'distance': dist}
                except Exception as e:
                    print(f"DeepFace verification error: {e}")
                    return {'verified': False, 'distance': 1.0}

            # Next try face_recognition (may be unavailable on some hosts)
            if FACE_REC_AVAILABLE:
                try:
                    img1 = face_recognition.load_image_file(img1_path)
                    img2 = face_recognition.load_image_file(img2_path)
                    enc1 = face_recognition.face_encodings(img1)
                    enc2 = face_recognition.face_encodings(img2)
                    if not enc1 or not enc2:
                        return {'verified': False, 'distance': 1.0}
                    d = float(face_recognition.face_distance([enc2[0]], enc1[0])[0])
                    verified = d <= 0.6
                    return {'verified': verified, 'distance': max(0.0, min(1.0, d))}
                except Exception as e:
                    print(f"face_recognition verification error: {e}")
                    return {'verified': False, 'distance': 1.0}

            # Lightweight fallback: perceptual hash (imagehash)
            if IMAGEHASH_AVAILABLE:
                try:
                    from PIL import Image
                    import imagehash
                    h1 = imagehash.phash(Image.open(img1_path))
                    h2 = imagehash.phash(Image.open(img2_path))
                    d = int(h1 - h2)  # hamming distance, e.g. 0..64
                    max_bits = h1.hash.size
                    norm = d / float(max_bits) if max_bits else 1.0
                    # threshold: allow small hamming distances
                    verified = d <= 10
                    return {'verified': verified, 'distance': max(0.0, min(1.0, norm))}
                except Exception as e:
                    print(f"imagehash verification error: {e}")
                    return {'verified': False, 'distance': 1.0}

            # No verifier available
            return {'verified': False, 'distance': 1.0}

        # Check each known face
        for known_face in known_faces:
            known_path = os.path.join(KNOWN_DIR, known_face)
            person_name = os.path.splitext(known_face)[0]
            
            try:
                # Verify face using available verifier (DeepFace / face_recognition / imagehash)
                result = verify_images(temp_path, known_path)

                if result.get('verified'):
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
                            "confidence": float(max(0.0, min(1.0, 1 - result.get('distance', 1.0)))),
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
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port)

#include <WiFiClientSecure.h>
#include <HTTPClient.h>

// serverUrl MUST be https://...
const char* serverUrl = "https://your-project.up.railway.app/upload";

WiFiClientSecure client;
client.setInsecure(); // Accept Railway TLS (dev toleration). For production, use root CA/fingerprint.

HTTPClient https;
https.begin(client, serverUrl);
https.addHeader("Content-Type", "application/octet-stream");

// fb is your camera frame buffer (fb->buf, fb->len)
int response = https.POST(fb->buf, fb->len);
Serial.printf("Upload response: %d\n", response);

https.end();

config.pixel_format = PIXFORMAT_JPEG;
config.frame_size = FRAMESIZE_VGA;   // NOT UXGA — use VGA or smaller
config.jpeg_quality = 12;            // larger number => more compression => smaller bytes
config.fb_count = 1;                 // save memory

// after init, ensure sensor framesize:
sensor_t * s = esp_camera_sensor_get();
s->set_framesize(s, FRAMESIZE_VGA);