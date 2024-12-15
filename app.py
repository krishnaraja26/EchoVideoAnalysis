from flask import Flask, request, render_template, jsonify
import os
import torch
from models.video_analysis import load_video_analysis_model, predict_esv_edv
from torchvision import transforms
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
video_analysis_weights = "weights/model_weights.pth"
video_analysis_model = load_video_analysis_model(video_analysis_weights, device)

# Transform for video analysis
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

# Allowed file types for video frames (images)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file is an allowed image type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the heart health details page."""
    return render_template('heart_details.html')  # Serve heart health details page

@app.route('/video-analysis')
def video_analysis():
    """Render the video analysis page."""
    return render_template('video_analysis.html')  # Serve video analysis page
@app.route('/heart-details', methods=['GET'])
def heart_details():
    """Render the heart health details page."""
    return render_template('heart_details.html')
@app.route('/video_analysis', methods=['POST'])
def video_analysis_work():
    """Handle video analysis and ESV/EDV prediction."""
    
    # Check if 'video_folder' exists in the form data
    if 'video_folder' not in request.files:
        return jsonify({"error": "No video folder provided"}), 400

    video_folder = request.files.getlist('video_folder')

    if not video_folder:
        return jsonify({"error": "No files selected"}), 400

    # Save the video frames into the appropriate folder
    folder_name = 'uploaded_video_folder'  # Folder for this upload
    saved_folder = save_uploaded_files(video_folder, folder_name)

    # Process the frames and predict ESV/EDV
    esv, edv = predict_esv_edv(saved_folder, video_analysis_model, transform, device)
    
    # Return the results as JSON
    return jsonify({"ESV": float(esv), "EDV": float(edv), "EF": float((edv-esv)/edv)*(100)})


def save_uploaded_files(files, folder_name):
    """Save the uploaded video frames into a specific folder."""
    
    upload_folder = os.path.join(os.path.dirname(__file__), 'uploads', 'video_folder', folder_name)
    os.makedirs(upload_folder, exist_ok=True)  # Create folder if it doesn't exist

    file_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            file_paths.append(file_path)
            print(f"Saved file: {file_path}")
        else:
            print(f"Skipping file: {file.filename} (invalid file type)")

    return upload_folder  # Return path to the folder containing saved files


if __name__ == "__main__":
    app.run(debug=True)
