import torch
import torch.nn as nn
from torchvision import transforms
import os
import cv2
from pytorchvideo.models.hub import x3d_xs

# Custom X3D model
class CustomX3D(nn.Module):
    def __init__(self, pretrained=True):
        super(CustomX3D, self).__init__()
        self.x3d = x3d_xs(pretrained=pretrained)
        self.x3d.fc = nn.Identity()  # Remove classification layer

    def forward(self, x):
        return self.x3d(x)

# Video frame analysis model
class VideoAnalysisModel(nn.Module):
    def __init__(self, x3d_model, feature_dim=400, target_dim=2, hidden_dim=512):
        super(VideoAnalysisModel, self).__init__()
        self.x3d = x3d_model
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, target_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        features = self.x3d(x)
        x = features.view(features.size(0), -1)
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        predictions = self.fc_out(x)
        return predictions

def load_video_analysis_model(weight_path, device):
    x3d_model = CustomX3D(pretrained=False)
    model = VideoAnalysisModel(x3d_model)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device)

import torch
import cv2
import os
from torchvision import transforms

def predict_esv_edv(video_folder_path, model, transform, device):
    frames = []
    
    # Print the directory contents to debug
    print(f"Reading frames from: {video_folder_path}")
    print("Files in directory:", os.listdir(video_folder_path))
    
    # Check if there are any image files in the directory
    for frame_file in sorted(os.listdir(video_folder_path)):
        # Check for valid image extensions
        if frame_file.endswith(('.png', '.jpg', '.jpeg')):
            frame_path = os.path.join(video_folder_path, frame_file)
            
            # Read and process the frame
            frame = cv2.imread(frame_path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                frames.append(frame)
            else:
                print(f"Warning: {frame_file} is not a valid image or couldn't be read.")
    
    # If no frames were added, print an error message and raise an exception
    if not frames:
        print(f"Error: No valid frames were found in {video_folder_path}")
        raise RuntimeError("No frames to process.")

    # Resize frames
    frames = [transforms.Resize((160, 160))(frame) for frame in frames]
    
    # Stack frames into a tensor
    frames = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(frames)
    
    return predictions.cpu().numpy().flatten()
