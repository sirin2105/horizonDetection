import os
import torch.serialization
from ultralytics.nn.tasks import DetectionModel

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from skimage.feature import local_binary_pattern
from ultralytics import YOLO
import tempfile
import warnings
from sklearn.linear_model import RANSACRegressor

warnings.filterwarnings('ignore')

# ================================
# MODEL CONFIGURATION - SPECIFY PATH HERE
# ================================
SEGMENTATION_MODEL_PATH = "best.pt"  # Change this to your actual model path

# Set page config
st.set_page_config(
    page_title="Combined Sea Segmentation & YOLO Boat Detection",
    page_icon="🌊🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .step-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .metrics-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# SEGMENTATION MODEL LOADING
# ================================

@st.cache_resource
def load_segmentation_model():
    """Load segmentation model from predefined path (cached)"""
    
    if not os.path.exists(SEGMENTATION_MODEL_PATH):
        st.error(f"Segmentation model not found at: {SEGMENTATION_MODEL_PATH}")
        st.error("Please update the SEGMENTATION_MODEL_PATH variable in the code to point to your model file.")
        return None, None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Show loading progress
        with st.spinner('Loading segmentation model (this may take a while for large models)...'):
            model = smp.Unet(
                encoder_name='resnet50',
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,  # Changed to 2 classes for binary segmentation (sea vs no-sea)
                activation=None
            )
           
            torch.serialization.add_safe_globals([DetectionModel])

            state_dict = torch.load(SEGMENTATION_MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)  # ✅ Load directly

            #checkpoint = torch.load(SEGMENTATION_MODEL_PATH, map_location=device)
            #model.load_state_dict(checkpoint['model_state_dict'])  # load only model weights

            
            # Clear checkpoint from memory
            
            model = model.to(device)
            model.eval()
            
            # Set model to half precision if using GPU to save memory
            if device.type == 'cuda':
                model = model.half()
            
        st.success(f"✅ Segmentation model loaded from: {SEGMENTATION_MODEL_PATH}")
        return model, device
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        return None, None

# ================================
# YOLO MODEL LOADING
# ================================

@st.cache_resource
def load_yolo_model(model_path):
    """Load YOLO model (cached)"""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# ================================
# IMAGE PROCESSING FUNCTIONS
# ================================


def detect_horizon_from_mask_and_yolo(seg_mask, yolo_results, image, min_required_points=30):
    """
    Detect sea horizon line by combining segmentation mask and YOLO detections.
    
    seg_mask: 2D numpy array (H, W), binary values (1 = sea, 0 = no-sea)
    yolo_results: YOLO detection results object
    image: original image as NumPy array (H, W, 3)
    """
    height, width = seg_mask.shape

    # Step 1: Top-down and bottom-up transitions
    top_points = []
    bottom_points = []

    for x in range(width):
        # Top-down
        for y in range(1, height):
            if seg_mask[y-1, x] == 0 and seg_mask[y, x] == 1:
                top_points.append((x, y))
                break

        # Bottom-up
        for y in range(height - 2, -1, -1):
            if seg_mask[y+1, x] == 0 and seg_mask[y, x] == 1:
                bottom_points.append((x, y))
                break

    common_points = set()
    for p in top_points:
        common_points.add(p)
    for p in bottom_points:
        common_points.add(p)

    filtered_points = []
    for p in common_points: 
        x, y = p
        is_inside = False
        if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy().astype(int)
            for x1, y1, x2, y2 in boxes:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    is_inside = True
                    break
        if not is_inside:
            filtered_points.append((x, y))

        

    # Step 4: Fit RANSAC line
    xs = np.array([x for x, y in filtered_points]).reshape(-1, 1)
    ys = np.array([y for x, y in filtered_points])

    ransac = RANSACRegressor(residual_threshold=3.0, max_trials=100)
    ransac.fit(xs, ys)

    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_

    # Step 5: Draw line using cv2.line
    overlay = image.copy()
    x0, x1 = 0, width - 1
    y0 = int(slope * x0 + intercept)
    y1 = int(slope * x1 + intercept)

    # Ensure points are within image bounds
    if 0 <= y0 < height and 0 <= y1 < height:
        cv2.line(overlay, (x0, y0), (x1, y1), (0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    
    # for x, y in filtered_points:
    #     if 0 <= x < width and 0 <= y < height:
    #         cv2.circle(overlay, (x, y), radius=2, color=(255, 0, 255), thickness=-1)

    return overlay, ransac




def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def apply_scharr_edge(image):
    """Apply Scharr-Y edge detection"""
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharr_y = np.absolute(scharr_y)
    scharr_y = np.uint8(scharr_y / scharr_y.max() * 255)
    return scharr_y

def apply_lbp(image, radius=3, n_points=24):
    """Apply Local Binary Pattern"""
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp_normalized = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)
    return lbp_normalized

def create_3channel_input(grayscale_image):
    """Create 3-channel input from grayscale image"""
    if grayscale_image.dtype != np.uint8:
        grayscale_image = (grayscale_image * 255).astype(np.uint8)
    
    clahe_img = apply_clahe(grayscale_image)
    edge_img = apply_scharr_edge(grayscale_image)
    lbp_img = apply_lbp(grayscale_image)
    
    return np.stack([clahe_img, edge_img, lbp_img], axis=2), clahe_img, edge_img, lbp_img

def preprocess_image_for_segmentation(image_array):
    """Preprocess image for segmentation"""
    if len(image_array.shape) == 3:
        grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = image_array
    
    image_3ch, clahe_img, edge_img, lbp_img = create_3channel_input(grayscale)
    
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image_3ch)
    return transformed['image'].unsqueeze(0), grayscale, clahe_img, edge_img, lbp_img

def preprocess_image_for_yolo(image_array):
    """Preprocess image for YOLO"""
    if len(image_array.shape) == 3:
        grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = image_array
    
    _, clahe_img, edge_img, lbp_img = create_3channel_input(grayscale)
    three_channel = np.stack([clahe_img, edge_img, lbp_img], axis=2)
    
    return three_channel

# ================================
# INFERENCE FUNCTIONS
# ================================

def predict_segmentation(model, image_tensor, device):
    """Predict binary segmentation mask using sigmoid activation"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Use half precision if model is in half precision
        if next(model.parameters()).dtype == torch.float16:
            image_tensor = image_tensor.half()
        
        output = model(image_tensor)  # (1, 1, H, W) for binary segmentation
        probs = torch.sigmoid(output)
        pred_mask = (probs > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()  # (H, W), binary mask
        
        # Clear GPU memory
        del output, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    return pred_mask.astype(np.uint8)  # Optional: return as uint8 (0 or 1)


def predict_yolo(model, processed_image, conf_threshold=0.5):
    """Run YOLO inference"""
    if model is None:
        return None
    
    try:
        results = model(processed_image, conf=conf_threshold)
        return results
    except Exception as e:
        st.error(f"Error during YOLO inference: {e}")
        return None

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_segmentation_overlay(original_image, pred_mask, alpha=0.5):
    """Create segmentation overlay"""
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    overlay = original_image.copy()
    
    # Binary segmentation colors
    colors = {
        0: (128, 128, 128),   # no-sea (gray)
        1: (0, 100, 255),     # sea (blue)
    }
    
    for class_id, color in colors.items():
        mask = (pred_mask == class_id)
        overlay[mask] = color
    
    blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
    return blended

def draw_yolo_detections(image, results):
    """Draw YOLO detection results on image"""
    img_with_boxes = image.copy()
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw confidence score
            label = f'Boat: {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background rectangle for text
            cv2.rectangle(img_with_boxes, (x1, y1-label_height-10), (x1+label_width, y1), (0, 255, 0), -1)
            cv2.putText(img_with_boxes, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img_with_boxes

def create_combined_overlay(original_image, seg_mask, yolo_results, seg_alpha=0.3, yolo_alpha=0.7):
    """Create combined overlay with both segmentation and YOLO detections"""
    # Start with original image
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    if original_image.ndim == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # First add segmentation overlay
    seg_overlay = original_image.copy()
    colors = {
        0: (128, 128, 128),   # no-sea (gray)
        1: (0, 100, 255),     # sea (blue)
    }
    
    for class_id, color in colors.items():
        mask = (seg_mask == class_id)
        seg_overlay[mask] = color
    
    # Blend segmentation
    combined = cv2.addWeighted(original_image, 1 - seg_alpha, seg_overlay, seg_alpha, 0)
    
    # Add YOLO detections
    if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        confs = yolo_results[0].boxes.conf.cpu().numpy()
        
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 255, 0), 3)
            
            # Draw confidence score
            label = f'Boat: {conf:.2f}'
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background rectangle for text
            cv2.rectangle(combined, (x1, y1-label_height-10), (x1+label_width, y1), (255, 255, 0), -1)
            cv2.putText(combined, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return combined

def count_detections(results):
    """Count number of boats detected"""
    if results and len(results) > 0 and results[0].boxes is not None:
        return len(results[0].boxes)
    return 0

# ================================
# MAIN APPLICATION
# ================================

def main():
    # Header
    st.markdown('<div class="main-header">🌊🚢 Combined Sea Segmentation & YOLO Boat Detection</div>', unsafe_allow_html=True)
    
    # Initialize segmentation model automatically
    seg_model, device = load_segmentation_model()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model status
    st.sidebar.subheader("Model Status")
    seg_status = "✅ Loaded" if seg_model is not None else "❌ Not loaded"
    st.sidebar.write(f"Segmentation: {seg_status}")
    
    if seg_model is not None:
        st.sidebar.success(f"Model loaded from:\n{SEGMENTATION_MODEL_PATH}")
    else:
        st.sidebar.error("Segmentation model failed to load!")
        st.sidebar.error("Please check the SEGMENTATION_MODEL_PATH in the code.")
    
    # YOLO model configuration
    st.sidebar.subheader("YOLO Model Configuration")
    yolo_model_option = st.sidebar.selectbox(
        "Select YOLO Model Source",
        ["Upload trained model", "Use pre-trained YOLOv8"]
    )
    
    yolo_model = None
    if yolo_model_option == "Upload trained model":
        yolo_model_file = st.sidebar.file_uploader(
            "Upload YOLO Model (.pt file)",
            type=['pt'],
            key="yolo_model"
        )
        if yolo_model_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(yolo_model_file.read())
                yolo_model = load_yolo_model(tmp_file.name)
    else:
        model_size = st.sidebar.selectbox("Select YOLOv8 Model Size", ["n", "s", "m", "l", "x"])
        if st.sidebar.button("Load Pre-trained YOLO Model"):
            yolo_model = load_yolo_model(f'yolov8{model_size}.pt')
    
    yolo_status = "✅ Loaded" if yolo_model is not None else "❌ Not loaded"
    st.sidebar.write(f"YOLO: {yolo_status}")
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    seg_alpha = st.sidebar.slider("Segmentation Overlay Alpha", 0.0, 1.0, 0.3, 0.05)
    
    # CLAHE parameters
    st.sidebar.subheader("CLAHE Parameters")
    clip_limit = st.sidebar.slider("Clip Limit", 1.0, 5.0, 2.0, 0.1)
    tile_grid_x = st.sidebar.slider("Tile Grid X", 2, 16, 8, 1)
    tile_grid_y = st.sidebar.slider("Tile Grid Y", 2, 16, 8, 1)
    
    # Clear cache button
    if st.sidebar.button("🗑️ Clear Cache"):
        st.cache_resource.clear()
        # Also clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.sidebar.success("Cache cleared!")
    
    # Memory usage info
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        st.sidebar.info(f"GPU Memory: {gpu_memory:.1f} GB available")
    
    # Main content
    st.header("Upload Images")
    
    uploaded_files = st.file_uploader(
        "Choose images for processing",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        accept_multiple_files=True,
        help="Upload one or more images for combined sea segmentation and boat detection"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} images")
        
        # Process images
        if st.button("🔍 Process Images", type="primary"):
            # Check if models are loaded
            if seg_model is None:
                st.error("Segmentation model is not loaded! Please check the model path in the code.")
                return
            
            if yolo_model is None:
                st.error("Please load a YOLO model first!")
                return
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                # Load image
                image = Image.open(uploaded_file)
                image = image.resize((640, 512))
                image_array = np.array(image)
                
                # Convert to grayscale if needed
                if len(image_array.shape) == 3:
                    if image_array.shape[2] == 3:
                        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray_image = image_array[:,:,0]
                else:
                    gray_image = image_array
                
                # Create expandable section for each image
                with st.expander(f"📸 {uploaded_file.name} - Results", expanded=True):
                    
                    # Preprocessing
                    seg_tensor, grayscale, clahe_img, edge_img, lbp_img = preprocess_image_for_segmentation(image_array)
                    yolo_processed = preprocess_image_for_yolo(image_array)
                    
                    # Run inference
                    seg_mask = predict_segmentation(seg_model, seg_tensor, device)
                    yolo_results = predict_yolo(yolo_model, yolo_processed, conf_threshold)
                    
                    # Create visualization grid
                    st.markdown('<div class="step-header">🔧 Processing Pipeline</div>', unsafe_allow_html=True)
                    
                    # First row: Original and processed images
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.subheader("Original Image")
                        st.image(gray_image, caption="Original Image")
                    
                    with col2:
                        st.subheader("CLAHE Enhancement")
                        st.image(clahe_img, caption="Contrast Enhanced")
                    
                    with col3:
                        st.subheader("Edge Detection")
                        st.image(edge_img, caption="Scharr-Y Edges")
                    
                    with col4:
                        st.subheader("LBP Texture")
                        st.image(lbp_img, caption="Local Binary Pattern")
                    
                    # Second row: Results
                    st.markdown('<div class="step-header">🎯 Analysis Results</div>', unsafe_allow_html=True)
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        st.subheader("Binary Sea Segmentation")
                        # Create colored segmentation mask
                        seg_color = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
                        seg_color[seg_mask == 0] = [128, 128, 128]  # no-sea (gray)
                        seg_color[seg_mask == 1] = [0, 100, 255]    # sea (blue)
                        
                        st.image(seg_color, caption="Sea (Blue) vs No-Sea (Gray)")
                        
                        # Segmentation metrics
                        total_pixels = seg_mask.size
                        sea_pixels = np.sum(seg_mask == 1)
                        no_sea_pixels = np.sum(seg_mask == 0)
                        
                        st.metric("Sea Coverage", f"{(sea_pixels/total_pixels)*100:.1f}%")
                        st.metric("No-Sea Coverage", f"{(no_sea_pixels/total_pixels)*100:.1f}%")
                    
                    with res_col2:
                        st.subheader("YOLO Boat Detection")
                        # Convert grayscale to RGB for YOLO visualization
                        if len(gray_image.shape) == 2:
                            original_rgb = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
                        else:
                            original_rgb = gray_image
                        
                        yolo_viz = draw_yolo_detections(original_rgb, yolo_results)
                        st.image(yolo_viz, caption="Detected Boats")
                        
                        # YOLO metrics
                        boat_count = count_detections(yolo_results)
                        st.metric("Boats Detected", boat_count)
                        
                        if yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
                            avg_conf = float(yolo_results[0].boxes.conf.mean()) if len(yolo_results[0].boxes.conf) > 0 else 0
                            st.metric("Average Confidence", f"{avg_conf:.2f}")
                        else:
                            st.metric("Average Confidence", "0.00")
                    
                    with res_col3:
                        st.subheader("Combined Overlay")
                        # Horizon detection and overlay
                        horizon_overlay, ransac_model = detect_horizon_from_mask_and_yolo(seg_mask, yolo_results, original_rgb)
                        st.image(horizon_overlay, caption="Horizon Overlay")

                        # Combined metrics
                        st.metric("Image Size", f"{gray_image.shape[1]}×{gray_image.shape[0]}")
                        
                        # Sea area with boats
                        if boat_count > 0 and yolo_results and len(yolo_results) > 0 and yolo_results[0].boxes is not None:
                            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                            boats_in_sea = 0
                            for box in boxes:
                                x1, y1, x2, y2 = box.astype(int)
                                # Check if boat center is in sea area
                                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                                if 0 <= center_y < seg_mask.shape[0] and 0 <= center_x < seg_mask.shape[1]:
                                    if seg_mask[center_y, center_x] == 1:  # sea class
                                        boats_in_sea += 1
                            
                            st.metric("Boats in Sea Area", boats_in_sea)
                        else:
                            st.metric("Boats in Sea Area", "0")
                    
                    # # Detection details
                    # if boat_count > 0:
                    #     st.markdown('<div class="step-header">📋 Detection Details</div>', unsafe_allow_html=True)
                        
                    #     boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
                    #     confs = yolo_results[0].boxes.conf.cpu().numpy()
                        
                    #     for i, (box, conf) in enumerate(zip(boxes, confs)):
                    #         x1, y1, x2, y2 = box.astype(int)
                    #         width = x2 - x1
                    #         height = y2 - y1
                    #         center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            
                    #         # Check if boat is in sea area
                    #         in_sea = "Unknown"
                    #         if 0 <= center_y < seg_mask.shape[0] and 0 <= center_x < seg_mask.shape[1]:
                    #             in_sea = "Yes" if seg_mask[center_y, center_x] == 1 else "No"
                            
                    #         det_col1, det_col2, det_col3, det_col4 = st.columns(4)
                            
                    #         with det_col1:
                    #             st.write(f"**Boat {i+1}**")
                    #             st.write(f"Confidence: {conf:.3f}")
                            
                    #         with det_col2:
                    #             st.write(f"**Position**")
                    #             st.write(f"Center: ({center_x}, {center_y})")
                            
                    #         with det_col3:
                    #             st.write(f"**Size**")
                    #             st.write(f"{width}×{height} pixels")
                            
                    #         with det_col4:
                    #             st.write(f"**Location**")
                    #             st.write(f"In Sea: {in_sea}")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            st.success("Processing completed for all images!")
    
    # Information section
    with st.expander("ℹ️ About This System"):
        st.markdown(f"""
        ### Combined Sea Segmentation & YOLO Boat Detection System
        
        This system combines semantic segmentation and object detection to provide comprehensive maritime scene analysis.
        
        **Model Configuration:**
        - Segmentation model path: `{SEGMENTATION_MODEL_PATH}`
        - Model is automatically loaded and cached on startup
        - No need to upload or specify model path in UI
        
        **Processing Pipeline:**
        1. **Image Preprocessing**: CLAHE enhancement, Scharr-Y edge detection, Local Binary Pattern
        2. **Sea Segmentation**: Binary classification of sea vs no-sea areas using DeepLabV3+
        3. **Boat Detection**: YOLO-based detection of boat objects
        4. **Combined Analysis**: Overlay results to show boats in context of sea areas
        
        **Key Features:**
        - Binary sea segmentation (sea vs no-sea)
        - YOLO-based boat detection
        - Combined visualization with overlays
        - Detailed analysis of boat locations relative to sea areas
        - Comprehensive metrics and statistics
        - Support for multiple image formats
        - Persistent model loading (no re-loading on each run)
        
        **Model Requirements:**
        - Segmentation model: DeepLabV3+ with ResNet50 backbone (binary classification, ~372 MB)
        - YOLO model: YOLOv8 or custom trained model
        
        **Performance Optimizations:**
        - Cached model loading prevents re-loading
        - GPU memory optimization with half precision
        - Automatic memory cleanup after inference
        - Batch processing support
        
        **Supported Formats:** PNG, JPG, JPEG, TIFF, TIF
        """)

if __name__ == "__main__":
    main()
