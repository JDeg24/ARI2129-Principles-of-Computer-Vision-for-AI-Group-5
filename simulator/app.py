import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="Edge Detection Simulator", layout="wide")

st.title("Interactive Edge Detection Simulator")
st.write("Compare Sobel, Prewitt, Laplacian of Gaussian, and Canny edge detection.")

# Sidebar image input
st.sidebar.header("Image Input")

input_mode = st.sidebar.radio(
    "Choose image source",
    ["Upload image", "Use built-in sample"]
)

uploaded_file = None

# Only show file uploader when the user chooses upload mode.
if input_mode == "Upload image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

# Sidebar parameters
st.sidebar.header("Parameters")

blur_kernel = st.sidebar.slider(
    "Gaussian blur kernel size", 1, 15, 5, step=2,
    help="Controls how much smoothing is applied before LoG and Canny. Higher values reduce noise but can blur fine edges."
    )

noise_level = st.sidebar.slider(
    "Noise level", 0, 50, 0,
    help="Adds random brightness changes to the grayscale image to test failure cases."
    )
canny_low = st.sidebar.slider(
    "Canny low threshold", 0, 255, 50,
    help="Controls which weak edges may be kept if connected to strong edges. Lower values keep more weak edges."
    )
canny_high = st.sidebar.slider(
    "Canny high threshold", 0, 255, 150,
    help="Controls which edges are immediately accepted as strong edges. Higher values keep only stronger boundaries."
    )
log_threshold = st.sidebar.slider(
    "LoG threshold", 0, 255, 20,
    help="Controls how strong the Laplacian response must be to appear as an edge. Higher values remove weaker details."
    )
compare_noise = st.sidebar.checkbox("Compare clean vs noisy", value=False)

def load_image(file):
    """Load uploaded image and convert it to RGB."""
    image = Image.open(file).convert("RGB")
    return np.array(image)

def create_sample_image():
    """Load the built-in sample image from the project assets folder."""
    image = cv2.imread("simulator/assets/sample_images/sample.jpg")

    if image is None:
        st.error("Built-in sample image not found. Check simulator/assets/sample_images/sample.jpg")
        st.stop()

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_grayscale(image):
    """Convert RGB image to grayscale because edge detectors use intensity changes."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def add_noise(gray, noise_level):
    """Add Gaussian noise to demonstrate detector sensitivity and failure cases."""
    if noise_level == 0:
        return gray

    noise = np.random.normal(0, noise_level, gray.shape)
    noisy_image = gray.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def apply_sobel(gray):
    """Apply Sobel using horizontal and vertical weighted gradient kernels."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_prewitt(gray):
    """Apply Prewitt using uniform horizontal and vertical gradient kernels."""
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])

    kernel_y = np.array([[1, 1, 1],
                         [0, 0, 0],
                         [-1, -1, -1]])

    gx = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def apply_log(gray, blur_kernel, threshold):
    """Apply LoG: Gaussian smoothing followed by Laplacian second derivative."""
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F)

    log_abs = np.absolute(log)
    log_norm = cv2.normalize(log_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, log_binary = cv2.threshold(log_norm, threshold, 255, cv2.THRESH_BINARY)
    return log_binary

def apply_canny(gray, blur_kernel, low, high):
    """Apply Canny after smoothing, using low and high hysteresis thresholds."""
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    return cv2.Canny(blurred, low, high)

# App runs when an uploaded image exists or the sample image mode is selected.
if uploaded_file is not None or input_mode == "Use built-in sample":

    # Load either the uploaded image or the built-in sample image.
    if input_mode == "Use built-in sample":
        image = create_sample_image()
    else:
        image = load_image(uploaded_file)

    gray = to_grayscale(image)

    # Keep a clean version and create a noisy version so the checkbox can compare both.
    clean_gray = gray
    noisy_gray = add_noise(gray, noise_level)

    # If comparison mode is off, the app uses the noisy image as the main processed input.
    processed_gray = noisy_gray

    # Apply all four detectors to the processed grayscale image.
    sobel = apply_sobel(processed_gray)
    prewitt = apply_prewitt(processed_gray)
    log = apply_log(processed_gray, blur_kernel, log_threshold)
    canny = apply_canny(processed_gray, blur_kernel, canny_low, canny_high)

    # When comparison mode is on, compute a second set of outputs from the clean image.
    if compare_noise:
        clean_sobel = apply_sobel(clean_gray)
        clean_prewitt = apply_prewitt(clean_gray)
        clean_log = apply_log(clean_gray, blur_kernel, log_threshold)
        clean_canny = apply_canny(clean_gray, blur_kernel, canny_low, canny_high)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Grayscale / Noisy Image")
        st.image(processed_gray, width="stretch")

    st.divider()
    st.subheader("Edge Detection Comparison")

    if compare_noise:
        st.write("Clean image outputs")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.write("Sobel")
            st.image(clean_sobel, width="stretch")
            st.caption("Weighted gradient detector. Usually smoother than Prewitt but still produces thick edges.")

        with c2:
            st.write("Prewitt")
            st.image(clean_prewitt, width="stretch")
            st.caption("Uniform gradient detector. Fast and simple, but usually more sensitive to noise.")

        with c3:
            st.write("Laplacian of Gaussian")
            st.image(clean_log, width="stretch")
            st.caption("Smooths first, then detects second-derivative changes. Can highlight contours but may over-detect.")

        with c4:
            st.write("Canny")
            st.image(clean_canny, width="stretch")
            st.caption("Multi-stage detector. Produces thin, cleaner edges using thresholds and hysteresis.")

        st.write("Noisy image outputs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.write("Sobel")
        st.image(sobel, width="stretch")
        st.caption("Weighted gradient detector. Usually smoother than Prewitt but still produces thick edges.")

    with c2:
        st.write("Prewitt")
        st.image(prewitt, width="stretch")
        st.caption("Uniform gradient detector. Fast and simple, but usually more sensitive to noise.")

    with c3:
        st.write("Laplacian of Gaussian")
        st.image(log, width="stretch")
        st.caption("Smooths first, then detects second-derivative changes. Can highlight contours but may over-detect.")

    with c4:
        st.write("Canny")
        st.image(canny, width="stretch")
        st.caption("Multi-stage detector. Produces thin, cleaner edges using thresholds and hysteresis.")

    st.divider()
    
    st.subheader("Current Settings Insight")

    st.markdown(f"""
    Blur kernel: {blur_kernel} → {'More smoothing' if blur_kernel > 7 else 'Less smoothing'}  
    Noise level: {noise_level} → {'High noise (harder detection)' if noise_level > 20 else 'Low noise'}  
    Canny thresholds: {canny_low}-{canny_high} → {'Strict edges' if canny_high > 150 else 'More edges detected'}  
    LoG threshold: {log_threshold} → {'Only strong edges' if log_threshold > 30 else 'More details detected'}  
    Compare clean vs noisy: {'On — showing clean and noisy outputs separately' if compare_noise else 'Off — showing one processed output'}
    """)

    st.subheader("Parameter Sensitivity and Failure Modes")

    st.write(
        """
        Increasing noise shows how simple gradient methods such as Sobel and Prewitt can produce many false edges.
        LoG may also over-detect edges because second derivatives are sensitive to small intensity fluctuations.
        Canny is usually more stable because it smooths the image first and uses double thresholding with hysteresis.
        """
    )

    st.subheader("Method Comparison")

    # This table links the visual outputs to the theory from the study notes.
    st.table({
        "Method": ["Sobel", "Prewitt", "LoG", "Canny"],
        "Type": ["First-order gradient", "First-order gradient", "Second derivative", "Multi-stage"],
        "Noise Sensitivity": ["Medium", "High", "High", "Low"],
        "Edge Quality": ["Smooth but thick", "Rougher and noisier", "Detailed but noise-sensitive", "Thin and clean"],
        "Key Strength": [
            "Balances smoothing and detection",
            "Simple and fast",
            "Detects contour-like structures",
            "Best overall edge quality"
        ]
    })


else:
    st.info("Upload an image from the sidebar or choose the built-in sample to begin.")