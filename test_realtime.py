import cv2
import torch
from network import TransformNetwork
from images_unit import imload_realtime, postprocess_image

def load_transform_network(model_path, device):
    transform_network = TransformNetwork()
    transform_network.load_state_dict(torch.load(model_path, map_location=device))
    transform_network = transform_network.to(device)
    transform_network.eval()  # Set model to evaluation mode
    return transform_network

def real_time_style_transfer(model_path, cuda_device_no=0, imsize=256):
    # Set device
    device = torch.device("cuda" if cuda_device_no >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    transform_network = load_transform_network(model_path, device)

    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 is the default webcam
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        # flip frame
        frame = cv2.flip(frame, 1)
        # Resize frame for processing
        input_image = imload_realtime(frame, imsize).to(device)

        # Apply style transfer
        with torch.no_grad():
            output_image = transform_network(input_image)

        # Post-process and display output
        stylized_frame = postprocess_image(output_image)
        cv2.imshow('Real-Time Style Transfer', stylized_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define paths
model_path = "Path&original_img/model/anime.pth"
cuda_device_no = 0  # Use GPU 0, or set -1 for CPU

# Run the real-time style transfer
real_time_style_transfer(model_path, cuda_device_no)