import cv2, argparse
from vittrack import VitTrack

VIDEO_PATH = 'pexels_videos/aerial-view-of-waterskiing-adventure-on-lake-32130802.mp4'  # Replace with your video path

def main():
    backend_target_pairs = [
        [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
        [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA],
        [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA_FP16],
        [cv2.dnn.DNN_BACKEND_TIMVX, cv2.dnn.DNN_TARGET_NPU],
        [cv2.dnn.DNN_BACKEND_CANN, cv2.dnn.DNN_TARGET_NPU]
    ]

    parser = argparse.ArgumentParser(description="VIT track OpenCV API")
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the input video. Omit for using the default camera.')
    parser.add_argument('--model_path', type=str, default='object_tracking_vittrack_2023sep.onnx',
                        help='Path to the model file.')
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                        help='''Choose a backend-target pair:
                                0: OpenCV implementation + CPU,
                                1: CUDA + GPU (CUDA),
                                2: CUDA + GPU (CUDA FP16),
                                3: TIM-VX + NPU,
                                4: CANN + NPU
                        ''')
    args = parser.parse_args()

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    tracker = VitTrack(model_path=args.model_path, backend_id=backend_id, target_id=target_id)

    # Open the video capture (webcam or video file)
    cap = cv2.VideoCapture(VIDEO_PATH)  # Use a path like 'video.mp4' to track on a file

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame from source.")
        return

    # Let user select the bounding box
    bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object")

    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker
        success, box, _ = tracker.infer(frame)

        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display the result
        cv2.imshow("ViTTrack Tracker", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
