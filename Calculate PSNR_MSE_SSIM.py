import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def evaluate_video(video_path):
    cap = cv2.VideoCapture(video_path)

    psnr_values = []
    ssim_values = []
    mse_values = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count > 0:
            # Convert frames to grayscale
            prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate PSNR
            psnr = cv2.PSNR(prev_frame, frame)
            psnr_values.append(psnr)

            # Calculate SSIM
            ssim_score = ssim(prev_frame_gray, frame_gray)
            ssim_values.append(ssim_score)

            # Calculate MSE
            mse = mean_squared_error(prev_frame_gray, frame_gray)
            mse_values.append(mse)

        prev_frame = frame.copy()
        frame_count += 1

    cap.release()

    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)

    return avg_psnr, avg_ssim, avg_mse

video_paths = ['BigBuckBunny.mp4']

# Open a text file to save the results
with open('metrics.txt', 'w') as file:
    for video_path in video_paths:
        avg_psnr, avg_ssim, avg_mse = evaluate_video(video_path)
        file.write(f"Video: {video_path}\n")
        file.write("Average PSNR: {}\n".format(avg_psnr))
        file.write("Average SSIM: {}\n".format(avg_ssim))
        file.write("Average MSE: {}\n".format(avg_mse))
        file.write("\n")

print("Metrics saved to 'metrics.txt'")
