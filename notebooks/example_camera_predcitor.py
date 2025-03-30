import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import sys
import time


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
from efficient_track_anything.efficienttam_camera_predictor import EfficientTAMCameraPredictor

checkpoint = "../checkpoints/efficienttam_ti_512x512.pt"
model_cfg = "configs/efficienttam/efficienttam_ti_512x512.yaml"

# EfficientTAM s - 11 FPS
# Efficient TAM s 512 - 30 FPS

predictor: EfficientTAMCameraPredictor = build_efficienttam_camera_predictor(model_cfg, checkpoint)

cap = cv2.VideoCapture("videos/bedroom.mp4")

if_init = False

# Set up matplotlib figure for real-time display
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 8))
display_img = ax.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
plt.axis('off')
plt.tight_layout()

# Define a keypress event handler
exit_program = False
def on_key_press(event):
    global exit_program
    if event.key == 'q':
        exit_program = True
fig.canvas.mpl_connect('key_press_event', on_key_press)

# For FPS and timing calculations
start_time = time.time()
frame_count = 0
fps_text = ax.text(10, 30, "FPS: 0", color='green', fontsize=20, backgroundcolor='black')
processing_time_text = ax.text(10, 60, "Processing Time: 0s", color='green', fontsize=20, backgroundcolor='black')

# Get video duration
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_duration = total_frames / fps if fps > 0 else 0
duration_text = ax.text(10, 90, f"Video Duration: {video_duration:.2f}s", color='green', fontsize=20, backgroundcolor='black')

while True and not exit_program:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width, height = frame.shape[:2][::-1]
    if not if_init:

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        # Let's add a positive click at (x, y) = (210, 350) to get started


        ##! add points, `1` means positive click and `0` means negative click
        # points = np.array([[660, 267]], dtype=np.float32)
        # labels = np.array([1], dtype=np.int32)

        # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
        # )

        ## ! add bbox
        points = np.array([[200, 300], [275, 175]], dtype=np.float32)
        labels = np.array([1, 0], dtype=np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points,
            labels=labels,
        )

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)

        all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
        frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
    
    # Calculate and display average FPS
    current_time = time.time()
    elapsed_time = current_time - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    fps_text.set_text(f"Avg FPS: {avg_fps:.1f}")
    
    # Update processing time
    processing_time_text.set_text(f"Processing Time: {elapsed_time:.2f}s")
    
    # Update the matplotlib display
    display_img.set_data(frame)
    plt.draw()
    plt.pause(0.0001)  # Add short pause to allow the GUI to update

# Final timing calculations
end_time = time.time()
total_processing_time = end_time - start_time
final_avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0

print(f"Processing complete:")
print(f"Total frames processed: {frame_count}")
print(f"Total processing time: {total_processing_time:.2f} seconds")
print(f"Video duration: {video_duration:.2f} seconds")
print(f"Average FPS: {final_avg_fps:.2f}")

cap.release()
plt.ioff()  # Turn off interactive mode
plt.close(fig)
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
