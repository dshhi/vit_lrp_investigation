import cv2
import os
import re
import argparse

def create_video_from_images(folder, output_video, fps):
    # Collect all PNG images in the specified folder
    images = [img for img in os.listdir(folder) if img.endswith('.png')]
    
    if not images:
        print("No PNG images found in the specified folder.")
        return

    # Sort images based on the numeric value of topk
    def extract_topk(image_name):
        match = re.search(r'top(\d+)', image_name)
        return int(match.group(1)) if match else float('inf')  # Use inf for unmatched filenames

    images.sort(key=extract_topk)

    # Get the width and height of the first image
    img_path = os.path.join(folder, images[0])
    img = cv2.imread(img_path)
    height, width, layers = img.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_name in images:
        img_path = os.path.join(folder, image_name)
        img = cv2.imread(img_path)

        # Extract the topk number from the filename
        topk = re.search(r'top(\d+)', image_name)
        if topk:
            topk_value = topk.group(1)
            label = f'topk = {topk_value}'
        else:
            label = 'topk = N/A'  # Handle case where topk is not found

        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Reduced font size for better fitting
        font_thickness = 2
        
        # Calculate text position
        text_x = 10
        text_y = height - 10
        
        # Add black border for the text
        cv2.putText(img, label, (text_x - 2, text_y - 2), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(img, label, (text_x + 2, text_y - 2), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(img, label, (text_x - 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness + 2)
        cv2.putText(img, label, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness + 2)

        # Add white text
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        # Write the frame into the video
        out.write(img)

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a video from images with labels.')
    parser.add_argument('folder', type=str, help='Folder containing the images')
    parser.add_argument('output_video', type=str, help='Output video file name')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the video')

    args = parser.parse_args()
    
    create_video_from_images(args.folder, args.output_video, args.fps)
