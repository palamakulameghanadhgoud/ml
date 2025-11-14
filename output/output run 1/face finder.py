import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow info messages

import cv2
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import shutil
try:
    import tkinter as tk
    from tkinter import filedialog
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
from sklearn.metrics.pairwise import cosine_similarity

def browse_for_image():
    """Open file dialog to select reference image from anywhere"""
    if not GUI_AVAILABLE:
        print("GUI not available. Please enter the full path to your image:")
        return input("Image path: ").strip().strip('"')
    
    try:
        root = tk.Tk()
        root.attributes('-topmost', True)  # Bring dialog to front
        root.withdraw()  # Hide the main window
        
        print("Opening file dialog... Please select your reference image.")
        
        # File dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Reference Face Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        if file_path:
            print(f"Selected: {file_path}")
        else:
            print("No file selected.")
            
        return file_path
        
    except Exception as e:
        print(f"Error opening file dialog: {e}")
        print("Please enter the full path to your image:")
        return input("Image path: ").strip().strip('"')

def extract_first_face_from_image(image_path, output_path):
    """Extract the first detected face from an image and save it to the output path."""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return False
        
        # Convert the image to RGB (OpenCV loads images in BGR format by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize the MTCNN face detector
        detector = MTCNN()
        
        # Detect faces in the image
        faces = detector.detect_faces(image_rgb)
        
        if not faces:
            print("No faces found in the image.")
            return False
        
        # Get the bounding box for the first detected face
        x1, y1, width, height = faces[0]['box']
        x2, y2 = x1 + width, y1 + height
        
        # Extract the face region
        face_image = image[y1:y2, x1:x2]
        
        # Save the extracted face image
        cv2.imwrite(output_path, face_image)
        
        return True
    
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

def find_similar_faces_in_output(ref_face_path):
    """Find faces in the output directory that are similar to the reference face."""
    similar_faces = []
    
    try:
        # Load the reference face image
        ref_image = cv2.imread(ref_face_path)
        if ref_image is None:
            print(f"Error loading reference image: {ref_face_path}")
            return similar_faces
        
        # Convert the reference image to RGB
        ref_image_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        
        # Initialize the MTCNN face detector
        detector = MTCNN()
        
        # Detect the reference face
        ref_faces = detector.detect_faces(ref_image_rgb)
        
        if not ref_faces:
            print("No face found in the reference image.")
            return similar_faces
        
        # Get the embedding for the reference face
        ref_face_embedding = ref_faces[0]['embedding']
        
        # Directory where output faces are saved
        output_dir = r'D:\ml projects\output_faces'
        
        if not os.path.exists(output_dir):
            print(f"Output directory not found: {output_dir}")
            return similar_faces
        
        # Iterate over all face images in the output directory
        for face_file in os.listdir(output_dir):
            face_path = os.path.join(output_dir, face_file)
            
            # Load the face image
            face_image = cv2.imread(face_path)
            if face_image is None:
                print(f"Error loading face image: {face_path}")
                continue
            
            # Convert the face image to RGB
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Detect the face
            faces = detector.detect_faces(face_image_rgb)
            
            if not faces:
                print(f"No face found in the image: {face_path}")
                continue
            
            # Get the embedding for the detected face
            face_embedding = faces[0]['embedding']
            
            # Compute the cosine similarity between the reference face and the detected face
            similarity = cosine_similarity([ref_face_embedding], [face_embedding])
            
            # If similarity is above a certain threshold, consider it a match (e.g., 0.9)
            if similarity >= 0.9:
                similar_faces.append(face_path)
                print(f"Found similar face: {face_path} (similarity: {similarity[0][0]:.4f})")
    
    except Exception as e:
        print(f"Error finding similar faces: {e}")
    
    return similar_faces

def save_face_data(face_paths, ref_face_path):
    """Save the face data (images and metadata) for the similar faces."""
    try:
        # Create a directory for the results using the current timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join(r'D:\ml projects\results', f"face_recognition_{timestamp}")
        os.makedirs(result_dir)
        
        # Copy the reference face to the results directory
        shutil.copy(ref_face_path, result_dir)
        
        # Save each similar face image to the results directory
        for face_path in face_paths:
            shutil.copy(face_path, result_dir)
        
        # Optionally, save metadata (e.g., paths of similar faces) to a text file
        metadata_path = os.path.join(result_dir, 'metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write("Reference Face:\n")
            f.write(f"{ref_face_path}\n\n")
            f.write("Similar Faces:\n")
            for face_path in face_paths:
                f.write(f"{face_path}\n")
        
        return result_dir
    
    except Exception as e:
        print(f"Error saving face data: {e}")
        return None

def main():
    print("Face Recognition System")
    print("======================")
    
    # Step 1: Browse for reference image
    print("\nStep 1: Select reference face image from your computer")
    print("A file dialog should open. If not, you'll be asked to enter the path manually.")
    
    selected_image = browse_for_image()
    
    if not selected_image:
        print("No image selected. Exiting.")
        return
    
    if not os.path.exists(selected_image):
        print(f"File not found: {selected_image}")
        return
    
    print(f"Selected image: {selected_image}")
    
    # Create reference directory
    ref_dir = r'D:\ml projects\reference_faces'
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    
    # Extract face from selected image
    image_name = os.path.splitext(os.path.basename(selected_image))[0]
    ref_face_path = os.path.join(ref_dir, f"reference_{image_name}.png")
    
    if extract_first_face_from_image(selected_image, ref_face_path):
        print(f"\nReference face saved: {ref_face_path}")
        
        # Step 2: Find similar faces
        print("\nStep 2: Searching for similar faces in output runs...")
        similar_faces = find_similar_faces_in_output(ref_face_path)
        
        if similar_faces:
            # Step 3: Save results
            print(f"\nStep 3: Saving {len(similar_faces)} similar faces...")
            result_dir = save_face_data(similar_faces, ref_face_path)
            print(f"Process completed! Results saved in: {result_dir}")
        else:
            print("No similar faces found.")
    else:
        print("Failed to extract reference face.")

if __name__ == "__main__":
    main()