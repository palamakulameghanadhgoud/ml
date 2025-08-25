import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

def extract_faces(filename, required_size=(224, 224)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    faces = []
    for result in results:
        x1, y1, width, height = result['box']
        
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = pixels[y1:y2, x1:x2]

        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        faces.append(face_array)
    return faces

frame_counter_file = 'frame_counter.txt'
if os.path.exists(frame_counter_file):
    with open(frame_counter_file, 'r') as f:
        start_frame = int(f.read().strip())
else:
    start_frame = 0

print(f"Starting processing from frame {start_frame}")
6
for current_frame in range(start_frame, 2777):
    frame_filename = f'owx/frame{current_frame}.jpg'
    
    if os.path.exists(frame_filename):
        print(f"Processing frame {current_frame}...")
        
        
        faces = extract_faces(frame_filename)

        output_base = r'D:\ml projects\output'
        output_dir = os.path.join(output_base, f'output run {current_frame + 1}')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        
        for i, face in enumerate(faces):
            image = Image.fromarray(face)
            filename = f'face_{i}.png'
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f'  Saved face {i} to {filepath}')

        print(f'Completed output run {current_frame + 1} ({len(faces)} faces found)')
        
      
        with open(frame_counter_file, 'w') as f:
            f.write(str(current_frame + 1))
    else:
        print(f'Frame {current_frame} not found: {frame_filename} - Skipping')

print("All frames processed successfully!")
