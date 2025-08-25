import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   

import cv2
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import time

class FaceTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.detector = MTCNN()
        
    def extract_face_features(self, image_path):
        """Extract face features for training"""
        try:
            print(f"Processing: {image_path}")
            
            # Load image using PIL first
            image = Image.open(image_path)
            image = image.convert('RGB')
            pixels = np.asarray(image)
            
            print(f"Image shape: {pixels.shape}")
            
            # Detect faces
            results = self.detector.detect_faces(pixels)
            print(f"MTCNN detected {len(results)} faces")
            
            if len(results) > 0:
                # Get the best face (highest confidence)
                result = max(results, key=lambda x: x['confidence'])
                print(f"Best face confidence: {result['confidence']:.3f}")
                
                x1, y1, width, height = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                
                # Extract face
                face = pixels[y1:y2, x1:x2]
                
                # Resize face to consistent size
                face_resized = cv2.resize(face, (128, 128))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
                feature_vector = face_gray.flatten()
                
                print(f"✓ Successfully extracted features: {len(feature_vector)} dimensions")
                return feature_vector
            else:
                print("✗ No faces detected")
                return None
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def show_file_dialog(self, image_num):
        """Show file dialog with better error handling"""
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            root.lift()
            root.attributes('-topmost', True)
            
            file_path = filedialog.askopenfilename(
                parent=root,
                title=f"Select Training Image {image_num} (Cancel when done)",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                    ("JPEG files", "*.jpg *.jpeg"),
                    ("PNG files", "*.png"),
                    ("All files", "*.*")
                ],
                initialdir=os.path.expanduser("~")
            )
            
            root.destroy()
            return file_path
            
        except Exception as e:
            print(f"Error with file dialog: {e}")
            return None
    
    def collect_training_images(self):
        """Let user select multiple training images of the same person"""
        training_images = []
        
        print("\n" + "="*50)
        print("TRAINING MODE - COLLECT IMAGES")
        print("="*50)
        print("You need to select 2-5 clear images of the SAME PERSON")
        print("These should be:")
        print("- Different angles of the same person")
        print("- Good lighting and clear face")
        print("- The person you want to find in the output runs")
        print("-" * 50)
        
        for i in range(5):  # Allow up to 5 training images
            print(f"\n>>> Ready to select training image {i+1}/5")
            input("Press ENTER to open file dialog...")
            
            file_path = self.show_file_dialog(i+1)
            
            if file_path and file_path.strip():
                if os.path.exists(file_path):
                    # Test if we can extract features before adding
                    print(f"Testing face detection in: {os.path.basename(file_path)}")
                    test_features = self.extract_face_features(file_path)
                    
                    if test_features is not None:
                        training_images.append(file_path)
                        print(f"✓ Added: {os.path.basename(file_path)}")
                    else:
                        print(f"✗ No clear face found in: {os.path.basename(file_path)}")
                        print("Please select a different image with a clear, visible face.")
                else:
                    print(f"✗ File not found: {file_path}")
            else:
                print("No file selected or cancelled.")
                break
        
        print(f"\nTotal valid images selected: {len(training_images)}")
        
        if len(training_images) < 2:
            print("❌ ERROR: Need at least 2 valid training images!")
            print("Please restart and select images with clear, visible faces.")
            return False
        
        print("✓ Sufficient training images collected!")
        return self.train_from_images(training_images)
    
    def train_from_images(self, training_images):
        """Train the model with user-provided images"""
        print(f"\n🔄 Training with {len(training_images)} images...")
        
        # Extract features from training images (positive samples)
        positive_features = []
        for i, img_path in enumerate(training_images):
            print(f"\nProcessing training image {i+1}/{len(training_images)}: {os.path.basename(img_path)}")
            features = self.extract_face_features(img_path)
            if features is not None:
                positive_features.append(features)
        
        if len(positive_features) < 2:
            print("❌ ERROR: Not enough valid training images!")
            return False
        
        print(f"✓ Successfully processed {len(positive_features)} training images")
        
        # Collect negative samples from output runs
        print("\n🔄 Collecting negative samples from output runs...")
        negative_features = []
        output_base = r'D:\ml projects\output'
        
        if not os.path.exists(output_base):
            print(f"❌ Output directory not found: {output_base}")
            return False
        
        # Get random faces from output runs as negative samples
        collected_negatives = 0
        target_negatives = len(positive_features) * 5  # 5x negative samples
        
        # Get available run directories
        run_dirs = []
        for item in os.listdir(output_base):
            if item.startswith('output run '):
                try:
                    run_number = int(item.split('output run ')[1])
                    run_dirs.append((run_number, item))
                except:
                    continue
        
        run_dirs.sort(key=lambda x: x[0])
        
        for run_number, run_dir in run_dirs:
            if collected_negatives >= target_negatives:
                break
                
            run_path = os.path.join(output_base, run_dir)
            
            if os.path.exists(run_path):
                face_files = [f for f in os.listdir(run_path) if f.endswith('.png')]
                
                # Take a few random faces from each run
                for filename in face_files[:3]:  # Max 3 per run
                    if collected_negatives >= target_negatives:
                        break
                        
                    face_path = os.path.join(run_path, filename)
                    features = self.extract_face_features(face_path)
                    
                    if features is not None:
                        # Check if it's too similar to our positive samples
                        is_similar = False
                        for pos_feat in positive_features:
                            similarity = cosine_similarity([features], [pos_feat])[0][0]
                            if similarity > 0.8:  # Too similar, skip
                                is_similar = True
                                break
                        
                        if not is_similar:
                            negative_features.append(features)
                            collected_negatives += 1
                            if collected_negatives % 20 == 0:
                                print(f"  Collected {collected_negatives} negative samples...")
        
        print(f"✓ Collected {len(negative_features)} negative samples")
        
        if len(negative_features) == 0:
            print("⚠️ Warning: No negative samples collected, creating artificial ones...")
            # Create some artificial negative samples by adding noise
            for pos_feat in positive_features:
                for _ in range(3):
                    noise = np.random.normal(0, 0.1, pos_feat.shape)
                    negative_features.append(pos_feat + noise)
        
        # Prepare training data
        X = np.array(positive_features + negative_features)
        y = np.array([1] * len(positive_features) + [0] * len(negative_features))
        
        print(f"🔄 Training model...")
        print(f"  Total samples: {len(X)}")
        print(f"  Positive (target person): {len(positive_features)}")
        print(f"  Negative (other people): {len(negative_features)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"✅ Model trained successfully!")
        return True
    
    def predict_match(self, face_features):
        """Predict if a face matches the trained person"""
        if not self.is_trained:
            return 0.0
        
        try:
            features_scaled = self.scaler.transform([face_features])
            probability = self.model.predict_proba(features_scaled)[0][1]
            return probability
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0
    
    def save_model(self, model_path):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to: {model_path}")
    
    def load_model(self, model_path):
        """Load a previously trained model"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            print(f"📂 Model loaded from: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

def find_faces_with_trained_model(trainer, similarity_threshold=0.81):  # Changed from 0.6 to 0.81
    """Find faces using the trained model"""
    if not trainer.is_trained:
        print("Model not trained!")
        return []
    
    similar_faces = []
    output_base = r'D:\ml projects\output'
    
    # Get all output runs
    output_runs = []
    for item in os.listdir(output_base):
        if item.startswith('output run '):
            try:
                run_number = int(item.split('output run ')[1])
                output_runs.append((run_number, item))
            except:
                continue
    
    output_runs.sort(key=lambda x: x[0])
    
    # Limit to first 40 output runs
    output_runs = output_runs[:100]
    
    print(f"\nSearching through {len(output_runs)} output runs (limited to first 40)...")
    print(f"Using confidence threshold: 70%+ only")  # Added info message
    
    for run_number, run_dir in output_runs:
        run_path = os.path.join(output_base, run_dir)
        found_match = False
        
        print(f"Checking {run_dir}...")
        
        if os.path.exists(run_path):
            face_files = [f for f in os.listdir(run_path) if f.endswith('.png')]
            
            for filename in face_files:
                face_path = os.path.join(run_path, filename)
                
                features = trainer.extract_face_features(face_path)
                if features is not None:
                    match_probability = trainer.predict_match(features)
                    
                    if match_probability >= similarity_threshold:
                        similar_faces.append({
                            'path': face_path,
                            'run': run_number,
                            'filename': filename,
                            'similarity_score': match_probability
                        })
                        found_match = True
                        print(f"  ✓ HIGH CONFIDENCE MATCH: {filename} (confidence: {match_probability:.3f})")
                    elif match_probability >= 0.5:  # Show near-misses for debugging
                        print(f"  ~ Low confidence skipped: {filename} (confidence: {match_probability:.3f})")
        
        if not found_match:
            print(f"  No 70%+ confidence matches in {run_dir}")
    
    return similar_faces

def save_results(similar_faces, person_name):
    """Save results to faces directory"""
    faces_dir = r'D:\ml projects\faces'
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)
    
    person_dir = os.path.join(faces_dir, person_name)
    counter = 1
    original_dir = person_dir
    while os.path.exists(person_dir):
        person_dir = f"{original_dir}_{counter}"
        counter += 1
    
    os.makedirs(person_dir)
    
    # Copy similar faces
    for i, face_data in enumerate(similar_faces):
        confidence = face_data['similarity_score']
        new_filename = f"match_{i+1:03d}_run{face_data['run']}_conf{confidence:.3f}.png"
        dest_path = os.path.join(person_dir, new_filename)
        shutil.copy2(face_data['path'], dest_path)
    
 
    summary_path = os.path.join(person_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Trained Face Recognition Results\n")
        f.write(f"Person: {person_name}\n")
        f.write(f"Total matches: {len(similar_faces)}\n\n")
        
        for i, face_data in enumerate(similar_faces):
            f.write(f"Match {i+1}: Run {face_data['run']}, Confidence: {face_data['similarity_score']:.3f}\n")
    
    print(f"\nResults saved to: {person_dir}")
    return person_dir

def main():
    print("🎯 TRAINED FACE RECOGNITION SYSTEM")
    print("=" * 50)
    
    trainer = FaceTrainer()
    model_path = r'D:\ml projects\trained_face_model.pkl'
    
    print("\nChoose option:")
    print("1. 🆕 Train new model with your images")
    print("2. 📂 Load existing trained model")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting training process...")
        
        if trainer.collect_training_images():
            trainer.save_model(model_path)
            
            person_name = input("\n👤 Enter a name for this person: ").strip()
            if not person_name:
                person_name = "unknown_person"
            
            print(f"\n🔍 Searching for {person_name} in output runs...")
            
            similar_faces = find_faces_with_trained_model(trainer)
            
            if similar_faces:
                save_results(similar_faces, person_name)
                print("✅ Training and search completed!")
            else:
                print("No matches found in output runs.")
        else:
            print("❌ Training failed.")
    
    elif choice == "2":
        print("\n📂 Loading existing model...")
        if trainer.load_model(model_path):
            print("✅ Model loaded successfully!")
            
            person_name = input("\n👤 Enter a name for this search: ").strip()
            if not person_name:
                person_name = "unknown_person"
            
            print(f"\n🔍 Searching for {person_name} in output runs...")
            
            similar_faces = find_faces_with_trained_model(trainer)
            
            if similar_faces:
                save_results(similar_faces, person_name)
                print("✅ Search completed!")
            else:
                print("No matches found in output runs.")
        else:
            print("❌ No trained model found. Please train a new model first.")
    
    else:
        print("❌ Invalid choice.")

if __name__ == "__main__":
    main()