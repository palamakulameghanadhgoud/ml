import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TensorFlow info messages

import cv2
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import time
from datetime import datetime
import openpyxl

class ImprovedVideoRecognizer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.detector = MTCNN()
        self.current_person_name = None
        self.positive_features = []

    def collect_training_images(self):
        """Let user select unlimited training images of the same person"""
        training_images = []

        print("\n" + "="*50)
        print("TRAINING MODE - COLLECT IMAGES")
        print("="*50)
        print("You can select as many clear images of the SAME PERSON as you want")
        print("These should be:")
        print("- Different angles of the same person")
        print("- Good lighting and clear face")
        print("- The person you want to find in the video")
        print("- More images = better accuracy")
        print("-" * 50)
        print("Keep selecting images until you're done, then cancel to proceed")

        image_count = 0
        while True:  # Unlimited loop
            image_count += 1
            print(f"\n>>> Ready to select training image {image_count}")
            print("Press ENTER to open file dialog, or type 'done' to finish:")
            
            user_input = input().strip().lower()
            if user_input == 'done':
                break

            file_path = self.show_file_dialog(image_count)

            if file_path and file_path.strip():
                if os.path.exists(file_path):
                    # Test if we can extract features before adding
                    print(f"Testing face detection in: {os.path.basename(file_path)}")
                    test_features = self.extract_face_features(file_path)

                    if test_features is not None:
                        training_images.append(file_path)
                        print(f"✓ Added: {os.path.basename(file_path)}")
                        print(f"✓ Total images so far: {len(training_images)}")
                    else:
                        print(f"✗ No clear face found in: {os.path.basename(file_path)}")
                        print("Please select a different image with a clear, visible face.")
                        image_count -= 1  # Don't count failed attempts
                else:
                    print(f"✗ File not found: {file_path}")
                    image_count -= 1  # Don't count failed attempts
            else:
                print("No file selected or cancelled.")
                if len(training_images) >= 2:
                    print(f"You have {len(training_images)} valid images. Ready to proceed?")
                    proceed = input("Type 'done' to finish, or ENTER to add more images: ").strip().lower()
                    if proceed == 'done':
                        break
                image_count -= 1  # Don't count cancelled selections

        print(f"\n📊 TRAINING SUMMARY:")
        print(f"Total valid images selected: {len(training_images)}")
        
        if len(training_images) < 2:
            print("❌ ERROR: Need at least 2 valid training images!")
            print("Please restart and select images with clear, visible faces.")
            return False
        elif len(training_images) < 5:
            print("⚠️  You have fewer than 5 images. More images usually mean better accuracy.")
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Please restart and add more training images.")
                return False
        else:
            print(f"✅ Excellent! {len(training_images)} images should provide good training data.")

        print("✓ Sufficient training images collected!")
        return self.train_from_images(training_images)

    def show_file_dialog(self, image_num):
        """Show file dialog"""
        try:
            root = tk.Tk()
            root.withdraw()
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

    def extract_face_features(self, image_path):
        """Extract face features using MTCNN"""
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            pixels = np.asarray(image)

            results = self.detector.detect_faces(pixels)

            if len(results) > 0:
                result = max(results, key=lambda x: x['confidence'])
                x1, y1, width, height = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                face = pixels[y1:y2, x1:x2]
                face_resized = cv2.resize(face, (128, 128))
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
                feature_vector = face_gray.flatten()

                return feature_vector

            return None

        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def extract_face_features_from_frame(self, frame_rgb):
        """Extract face features from video frame"""
        try:
            results = self.detector.detect_faces(frame_rgb)

            face_data = []
            for result in results:
                x1, y1, width, height = result['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height

                face = frame_rgb[y1:y2, x1:x2]

                if face.size > 0:
                    face_resized = cv2.resize(face, (128, 128))
                    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_RGB2GRAY)
                    feature_vector = face_gray.flatten()

                    face_data.append({
                        'features': feature_vector,
                        'bbox': (x1, y1, x2, y2),
                        'confidence': result['confidence']
                    })

            return face_data

        except Exception as e:
            print(f"Error extracting features from frame: {e}")
            return []

    def train_from_images(self, training_images):
        """Train model using positive samples and output run negatives"""
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

        self.positive_features = positive_features
        print(f"✓ Successfully processed {len(positive_features)} training images")

        # Collect negative samples from OUTPUT RUNS
        print("\n🔄 Collecting negative samples from OUTPUT RUNS...")
        negative_features = []
        output_base = r'D:\ml projects\output'

        if not os.path.exists(output_base):
            print(f"❌ Output directory not found: {output_base}")
            return False

        # Get all output run directories
        run_dirs = []
        for item in os.listdir(output_base):
            if item.startswith('output run '):
                try:
                    run_number = int(item.split('output run ')[1])
                    run_dirs.append((run_number, item))
                except:
                    continue

        run_dirs.sort(key=lambda x: x[0])
        print(f"Found {len(run_dirs)} output run directories")

        # Collect negatives with smart filtering
        collected_negatives = 0
        target_negatives = len(positive_features) * 10  # 10x negative samples
        similarity_threshold = 0.8  # Exclude faces too similar to target

        for run_number, run_dir in run_dirs[:200]:  # Use first 200 runs
            if collected_negatives >= target_negatives:
                break

            run_path = os.path.join(output_base, run_dir)

            if os.path.exists(run_path):
                face_files = [f for f in os.listdir(run_path) if f.endswith('.png')]

                # Take 2 faces per run for variety
                for filename in face_files[:2]:
                    if collected_negatives >= target_negatives:
                        break

                    face_path = os.path.join(run_path, filename)
                    features = self.extract_face_features(face_path)

                    if features is not None:
                        # Check if it's too similar to our positive samples
                        is_similar = False
                        max_similarity = 0
                        for pos_feat in positive_features:
                            similarity = cosine_similarity([features], [pos_feat])[0][0]
                            max_similarity = max(max_similarity, similarity)
                            if similarity > similarity_threshold:
                                is_similar = True
                                break

                        if not is_similar:
                            negative_features.append(features)
                            collected_negatives += 1
                            if collected_negatives % 50 == 0:
                                print(f"  Collected {collected_negatives} negative samples...")
                        else:
                            # This is good - we're filtering out similar faces
                            if max_similarity > 0.9:
                                print(f"  Filtered out very similar face (similarity: {max_similarity:.3f})")

        print(f"✓ Collected {len(negative_features)} negative samples from output runs")

        if len(negative_features) < 10:
            print("⚠️ Warning: Very few negative samples. Creating artificial ones...")
            # Create artificial negatives by adding noise
            original_negatives = len(negative_features)
            for pos_feat in positive_features:
                for noise_level in [0.2, 0.3, 0.4]:
                    if len(negative_features) >= target_negatives:
                        break
                    noise = np.random.normal(0, noise_level, pos_feat.shape)
                    artificial_negative = pos_feat + noise
                    artificial_negative = np.clip(artificial_negative, 0, 255)
                    negative_features.append(artificial_negative)

            print(f"✓ Added {len(negative_features) - original_negatives} artificial negatives")

        # Prepare training data
        X = np.array(positive_features + negative_features)
        y = np.array([1] * len(positive_features) + [0] * len(negative_features))

        print(f"🔄 Training improved model...")
        print(f"  Total samples: {len(X)}")
        print(f"  Positive (target person): {len(positive_features)}")
        print(f"  Negative (output faces): {len(negative_features)}")
        print(f"  Ratio: 1:{len(negative_features)/len(positive_features):.1f}")

        # Scale and train
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        print(f"✅ Enhanced model trained successfully!")
        return True

    def predict_face_match(self, features):
        """Predict if face matches the trained person"""
        if not self.is_trained:
            return 0.0

        try:
            features_scaled = self.scaler.transform([features])
            probability = self.model.predict_proba(features_scaled)[0][1]
            return probability
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0

    def check_person_in_video_cosine(self, video_path):
        """Check if person appears in video using direct cosine similarity"""
        print(f"\n🔍 Pre-checking video for target person using cosine similarity...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_check = min(300, total_frames)
        frame_skip = max(1, total_frames // frames_to_check)

        frame_count = 0
        checked_frames = 0

        while checked_frames < frames_to_check:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip == 0:
                checked_frames += 1

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_data_list = self.extract_face_features_from_frame(frame_rgb)

                for face_data in face_data_list:
                    features = face_data['features']

                    # Check cosine similarity to positive samples
                    for pos_feat in self.positive_features:
                        similarity = cosine_similarity([features], [pos_feat])[0][0]
                        if similarity >= 0.82:  # Lower threshold for checking
                            print(f"🎯 TARGET PERSON LIKELY FOUND! (Cosine similarity: {similarity:.3f})")
                            cap.release()
                            return True

                if checked_frames % 50 == 0:
                    print(f"  Checked {checked_frames}/{frames_to_check} frames...")

        cap.release()
        print(f"❌ Target person not detected in preview")
        return False

    def collect_training_images_from_folder(self, volunteer_id="24100880002"):
        """Load training images from volunteers folder instead of GUI selection"""
        volunteer_folder = rf"D:\ml projects\volunteers\{volunteer_id}"
        
        print("\n" + "="*50)
        print("TRAINING MODE - LOAD FROM VOLUNTEERS FOLDER")
        print("="*50)
        print(f"Loading training images from: {volunteer_folder}")
        
        if not os.path.exists(volunteer_folder):
            print(f"❌ Volunteer folder not found: {volunteer_folder}")
            return False
        
        # Get all image files from the volunteer folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        training_images = []
        
        for filename in os.listdir(volunteer_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(volunteer_folder, filename)
                training_images.append(full_path)
        
        if len(training_images) == 0:
            print(f"❌ No image files found in {volunteer_folder}")
            return False
        
        print(f"Found {len(training_images)} image files:")
        for i, img_path in enumerate(training_images, 1):
            print(f"  {i}. {os.path.basename(img_path)}")
        
        # Test each image for face detection
        valid_training_images = []
        for img_path in training_images:
            print(f"\nTesting face detection in: {os.path.basename(img_path)}")
            test_features = self.extract_face_features(img_path)
            
            if test_features is not None:
                valid_training_images.append(img_path)
                print(f"✓ Valid face found in: {os.path.basename(img_path)}")
            else:
                print(f"✗ No clear face found in: {os.path.basename(img_path)}")
        
        print(f"\n📊 TRAINING SUMMARY:")
        print(f"Total image files found: {len(training_images)}")
        print(f"Valid training images: {len(valid_training_images)}")
        
        if len(valid_training_images) < 2:
            print("❌ ERROR: Need at least 2 valid training images!")
            print("Please add more clear face images to the volunteers folder.")
            return False
        
        print(f"✅ Excellent! {len(valid_training_images)} valid training images found.")
        print("✓ Sufficient training images collected from volunteers folder!")
        
        return self.train_from_images(valid_training_images)

    def check_person_in_video_ml(self, video_path, threshold=0.80):
        """Check if person appears in video using ML model prediction with 80% threshold"""
        print(f"\n🔍 Pre-checking video for target person using ML model...")
        print(f"🎯 Using ML threshold: {threshold:.0%}")

        if not self.is_trained:
            print("❌ Model not trained! Cannot perform ML-based pre-check.")
            return False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_to_check = min(300, total_frames)
        frame_skip = max(1, total_frames // frames_to_check)

        frame_count = 0
        checked_frames = 0
        max_ml_confidence = 0.0
        best_detection_frame = 0

        while checked_frames < frames_to_check:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip == 0:
                checked_frames += 1

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_data_list = self.extract_face_features_from_frame(frame_rgb)

                for face_data in face_data_list:
                    features = face_data['features']
                    
                    # Use ML model prediction with 80% threshold
                    ml_confidence = self.predict_face_match(features)
                    
                    # Track the highest ML confidence found
                    if ml_confidence > max_ml_confidence:
                        max_ml_confidence = ml_confidence
                        best_detection_frame = frame_count
                    
                    # Use 80% threshold for pre-check
                    if ml_confidence >= threshold:
                        print(f"🎯 TARGET PERSON FOUND! (ML confidence: {ml_confidence:.3f} at frame {frame_count})")
                        cap.release()
                        return True

        cap.release()
        
        if max_ml_confidence > 0.6:
            print(f"🔶 Highest ML confidence found: {max_ml_confidence:.3f} at frame {best_detection_frame}")
            print(f"   This is below 80% threshold but person might still be present")
        else:
            print(f"❌ Target person not detected in preview (Max ML confidence: {max_ml_confidence:.3f})")
        
        return False

    def play_video_with_detection(self, video_path, confidence_threshold=0.80, frame_skip=3):
        """Play video with ML-primary detection using 80% threshold"""
        if not self.is_trained:
            print("❌ Model not trained!")
            return []

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"\n🎬 ENHANCED VIDEO DETECTION - 80% ML THRESHOLD")
        print(f"Video: {os.path.basename(video_path)}")
        print(f"🎯 Target: {self.current_person_name}")
        print(f"🔴 RED BOXES = TARGET PERSON (ML ≥ 80%)")
        print(f"🟢 GREEN BOXES = Potential Match (ML 60-79%)")
        print(f"📊 Primary detection: ML Model (80% threshold)")
        print(f"ML confidence threshold: {confidence_threshold:.0%}")
        print("\nControls: SPACE=Pause, Q=Quit, S=Save")
        print("-" * 60)

        frame_count = 0
        detections = []
        paused = False

        cv2.namedWindow('Enhanced Person Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Enhanced Person Detection', 1200, 800)

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ Video completed!")
                    break

                frame_count += 1

                if frame_count % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_data_list = self.extract_face_features_from_frame(frame_rgb)

                    for face_data in face_data_list:
                        features = face_data['features']
                        bbox = face_data['bbox']

                        # Method 1: ML Model prediction (PRIMARY GATE)
                        ml_probability = self.predict_face_match(features)

                        # Method 2: Cosine similarity (SECONDARY INFO ONLY)
                        max_cosine = 0
                        for pos_feat in self.positive_features:
                            cosine_sim = cosine_similarity([features], [pos_feat])[0][0]
                            max_cosine = max(max_cosine, cosine_sim)

                        # DETECTION Logic with 80% ML threshold
                        high_ml_confidence = ml_probability >= confidence_threshold  # 80%+
                        medium_ml_confidence = ml_probability >= 0.60 and ml_probability < confidence_threshold  # 60-79%
                        
                        # TARGET PERSON: MUST have 80%+ ML confidence
                        is_target_person = high_ml_confidence
                        
                        # POTENTIAL MATCH: 60-79% ML confidence
                        is_potential_match = (not is_target_person) and medium_ml_confidence

                        if is_target_person:
                            # RED BOX for TARGET PERSON (80%+ ML confidence)
                            detection = {
                                'frame': frame_count,
                                'bbox': bbox,
                                'ml_confidence': ml_probability,
                                'cosine_similarity': max_cosine,
                                'timestamp': frame_count / fps,
                                'method': 'ML',
                                'detection_type': 'TARGET'
                            }
                            detections.append(detection)

                            print(f"🔴 TARGET PERSON: {self.current_person_name} | ML: {ml_probability:.3f} ✓ (≥80%) | Cosine: {max_cosine:.3f}")

                            # Draw RED detection box
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # RED

                            # RED background for target person label
                            label = f"TARGET: {self.current_person_name}"
                            score_text = f"ML:{ml_probability:.2f} ✓ Cos:{max_cosine:.2f}"

                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            cv2.rectangle(frame, (x1, y1-60), (x1 + max(label_size[0], 220), y1), (0, 0, 255), -1)

                            cv2.putText(frame, label, (x1 + 5, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(frame, score_text, (x1 + 5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                            # RED detection indicator
                            cv2.circle(frame, (x1+15, y1+15), 8, (0, 0, 255), -1)

                        elif is_potential_match:
                            # GREEN BOX for potential matches (60-79% ML confidence)
                            print(f"🟢 POTENTIAL MATCH: ML: {ml_probability:.3f} ~ (60-79%) | Cosine: {max_cosine:.3f}")

                            # Draw GREEN detection box
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # GREEN

                            # GREEN background for potential match label
                            label = f"POTENTIAL: {self.current_person_name}"
                            score_text = f"ML:{ml_probability:.2f} ~ Cos:{max_cosine:.2f}"

                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (x1, y1-50), (x1 + max(label_size[0], 200), y1), (0, 255, 0), -1)

                            cv2.putText(frame, label, (x1 + 5, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, score_text, (x1 + 5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                            # GREEN detection indicator
                            cv2.circle(frame, (x1+10, y1+10), 6, (0, 255, 0), -1)
                        
                        # DEBUG: Show other faces for debugging
                        else:
                            if ml_probability > 0.4:
                                print(f"   Low confidence face: ML: {ml_probability:.3f} | Cosine: {max_cosine:.3f}")

            # Enhanced status overlay
            status_text = f"Frame: {frame_count}/{total_frames} | TARGET DETECTIONS: {len(detections)} | Person: {self.current_person_name}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            mode_text = f"🔴 RED = TARGET (ML≥80%) | 🟢 GREEN = POTENTIAL (ML 60-79%) | Volunteer: 24100880002"
            cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Threshold display
            threshold_text = f"ML Target=80% | ML Potential=60-79% | Training from volunteers folder"
            cv2.putText(frame, threshold_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow('Enhanced Person Detection', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n🛑 Quitting...")
                break
            elif key == ord(' '):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord('s'):
                if detections:
                    self.save_detections(detections, video_path)
                else:
                    print("No target person detections to save yet")

        cap.release()
        cv2.destroyAllWindows()

        # Enhanced summary
        print(f"\n📊 DETECTION SUMMARY:")
        print(f"  Target: {self.current_person_name} (Volunteer 24100880002)")
        print(f"  🔴 HIGH CONFIDENCE (ML ≥ 80%): {len(detections)}")

        if detections:
            print(f"  Average ML confidence: {np.mean([d['ml_confidence'] for d in detections]):.3f}")
            print(f"  Average cosine similarity: {np.mean([d['cosine_similarity'] for d in detections]):.3f}")

            # Show detection breakdown
            print(f"\n📈 DETECTION BREAKDOWN:")
            for i, detection in enumerate(detections[:10]):  # Show first 10
                timestamp = detection['timestamp']
                ml_conf = detection['ml_confidence']
                cos_sim = detection['cosine_similarity']
                print(f"  {i+1:2d}. Time: {timestamp:6.1f}s | ML: {ml_conf:.3f} | Cos: {cos_sim:.3f}")

            if len(detections) > 10:
                print(f"  ... and {len(detections)-10} more detections")

            save_choice = input("\n💾 Save TARGET PERSON results? (y/n): ").strip().lower()
            if save_choice == 'y':
                self.save_detections(detections, video_path)

        return detections

    def save_detections(self, detections, video_path):
        """Save detection results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        results_dir = rf'D:\ml projects\enhanced_detections\{self.current_person_name}_{video_name}_{timestamp}'
        os.makedirs(results_dir, exist_ok=True)

        report_path = os.path.join(results_dir, "enhanced_detection_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"ENHANCED VIDEO DETECTION REPORT\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Target: {self.current_person_name}\n")
            f.write(f"Video: {os.path.basename(video_path)}\n")
            f.write(f"Total detections: {len(detections)}\n")
            f.write(f"Detection method: ML Model + Cosine Similarity\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            ml_detections = sum(1 for d in detections if d['method'] == 'ML')
            cosine_detections = sum(1 for d in detections if d['method'] == 'Cosine')

            f.write(f"Detection breakdown:\n")
            f.write(f"  ML Model detections: {ml_detections}\n")
            f.write(f"  Cosine similarity detections: {cosine_detections}\n")
            f.write(f"  Average ML confidence: {np.mean([d['ml_confidence'] for d in detections]):.3f}\n")
            f.write(f"  Average cosine similarity: {np.mean([d['cosine_similarity'] for d in detections]):.3f}\n\n")

            for i, detection in enumerate(detections):
                f.write(f"Detection {i+1}:\n")
                f.write(f"  Frame: {detection['frame']}\n")
                f.write(f"  Time: {detection['timestamp']:.2f}s\n")
                f.write(f"  ML Confidence: {detection['ml_confidence']:.3f}\n")
                f.write(f"  Cosine Similarity: {detection['cosine_similarity']:.3f}\n")
                f.write(f"  Detection Method: {detection['method']}\n")
                f.write(f"  Detection Type: {detection['detection_type']}\n")
                f.write(f"  Bounding Box: {detection['bbox']}\n\n")

        print(f"\n💾 Enhanced results saved to: {results_dir}")

    def save_model(self, person_name):
        """Save the trained model to a pickle file"""
        if not self.is_trained:
            print("❌ No trained model to save!")
            return False
        
        # Create models directory if it doesn't exist
        models_dir = r'D:\ml projects\trained_models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{person_name}_{timestamp}_model.pkl"
        model_path = os.path.join(models_dir, model_filename)
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'positive_features': self.positive_features,
            'person_name': person_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_positive_samples': len(self.positive_features),
            'model_type': 'RandomForestClassifier'
        }
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✅ Model saved successfully!")
            print(f"📁 Saved to: {model_path}")
            print(f"👤 Person: {person_name}")
            print(f"📊 Training samples: {len(self.positive_features)}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False

    def load_model(self, model_path=None, person_name=None):
        """Load a previously trained model"""
        models_dir = r'D:\ml projects\trained_models'
        
        if model_path is None:
            # Let user select a model file
            if not os.path.exists(models_dir):
                print(f"❌ Models directory not found: {models_dir}")
                return False
            
            # List available models
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            
            if not model_files:
                print("❌ No saved models found!")
                return False
            
            print("\n📁 Available trained models:")
            for i, filename in enumerate(model_files, 1):
                print(f"  {i}. {filename}")
            
            try:
                choice = int(input(f"\nSelect model (1-{len(model_files)}): ")) - 1
                if 0 <= choice < len(model_files):
                    model_path = os.path.join(models_dir, model_files[choice])
                else:
                    print("❌ Invalid selection!")
                    return False
            except ValueError:
                print("❌ Invalid input!")
                return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load model components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.positive_features = model_data['positive_features']
            self.current_person_name = model_data['person_name']
            self.is_trained = True
            
            print(f"✅ Model loaded successfully!")
            print(f"📁 From: {os.path.basename(model_path)}")
            print(f"👤 Person: {model_data['person_name']}")
            print(f"📅 Training date: {model_data['training_date']}")
            print(f"📊 Training samples: {model_data['num_positive_samples']}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def list_saved_models(self):
        """List all saved models"""
        models_dir = r'D:\ml projects\trained_models'
        
        if not os.path.exists(models_dir):
            print(f"❌ Models directory not found: {models_dir}")
            return
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            print("❌ No saved models found!")
            return
        
        print(f"\n📁 Found {len(model_files)} saved models:")
        print("-" * 60)
        
        for filename in model_files:
            model_path = os.path.join(models_dir, filename)
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                print(f"📄 {filename}")
                print(f"   👤 Person: {model_data['person_name']}")
                print(f"   📅 Date: {model_data['training_date']}")
                print(f"   📊 Samples: {model_data['num_positive_samples']}")
                print()
            except:
                print(f"📄 {filename} (corrupted)")

    def select_video_file(self):
        """Select video file"""
        try:
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                    ("MP4 files", "*.mp4"),
                    ("AVI files", "*.avi"),
                    ("All files", "*.*")
                ],
                initialdir=r'D:\ml projects'
            )
            
            root.destroy()
            
            if video_path and os.path.exists(video_path):
                print(f"✓ Selected video: {os.path.basename(video_path)}")
                return video_path
            else:
                return None
                
        except Exception as e:
            print(f"Error selecting video: {e}")
            return None

    def collect_training_images_from_selected_folder(self):
        """Let user select a folder and use all images inside for training"""
        try:
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            folder_path = filedialog.askdirectory(
                title="Select Folder Containing Training Images",
                initialdir=os.path.expanduser("~")
            )
            root.destroy()
        except Exception as e:
            print(f"Error selecting folder: {e}")
            return False

        if not folder_path or not os.path.exists(folder_path):
            print("❌ No folder selected or folder does not exist.")
            return False

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
        training_images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]

        print(f"\nFound {len(training_images)} image files in selected folder: {folder_path}")
        if len(training_images) < 2:
            print("❌ Need at least 2 valid training images!")
            return False

        valid_training_images = []
        for img_path in training_images:
            print(f"Testing face detection in: {os.path.basename(img_path)}")
            test_features = self.extract_face_features(img_path)
            if test_features is not None:
                valid_training_images.append(img_path)
                print(f"✓ Valid face found in: {os.path.basename(img_path)}")
            else:
                print(f"✗ No clear face found in: {os.path.basename(img_path)}")

        print(f"\nTotal valid training images: {len(valid_training_images)}")
        if len(valid_training_images) < 2:
            print("❌ Not enough valid images for training.")
            return False

        return self.train_from_images(valid_training_images)

def main():
    print("🚀 ENHANCED VIDEO FACE RECOGNITION")
    print("=" * 60)
    print("🎯 Dual Detection: ML Model + Cosine Similarity")
    print("🔴 RED BOXES = Target Person (High Confidence)")
    print("🟢 GREEN BOXES = Potential Match (Medium Confidence)")
    print("📊 Uses extensive output run training data")

    recognizer = ImprovedVideoRecognizer()
    model_path = r'D:\ml projects\trained_models\video_recognizer_model.pkl'

    # Step 1: Try to load existing model
    if os.path.exists(model_path):
        print("\n📂 Found existing trained model. Loading...")
        if recognizer.load_model(model_path):
            print("✅ Model loaded successfully!")
        else:
            print("❌ Failed to load model. Will train a new one.")
    else:
        print("\n🆕 No trained model found. Starting training process...")
        print("Choose training input method:")
        print("1. Select images one by one")
        print("2. Select a folder containing all training images")
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "2":
            if not recognizer.collect_training_images_from_selected_folder():
                print("❌ Training failed")
                return
        else:
            if not recognizer.collect_training_images():
                print("❌ Training failed")
                return
        recognizer.current_person_name = input("\n👤 Enter person name: ").strip() or "unknown_person"
        recognizer.save_model(recognizer.current_person_name)
        print("✅ Model trained and saved!")

    # Step 2: Select video
    print("\n🎬 STEP 2: Select video file")
    video_path = recognizer.select_video_file()
    if not video_path:
        print("❌ No video selected")
        return

    # Step 3: Quick check if person appears in video
    print("\n🔍 STEP 3: Quick video check")
    person_found = recognizer.check_person_in_video_cosine(video_path)

    if not person_found:
        print("⚠️ Target person not found in quick check")
        proceed = input("Continue with full detection anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return
    else:
        print("✅ Target person detected in preview!")

    # Step 4: Configure and start detection
    print("\n⚙️ STEP 4: Configure detection")
    confidence = float(input("ML confidence threshold (0.75): ") or "0.75")
    frame_skip = int(input("Process every N frames (3): ") or "3")

    print(f"\n🎯 Starting enhanced detection for: {recognizer.current_person_name}")
    print("🔴 RED = High confidence target person")
    print("🟢 GREEN = Medium confidence potential match")

    detections = recognizer.play_video_with_detection(video_path, confidence, frame_skip)

    print("✅ Enhanced detection completed!")

if __name__ == "__main__":
    main()