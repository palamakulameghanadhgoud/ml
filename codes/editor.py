import os
import shutil
from pathlib import Path
from datetime import datetime

def consolidate_output_images():
    """Consolidate all images from output run folders into one directory"""
    

    output_base = r'D:\ml projects\output'
    consolidated_dir = r'D:\ml projects\consolidated_faces'

    if not os.path.exists(consolidated_dir):
        os.makedirs(consolidated_dir)
        print(f"✅ Created consolidated directory: {consolidated_dir}")
    else:
        print(f"📁 Using existing directory: {consolidated_dir}")
    
    
    if not os.path.exists(output_base):
        print(f"❌ Output directory not found: {output_base}")
        return False
    
    output_runs = []
    for item in os.listdir(output_base):
        if item.startswith('output run '):
            try:
                run_number = int(item.split('output run ')[1])
                output_runs.append((run_number, item))
            except ValueError:
                continue
    

    output_runs.sort(key=lambda x: x[0])
    
    print(f"\n📊 Found {len(output_runs)} output run directories")
    print(f"🎯 Consolidating all face images into: {consolidated_dir}")
    print("-" * 60)
    
    total_images = 0
    processed_runs = 0
    skipped_files = 0
    
    for run_number, run_dir in output_runs:
        run_path = os.path.join(output_base, run_dir)
        
        if os.path.exists(run_path):
         
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
            image_files = []
            
            for file in os.listdir(run_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file)
            
            if image_files:
                print(f"Processing {run_dir}: {len(image_files)} images")
                
                for i, filename in enumerate(image_files):
                    source_path = os.path.join(run_path, filename)
                    
                   
                    file_ext = os.path.splitext(filename)[1]
                    new_filename = f"run{run_number:04d}_face{i+1:02d}{file_ext}"
                    dest_path = os.path.join(consolidated_dir, new_filename)
                    
                   
                    counter = 1
                    original_dest = dest_path
                    while os.path.exists(dest_path):
                        name_part = os.path.splitext(new_filename)[0]
                        dest_path = os.path.join(consolidated_dir, f"{name_part}_v{counter}{file_ext}")
                        counter += 1
                    
                    try:
                        shutil.copy2(source_path, dest_path)
                        total_images += 1
                    except Exception as e:
                        print(f"  ⚠️ Error copying {filename}: {e}")
                        skipped_files += 1
                
                processed_runs += 1
            else:
                print(f"Skipping {run_dir}: No image files found")
        else:
            print(f"⚠️ Directory not found: {run_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 CONSOLIDATION SUMMARY")
    print("="*60)
    print(f"✅ Processed runs: {processed_runs}/{len(output_runs)}")
    print(f"✅ Total images consolidated: {total_images}")
    print(f"⚠️ Skipped files (errors): {skipped_files}")
    print(f"📁 Consolidated directory: {consolidated_dir}")
    print(f"💾 Total size: {get_directory_size(consolidated_dir):.2f} MB")
    
    return True

def get_directory_size(directory):
    """Get directory size in MB"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        print(f"Error calculating directory size: {e}")
        return 0
    
    return total_size / (1024 * 1024)  # Convert to MB

def create_consolidated_index():
    """Create an index file showing which images came from which runs"""
    consolidated_dir = r'D:\ml projects\consolidated_faces'
    index_file = os.path.join(consolidated_dir, 'consolidation_index.txt')
    
    if not os.path.exists(consolidated_dir):
        print("❌ Consolidated directory not found. Run consolidation first.")
        return False
    
   
    files_by_run = {}
    
    for filename in os.listdir(consolidated_dir):
        if filename.startswith('run') and not filename.endswith('.txt'):
            
            try:
                run_part = filename.split('_')[0]  
                run_number = int(run_part[3:])  
                if run_number not in files_by_run:
                    files_by_run[run_number] = []
                files_by_run[run_number].append(filename)
            except (ValueError, IndexError):
                continue
    
    # Write index file
    try:
        with open(index_file, 'w') as f:
            f.write("CONSOLIDATED FACES INDEX\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total runs processed: {len(files_by_run)}\n")
            f.write(f"Total images: {sum(len(files) for files in files_by_run.values())}\n\n")
            
            for run_number in sorted(files_by_run.keys()):
                f.write(f"Output Run {run_number}: {len(files_by_run[run_number])} images\n")
                for filename in sorted(files_by_run[run_number]):
                    f.write(f"  {filename}\n")
                f.write("\n")
        
        print(f"✅ Index file created: {index_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating index file: {e}")
        return False

def main():
    print("🔄 OUTPUT FOLDER CONSOLIDATION TOOL")
    print("=" * 50)
    print("This tool will consolidate all face images from output run folders")
    print("into a single directory for easier processing.")
    print()
    
    choice = input("Choose option:\n1. Consolidate images\n2. Create index file\n3. Both\n\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting image consolidation...")
        if consolidate_output_images():
            print("\n✅ Consolidation completed successfully!")
        else:
            print("\n❌ Consolidation failed!")
    
    elif choice == "2":
        print("\n📝 Creating index file...")
        if create_consolidated_index():
            print("\n✅ Index file created successfully!")
        else:
            print("\n❌ Index file creation failed!")
    
    elif choice == "3":
        print("\n🚀 Starting image consolidation...")
        if consolidate_output_images():
            print("\n📝 Creating index file...")
            if create_consolidated_index():
                print("\n✅ Both consolidation and index creation completed!")
            else:
                print("\n⚠️ Consolidation completed but index creation failed!")
        else:
            print("\n❌ Consolidation failed!")
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
