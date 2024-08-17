import os
import requests
from zipfile import ZipFile
import shutil
import pandas as pd

# URLs and file paths
data_files = {
    "2019": [
        ("images/images.zip", "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"),
        ("metadata/metadata.csv", "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"),
        ("metadata/ground_truth.csv", "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"),
    ],
    "2020": [
        ("images/images.zip", "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_JPEG.zip"),
        ("metadata/ground_truth.csv", "https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Training_GroundTruth_v2.csv"),
    ],
    "2024": [
        ("images/images.zip", "https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Training_Input.zip"),
        ("metadata/supplement.csv", "https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Training_Supplement.csv"),
        ("metadata/ground_truth.csv", "https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Training_GroundTruth.csv"),
    ],
}

def download_file(url, file_path):
    response = requests.get(url, stream=True)
    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

def unzip_file(zip_path, extract_to):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.remove(zip_path)  # Delete the zip file after extraction

    # Move files from subdirectories to the main images directory
    for root, dirs, files in os.walk(extract_to):
        if root != extract_to:
            for file in files:
                if file.lower() not in ['license', 'attribution.txt']:
                    shutil.move(os.path.join(root, file), os.path.join(extract_to, file))
    
    # Remove empty directories and unnecessary files
    for root, dirs, files in os.walk(extract_to, topdown=False):
        for file in files:
            if file.lower() in ['license', 'attribution.txt']:
                os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

    # Handle csv and txt files
    handle_files(extract_to, os.path.join(extract_to, '../metadata'))

def handle_files(images_dir, metadata_dir):
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.csv'):
                move_file(file_path, metadata_dir)
            elif file.endswith('.txt'):
                os.remove(file_path)

def move_file(file_path, metadata_dir):
    base_name = os.path.basename(file_path)
    new_file_path = os.path.join(metadata_dir, base_name)
    
    # Rename file if it already exists
    if os.path.exists(new_file_path):
        base, ext = os.path.splitext(base_name)
        counter = 1
        while os.path.exists(new_file_path):
            new_file_path = os.path.join(metadata_dir, f"{base}{counter}{ext}")
            counter += 1
    
    shutil.move(file_path, new_file_path)

def rename_and_cleanup_files(base_dir='data'):
    for year in os.listdir(base_dir):
        year_images_dir = os.path.join(base_dir, year, 'images')
        year_metadata_dir = os.path.join(base_dir, year, 'metadata')

        if year == '2019':
            if os.path.exists(year_images_dir):
                for img_file in os.listdir(year_images_dir):
                    if img_file.endswith('_downsampled.jpg'):
                        new_name = img_file.replace('_downsampled', '')
                        os.rename(os.path.join(year_images_dir, img_file), os.path.join(year_images_dir, new_name))
                        print(f"Renamed image {img_file} to {new_name} in {year} images folder.")

            if os.path.exists(year_metadata_dir):
                metadata_file = os.path.join(year_metadata_dir, 'metadata.csv')
                if os.path.exists(metadata_file):
                    df = pd.read_csv(metadata_file)
                    df[0] = df[0].str.replace('_downsampled', '')
                    df.to_csv(metadata_file, index=False)
                    print(f"Removed '_downsampled' from isic_id in {year} metadata.")

def combine_metadata(base_dir='data'):
    for year in os.listdir(base_dir):
        year_dir = os.path.join(base_dir, year, 'metadata')
        
        if os.path.exists(year_dir) and os.path.isdir(year_dir):
            csv_files = [f for f in os.listdir(year_dir) if f.endswith('.csv') and f != 'metadata.csv']
            
            if len(csv_files) == 0:
                print(f"No CSV files found for {year}.")
                continue
            
            combined_df = pd.DataFrame()

            for i, csv_file in enumerate(csv_files):
                csv_path = os.path.join(year_dir, csv_file)
                df = pd.read_csv(csv_path, header=None if year == '2016' else 0)
                df.columns = ['isic_id'] + list(df.columns[1:]) if year != '2016' else ['isic_id', 'MEL']
                
                if i == 0:
                    combined_df = df
                else:
                    combined_df = pd.concat([combined_df, df.drop(columns=['isic_id'])], axis=1)

            combined_csv_path = os.path.join(year_dir, 'metadata.csv')
            combined_df.to_csv(combined_csv_path, index=False)
            print(f"Combined CSV files into {combined_csv_path}.")

            for csv_file in csv_files:
                os.remove(os.path.join(year_dir, csv_file))

def remove_duplicates(base_dir='data'):
    isic_id_set = set()
    years = ['2024', '2020', '2019']  # process years in reverse order
    removed_entries = []

    for year in years:
        year_dir = os.path.join(base_dir, year)
        metadata_file = os.path.join(year_dir, 'metadata', 'metadata.csv')
        images_dir = os.path.join(year_dir, 'images')
        
        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file)
            
            rows_to_drop = []
            for index, row in df.iterrows():
                isic_id = row['isic_id']
                if isic_id in isic_id_set:
                    rows_to_drop.append(index)
                    img_file = os.path.join(images_dir, f"{isic_id}.jpg")
                    if os.path.exists(img_file):
                        os.remove(img_file)
                        removed_entries.append((year, isic_id, 'image'))
                else:
                    isic_id_set.add(isic_id)

            df.drop(rows_to_drop, inplace=True)
            df.to_csv(metadata_file, index=False)
            removed_entries.extend([(year, isic_id, 'metadata') for index in rows_to_drop])
            print(f"Processed duplicates for {year}.")

    print(f"Total removed entries: {len(removed_entries)}")
    return removed_entries

def main():
    base_dir = 'data'

    # Check if the base directory already exists
    if os.path.exists(base_dir):
        overwrite = input("The data directory already exists. Do you want to overwrite it? This will delete all existing data. (y/n): ")
        if overwrite.lower() == 'y':
            shutil.rmtree(base_dir)  # Delete the entire data directory
        else:
            print("Exiting without downloading any data.")
            return  # Exit the script

    os.makedirs(base_dir)  # Create the base directory

    # Proceed with downloading and extracting files
    for year, files in data_files.items():
        year_dir = os.path.join(base_dir, year)
        images_dir = os.path.join(year_dir, 'images')
        metadata_dir = os.path.join(year_dir, 'metadata')

        os.makedirs(images_dir)
        os.makedirs(metadata_dir)

        for relative_path, url in files:
            file_path = os.path.join(year_dir, relative_path)

            print(f"Downloading {url} to {file_path}...")
            download_file(url, file_path)

            if file_path.endswith('.zip'):
                print(f"Unzipping {file_path}...")
                unzip_file(file_path, images_dir)

if __name__ == "__main__":
    main()
    rename_and_cleanup_files()
    combine_metadata()
    remove_duplicates()