import os
import requests
from zipfile import ZipFile
import shutil

# URLs and file paths
data_files = {
    "2016": [
        ("images/images.zip", "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip"),
        ("metadata/ground_truth.csv", "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv"),
    ],
    "2017": [
        ("images/images.zip", "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip"),
        ("metadata/ground_truth.csv", "https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part3_GroundTruth.csv"),
    ],
    "2018": [
        ("images/images.zip", "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip"),
        ("metadata/lesion_groupings.csv", "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip"),
    ],
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

# Placeholder for combining files later
# def combine_files():
#     pass