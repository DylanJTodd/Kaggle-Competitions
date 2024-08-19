import os
import requests
from zipfile import ZipFile
import shutil
import pandas as pd
from PIL import Image

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

    for root, dirs, files in os.walk(extract_to):
        if root != extract_to:
            for file in files:
                if file.lower() not in ['license', 'attribution.txt']:
                    shutil.move(os.path.join(root, file), os.path.join(extract_to, file))
    
    for root, dirs, files in os.walk(extract_to, topdown=False):
        for file in files:
            if file.lower() in ['license', 'attribution.txt']:
                os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))

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
                    df['image'] = df['image'].str.replace('_downsampled', '')
                    df.to_csv(metadata_file, index=False)
                    print(f"Removed '_downsampled' from isic_id in {year} metadata.")
                    
def combine_metadata(base_dir='data'):
    column_a_mapping = {
        '2019': 'image',
        '2020': 'image_name',
        '2024': 'isic_id'
    }

    # Iterate through each year folder
    for year in ['2019', '2020', '2024']:
        metadata_dir = os.path.join(base_dir, year, 'metadata')
        
        combined_df = None
        
        for file_name in os.listdir(metadata_dir):
            file_path = os.path.join(metadata_dir, file_name)
            
            df = pd.read_csv(file_path)
            
            column_a_name = column_a_mapping[year]
            df = df.rename(columns={column_a_name: 'isic_id'})
            
            if year == '2019':
                df['isic_id'] = df['isic_id'].str.replace('_downsampled', '')
            
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.merge(combined_df, df, on='isic_id', how='outer')
        
        output_file_path = os.path.join(metadata_dir, 'metadata.csv')
        combined_df.to_csv(output_file_path, index=False)
        
        for file_name in os.listdir(metadata_dir):
            if file_name != 'metadata.csv':  # Keep the newly created metadata.csv
                file_path = os.path.join(metadata_dir, file_name)
                os.remove(file_path)

def remove_duplicates(base_dir='data'):
    isic_id_set = set()
    years = ['2024', '2020', '2019']
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

def crop_resize_images(base_dir = 'data'):
    years = ["2019", "2020"]

    for year in years:
        images_directory = os.path.join(base_dir, year, "images")
        
        for file in os.listdir(images_directory):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(images_directory, file)
                
                with Image.open(image_path) as img:
                    width, height = img.size

                    square_size = min(width, height)

                    left = (width - square_size) / 2
                    top = 0
                    right = (width + square_size) / 2
                    bottom = height

                    img_cropped = img.crop((left, top, right, bottom))
                    img_resized = img_cropped.resize((127, 127), Image.Resampling.LANCZOS)
                    img_resized.save(image_path)

import os
import pandas as pd
import numpy as np

def adjusting_metadata(base_dir='data'):
    anatom_site_general_encoding = {}
    encoding_counter = 0

    for year in os.listdir(base_dir):
        year_dir = os.path.join(base_dir, year, 'metadata')
        metadata_file = os.path.join(year_dir, 'metadata.csv')
        
        if os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file, header=None if year == '2016' else 0)
            df.columns = ['isic_id'] + list(df.columns[1:]) if year != '2016' else ['isic_id', 'MEL']
            
            try:
                if year == '2019':
                    try:
                        df = df.drop(columns=['NV', 'AK', 'BKL', 'VASC', 'UNK', 'lesion_id'])
                    except Exception as e:
                        print(f"Error dropping columns for {year}: {e}")
                    try:
                        for value in df['anatom_site_general'].unique():
                            if value not in anatom_site_general_encoding:
                                anatom_site_general_encoding[value] = encoding_counter
                                encoding_counter += 1
                        df['anatom_site_general'] = df['anatom_site_general'].map(anatom_site_general_encoding)
                        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
                        df['malignant'] = df[['MEL', 'BCC', 'SCC']].max(axis=1)
                        df = df[['isic_id', 'age_approx', 'sex', 'anatom_site_general', 'malignant']]
                    except Exception as e:
                        print(f"Error encoding columns for {year}: {e}")

                        df = df.replace({np.nan: -1})

                elif year == '2020':
                    try:
                        df = df.drop(columns=['patient_id', 'diagnosis', 'benign_malignant', 'patient_id.1', 'sex.1', 'age_approx.1', 'anatom_site_general_challenge.1', 'diagnosis.1', 'benign_malignant.1', 'target.1'])
                    except Exception as e:
                        print(f"Error dropping columns for {year}: {e}")

                    try:
                        df['anatom_site_general_challenge'] = df['anatom_site_general_challenge'].map(anatom_site_general_encoding)
                        df = df.rename(columns={'anatom_site_general_challenge': 'anatom_site_general', 'target': 'malignant'})
                        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
                        df = df[['isic_id', 'age_approx', 'sex', 'anatom_site_general', 'malignant']]
                    except Exception as e:
                        print(f"Error encoding columns for {year}: {e}")

                    df = df.replace({np.nan: -1})


                elif year == '2024':
                    try:
                        columns_to_remove = ['attribution', 'copyright_license', 'lesion_id', 'iddx_full', 'iddx_1', 'iddx_2', 'iddx_3', 'iddx_4', 'iddx_5', 
                                            'mel_mitotic_index', 'mel_thick_mm', 'tbp_lv_dnn_lesion_confidence', 'patient_id', 'clin_size_long_diam_mm', 
                                            'image_type', 'tbp_tile_type', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 
                                            'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 
                                            'tbp_lv_deltaL', 'tbp_lv_eccentricity', 'tbp_lv_location', 'tbp_lv_location_simple', 'tbp_lv_minorAxisMM', 
                                            'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt', 
                                            'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_color_std_mean',
                                            'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM']
                        df = df.drop(columns=columns_to_remove)
                    except Exception as e:
                        print(f"Error dropping columns for {year}: {e}")

                    try:
                        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
                        for value in df['anatom_site_general'].unique():
                            if value not in anatom_site_general_encoding:
                                anatom_site_general_encoding[value] = encoding_counter
                                encoding_counter += 1
                        df['anatom_site_general'] = df['anatom_site_general'].map(anatom_site_general_encoding)
                        columns_order = ['isic_id', 'age_approx', 'sex'] + [col for col in df.columns if col not in ['isic_id', 'age_approx', 'sex', 'malignant']] + ['malignant']
                        df = df[columns_order]
                    except Exception as e:
                        print(f"Error encoding columns for {year}: {e}")

                    df = df.replace({np.nan: -1})

                df = df.loc[:, ~df.columns.duplicated()]
                df.to_csv(metadata_file, index=False)
                print(f"Adjusted metadata for {year}.")
            except Exception as e:
                print(f"Error processing {year}: {e}")

def fill_missing_values(base_dir='data'):
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.isdir(folder_path):
            metadata_file = os.path.join(folder_path, 'metadata', 'metadata.csv')
            
            if os.path.exists(metadata_file):
                df = pd.read_csv(metadata_file)
                df.fillna(-1, inplace=True)
                
                df.to_csv(metadata_file, index=False)
                print(f'Processed and updated: {metadata_file}')
            else:
                print(f'metadata.csv not found in {folder_path}/metadata')

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
    print("Downloading and extracting data...")
    main()
    print("Data download and extraction completed.")
    print("Renaming and cleaning up files...")
    rename_and_cleanup_files()
    print("Renaming and cleaning up files completed.")
    print("Combining loose metadata files...")
    combine_metadata()
    print("Combining loose metadata files completed.")
    print("Adjusting metadata and removing duplicates...")
    adjusting_metadata()
    remove_duplicates()
    print("Adjusting metadata and removing duplicates completed.")
    print("Cropping and resizing images.")
    print("This may take a while...")
    crop_resize_images()
    print("Cropping and resizing images completed.")
    print("Filling missing values with -1.")
    fill_missing_values()
    print("Filling missing values completed.")
    print("Everything completed successfully.")