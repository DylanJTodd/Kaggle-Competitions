import os
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm

def rename_and_cleanup_files(base_dir='data'):
    for year in os.listdir(base_dir):
        year_images_dir = os.path.join(base_dir, year, 'images')
        year_metadata_dir = os.path.join(base_dir, year, 'metadata')

        if year == '2017' and os.path.exists(year_images_dir):
            for img_file in os.listdir(year_images_dir):
                if img_file.endswith('_superpixels.png'):
                    os.remove(os.path.join(year_images_dir, img_file))
                    print(f"Deleted image {img_file} in {year} images folder.")

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
                    df['isic_id'] = df['isic_id'].str.replace('_downsampled', '')
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
                if year == '2016':
                    df['MEL'] = df['MEL'].map({'benign': 0.0, 'malignant': 1.0})
                elif year == '2017':
                    try:
                        df = df.drop(columns=['seborrheic_keratosis'])
                    except Exception as e:
                        print(f"Error dropping columns for {year}: {e}")

                    try:
                        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
                    except Exception as e:
                        print(f"Error encoding columns for {year}: {e}")
                        
                    df['age_approximate'] = df['age_approximate'].replace({'unknown': -1})
                    df = df.replace({np.nan: -1})

                elif year == '2018':
                    try:
                        df = df.drop(columns=['NV', 'BKL', 'DF', 'VASC'])
                        df['malignant'] = df[['MEL', 'BCC', 'AKIEC']].max(axis=1)
                        df = df[['isic_id', 'malignant']]
                    except Exception as e:
                        print(f"Error processing {year}: {e}")

                elif year == '2019':
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
                                            'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z']
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

def remove_duplicates(base_dir='data'):
    isic_id_set = set()
    years = ['2024', '2020', '2019', '2018', '2017', '2016']  # process years in reverse order
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

    # Remove year directories with empty images folder
    for year in years:
        year_images_dir = os.path.join(base_dir, year, 'images')
        if os.path.exists(year_images_dir) and len(os.listdir(year_images_dir)) == 0:
            os.rmdir(year_images_dir)
            year_dir = os.path.join(base_dir, year)
            if len(os.listdir(year_dir)) == 0:
                os.rmdir(year_dir)
                print(f"Deleted empty year directory: {year}")

    print(f"Total removed entries: {len(removed_entries)}")
    return removed_entries

def preprocess_images(base_dir='data', target_size=(139, 139)):
    for year in os.listdir(base_dir):
        year_images_dir = os.path.join(base_dir, year, 'images')

        if os.path.exists(year_images_dir):
            for img_file in os.listdir(year_images_dir):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(year_images_dir, img_file)
                    image = cv2.imread(img_path)

                    # Resize image
                    image = cv2.resize(image, target_size)

                    # Convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Normalize image
                    image = image / 255.0

                    # Save preprocessed image
                    cv2.imwrite(img_path, (image * 255).astype(np.uint8))
                    print(f"Preprocessed image {img_file} in {year} images folder.")

def augment_images(base_dir='data', target_size=(139, 139)):
    augmentation_transforms_malignant = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.Resize(height=target_size[0], width=target_size[1])
    ])

    augmentation_transforms_non_malignant = A.Compose([
        A.Flip(p=0.5),
        A.Resize(height=target_size[0], width=target_size[1])
    ])

    augment_index = 1  # Initialize augment index

    for year in ['2019', '2020', '2024']:  # Augment these years
        year_images_dir = os.path.join(base_dir, year, 'images')
        metadata_file = os.path.join(base_dir, year, 'metadata', 'metadata.csv')

        if os.path.exists(year_images_dir) and os.path.exists(metadata_file):
            df = pd.read_csv(metadata_file)
            malignant_column = 'malignant' if 'malignant' in df.columns else None

            if malignant_column is None:
                print(f"No malignant column found for {year}.")
                continue

            new_rows = []  # List to store new rows for concatenation

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Augmenting {year}"):
                isic_id = row['isic_id']
                malignant = row[malignant_column]

                img_file = os.path.join(year_images_dir, f"{isic_id}.jpg")
                if not os.path.exists(img_file):
                    continue

                image = cv2.imread(img_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                augment_count = 5 if malignant else 1

                for _ in range(augment_count):
                    transform = augmentation_transforms_malignant if malignant else augmentation_transforms_non_malignant
                    augmented = transform(image=image)['image']

                    # Convert back to BGR before saving
                    augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

                    # Name augmented image
                    augment_name = f"augment_{augment_index:08d}"
                    augmented_img_path = os.path.join(year_images_dir, f"{augment_name}.jpg")
                    cv2.imwrite(augmented_img_path, augmented)
                    print(f"Saved augmented image {augmented_img_path}.")

                    # Create new metadata row
                    new_row = row.copy()
                    new_row['isic_id'] = augment_name
                    new_rows.append(new_row)  # Add new row to list

                    augment_index += 1  # Increment augment index

            # Concatenate new rows to the DataFrame
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

            # Save updated metadata
            df.to_csv(metadata_file, index=False)
            print(f"Updated metadata for {year}.")
            
if __name__ == "__main__":
    rename_and_cleanup_files()
    combine_metadata()
    adjusting_metadata()
    removed_entries = remove_duplicates()
    preprocess_images()
    augment_images()
