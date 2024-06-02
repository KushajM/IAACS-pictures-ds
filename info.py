import os
import pandas as pd

images_dir = 'images'
csv_file = 'public/test_set.csv'

image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

df = pd.read_csv(csv_file)

data = { 'Actual Name': [], 'New Name': [], 'Score': []}

count_index = 1

for image_file in image_files:
    if image_file in df['image'].values:
        score = df.loc[df['image'] == image_file, 'score'].values[0]
        data['New Name'].append(count_index)
        data['Actual Name'].append(image_file)
        data['Score'].append(score)
        count_index += 1
        if count_index > 50:
            break

result_df = pd.DataFrame(data)

result_df.to_csv('info.csv', index=False)

print(result_df)

for index, row in result_df.iterrows():
    old_name = row['Actual Name']
    new_name = f"{row['New Name']}.jpg"
    old_path = os.path.join(images_dir, old_name)
    new_path = os.path.join(images_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} to {new_name}")
    else:
        print("Error")

