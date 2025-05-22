import pandas as pd

df = pd.read_csv('/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/TrainingGroundTruth.csv')

ids = df['image'].tolist()
img_root = '/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/train_data'
with open('/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/train.txt', 'w') as f:
    for img_id in ids:
        f.write(f'{img_root}/{img_id}.jpg' + '\n')
f.close()
