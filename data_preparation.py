import os
from lxml import etree
import pandas as pd

#function to map the levels to CEFR levels 
def cefr_level_mapper(level):
    level = int(level)  
    if 1 <= level <= 3:
        return "A1"
    elif 4 <= level <= 6:
        return "A2"
    elif 7 <= level <= 9:
        return "B1"
    elif 10 <= level <= 12:
        return "B2"
    elif 13 <= level <= 15:
        return "C1"
    elif level == 16:
        return "C2"
    else:
        return None

data = []

xml_file_path = '/filepath/EFCAMDAT_Database.xml'

tree = etree.parse(xml_file_path, parser=etree.XMLParser(recover=True))
root = tree.getroot()

for writing in root.findall('.//writing'):
    level = writing.get('level')
    text = writing.find('text').text
    if text:  
        text = text.strip()
        mapped_level = cefr_level_mapper(level) 
        data.append({'CEFR_Level': mapped_level, 'Text': text})

df = pd.DataFrame(data)

if df.empty:
    print("DataFrame is empty.")
else:
    csv_file_path = '/filepath/efcamdat_data.csv'
  
    df.to_csv(csv_file_path, index=False)
    print(f"Data extraction and saving completed. CSV file created at {csv_file_path}")

#Create subset 
subset_df = df.sample(frac=0.025, random_state=42)

#save downsized set
subset_df.to_csv('/filepath/ds_set.csv', index=False)


