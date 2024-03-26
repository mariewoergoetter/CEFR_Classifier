def cefr_level_mapper(level):
    """Map numeric levels to CEFR levels."""
    level = int(level)  # Ensure level is an integer
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

xml_file_path = '/Users/mlwfee/Desktop/CEFR_Classifier/data/EFCAMDAT_Database.xml'

tree = etree.parse(xml_file_path, parser=etree.XMLParser(recover=True))
root = tree.getroot()

for writing in root.findall('.//writing'):
    level = writing.get('level')
    text = writing.find('text').text
    if text:  # Ensure text is not None
        text = text.strip()  # Remove any leading/trailing whitespace
        mapped_level = cefr_level_mapper(level)  # Map numeric level to CEFR level
        data.append({'CEFR_Level': mapped_level, 'Text': text})

df = pd.DataFrame(data)

if df.empty:
    print("DataFrame is empty.")
else:
    csv_file_path = '/Users/mlwfee/Desktop/CEFR_Classifier/data/efcamdat_data.csv'
  
    df.to_csv(csv_file_path, index=False)
    print(f"Data extraction and saving completed. CSV file created at {csv_file_path}")


subset_df = df.sample(frac=0.025, random_state=42)

subset_df.to_csv('/Users/mlwfee/Desktop/CEFR_Classifier/data//ds_set.csv', index=False)


