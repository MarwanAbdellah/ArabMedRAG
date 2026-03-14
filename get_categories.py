import pandas as pd
df = pd.read_csv('data/concatenated_df.csv')
with open('categories_out.txt', 'w', encoding='utf-8') as f:
    f.write('---Arabic---\n')
    f.write('\n'.join(df['category'].dropna().unique()) + '\n')
    f.write('---English---\n')
    f.write('\n'.join(df['category_en'].dropna().unique()) + '\n')
