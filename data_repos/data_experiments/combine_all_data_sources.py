import pandas as pd
import glob, os
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../data/arab_observer_1960_1962_stopwords_ner_congo.csv')
df_hathi = pd.read_csv('../data/hathi_trust_1963_1966_stopwords_ner_congo.csv')

df = df.drop(['Unnamed: 0', 'index', 'google_vision_text', 'file_name', 'vol'], axis=1)

df_hathi.rename(columns={'first_month': 'month'}, inplace=True)
df_hathi.rename(columns={'first_month_index': 'month_index'}, inplace=True)
df_hathi.rename(columns={'page': 'page_number'}, inplace=True)
df_hathi = df_hathi.drop(['Unnamed: 0', 'level_0', 'htrc_vol', 'index', 'lowercase','second_month', 'second_month_index'], axis=1)
df_hathi['day'] = '01'
df_hathi['date'] = df_hathi.year.astype(str)+ '-' +df_hathi.month.astype(str) +'-' + df_hathi.day.astype(str)
df_hathi['string_date'] = df_hathi.date.astype(str)
df['string_date'] = df.date.astype(str)


df_1 = df.append(df_hathi, ignore_index=True)

df_1.month[df_1.month == 'Jan'] = 'January'
df_1.month[df_1.month == 'Aug'] = 'August'
df_1.month[df_1.month == 'Oct'] = 'October'
df_1.month[df_1.month == 'Sept'] = 'September'
df_1.month[df_1.month == 'Sep'] = 'September'
df_1.month[df_1.month == 'Apr'] = 'April'
df_1.month[df_1.month == 'Jul'] = 'July'
df_1['date'] = df_1.year.astype(str) +'-'+df_1.month+'-'+df_1.day.astype(str)
df_1['datetime'] = pd.to_datetime(df_1['date'], format='%Y%m%d', errors='ignore')
df_1.to_csv('../data/combined_all_data_ner_congo.csv')
