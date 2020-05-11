import pandas as pd
import matplotlib.pyplot as plt
questions = pd.read_csv("survey_results_schema.csv", header=0)
df = pd.read_csv("survey_results_public.csv", header=0,
                 usecols=['WorkWeekHrs', 'CompTotal', 'CurrencySymbol', 'Age', 'Gender'])

df = df.dropna()
df = df.loc[df['CurrencySymbol'] == 'EUR']
# Change value in column WorkWeekHrs to get a Full Time Equivalent (FTE)
df['WorkWeekHrs'] = df['WorkWeekHrs'] / 40
# Calculation a Annual Total Salary for FTE
df['ATS'] = round(df['CompTotal'] / df['WorkWeekHrs'], 0)
