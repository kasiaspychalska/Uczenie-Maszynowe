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

plt.scatter(df['Age'], df['ATS'])
plt.title('Annual Total Salary depending on age')
plt.xlabel('Age')
plt.ylabel('Annual Total Salary \n [EUR]')
plt.show()

# Second plot
ans5 = [df.loc[df['Gender'] == 'Man', 'ATS'],
        df.loc[df['Gender'] == 'Woman', 'ATS']]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
# Create the boxplot
bp = ax.boxplot(ans5, labels=list('MF'))
plt.title('Annual Total Salary by gender')
plt.ylabel('Annual Total Salary \n [EUR]')
plt.show()
