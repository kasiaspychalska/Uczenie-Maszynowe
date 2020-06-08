import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("survey_results_public.csv", header=0,
                 usecols=['WorkWeekHrs', 'ConvertedComp', 'CompTotal', 'CodeRevHrs', 'YearsCode', 'Age', 'Age1stCode'])
df = df.dropna()
df.loc[df['YearsCode'] == 'Less than 1 year'] = 0
df.loc[df['YearsCode'] == 'More than 50 years'] = 51
df.loc[df['Age1stCode'] == 'Younger than 5 years'] = 0
df.loc[df['Age1stCode'] == 'Older than 85'] = 86
df['YearsCode'] = df['YearsCode'].astype("float64")
df['Age1stCode'] = df['Age1stCode'].astype('float64')
print(df.corr(method='pearson'))

fig, ((ax1, ax2, ax3, ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11, ax12, ax13, ax14),
      (ax15, ax16, ax17, ax18, ax19, ax20, ax21)) = plt.subplots(3, 7)
ax1.scatter(df['WorkWeekHrs'], df['ConvertedComp'])
ax1.set_title('WorkWeekHrs/ConvertedComp')
ax2.scatter(df['WorkWeekHrs'], df['CompTotal'])
ax2.set_title('WorkWeekHrs/CompTotal')
ax3.scatter(df['WorkWeekHrs'], df['CodeRevHrs'])
ax3.set_title('WorkWeekHrs/CodeRevHrs')
ax4.scatter(df['WorkWeekHrs'], df['YearsCode'])
ax4.set_title('WorkWeekHrs/YearsCode')
ax5.scatter(df['WorkWeekHrs'], df['Age'])
ax5.set_title('WorkWeekHrs/Age')
ax6.scatter(df['WorkWeekHrs'], df['Age1stCode'])
ax6.set_title('WorkWeekHrs/Age1stCode')
ax7.scatter(df['ConvertedComp'], df['CompTotal'])
ax7.set_title('ConvertedComp/CompTotal')
ax8.scatter(df['ConvertedComp'], df['CodeRevHrs'])
ax8.set_title('ConvertedComp/CodeRevHrs')
ax9.scatter(df['ConvertedComp'], df['YearsCode'])
ax9.set_title('ConvertedComp/YearsCode')
ax10.scatter(df['ConvertedComp'], df['Age'])
ax10.set_title('ConvertedComp/Age')
ax11.scatter(df['ConvertedComp'], df['Age1stCode'])
ax11.set_title('ConvertedComp/Age1stCode')
ax12.scatter(df['CompTotal'], df['CodeRevHrs'])
ax12.set_title('CompTotal/CodeRevHrs')
ax13.scatter(df['CompTotal'], df['YearsCode'])
ax13.set_title('CompTotal/YearsCode')
ax14.scatter(df['CompTotal'], df['Age'])
ax14.set_title('CompTotal/Age')
ax15.scatter(df['CompTotal'], df['Age1stCode'])
ax15.set_title('CompTotal/Age1stCode')
ax16.scatter(df['CodeRevHrs'], df['YearsCode'])
ax16.set_title('CodeRevHrs/YearsCode')
ax17.scatter(df['CodeRevHrs'], df['Age'])
ax17.set_title('CodeRevHrs/Age')
ax18.scatter(df['CodeRevHrs'], df['Age1stCode'])
ax18.set_title('CodeRevHrs/Age1stCode')
ax19.scatter(df['YearsCode'], df['Age'])
ax19.set_title('YearsCode/Age')
ax20.scatter(df['YearsCode'], df['Age1stCode'])
ax20.set_title('YearsCode/Age1stCode')
ax21.scatter(df['Age'], df['Age1stCode'])
ax21.set_title('Age/Age1stCode')

# Zmienną zależną y będzie kolumna YearsCode, 
# a zmiennymi niezależnymi x1 oraz x2 odpowiednio Age1stCode oraz Age
