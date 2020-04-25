import pandas as pd
df = pd.read_csv("train.tsv", delimiter='\t', encoding="utf-8",
                 names=["Cost", "NumberOfRooms", "Area", "Floor", "Address", "Description"])
df2 = pd.read_csv("description.csv", delimiter=',', header=0, encoding='utf-8')
# Adding a column from the second data frame to the first data frame using marge function (equivalent join in SQL)
df = pd.merge(df, df2, left_on='Floor', right_on='liczba', how='left')
with open('out2.csv', 'w', encoding="utf-8") as csvfile:
    df.to_csv(csvfile, header=False, index=False, line_terminator='\n')
