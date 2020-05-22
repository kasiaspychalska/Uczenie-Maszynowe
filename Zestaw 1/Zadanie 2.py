import pandas as pd
df = pd.read_csv("train.tsv", delimiter='\t',
                 names=["Cost", "NumberOfRooms", "Area", "Floor", "Address", "Description"])
df['CostOfSquareMeter'] = df.Cost/df.Area
# Select rows from data frame
df2 = df[(df.NumberOfRooms >= 3) &
         (df.CostOfSquareMeter < df.CostOfSquareMeter.mean())]
# Select columns from data frame
df2 = df2[['NumberOfRooms', 'Cost', 'CostOfSquareMeter']]
with open('out1.csv', 'w', encoding="utf-8") as csvfile:
    df2.to_csv(csvfile, index=False, line_terminator='\n')
