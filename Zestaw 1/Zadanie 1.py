import pandas as pd
df = pd.read_csv("train.tsv", delimiter='\t',
                 names=["Cost", "NumberOfRooms", "Area", "Floor", "Address", "Description"])
with open('out0.csv', 'w') as csvfile:
    csvfile.write(str(df.Cost.mean().round(0)))
