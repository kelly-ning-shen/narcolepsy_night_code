import pandas as pd

df = pd.read_excel(io='diagnosis\participants_2.5min.xlsx')

for index, row in df.iterrows():
    # print(row['participant'], row['segnum'])
    participant = row['participant'] # str
    segnum = row['segnum'] # int
    with open('participant_2.5min.txt','a') as f:
        for i in range(segnum):
            f.write(participant)
            f.write('\n')