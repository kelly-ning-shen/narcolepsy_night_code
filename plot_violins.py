import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid")

# Load the example tips dataset
df = pd.read_excel('diagnosis\modelcomparision_2.5min.xlsx')

# Draw a nested violinplot and split the violins for easier comparison
fig, ax = plt.subplots()
fig = sns.violinplot(data=df, x="participant", y="segment", hue="model",
               split=True, inner="quart", linewidth=0.5,edgealpha=0.5,scale='count',cut=0,
               palette={"Multitask": "b", "One-Phase": ".85"})
sns.despine(left=True)
plt.xticks(rotation = 90)

plt.show()
plot = fig.get_figure()
plot.savefig('test2_1.png',dpi=400) # save!