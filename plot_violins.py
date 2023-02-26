import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

my_colors = ['#3494BA', '#FF9999']
# sns.set_palette(my_colors)

sns.set_theme(style="ticks")

# Load the example tips dataset
df = pd.read_excel('diagnosis\modelcomparision_2.5min_multicnnc2cm.xlsx')

# Draw a nested violinplot and split the violins for easier comparison
fig, ax = plt.subplots(figsize=(20,4))
fig = sns.boxplot(x='Participant', y='Likelihood of NT1 on 2.5 min segment', data=df, hue='Diagnosis', 
                    flierprops={'marker':'+', 'color':'gray', 'alpha': 0.3}, palette=my_colors)
# fig = sns.violinplot(data=df, x="participant", y="segment", hue="model",
#                split=True, inner="quart", linewidth=0.5,edgealpha=0.5,scale='count',cut=0,
#                palette={"Multitask": "b", "One-Phase": ".85"})
sns.despine(left=True)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.show()
plot = fig.get_figure()
plot.savefig('test3_3.png',dpi=220) # save!