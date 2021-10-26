from matplotlib import pyplot as plt
import seaborn as sns


def facet_grid(data,col,hue,palette,xlabel):
    plt.Figure(figsize=(12, 6), dpi=1000)
    a = sns.FacetGrid(data=data, col=col, hue=hue, palette=palette, height=6, aspect=1.5,
                      margin_titles=True)
    a.map(plt.hist, xlabel)
    plt.title(xlabel+" "+"vs"+" "+ col+" "+"Classified By"+" "+hue)
    plt.legend()
    plt.savefig("output/"+xlabel+" "+"vs"+" "+ col+" "+"Classified By"+" "+hue+".jpeg")
    plt.show()


