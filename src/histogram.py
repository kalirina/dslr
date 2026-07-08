import pandas as pd
import math
import matplotlib.pyplot as plt


def main():
    """
    To answer the question:
    'Which Hogwarts course has a homogeneous score
    distribution between all four houses?'
    We diplay an histogram for each numeric feature showing
    the grades of students divided in the different houses
    """
    df = pd.read_csv('../datasets/dataset_train.csv')
    df = df.drop('Index', axis='columns')
    numeric_df = df.select_dtypes(include=['number'])
    features = numeric_df.columns

    gryffindor = df[df['Hogwarts House'] == 'Gryffindor'][features]
    ravenclaw = df[df['Hogwarts House'] == 'Ravenclaw'][features]
    hufflepuff = df[df['Hogwarts House'] == 'Hufflepuff'][features]
    slytherin = df[df['Hogwarts House'] == 'Slytherin'][features]

    num_features = len(features)
    cols = math.ceil(math.sqrt(num_features))
    rows = math.ceil(num_features / cols)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    ax = ax.flatten()
    n = 0
    for feature in features:
        ax[n].hist(gryffindor[feature], bins=30, alpha=0.4, label='Gryffindor')
        ax[n].hist(ravenclaw[feature], bins=30, alpha=0.4, label='Ravenclaw')
        ax[n].hist(hufflepuff[feature], bins=30, alpha=0.4, label='Hufflepuff')
        ax[n].hist(slytherin[feature], bins=30, alpha=0.4, label='Slytherin')

        ax[n].set_title(feature)
        ax[n].legend()
        n += 1

    for i in range(n, len(ax)):
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
