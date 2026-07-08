import sys
import math
import pandas as pd
import matplotlib.pyplot as plt


def diaply_all_hist(gryffindor, ravenclaw, hufflepuff, slytherin):
    """We diplay an histogram for each numeric feature showing
    the grades of students divided in the different houses"""
    features = gryffindor.columns
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


def display_single_hist(gryffindor, ravenclaw, hufflepuff, slytherin):
    """Only displays the hist of the only subject where the four houses
    have homogeneous score distribution """
    plt.figure(figsize=(10,10))

    plt.hist(gryffindor['Care of Magical Creatures'], bins=30, alpha=0.4, label='Gryffindor')
    plt.hist(ravenclaw['Care of Magical Creatures'], bins=30, alpha=0.4, label='Ravenclaw')
    plt.hist(hufflepuff['Care of Magical Creatures'], bins=30, alpha=0.4, label='Hufflepuff')
    plt.hist(slytherin['Care of Magical Creatures'], bins=30, alpha=0.4, label='Slytherin')
    plt.legend()
    plt.title('Care of Magical Creatures')
    
    plt.show()


def main():
    """We have to answer the question
    'Which Hogwarts course has a homogeneous score
    distribution between all four houses?'
    using histograms"""

    df = pd.read_csv('../datasets/dataset_train.csv')
    df = df.drop('Index', axis='columns')
    numeric_df = df.select_dtypes(include=['number'])

    gryffindor = df[df['Hogwarts House'] == 'Gryffindor'][numeric_df.columns]
    ravenclaw = df[df['Hogwarts House'] == 'Ravenclaw'][numeric_df.columns]
    hufflepuff = df[df['Hogwarts House'] == 'Hufflepuff'][numeric_df.columns]
    slytherin = df[df['Hogwarts House'] == 'Slytherin'][numeric_df.columns]

    if len(sys.argv) > 1 and sys.argv[1] == '-all':
        diaply_all_hist(gryffindor, ravenclaw, hufflepuff, slytherin)
    else:
        display_single_hist(gryffindor, ravenclaw, hufflepuff, slytherin)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
