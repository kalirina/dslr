import matplotlib.pyplot as plt
import pandas as pd
import json
import sys


def clean_model_json(useless_features):
    """
    Opens model.json, deletes the useless features,
    and saves the cleaned version back to the file.
    """
    try:
        with open('model.json', 'r') as file:
            model_data = json.load(file)

        model_data['features'] = [
            f for f in model_data['features'] if f not in useless_features
        ]

        for feature in useless_features:
            model_data['mean'].pop(feature, None)
            model_data['std'].pop(feature, None)

        with open('model.json', 'w') as file:
            json.dump(model_data, file, indent=4)

    except FileNotFoundError:
        print("Warning: model.json not found.")


def main():
    """Displays the pairplot of the whole dataframe to
    help us choose the best features to feed the model"""
    df = pd.read_csv('datasets/dataset_train.csv')
    df = df.drop('Index', axis='columns')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    houses = df['Hogwarts House'].dropna().unique()

    house_colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'gold'
    }

    n_features = len(numeric_cols)
    fig, axes = plt.subplots(nrows=n_features,
                             ncols=n_features, figsize=(20, 20))

    for i, row_col in enumerate(numeric_cols):
        for j, col_col in enumerate(numeric_cols):
            ax = axes[i, j]

            # Diagonal
            if i == j:
                for house in houses:
                    s = 'Hogwarts House'
                    grades = df[df[s] == house][row_col].dropna()
                    ax.hist(grades, bins=20, alpha=0.5,
                            color=house_colors[house], label=house)

            # The rest
            else:
                for house in houses:
                    grades = df[df['Hogwarts House'] == house]
                    ax.scatter(grades[col_col], grades[row_col],
                               alpha=0.5, s=15, color=house_colors[house])

            if i < n_features - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel(col_col, rotation=20)

            if j > 0:
                ax.set_yticks([])
            else:
                ax.set_ylabel(row_col, rotation=20, ha='right')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='Hogwarts House')

    # plt.savefig('graphs/pair_plot.png', dpi=300, bbox_inches='tight')

    useless_features = [
        'Care of Magical Creatures',     # Overlapping hist
        'Arithmancy',                    # Overlapping hist
        'Defense Against the Dark Arts'  # Redundant (with Astronomy)
    ]
    clean_model_json(useless_features)

    plt.show()


if __name__ == '__main__':
    try:
        assert len(sys.argv) == 1, "Invalid number of arguments."
        main()
    except Exception as e:
        print(e)
