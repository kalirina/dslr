import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def main():
    """We have to answer the question
    'What are the two features that are similar?'
    using a scatterplot
    """

    df = pd.read_csv('datasets/dataset_train.csv')
    df = df.drop('Index', axis='columns')
    numeric_df = df.select_dtypes(include=['number'])

    correlation = numeric_df.corr().abs()
    # Drop the diagonal of 1.0
    # .triu stands for upper triangle
    upper_triangle_mask = np.triu(np.ones(correlation.shape), k=1).astype(bool)
    upper_triangle = correlation.where(upper_triangle_mask)

    top_features = upper_triangle.unstack().idxmax()[0:2]
    top_score = upper_triangle.unstack().max()

    print(f"Most similar features: {top_features[0]} and {top_features[1]}")
    print(f"Correlation score: {top_score:.3f}")

    plt.figure(figsize=(10, 10))
    plt.scatter(df[top_features[0]], df[top_features[1]])

    plt.xlabel(top_features[0])
    plt.ylabel(top_features[1])

    plt.title('Most similar features')
    plt.savefig('graphs/scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    try:
        assert len(sys.argv) == 1, "Invalid number of arguments"
        main()
    except Exception as e:
        print(e)
