import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    """We have to answer the question
    'What are the two features that are similar?'
    using a scatterplot
    """

    df = pd.read_csv('../datasets/dataset_train.csv')
    df = df.drop('Index', axis='columns')
    numeric_df = df.select_dtypes(include=['number'])
    
    correlation = numeric_df.corr().abs()
    #drop the diagonal of 1.0
    upper_triangle_mask = np.triu(np.ones(correlation.shape), k=1).astype(bool)
    upper_triangle = correlation.where(upper_triangle_mask)

    top_features = upper_triangle.unstack().idxmax()
    top_score = upper_triangle.unstack().max()

    print(f"The two most similar features are: {top_features[0]} and {top_features[1]}")
    print(f"Correlation score: {top_score:.4f}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)