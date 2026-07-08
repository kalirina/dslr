import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    """Displays the pairplot of the whole dataframe to
    help us choose the best features to feed the model"""
    df = pd.read_csv('../datasets/dataset_train.csv')
    df = df.drop('Index', axis='columns')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    columns_to_plot = numeric_cols + ['Hogwarts House']
    plot_df = df[columns_to_plot]

    sns.pairplot(plot_df, hue='Hogwarts House', diag_kind='hist')
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)