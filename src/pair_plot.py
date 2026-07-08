import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    """ """
    df = pd.read_csv('../datasets/dataset_train.csv')
    df = df.drop('Index', axis='columns')
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # 2. Add 'Hogwarts House' to our list of columns to keep
    columns_to_plot = numeric_cols + ['Hogwarts House']
    
    # 3. Create a new dataframe using only those specific columns
    plot_df = df[columns_to_plot]

    sns.pairplot(plot_df, hue='Hogwarts House', diag_kind='hist')
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)