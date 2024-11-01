from common import *
from plotting import *
import seaborn as sns
import numpy as np


def calculate_class_values_dataframe(df, column_name):
    score_classes = {
        'A': range(16, 21),
        'B': range(14, 16),
        'C': range(12, 14),
        'D': range(10, 12),
        'F': range(0, 10),
    }

    # Assuming 'Y' is your original array with grades
    df_mapped = pd.DataFrame()
    df_mapped[column_name] = [category for grade in df[column_name] for category, scores in score_classes.items() if grade in scores]

    return df_mapped

def calculate_binary_values_dataframe(df, column_name):
    df_binary = pd.DataFrame()
    df_binary[column_name] = (df[column_name] >= 10).astype(int)
    return df_binary



def show_mean_and_std(df):

    df['G3'] = calculate_binary_values_dataframe(df, 'G3')

    continuous_predictor_names = ['age', 'G1', 'G2']

    kontinualni_prediktori = df[continuous_predictor_names]

    # Izračunavanje srednjih vrednosti i standardnih devijacija za pozitivne i negativne primere
    mean_positives = kontinualni_prediktori[df['G3'] == 1].mean()
    mean_negatives = kontinualni_prediktori[df['G3'] == 0].mean()

    std_positives = kontinualni_prediktori[df['G3'] == 1].std()
    std_negatives = kontinualni_prediktori[df['G3'] == 0].std()

    # Vizualizacija
    bar_width = 0.35
    index = np.arange(len(kontinualni_prediktori.columns))

    plt.bar(index, mean_positives, bar_width, label='Pozitivni', yerr=std_positives)
    plt.bar(index + bar_width, mean_negatives, bar_width, label='Negativni', yerr=std_negatives)

    plt.xlabel('Kontinualni Prediktori')
    plt.ylabel('Srednje vrednosti')
    plt.title('Srednje vrednosti i standardne devijacije za pozitivne i negativne primere (kontinualni prediktori)')
    plt.xticks(index + bar_width / 2, kontinualni_prediktori.columns)
    plt.legend()

    plt.show()


def count_plot_binary(df):

    df['G3'] = calculate_binary_values_dataframe(df, 'G3')
    df['G1'] = calculate_binary_values_dataframe(df, 'G1')
    df['G2'] = calculate_binary_values_dataframe(df, 'G2')
    # Define the bin edges for the categories
    bins = [0, 5, 15, float('inf')]  # The last bin includes everything greater than 15

    # Define the labels for each category
    labels = ['0-5', '5-15', '>15']

    # Create a new column 'absences_category' based on the bins and labels
    df['absences'] = pd.cut(df['absences'], bins=bins, labels=labels, right=False)


    # Postavljanje veličine figure na osnovu broja prediktora
    num_predictors = len(df.columns) - 1

    # Broj figura koje želite po redu
    num_figures = (num_predictors - 1) // 4 + 1

    # Iteracija kroz svaku grupu od 4 prediktora
    for fig_num in range(num_figures):
        start_index = fig_num * 4
        end_index = min((fig_num + 1) * 4, num_predictors)

        fig, axes = plt.subplots(1, end_index - start_index, figsize=(5 * (end_index - start_index), 5))

        # Petlja kroz prediktore i pravljenje count plot-a
        for i, predictor in enumerate(df.columns[:-1][start_index:end_index]):
            sns.countplot(x=predictor, hue='G3', data=df, ax=axes[i])
            axes[i].set_title(f'Count Plot za {predictor}')
            axes[i].tick_params(axis='x')  # Rotate x-axis labels vertically

        plt.show()


def correlation(df):

    continuous_predictor_names = ['absences', 'G1', 'G2', 'G3']

    new_df = df[continuous_predictor_names]
    correlation_matrix = new_df.corr()

    # Create a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

    # Display the heatmap
    plt.title(f'Correlation Heatmap with G3')
    plt.show()




if __name__ == '__main__':
    df = load_data_frame()
    # show_mean_and_std(df)
    # count_plot_binary(df)
    correlation(df)









