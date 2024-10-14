import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, ttest_ind, mannwhitneyu, chi2_contingency

class DataInspection:
    def __init__(self):
        self.df = None  # DataFrame will be loaded and stored here
        self.column_types = {}

    def load_csv(self, file_path):
        """Loads the dataset from the given file path."""
        try:
            self.df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully from {file_path}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except Exception as e:
            raise Exception(f"An error occurred while loading the file: {e}")
        self.column_types = self.list_column_types()

    def list_column_types(self):
        """This function will check if the type of data is numeric ordinal, non-numeric ordinal, interval, or nominal."""
        col_types = {}
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].nunique() > 10: 
                    col_types[col] = 'interval'
                else:
                    col_types[col] = 'ordinal'
            else:
                col_types[col] = 'nominal'
        return col_types

    def select_variable(self, data_type, allow_skip=False):
        available_vars = [col for col, dtype in self.column_types.items() if dtype == data_type]
        if allow_skip and not available_vars:
            return None
        
        print(f"Available {data_type} variables: {available_vars}")
        selected_var = input(f"Please select a {data_type} variable: ").strip()
        
        if selected_var in available_vars:
            return selected_var
        else:
            print("Invalid choice.")
            return self.select_variable(data_type, allow_skip)

    def plot_qq_histogram(self, data, title):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sm.qqplot(data, line='s')
        plt.title(f'Q-Q Plot: {title}')

        plt.subplot(1, 2, 2)
        sns.histplot(data, kde=True)
        plt.title(f'Histogram: {title}')
        plt.show()

    def check_normality(self, data, size_limit=2000):
        data = data.dropna()
        if len(data) < size_limit:
            stat, p_value = stats.shapiro(data)
            test_type = "Shapiro-Wilk Test"
        else:
            result = stats.anderson(data, dist='norm')
            stat, p_value = result.statistic, None  # Anderson-Darling doesn't provide p-value
            test_type = "Anderson-Darling Test"
        
        print(f"{test_type} - Statistic: {stat}, p-value: {p_value}")
        return stat, p_value

    def check_skewness(self, data):
        skewness = stats.skew(data.dropna())
        print(f"Skewness: {skewness}")
        return np.abs(skewness) > 1

    def hypothesis_test(self, continuous_var, categorical_var, skewed):
        data = self.df[[continuous_var, categorical_var]].dropna()
        groups = [data[continuous_var][data[categorical_var] == g] for g in data[categorical_var].unique()]

        if skewed:
            stat, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"
        else:
            stat, p_value = stats.f_oneway(*groups)
            test_name = "ANOVA"

        print(f"{test_name} - Statistic: {stat}, p-value: {p_value}")
        return stat, p_value

    def perform_regression(self, x_var, y_var):
        """Perform linear regression between two interval variables."""
        X = self.df[x_var].dropna()
        Y = self.df[y_var].dropna()
        
        # Ensure both variables have the same length
        min_length = min(len(X), len(Y))
        X = X[:min_length]
        Y = Y[:min_length]
        
        slope, intercept, r_value, p_value, std_err = linregress(X, Y)
        
        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R-squared: {r_value**2:.4f}")
        print(f"P-value: {p_value:.15f}")
        print(f"Standard error: {std_err:.4f}")

    def t_test_or_mannwhitney(self, continuous_var, categorical_var):
        """Perform t-test or Mann-Whitney U test based on normality."""
        data = self.df[[continuous_var, categorical_var]].dropna()
        groups = [data[continuous_var][data[categorical_var] == g] for g in data[categorical_var].unique()]
        
        # Check normality
        test_type, _, _ = self.check_normality(data[continuous_var])
        
        if test_type == 'Shapiro-Wilk':
            stat, p_value = ttest_ind(*groups)
            test_name = "t-test"
        else:
            stat, p_value = mannwhitneyu(*groups)
            test_name = "Mann-Whitney U Test"
        
        print(f"{test_name} - Statistic: {stat:.4f}, p-value: {p_value:.15f}")
        return stat, p_value

    def chi_square_test(self, categorical_var_1, categorical_var_2):
        """Perform Chi-square test between two categorical variables."""
        contingency_table = pd.crosstab(self.df[categorical_var_1], self.df[categorical_var_2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        print(f"chi2 = {chi2:.4f}, p-value = {p:.15f}, dof = {dof}")
        return chi2, p, dof, expected

def main():
    analysis = DataInspection()
    file_path = input("Enter the path to the dataset (CSV file): ")
    analysis.load_csv(file_path)

    continuous_var = analysis.select_variable("interval")
    data = analysis.df[continuous_var]

    stat, p_value = analysis.check_normality(data)
    analysis.plot_qq_histogram(data, continuous_var)

    categorical_var = analysis.select_variable("nominal")
    skewed = analysis.check_skewness(data)

    null_hyp = input("Enter the null hypothesis: ")
    analysis.hypothesis_test(continuous_var, categorical_var, skewed)

    while True:
        print("\nSelect a test to perform:")
        print("1. t-test or Mann-Whitney U Test")
        print("2. Chi-square Test")
        print("3. Linear Regression")
        print("4. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == '1':
            continuous_var = analysis.select_variable("interval")
            categorical_var = analysis.select_variable("nominal", max_categories=2)
            if continuous_var and categorical_var:
                analysis.t_test_or_mannwhitney(continuous_var, categorical_var)
        elif choice == '2':
            categorical_var_1 = analysis.select_variable("nominal")
            categorical_var_2 = analysis.select_variable("nominal")
            if categorical_var_1 and categorical_var_2:
                analysis.chi_square_test(categorical_var_1, categorical_var_2)
        elif choice == '3':
            x_var = analysis.select_variable("interval")
            y_var = analysis.select_variable("interval")
            if x_var and y_var:
                analysis.perform_regression(x_var, y_var)
        elif choice == '4':
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
