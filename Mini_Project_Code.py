import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, chi2_contingency, shapiro
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

class DataAnalysis:
    def __init__(self):
        # Load dataset from user input
        file_path = input("Please enter the path to your dataset (CSV format): ")
        self.df = pd.read_csv(file_path)

        # Automatically identify variable types
        self.variable_types = self.classify_variable_types()

        # Display statistics in vertical format
        self.display_statistics()

        # Start interactive session
        self.interactive_session()

    def classify_variable_types(self):
        """Classify variables into Ratio, Nominal, or Ordinal."""
        variable_types = {}
        for col in self.df.columns:
            unique_vals = self.df[col].nunique()
            if self.df[col].dtype in [np.float64, np.int64]:
                if unique_vals > 2:  # Assuming numerical variables with many unique values are Ratio
                    variable_types[col] = "Ratio"
                else:
                    variable_types[col] = "Ordinal"
            else:
                variable_types[col] = "Nominal"  # Categorical variables
        return variable_types

    def display_statistics(self):
        """Display basic statistics for each variable in vertical format."""
        print("\nSummary of Variables:")
        print(f"{'Variable':<20}{'Type':<10}{'Mean / Median / Mode':<30}{'Kurtosis':<10}{'Skewness':<10}")
        print("=" * 80)
    
        for col, var_type in self.variable_types.items():
            if var_type == "Ratio":
               mean = self.df[col].mean()
               median = self.df[col].median()
               mode = self.df[col].mode()[0] if not self.df[col].mode().empty else 'NA'
               kurtosis = self.df[col].kurtosis()
               skewness = self.df[col].skew()
               print(f"{col:<20}{var_type:<10}{f'{mean:.2f} / {median:.2f} / {mode}':<30}{kurtosis:<10.2f}{skewness:<10.2f}")
            else:
               mode = self.df[col].mode()[0] if not self.df[col].mode().empty else 'NA'
               print(f"{col:<20}{var_type:<10}{mode:<30}{'NA':<10}{'NA':<10}")

    def interactive_session(self):
        """Start an interactive session with the user."""
        while True:
            print("\nHow do you want to analyze your data?")
            print("1. Plot variable distribution")
            print("2. Conduct ANOVA (with QQ Plot)")
            print("3. Conduct t-Test")
            print("4. Conduct Chi-Square Test")
            print("5. Conduct Regression")
            print("6. Conduct Sentiment Analysis")
            print("7. Quit")

            choice = input("Enter your choice (1-7): ")
            
            if choice == '1':
                self.plot_variable_distribution()
            elif choice == '2':
                self.conduct_anova()
            elif choice == '3':
                self.conduct_t_test()
            elif choice == '4':
                self.conduct_chi_square()
            elif choice == '5':
                self.conduct_regression()
            elif choice == '6':
                self.conduct_sentiment_analysis()
            elif choice == '7':
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 7.")

    def plot_variable_distribution(self):
        """Allow the user to choose variables to plot their distribution."""
        while True:
            print("\nFollowing variables are available for plot distribution:")
            for idx, col in enumerate(self.df.columns, 1):
                print(f"{idx}. {col}")
            print("BACK")
            print("QUIT")
            
            choice = input("Enter your choice: ")
            
            if choice.lower() == "back":
                break
            elif choice.lower() == "quit":
                print("Exiting...")
                exit()
            else:
                try:
                    idx = int(choice) - 1
                    selected_var = self.df.columns[idx]
                    self.show_distribution_plot(selected_var)
                except (ValueError, IndexError):
                    print("Invalid choice, please try again.")

    def show_distribution_plot(self, var):
        """Display the appropriate distribution plot for the selected variable."""
        var_type = self.variable_types[var]
        plt.figure(figsize=(8, 6))
        if var_type == "Ratio":
            plt.hist(self.df[var].dropna(), bins=10, color='skyblue', edgecolor='black')
            plt.title(f"Distribution plot for {var} (Histogram)")
        else:
            sns.boxplot(data=self.df, y=var)
            plt.title(f"Distribution plot for {var} (Boxplot)")
        plt.show()

    def conduct_anova(self):
        """Perform ANOVA analysis and display QQ plot."""
        print("Available categorical (Nominal/Ordinal) variables:")
        cat_vars = [var for var, vtype in self.variable_types.items() if vtype in ['Nominal', 'Ordinal']]
        print(cat_vars)
        cat_var = input("Select a categorical variable for ANOVA: ")

        print("Available numerical (Ratio) variables:")
        num_vars = [var for var, vtype in self.variable_types.items() if vtype == 'Ratio']
        print(num_vars)
        num_var = input("Select a numerical variable for ANOVA: ")

        if cat_var in self.df.columns and num_var in self.df.columns:
            groups = [self.df[self.df[cat_var] == level][num_var].dropna() for level in self.df[cat_var].unique()]
            
            # Normality check for each group
            normality_results = [shapiro(group) for group in groups]
            for i, (stat, p) in enumerate(normality_results):
                print(f"Group '{self.df[cat_var].unique()[i]}': W = {stat:.4f}, p = {p:.4f} (Normality Test)")
            
            if all(p > 0.05 for _, p in normality_results):  # p > 0.05 means normal
                print("All groups are normally distributed.")
            else:
                print("At least one group is not normally distributed.")
            
            f_stat, p_value = f_oneway(*groups)
            print(f"ANOVA results: F = {f_stat:.4f}, p = {p_value:.4f}")
            
            # Generate QQ plot for normality check
            self.plot_qq(num_var)
        else:
            print("Invalid selection of variables.")

    def conduct_anova(self):
        """Perform ANOVA analysis, check for normality using Shapiro-Wilk, and display QQ plot."""
        # Prompt user to select categorical and numerical variables for ANOVA
        print("Available categorical (Nominal/Ordinal) variables:")
        cat_vars = [var for var, vtype in self.variable_types.items() if vtype in ['Nominal', 'Ordinal']]
        print(cat_vars)
        cat_var = input("Select a categorical variable for ANOVA: ")

        print("Available numerical (Ratio) variables:")
        num_vars = [var for var, vtype in self.variable_types.items() if vtype == 'Ratio']
        print(num_vars)
        num_var = input("Select a numerical variable for ANOVA: ")

        if cat_var in self.df.columns and num_var in self.df.columns:
            # Separate the data into groups based on the categorical variable
            groups = [self.df[self.df[cat_var] == level][num_var].dropna() for level in self.df[cat_var].unique()]

            # Normality check for each group using Shapiro-Wilk test
            print("\nNormality Test (Shapiro-Wilk) Results:")
            normality_results = [stats.shapiro(group) for group in groups]
            for i, (stat, p) in enumerate(normality_results):
                print(f"Group '{self.df[cat_var].unique()[i]}': W = {stat:.4f}, p = {p:.4f}")
            
            # Determine if all groups are normally distributed
            if all(p > 0.05 for _, p in normality_results):  # p > 0.05 means normally distributed
                print("All groups are normally distributed.")
            else:
                print("At least one group is not normally distributed.")

            # Perform ANOVA test
            f_stat, p_value = f_oneway(*groups)
            print(f"\nANOVA results: F = {f_stat:.4f}, p = {p_value:.4f}")

            # Display QQ plot for residuals
            self.plot_qq(num_var)
        else:
            print("Invalid selection of variables.")

    def plot_qq(self, num_var):
        """Generate QQ plot and histogram for normality check."""
        plt.figure(figsize=(12, 6))

        # QQ plot on the left
        plt.subplot(1, 2, 1)
        stats.probplot(self.df[num_var].dropna(), dist="norm", plot=plt)
        plt.title(f"QQ Plot for {num_var}")

        # Histogram with KDE on the right
        plt.subplot(1, 2, 2)
        sns.histplot(self.df[num_var].dropna(), kde=True)
        plt.title(f"Histogram for {num_var}")

        # Show the plots
        plt.show()

    def conduct_t_test(self):
        """Perform t-Test analysis."""
        print("Available categorical variables (with 2 unique values):")
        cat_vars = [var for var, vtype in self.variable_types.items() if vtype in ['Nominal', 'Ordinal'] and self.df[var].nunique() == 2]
        print(cat_vars)
        cat_var = input("Select a categorical variable for t-Test: ")

        print("Available numerical (Ratio) variables:")
        num_vars = [var for var, vtype in self.variable_types.items() if vtype == 'Ratio']
        print(num_vars)
        num_var = input("Select a numerical variable for t-Test: ")

        if cat_var in self.df.columns and num_var in self.df.columns:
            group1 = self.df[self.df[cat_var] == self.df[cat_var].unique()[0]][num_var].dropna()
            group2 = self.df[self.df[cat_var] == self.df[cat_var].unique()[1]][num_var].dropna()

            # Normality checks
            stat1, p1 = shapiro(group1)
            stat2, p2 = shapiro(group2)
            print(f"Group 1: W = {stat1:.4f}, p = {p1:.4f} (Normality Test)")
            print(f"Group 2: W = {stat2:.4f}, p = {p2:.4f} (Normality Test)")

            if p1 > 0.05 and p2 > 0.05:  # Both groups are normally distributed
                print("Both groups are normally distributed.")
                # Perform t-Test
                t_stat, p_value = ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
                print(f"t-Test results: t = {t_stat:.4f}, p = {p_value:.4f}")
                
                # Boxplot to visualize relationship
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=self.df[cat_var], y=self.df[num_var])
                plt.title(f"Boxplot of {num_var} by {cat_var}")
                plt.show()
            else:
                print("At least one group is not normally distributed. Consider using a non-parametric test.")
        else:
            print("Invalid selection of variables.")

    def conduct_chi_square(self):
        """Perform Chi-Square test."""
        print("Available categorical variables:")
        cat_vars = [var for var, vtype in self.variable_types.items() if vtype == 'Nominal']
        print(cat_vars)

        # Select 2 cat_var
        cat_var1 = input("Select the first categorical variable: ")
        cat_var2 = input("Select the second categorical variable: ")

        if cat_var1 in self.df.columns and cat_var2 in self.df.columns:
           contingency_table = pd.crosstab(self.df[cat_var1], self.df[cat_var2])
           chi2, p, dof, expected = chi2_contingency(contingency_table)
           print(f"Chi-Square results: chi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}")

           # Show Bar Chart
           contingency_table.plot(kind='bar', stacked=True)
           plt.title(f"Bar Chart of {cat_var1} vs {cat_var2}")
           plt.xlabel(cat_var1)
           plt.ylabel("Count")
           plt.show()
        else:
           print("Invalid selection of variables.")

    def conduct_regression(self):
        """Conduct a regression analysis."""
        print("Available dependent (Ratio) variables:")
        dep_vars = [var for var, vtype in self.variable_types.items() if vtype == 'Ratio']
        print(dep_vars)
        dep_var = input("Select a dependent variable for regression: ")

        print("Available independent (Ratio) variables:")
        indep_vars = [var for var, vtype in self.variable_types.items() if vtype == 'Ratio' and var != dep_var]
        print(indep_vars)
        indep_var = input("Select an independent variable for regression: ")

        if dep_var in self.df.columns and indep_var in self.df.columns:
            X = self.df[[indep_var]]
            y = self.df[dep_var]
            model = sm.OLS(y, sm.add_constant(X)).fit()
            print(model.summary())

            # Scatter plot to visualize relationship
            plt.figure(figsize=(8, 6))
            plt.scatter(self.df[indep_var], self.df[dep_var], alpha=0.6)
            plt.title(f"Scatter Plot of {dep_var} vs {indep_var}")
            plt.xlabel(indep_var)
            plt.ylabel(dep_var)
            plt.plot(X, model.predict(sm.add_constant(X)), color='red', linewidth=2)  # regression line
            plt.show()
        else:
            print("Invalid selection of variables.")

    def conduct_sentiment_analysis(self):
        """Placeholder for sentiment analysis method."""
        print("Sentiment analysis is not implemented yet.")

if __name__ == "__main__":
    DataAnalysis()
