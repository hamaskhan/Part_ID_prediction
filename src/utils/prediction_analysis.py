import pandas as pd

class ResultProcessor:
    """
    A class to process result data from a CSV file, compute statistics, and save the results.

    Attributes:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        df (pd.DataFrame): DataFrame to hold the loaded data.
        result_df (pd.DataFrame): DataFrame to hold the processed results.
        overall_avg_correct_percentage (float): Overall average correct percentage across all organizations and part_ids.
    """

    def __init__(self, input_file, output_file):
        """
        Initialize the ResultProcessor with file paths.

        Args:
            input_file (str): Path to the input CSV file.
            output_file (str): Path to the output CSV file.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.df = None
        self.result_df = None
        self.overall_avg_correct_percentage = None

    def load_data(self):
        """
        Load the dataset from the specified CSV file.
        """
        self.df = pd.read_csv(self.input_file)

    def process_data(self):
        """
        Process the data to compute statistics and generate the result DataFrame.
        The statistics include total count, correct count, incorrect count, 
        and correct percentage for each organization and part_id.
        """
        result_list = []

        # Group the data by organization and part_id
        grouped = self.df.groupby(['organization', 'part_id'])

        # Iterate over each group
        for (org, part), group in grouped:
            total_count = len(group)
            correct_count = len(group[group['correct/incorrect'] == 'correct'])
            incorrect_count = len(group[group['correct/incorrect'] == 'incorrect'])
            correct_percentage = (correct_count / total_count) * 100 if total_count > 0 else 0

            # Append the results to the result_list
            result_list.append({
                'organization': org,
                'part_id': part,
                'total_count': total_count,
                'correct_count': correct_count,
                'incorrect_count': incorrect_count,
                'correct_percentage': correct_percentage
            })

        # Convert the list to a DataFrame
        self.result_df = pd.DataFrame(result_list)

        # Calculate the average correct percentage for each organization
        avg_correct_percentage = self.result_df.groupby('organization')['correct_percentage'].mean().reset_index()
        avg_correct_percentage.rename(columns={'correct_percentage': 'avg_correct_percentage'}, inplace=True)

        # Merge the average correct percentage back into the result DataFrame
        self.result_df = self.result_df.merge(avg_correct_percentage, on='organization', how='left')

        # Sort the DataFrame by organization and total_count
        self.result_df = self.result_df.sort_values(by=['organization', 'total_count'])

    def save_results(self):
        """
        Save the processed results to the specified output CSV file.
        """
        self.result_df.to_csv(self.output_file, index=False)

    def calculate_overall_average(self):
        """
        Calculate and return the overall average correct percentage across all organizations and part_ids.

        Returns:
            float: The overall average correct percentage.
        """
        self.overall_avg_correct_percentage = self.result_df['correct_percentage'].mean()
        return self.overall_avg_correct_percentage

    def print_overall_average(self):
        """
        Print the overall average correct percentage.
        """
        if self.overall_avg_correct_percentage is None:
            self.calculate_overall_average()
        print(f"Overall average correct percentage for all part_ids across all organizations: {self.overall_avg_correct_percentage:.2f}%")

    def results_analysis(self):
        """
        Perform the entire analysis process: load data, process data, save results, and print the overall average correct percentage.
        """
        self.load_data()
        self.process_data()
        self.save_results()
        self.print_overall_average()

# Usage example:
if __name__ == "__main__":
    processor = ResultProcessor('output/output.csv', 'output/processed_results.csv')
    processor.results_analysis()
