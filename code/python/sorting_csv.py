import pandas as pd

# Function to sort CSV data and move 'class' column to the end
def sort_csv(input_csv_path, output_csv_path, sort_column):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Sort the DataFrame by the specified column
    sorted_df = df.sort_values(by=sort_column, ascending=True)

    # Move the 'class' column to the end
    if 'class' in sorted_df.columns:
        class_column = sorted_df.pop('class')
        sorted_df['class'] = class_column

    # Write the sorted DataFrame back to a new CSV file
    sorted_df.to_csv(output_csv_path, index=False)
    print(f"Data sorted by {sort_column} and saved to {output_csv_path}")

# Example usage
input_csv_path = '/content/drive/MyDrive/Dataset2/frames_00_processed/gait_data.csv'  # Path to your input CSV file
output_csv_path = '/content/drive/MyDrive/Dataset2/frames_00_processed/gait_data_sorted.csv'  # Path to save the sorted CSV file
sort_column = 'class'  # Replace with the column name you want to sort by

sort_csv(input_csv_path, output_csv_path, sort_column)
