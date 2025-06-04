import pandas as pd

try:
    # Read the CSV file
    df = pd.read_csv('fa_reports.csv')

    # Filter the DataFrame
    filtered_df = df[df['FA_Report id'] == 'US232389.pptx']

    # Print the filtered data
    if not filtered_df.empty:
        print(filtered_df)
    else:
        print("No matching rows found for FA_Report id = US232389.pptx")

except FileNotFoundError:
    print("Error: fa_reports.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")
