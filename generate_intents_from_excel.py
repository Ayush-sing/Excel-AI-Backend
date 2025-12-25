import pandas as pd

excel_file = "sample_data_large.xlsx"
existing_intents_file = "excel_intents.csv"
output_file = "excel_intents_extended.csv"

INTENT_TEMPLATES = {
    "sum_column": ["sum {col}", "total {col}", "add up {col}"],
    "average_column": ["average {col}", "mean {col}", "find average of {col}"],
    "max_column": ["maximum {col}", "highest {col}", "find max of {col}"],
    "min_column": ["minimum {col}", "lowest {col}", "find min of {col}"],
    "count_rows": ["count rows", "how many rows", "number of rows"],
    "filter_rows": ["filter where {col} is high", "show rows where {col} is large"],
    "describe_data": ["describe data", "summary of dataset", "show statistics"],
    "value_counts": ["unique values in {col}", "value counts of {col}"],
    "create_chart": ["chart of {col}", "plot {col}", "bar chart for {col}"]
}

df = pd.read_excel(excel_file)
columns = list(df.columns)

new_rows = []
for intent, templates in INTENT_TEMPLATES.items():
    for col in columns:
        for phrase in templates:
            msg = phrase.format(col=col) if "{col}" in phrase else phrase
            new_rows.append({"message": msg, "intent": intent})

base = pd.read_csv(existing_intents_file)
extended = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates()
extended.to_csv(output_file, index=False)
print(f"✅ Extended training data saved as {output_file} with {len(extended)} rows")
