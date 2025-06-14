import pandas as pd
import json

def get_location_details(location):
    try:
        # Load CSV file (static)
        csv_file = "Location_Category.csv"  # Replace with your actual CSV file path
        df = pd.read_csv(csv_file)
        
        # Ensure required columns exist
        required_columns = {"Location", "revised proposed cat", "Revised Cap in cr"}
        if not required_columns.issubset(df.columns):
            raise ValueError("CSV file is missing one or more required columns.")
        
        # Filter based on location
        result = df[df["Location"].str.lower() == location.lower()]
        
        if result.empty:
            return json.dumps({"message": "No data found for the given location."})
        else:
            # Rename columns and convert to JSON
            result = result.rename(columns={"revised proposed cat": "proposed cat", "Revised Cap in cr": "Cap in cr"})
            return result[["Location", "proposed cat", "Cap in cr"]].to_json(orient="records")
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    location = "Ahmedabad"  # Example location
    print(get_location_details(location))