import json


def load_json(file_path):
    """Loads JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
file_path = "prompts/CreditAi.json"  # Update this with the actual file path
data = load_json(file_path)

def Mitigations(product=None, underwriting_method=None, profile=None):
    """Filters the JSON data based on provided criteria."""
    filtered_data = []
    
    for entry in data["Mitigation"]:
        if product and entry.get("Product") != product:
            continue
        if underwriting_method and entry.get("Underwriting Method") != underwriting_method:
            continue
        if profile and entry.get("Profile") != profile:
            continue
        
        filtered_data.append(entry)
    
    return filtered_data

# Example usage

# filtered_data = Mitigations(product="HL", underwriting_method="Cash Profit Method" , profile="SEP")

# if filtered_data:
#     print(json.dumps(filtered_data, indent=4))
# else:
#     print("No matching data found.")
