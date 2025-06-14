import json

import os
json_file = os.path.join("prompts", "product_faq.json")
def get_qna_by_location_from_file( location_input):
    with open(json_file, 'r',encoding="utf-8") as file:
        data = json.load(file)
    
    location_input_lower = location_input.strip().lower()
    results = []

    for entry in data:
        # Split by common delimiters: &, /, + 
        locations = [loc.strip().lower() for delim in ["&", "/", "+"] 
                     for loc in entry["Location"].split(delim)]
        
        # Check if input matches or partially exists in any location
        if any(location_input_lower in loc for loc in locations):
            results.append({
                "Location": entry["Location"],
                "Question": entry["Question"],
                "Answer": entry["Answer"]
            })
    
    return results

