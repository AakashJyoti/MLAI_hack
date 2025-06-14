import json
from difflib import get_close_matches
# Dictionary mapping each LOB to a dictionary of products and currency factors
LOB_PRODUCTS_CURRENCY = {
    "ABSLI": {
        "ABSLI (Individual Insurance): First Year Premium": 50,
        "ABSLI (Individual Insurance): First Year Commission": 8.5,
        "Group Insurance - Fund: AUM Traditional": 800,
        "Group Insurance - Fund: AUM ULIP": 1600,
        "Group Insurance - Term: First Year Premium": 60
    },
    "ABML": {
        "Equity, Derivatives, Commodities: Gross Brokerage": 70,
        "PMS: AUM": 800
    },
    "ABHI": {
        "Individual Insurance: First Year Premium": 21,
        "Group Insurance: First Year Premium": 700
    },
    "ABFL": {
        "Retail Lending: Disbursement": 600,
        "Mortgage - SME - SEG: Disbursement (Per Case Capped at 5 CR)": 600,
        "Mid Corporate: Disbursement": 600,
        "CMG (Loan against Shares & Securities)": 600,
        "Wealth - Digital Gold: 0 - 50 Lacs": 5000,
        "Wealth - Digital Gold: 50 Lacs - 1 Cr": 2000,
        "Wealth - Digital Gold: > 1 Cr": 1000
    }
}

def get_available_products(lob):
    """
    Returns a list of products available for a given LOB.
    """
    return list(LOB_PRODUCTS_CURRENCY.get(lob, {}).keys())

def find_best_matching_product(lob, product):
    """
    Returns a list of products available for a given LOB.
    """
    if lob not in LOB_PRODUCTS_CURRENCY:
        return None, "Invalid LOB"
    
    products = list(LOB_PRODUCTS_CURRENCY[lob].keys())
    if product in products:
        return product, None
    
    matches = get_close_matches(product, products, n=1, cutoff=0.6)
    return (matches[0], None) if matches else (None, "Invalid Product")


def calculate_primary_qualification(primary_ytd_abhfl_business):
    """
    Calculate primary qualification points and gate shortfall.
    """
    primary_ytd_abhfl_business *= 10**7  # Convert to absolute value
    primary_gate_criteria = 12 * 10**7
    primary_gate_shortfall = max(0, primary_gate_criteria - primary_ytd_abhfl_business)
    primary_points = primary_ytd_abhfl_business / 500
    return primary_points, primary_gate_shortfall

def calculate_secondary_business_points(lob, product, secondary_business_in_cr):
    """
    Calculate secondary business points based on LOB, product, and currency factor.
    """
    if not lob or not product or lob not in LOB_PRODUCTS_CURRENCY or product not in LOB_PRODUCTS_CURRENCY[lob]:
        return {"Error": "Invalid or missing LOB/Product"}

    currency_factor = LOB_PRODUCTS_CURRENCY[lob][product]
    business_in_absolute = secondary_business_in_cr * 10**7
    secondary_points = business_in_absolute / currency_factor if currency_factor else 0

    return {
        "LOB": lob,
        "Product": product,
        "SecondaryBusinessCr": secondary_business_in_cr,
        "BusinessInAbsolute": business_in_absolute,
        "SecondaryPoints": secondary_points,
        "CurrencyFactor": currency_factor
    }

# def calculate_ils_achievement(secondary_points, primary_points):
#     """
#     Determine ILS achievement levels based on primary and secondary points.
#     """
#     total_points = secondary_points + primary_points
#     ils_levels = {1: (160000, 30000), 2: (350000, 70000), 3: (550000, 110000), 4: (750000, 150000)}

    
#     return {
#     level: "Achieved" if total_points >= t and secondary_points >= s 
#     else (
#         f"Total: {max(0, t - total_points):.0f}, Secondary: {max(0, s - secondary_points):.0f}"
#         + (" ,Pending" if max(0, t - total_points) > 0 or max(0, s - secondary_points) > 0 else "")
#     )
#     for level, (t, s) in ils_levels.items()
# }
def calculate_ils_achievement(secondary_points, primary_points, currency_factor=None):
    """
    Determine ILS achievement levels based on primary and secondary points.
    If not achieved, also calculates the shortfall business (in crores) as:
    
    Shortfall Business = (Remaining Points) * (currency_factor) / 10**7
    
    Returns a dictionary where each level has either:
      - "Achieved", or
      - A dictionary with status "Pending" plus shortfall details.
    """
    total_points = secondary_points + primary_points
    # Each level: (total points required, secondary points required)
    ils_levels = {1: (160000, 30000), 2: (350000, 70000), 3: (550000, 110000), 4: (750000, 150000)}
    result = {}
    
    for level, (req_total, req_secondary) in ils_levels.items():
        if secondary_points >= req_secondary and total_points >= req_total:
            result[level] = "Achieved"
        else:
            # Calculate remaining points (if negative, becomes 0)
            remaining_secondary = max(0, req_secondary - secondary_points)
            remaining_total = max(0, req_total - total_points)
            # Compute shortfall business if currency factor is available
            if currency_factor is not None:
                secondary_business_shortfall = (remaining_secondary * currency_factor) / (10**7)
                print(secondary_business_shortfall)
                total_business_shortfall = (remaining_total * currency_factor) / (10**7)
            else:
                secondary_business_shortfall = None
                total_business_shortfall = None
            
            shortfall_details = {
                "status": "Pending",
                "total_points_shortfall": f"{remaining_total:.0f} points",
                "total_business_shortfall": f"{total_business_shortfall:.2f} Cr" if total_business_shortfall is not None else "N/A",
            }
            
            # Only include secondary shortfall if it's not already met
            if secondary_points < req_secondary:
                shortfall_details["secondary_points_shortfall"] = f"{remaining_secondary:.0f} points"
                shortfall_details["secondary_business_shortfall"] = f"{secondary_business_shortfall:.2f} Cr" if secondary_business_shortfall is not None else "N/A"
            result[level] = shortfall_details
    return result
# def calculate_ils_qualification_status(primary_gate_shortfall, secondary_points, primary_points):
#     """
#     Determine ILS qualification status.
#     """
#     total_points = secondary_points + primary_points
#     ils_levels = {1: (160000, 30000), 2: (350000, 70000), 3: (550000, 110000), 4: (750000, 150000)}

#     return {
#         level: "Qualified" if primary_gate_shortfall == 0 and total_points >= t and secondary_points >= s 
#         else "Pending"
#         for level, (t, s) in ils_levels.items()
#     }
def calculate_ils_qualification_status(primary_gate_shortfall, secondary_points, primary_points , currency_factor=None):
    """
    Determine ILS qualification status.
    """
    total_points = secondary_points + primary_points
    # Each level: (total points required, secondary points required)
    ils_levels = {1: (160000, 30000), 2: (350000, 70000), 3: (550000, 110000), 4: (750000, 150000)}
    result = {}
    
    for level, (req_total, req_secondary) in ils_levels.items():
        if primary_gate_shortfall == 0 and total_points >= req_total and secondary_points >= req_secondary:
            result[level] = "Achieved"
        else:
            
            remaining_total = max(0, req_total - total_points)
            # Compute shortfall business if currency factor is available
            if currency_factor is not None:
                # secondary_business_shortfall = (remaining_secondary * currency_factor) / (10**7)
                # print(secondary_business_shortfall)
                total_business_shortfall = (remaining_total * currency_factor) / (10**7)
            else:
                # secondary_business_shortfall = None
                total_business_shortfall = None
            
            result[level] = {
                "status": "Pending",
                "primary_gate_shortfall": f"{primary_gate_shortfall:.0f} points",
                "total_points_shortfall": f"{remaining_total:.0f} points",
                "total_business_shortfall": f"{total_business_shortfall:.2f} Cr" if total_business_shortfall is not None else "N/A",
            }
    return result

def select_calculator(secondary_lob=None, product=None, secondary_business_cr=None, primary_ytd_abhfl_business=None):
    """
    Main function to calculate business points, ILS qualification, and achievements.
    """
    lob = secondary_lob
    if lob and not product:
        output = {f"AvailableProducts in {lob} LOB": get_available_products(lob)}
        
        return output
    
    output = {}
    # Ensure secondary_points is defined (default to 0)
    secondary_points = 0

    # Primary Calculation
    if primary_ytd_abhfl_business is not None:
        primary_points, primary_gate_shortfall = calculate_primary_qualification(primary_ytd_abhfl_business)
        output["Primary Calculation"] = {
            "PrimaryPoints": primary_points,
            "PrimaryGateShortfall": primary_gate_shortfall
        }
    else:
        primary_points, primary_gate_shortfall = 0, float('inf')  # Default values if missing

    # Secondary Calculation
    if lob and product and secondary_business_cr is not None:
        product ,error= find_best_matching_product(lob, product)
        if error:
            output["Error"] = error
        print(product)
        secondary_business_result = calculate_secondary_business_points(lob, product, secondary_business_cr)
        if "Error" in secondary_business_result:
            output["Error"] = secondary_business_result["Error"]
        else:
            secondary_points = secondary_business_result["SecondaryPoints"]
            currency_factor = secondary_business_result["CurrencyFactor"]
            output["Secondary Calculation"] = secondary_business_result

    else:
        secondary_points = 0  # Default value if missing

    # Calculate total points if any valid inputs exist
    # if primary_points or secondary_points:
    total_points = primary_points + secondary_points
    output["Total Points"] = total_points
    output["Secondary Qualification for ILS"] = calculate_ils_achievement(secondary_points, primary_points , currency_factor)
    output["ILS Qualification Status"] = calculate_ils_qualification_status(primary_gate_shortfall, secondary_points, primary_points , currency_factor)

    return output

if __name__ == "__main__":
# #     # Example usage with partial inputs
#     # print(select_calculator(lob="ABSLI", product="ABSLI (Individual Insurance): First Year Premium" , secondary_business_cr=0))
    print(select_calculator(secondary_lob="ABSLI", product="ABSLI (Individual Insurance): First Year Premium", secondary_business_cr=0.5, primary_ytd_abhfl_business=12))
