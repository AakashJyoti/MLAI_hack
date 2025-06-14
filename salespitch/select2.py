import json

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
        "Equity, Derivatives, Commodities: Gorss Brokerage": 70,
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

def calculate_primary_qualification(ytfl_abhfl):
    ytfl_abhfl *= 10 ** 7
    primary_gate_criteria = 12 * 10 ** 7
    primary_gate_shortfall = max(0, primary_gate_criteria - ytfl_abhfl)
    primary_points = ytfl_abhfl / 500
    return primary_points, primary_gate_shortfall

def calculate_secondary_business_points(lob, product, secondary_business_in_cr):
    currency_factor = LOB_PRODUCTS_CURRENCY.get(lob, {}).get(product, 1)
    business_in_absolute = secondary_business_in_cr * 10 ** 7
    secondary_points = business_in_absolute / currency_factor if currency_factor else 0
    return {
        "LOB": lob,
        "Product": product,
        "SecondaryBusinessCr": secondary_business_in_cr,
        "BusinessInAbsolute": business_in_absolute,
        "SecondaryPoints": secondary_points,
        "CurrencyFactor": currency_factor
    }

def calculate_ils_achievement(secondary_points, primary_points):
    total_points = secondary_points + primary_points
    ils_levels = {1: (160000, 30000), 2: (350000, 70000), 3: (550000, 110000), 4: (750000, 150000)}
    return {level: "Achieved" if total_points >= t and secondary_points >= s else f"Total: {t - total_points:.0f}, Secondary: {s - secondary_points:.0f}" 
            for level, (t, s) in ils_levels.items()}

def calculate_ils_qualification_status(primary_gate_shortfall, secondary_points, primary_points, lob, product):
    total_points = secondary_points + primary_points
    ils_levels = {1: (160000, 30000), 2: (350000, 70000), 3: (550000, 110000), 4: (750000, 150000)}
    return {level: "Qualified" if primary_gate_shortfall == 0 and total_points >= t and secondary_points >= s else "Pending"
            for level, (t, s) in ils_levels.items()}

def main():
    lob_input = "ABSLI"
    product_input = "CMG (Loan against Shares & Securities)"
    secondary_business_cr_input = 2
    ytfl_abhfl_input = 1

    primary_points, primary_gate_shortfall = calculate_primary_qualification(ytfl_abhfl_input)
    secondary_business_result = calculate_secondary_business_points(lob_input, product_input, secondary_business_cr_input)
    secondary_points = secondary_business_result["SecondaryPoints"]
    total_points = secondary_points + primary_points
    ils_result = calculate_ils_achievement(secondary_points, primary_points)
    ils_qualification_status = calculate_ils_qualification_status(primary_gate_shortfall, secondary_points, primary_points, lob_input, product_input)

    output = {
        "Primary Calculation": {"PrimaryPoints": primary_points, "PrimaryGateShortfall": primary_gate_shortfall},
        "Secondary Calculation": secondary_business_result,
        "Total Points": total_points,
        "Secondary Points for ILS": ils_result,
        "ILS Qualification Status": ils_qualification_status
    }
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()
