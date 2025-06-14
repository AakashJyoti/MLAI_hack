import json
import math
import numpy_financial as npf # Assuming this is the correct library for financial calculations

def calculate_foir(net_annual_income):
    if net_annual_income <= 0:
        return "Require Monthly Income"
    elif net_annual_income <= 500000:
        return 0.60
    elif net_annual_income <= 1200000:
        return 0.65
    elif net_annual_income <= 2400000:
        return 0.70
    else:
        return 0.75

def calculate_eligible_emi(net_monthly_income, foir, obligations):
    if isinstance(foir, str):
        return "Not Eligible"
    else:
        return max(net_monthly_income * foir - obligations, 0)

def step_up_calculator(net_monthly_income=None, obligations=None, working_sector=None, 
                       total_tenure=None, rate=None, primary_tenure=None):
    output = {}

    if net_monthly_income:
        net_annual_income = net_monthly_income * 12
        output["Net Annual Income"] = f"₹ {net_annual_income:,.2f}"

        foir = calculate_foir(net_annual_income)
        output["FOIR"] = foir if isinstance(foir, str) else f"{foir:.2%}"

        if obligations is not None:
            eligible_emi = calculate_eligible_emi(net_monthly_income, foir, obligations)
            output["Eligible EMI"] = eligible_emi if isinstance(eligible_emi, str) else f"₹ {eligible_emi:,.2f}"

    if working_sector:
        print(f"Original working sector: '{working_sector}'")
        working_sector = working_sector.lower()
        sector_increase = {"manufacturing": 0.08, "service": 0.08, "industrial": 0.06}
        expected_increase_income = sector_increase.get(working_sector, 0)

        output["Expected Increase in Income"] = f"{expected_increase_income:.2%}"
        output["working sector"] = f"{working_sector}"

    if total_tenure and primary_tenure:
        secondary_tenure = total_tenure - primary_tenure
        output["Secondary Tenure"] = secondary_tenure

    if rate and total_tenure and 'Eligible EMI' in output:
        eligible_emi = float(output['Eligible EMI'].replace('₹ ', '').replace(',', ''))
        
        # Calculate standard loan eligibility using numfinancial.pv
        standard_eligibility = npf.pv(rate / 12 / 100, total_tenure, -eligible_emi)
        
        # Calculate step-up loan eligibility
        if primary_tenure:
            eligibility_primary = npf.pv(rate / 12 / 100, primary_tenure, -eligible_emi)
            eligibility_secondary = npf.pv(rate / 12 / 100, secondary_tenure, -(eligible_emi * (1 + expected_increase_income)))
            total_loan_eligibility = eligibility_primary + eligibility_secondary
            
            output["Loan Eligibility"] = {
                "Primary Tenure": f"₹ {eligibility_primary:,.2f}",
                "Secondary Tenure": f"₹ {eligibility_secondary:,.2f}",
                "Total (Step-Up)": f"₹ {total_loan_eligibility:,.2f}",
                "Standard": f"₹ {standard_eligibility:,.2f}"
            }
            
            increase_in_eligibility = (total_loan_eligibility / standard_eligibility) - 1
            output["Increase in Eligibility"] = f"{increase_in_eligibility:.2%}"
        else:
            output["Loan Eligibility"] = {
                "Standard": f"₹ {standard_eligibility:,.2f}"
            }

        # Calculate EMIs
        primary_tenure_emi = eligible_emi
        secondary_tenure_emi = eligible_emi * (1 + expected_increase_income)
        change_in_emi = (secondary_tenure_emi / primary_tenure_emi) - 1 if primary_tenure_emi > 0 else 0

        output["EMI Details"] = {
            "Primary Tenure EMI": f"₹ {primary_tenure_emi:,.2f}",
            "Secondary Tenure EMI": f"₹ {secondary_tenure_emi:,.2f}",
            "Change in EMI": f"{change_in_emi:.2%}"
        }

    return json.dumps(output, indent=2)

# Example usage
# if __name__ == "__main__":
#     result = step_up_calculator(
#         net_monthly_income=40000,
#         obligations=10000,
#         working_sector="Industrial",
#         total_tenure=240,
#         rate=9.05,
#         primary_tenure=36
#     )
#     print(result)
