import numpy_financial as npf
import json
from datetime import datetime
import math
def calculate_emi(principal, roi, tenure_months):
    if principal and roi and tenure_months:
        emi = npf.pmt(roi / 12, tenure_months, -principal)
        return round(float(emi))
    return None

def calculate_total_savings(original_emi, revised_emi, mob):
    if original_emi and revised_emi and mob:
        total_savings = (original_emi * mob) - (revised_emi * mob)
        return total_savings
    return None

def calculate_eligible_loan_amount(sanction_amount, mob):
    if sanction_amount and mob is not None:
        if mob <= 12:
            multiplier = 1.0
        elif 12 < mob <= 18:
            multiplier = 1.10
        elif 18 < mob <= 24:
            multiplier = 1.15
        elif 24 < mob <= 36:
            multiplier = 1.20
        else:
            multiplier = 1.30
        eligible_loan_amount = sanction_amount * multiplier
        return round(eligible_loan_amount)
    return None

def calculate_top_up_amount(eligible_loan_amount, sanction_amount):
    if eligible_loan_amount and sanction_amount:
        top_up_amount = eligible_loan_amount - sanction_amount
        return round(top_up_amount)
        # return math.floor(top_up_amount)
    return None

def calculate_mob(month_of_disbursement):
    if month_of_disbursement:
        current_date = datetime.now()
        disbursement_date = datetime.strptime(month_of_disbursement, '%b-%y')
        mob = (current_date.year - disbursement_date.year) * 12 + (current_date.month - disbursement_date.month)
        # return round(mob)
        return math.ceil(mob) + 1
    return None

def calculate_original_emi(sanction_amount, roi, tenure_months):
    return calculate_emi(sanction_amount, roi, tenure_months)

def bts_calc(sanction_amount=None, tenure_remaining_months=None, existing_roi=None, abhfl_roi=None, month_of_disbursement=None):
    print(sanction_amount,tenure_remaining_months,existing_roi,abhfl_roi,month_of_disbursement)
    output = {}
    if isinstance(sanction_amount, str):
        sanction_amount = int(sanction_amount.replace(',', ''))
    mob = calculate_mob(month_of_disbursement)
    if mob is not None:
        output["Months on Book (MoB)"] = mob

    if sanction_amount is not None and  tenure_remaining_months is not None and existing_roi is not None:
        existing_roi = existing_roi / 100
        original_emi = calculate_original_emi(sanction_amount, existing_roi,  tenure_remaining_months)
        if original_emi is not None:
            output["Original EMI"] = f"₹ {original_emi:,.2f}"
            output["Existing ROI"] = f"{existing_roi * 100}%"
            output["Tenure remaining (months)"] =  tenure_remaining_months

    if sanction_amount is not None and  tenure_remaining_months is not None and abhfl_roi is not None:
        abhfl_roi = abhfl_roi / 100
        revised_emi = calculate_emi(sanction_amount, abhfl_roi,  tenure_remaining_months)
        if revised_emi is not None:
            output["Revised EMI"] = f"₹ {revised_emi:,.2f}"
            output["ABHFL ROI"] = f"{abhfl_roi * 100}%"

            if "Original EMI" in output and mob is not None:
                total_savings = calculate_total_savings(original_emi, revised_emi,  tenure_remaining_months)
                if total_savings is not None:
                    output["Total amount saved with BT"] = f"₹ {total_savings:,.2f}"

    if sanction_amount is not None and mob is not None:
        new_eligible_loan_amount = calculate_eligible_loan_amount(sanction_amount, mob)
        if new_eligible_loan_amount is not None:
            output["New eligible Loan amount"] = f"₹ {new_eligible_loan_amount:,.2f}"

            top_up_amount_eligible = calculate_top_up_amount(new_eligible_loan_amount, sanction_amount)
            if top_up_amount_eligible is not None:
                output["Top Up amount eligible"] = f"₹ {top_up_amount_eligible:,.2f}"

    if sanction_amount is not None:
        output["Existing Sanction Amount"] = f"₹ {sanction_amount:,.2f}"

    print(json.dumps(output, indent=4))
    return output

# print(bts_calc(4000000 , 180 , 10 , 5 , "Nov-24"))