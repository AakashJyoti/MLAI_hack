import json
import numpy_financial as npf

def loan_details(principal, tenure_months, roi):
    monthly_interest_rate = roi / 12 / 100
    emi = npf.pmt(monthly_interest_rate, tenure_months, -principal)
    total_emi_paid = emi * tenure_months
    total_interest_paid = total_emi_paid - principal
    return emi, total_emi_paid, principal, total_interest_paid

def interest_amount(emi, tenure, loan_amount):
    interest = emi * tenure - loan_amount
    return interest

def loan_percentage(loan_amount, interest_amount):
    sum_L_I = loan_amount + interest_amount
    principal_percentage = loan_amount / sum_L_I * 100
    return principal_percentage

def interest_percentage(loan_amount, interest_amount):
    sum_L_I = loan_amount + interest_amount
    interest_percentage = interest_amount / sum_L_I * 100
    return interest_percentage

def interest_repaid(loan_amount, percentage):
    if loan_amount > 0:
        return loan_amount * percentage / 12 / 100  # Adjusted percentage calculation
    else:
        return 0

def principal_repaid(emi, interest_repaid):
    pri_repaid = emi - interest_repaid
    return pri_repaid

def closing_principal_outstanding(loan_amount, principal_repaid):
    clos_principal_outstand = loan_amount - principal_repaid
    return clos_principal_outstand

def emi_calc(principal=None, tenure_months=None, roi=None, emi=None, percentage=None):
    output = {}
    print(principal, tenure_months, roi, emi, percentage)
    if isinstance(principal, str):
        principal = int(principal.replace(',', ''))

    if isinstance(emi, str):
        emi = float(emi.replace(',', ''))

    if isinstance(percentage, str):
        percentage = float(percentage.replace(',', ''))
    print(principal , roi)
    if principal and tenure_months and roi:
        print("EMI calculation")
        # Loan details
        emi, total_emi_paid, principal, total_interest_paid = loan_details(principal, int(tenure_months), float(roi))
        output["Loan Details"] = {
            "EMI": f"₹ {emi:,.2f}",
            "Total EMI paid": f"₹ {total_emi_paid:,.2f}",
            "Principal": f"₹ {principal:,.2f}",
            "Total interest paid": f"₹ {total_interest_paid:,.2f}"
        }

    if emi and tenure_months and principal:
        # Interest amount
        interest = interest_amount(emi, int(tenure_months), principal)
        output["Interest Amount"] = f"₹ {interest:,.2f}"

        # Loan and interest percentages
        principal_percentage = loan_percentage(principal, interest)
        interest_percentage_value = interest_percentage(principal, interest)
        output["Loan Percentages"] = {
            "Principal Percentage": f"{principal_percentage:.2f}%",
            "Interest Percentage": f"{interest_percentage_value:.2f}%"
        }

    if principal and percentage:
        # Repaid interest
        repaid_interest = interest_repaid(principal, percentage)
        output["Repaid Interest"] = f"₹ {repaid_interest:,.2f}"

    if emi and principal and percentage:
        # Repaid interest
        repaid_interest = interest_repaid(principal, percentage)
        
        # Principal repaid
        principal_repaid_value = principal_repaid(emi, repaid_interest)
        output["Principal Repaid"] = f"₹ {principal_repaid_value:,.2f}"

        # Closing principal outstanding
        closing_principal = closing_principal_outstanding(principal, principal_repaid_value)
        output["Closing Principal Outstanding"] = f"₹ {closing_principal:,.2f}"

    print(json.dumps(output, indent=4))
    return output

# Example usage:
# emi_calc(principal=1000000, tenure_months=240, roi=8.6)
