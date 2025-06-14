from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy_financial as npf
import json

def calculate_age(dob):
    today = datetime.today()
    dob = datetime.strptime(dob, "%d %B %Y")
    age = relativedelta(today, dob)
    return age.years + age.months / 12 + age.days / 365.25

def get_foir(customer_type, net_monthly_income):
    if customer_type == "salaried":
        if net_monthly_income > 200000:
            return 75
        elif net_monthly_income > 100000:
            return 70
        elif net_monthly_income > 41500:
            return 65
        else:
            return 60
    else:  # Self-Employed
        return 80

def max_loan_amount(rate, tenure_months, emi):
    max_loan = npf.pv(rate / 12, tenure_months, -emi)
    return max_loan

# def count_property_value(down_payment):
#     if down_payment < 330000:
#         return down_payment / 0.10
#     elif down_payment < 1875000:
#         return down_payment / 0.20
#     else:
#         return down_payment / 0.25


def home_loan_eligibility(customer_type=None, dob=None, net_monthly_income=None, current_monthly_emi=None, roi=None):
    output = {}

    if dob:
        current_age = calculate_age(dob)
        output["Current Age (yrs)"] = f"{current_age:.2f}"

    if customer_type and net_monthly_income:
        foir = get_foir(customer_type.lower(), net_monthly_income)
        output["FOIR"] = f"{foir}%"

    if customer_type:
        max_age_at_maturity = 60 if customer_type.lower() == "salaried" else 70
        max_tenure_years = 30 if customer_type.lower() == "salaried" else 20
        if dob:
            max_tenure_to_be_offered_years = min(max_age_at_maturity - current_age, max_tenure_years)
            max_tenure_months = int(max_tenure_to_be_offered_years * 12)
            output["Max Tenure to be offered"] = f"{max_tenure_months} months"
            output["Max tenure in years"] = f"{max_tenure_months / 12:.2f} years"

    if net_monthly_income and current_monthly_emi is not None and roi is not None and customer_type:
        amount_available_for_emi = (net_monthly_income * foir / 100) - current_monthly_emi
        output["Amount available for EMI"] = f"₹ {amount_available_for_emi:,.2f}"

        if "Max Tenure to be offered" in output:
            max_loan = max_loan_amount(roi / 100, max_tenure_months, amount_available_for_emi)
            output["Max Loan Amount"] = f"₹ {max_loan:,.2f}"

    # if down_payment is not None:
    #     property_value = count_property_value(down_payment)
    #     output["Property Value"] = f"₹ {property_value:,.2f}"

    output["Customer Type"] = customer_type
    output["Date of Birth"] = dob
    output["Net Monthly Income"] = f"₹ {net_monthly_income:,.2f}" if net_monthly_income else None
    output["Current Monthly EMI"] = f"₹ {current_monthly_emi:,.2f}" if current_monthly_emi is not None else None
    output["ROI"] = f"{roi}%" if roi is not None else None

    print(json.dumps(output, indent=4))
    return output

# def home_loan_eligibility(customer_type, dob, net_monthly_income, current_monthly_emi, roi, down_payment):
#     # Define maximum age at maturity and maximum tenure years based on customer type
#     print("Customer Type:  ",customer_type)
#     max_age_at_maturity = 60 if customer_type == "salaried" else 70
#     max_tenure_years = 30 if customer_type == "salaried" else 20
#
#     # Get FOIR based on customer type and net monthly income
#     foir = get_foir(customer_type, net_monthly_income)
#
#     # Calculate current age
#     current_age = calculate_age(dob)
#
#     # Calculate max tenure to be offered based on the given condition
#     max_tenure_to_be_offered_years = min(max_age_at_maturity - current_age, max_tenure_years)
#     max_tenure_months = int(max_tenure_to_be_offered_years * 12)
#
#     # Calculate the amount available for EMI
#     amount_available_for_emi = (net_monthly_income * foir / 100) - current_monthly_emi
#
#     # Calculate the maximum loan amount
#     max_loan = max_loan_amount(roi / 100, max_tenure_months, amount_available_for_emi)
#
#     # Calculate property value
#     property_value = count_property_value(down_payment)
#
#     # Prepare JSON output
#     output = {
#         "Customer Type": customer_type,
#         "Date of Birth": dob,
#         "Net Monthly Income": net_monthly_income,
#         "Current Monthly EMI": current_monthly_emi,
#         "FOIR": f"{foir}%",
#         "Amount available for EMI": amount_available_for_emi,
#         "Max Loan Amount": f"₹ {max_loan:,.2f}",
#         "ROI": f"{roi}%",
#         "Current Age (yrs)": f"{current_age:.2f}",
#         "Max Tenure allowed as per Customer Type": f"{max_tenure_years * 12} months",
#         "Max Tenure to be offered": f"{max_tenure_months} months",
#         "Max tenure in years": f"{max_tenure_months / 12:.2f} years",
#         "Property Value": f"₹ {property_value:,.2f}"
#     }
#
#     print(output)
#     return output

# # Example usage
# main("Salaried", "12 May 2000", 200000, 12000, 9, 2000000)


    # Input values
#     customer_type = "Salaried"
#     dob = "12 May 2000"
#     net_monthly_income = 200000
#     current_monthly_emi = 12000
#     roi = 9
#     down_payment = 2000000
#     max_age_at_maturity = 60 if customer_type == "Salaried" else 70
#     max_tenure_years = 30 if customer_type == "Salaried" else 20

#     # Get FOIR based on customer type and net monthly income
#     foir = get_foir(customer_type, net_monthly_income)

#     # Calculated values
#     current_age = calculate_age(dob)

#     # Calculate max tenure to be offered based on the given condition
#     max_tenure_to_be_offered_years = min(max_age_at_maturity - current_age, max_tenure_years)
#     max_tenure_months = int(max_tenure_to_be_offered_years * 12)

#     # Calculate the amount available for EMI
#     amount_available_for_emi = (net_monthly_income * foir / 100) - current_monthly_emi

#     # Calculate the maximum loan amount
#     max_loan = max_loan_amount(roi / 100, max_tenure_months, amount_available_for_emi)
#     #calculte Property Value

#     property_value = count_property_value(down_payment)
#     print(f"Customer Type: {customer_type}")
#     print(f"Date of Birth: {dob}")
#     print(f"Net Monthly Income: {net_monthly_income}")
#     print(f"Current Monthly EMI: {current_monthly_emi}")
#     print(f"FOIR: {foir}%")
#     print(f"Amount available for EMI: {amount_available_for_emi}")
#     print(f"Max Loan Amount: ₹ {max_loan:,.2f}")
#     print(f"ROI: {roi}%")
#     print(f"Current Age (yrs): {current_age:.2f}")
#     print(f"Max Tenure allowed as per Customer Type: {max_tenure_years * 12} months")
#     print(f"Max Tenure to be offered: {max_tenure_months} months")
#     print(f"Max tenure in years: {max_tenure_months / 12:.2f} years")
#     print(f"Property Value: ₹ {property_value:,.2f}")

# if __name__ == "__main__":
#     main()
