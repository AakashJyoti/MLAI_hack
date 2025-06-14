from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import numpy_financial as npf

def calculate_age(dob):
    today = datetime.today()
    dob = datetime.strptime(dob, "%d %B %Y")
    age = relativedelta(today, dob)
    return round(age.years + age.months / 12 + age.days / 365.25)

def get_foir(customer_type, net_monthly_income):
    current_annual_income = net_monthly_income * 12
    if customer_type == "salaried":
        if net_monthly_income <= 0:
            return "Require Monthly Income"
        elif current_annual_income <= 500000:
            return 60
        elif current_annual_income <= 1200000:
            return 65
        elif current_annual_income <= 2400000:
            return 70
        elif current_annual_income > 2400000:
            return 75
    else:  # Self-Employed
        return 80


def get_eligible_emi(net_monthly_income, obligations, FOIR):
    if net_monthly_income != 0:
        eligible_emi = (net_monthly_income * (FOIR / 100)) - obligations
        return eligible_emi
    else:
        return "please provide a positive income"

def get_primary_tenure(salaried_son_age):
    result = (60 - salaried_son_age) * 12
    if result > 300:
        return 300
    else:
        return result

def get_secondary_tenure(primary_tenure, salaried_son_age):
    expression1 = 300 - primary_tenure
    expression2 = (60 - salaried_son_age) * 12
    minimum_value = min(expression1, expression2)
    return minimum_value


def calculate_present_value(roi, total_tenure, eligible_emi):
    # Convert ROI from percentage to decimal and calculate monthly rate
    monthly_rate = (roi / 100) / 12

    # Calculate total number of payments (months)
    total_payments = total_tenure * 12

    # Calculate Present Value using the formula
    if monthly_rate == 0:  # Handle case where ROI is 0
        present_value = eligible_emi * total_payments
    else:
        present_value = (eligible_emi * (1 - (1 + monthly_rate) ** -total_payments)) / monthly_rate

    return present_value



def calculate_primary_tenure(age):
    result = (60 - age) * 12
    return min(result, 300)



def calculate_pmt(annual_rate, tenure_months, loan_amount, other_costs):
    # Convert annual rate to monthly rate by dividing by 12
    monthly_rate = annual_rate / 12
    # Loan amount divided by other costs (assuming other costs affect the principal)
    adjusted_loan_amount = loan_amount / other_costs
    # Calculate monthly payment using the PMT formula
    monthly_payment = npf.pmt(monthly_rate, tenure_months, -adjusted_loan_amount)
    
    return round(monthly_payment, 2)

def calculate_secondary_tenure(primary_tenure, age):
    value1 = 300 - primary_tenure
    value2 = (60 - age) * 12
    return min(value1, value2)


def step_down(customer_type=None, salaried_son_dob=None, salaried_dad_dob=None, salaried_son_current_net_monthly_income=None,
              salaried_dad_current_net_monthly_income=None, salaried_dad_obligations=None, salaried_son_obligations=None,
              salaried_son_ROI=None, salaried_dad_ROI=None, salaried_dad_age=None, salaried_son_age=None):
    output = {
        "Son": {
        },
        "Dad": {
        }
    }
    if salaried_dad_age:
        output["Dad"]["Age"] = salaried_dad_age
        primary_tenure = get_primary_tenure(salaried_dad_age)
        output["Dad"]["primary_tenure"] = primary_tenure
        output["Son"]["primary_tenure"] = primary_tenure
    elif salaried_dad_dob:
        salaried_dad_age = calculate_age(dob=salaried_dad_dob)
        output["Dad"]["Age"] = salaried_dad_age

        # calculate primary tenure
        primary_tenure = get_primary_tenure(salaried_dad_age)
        output["Dad"]["primary_tenure"] = primary_tenure
        output["Son"]["primary_tenure"] = primary_tenure

    if salaried_son_age:
        output["Son"]["Age"] = salaried_son_age
        primary_tenure = output["Son"]["primary_tenure"]
        secondary_tenure = get_secondary_tenure(primary_tenure, salaried_son_age)
        output["Son"]["secondary_tenure"] = secondary_tenure
        output["Dad"]["secondary_tenure"] = 0
    elif salaried_son_dob:
        salaried_son_age = calculate_age(dob=salaried_son_dob)
        output["Son"]["Age"] = salaried_son_age

        primary_tenure = output["Son"]["primary_tenure"]
        secondary_tenure = get_secondary_tenure(primary_tenure, salaried_son_age)
        output["Son"]["secondary_tenure"] = secondary_tenure
        output["Dad"]["secondary_tenure"] = 0

    if salaried_son_current_net_monthly_income and customer_type:
        print(salaried_son_current_net_monthly_income)
        foir = get_foir(customer_type.lower(), salaried_son_current_net_monthly_income)
        output["Son"]["foir"] = foir

    if salaried_dad_current_net_monthly_income and customer_type:
        foir = get_foir(customer_type.lower(), salaried_dad_current_net_monthly_income)
        output["Dad"]["foir"] = foir

    if salaried_dad_current_net_monthly_income :
        foir = output["Dad"]["foir"]
        eligible_emi = get_eligible_emi(salaried_dad_current_net_monthly_income, salaried_dad_obligations, foir)
        output["Dad"]["eligible_emi"] = eligible_emi

    if salaried_son_current_net_monthly_income:
        foir = output["Son"]["foir"]
        eligible_emi = get_eligible_emi(salaried_son_current_net_monthly_income, salaried_son_obligations, foir)
        output["Son"]["eligible_emi"] = eligible_emi

    son_primary_tenure = output['Son']['primary_tenure']
    son_secondary_tenure = output['Son']['secondary_tenure']
    son_total_tenure = son_primary_tenure + son_secondary_tenure
    output["Son"]["total_tenure"] = son_total_tenure

    dad_primary_tenure = output['Dad']['primary_tenure']
    dad_secondary_tenure = output['Dad']['secondary_tenure']
    dad_total_tenure = dad_primary_tenure + dad_secondary_tenure
    output["Dad"]["total_tenure"] = dad_total_tenure


    # primary_tenure = calculate_primary_tenure(age)
    # secondary_tenure = calculate_secondary_tenure(primary_tenure, 30)
    # total_tenure = primary_tenure + secondary_tenure
    son_eligible_emi = output["Son"]["eligible_emi"]
    dad_eligible_emi = output["Dad"]["eligible_emi"]

    son_loan_amount = calculate_present_value(roi=salaried_son_ROI, total_tenure=son_total_tenure / 12, eligible_emi=son_eligible_emi)
    dad_loan_amount = calculate_present_value(roi=salaried_dad_ROI, total_tenure=dad_total_tenure / 12, eligible_emi=dad_eligible_emi)

    output['Son']["loan_amount"] = son_loan_amount
    output["Dad"]["loan_amount"] = dad_loan_amount
    
    output["EMI for Primary Tenure"] = son_eligible_emi + dad_eligible_emi
    output["EMI for Secondary Tenure"] = son_eligible_emi
    output["Total Laon Amount"] = dad_loan_amount + son_loan_amount
    print(json.dumps(output, indent=4))
    return output

# # Sample input data
# customer_type = "Salaried"
# salaried_son_dob = "15 November 1994"  # Son's Date of Birth
# salaried_dad_dob = "15 November 1972" # Dad's Date of Birth
# salaried_son_current_net_monthly_income = 40231  # Son's Income
# salaried_dad_current_net_monthly_income = 76556  # Dad's Income
# salaried_dad_obligations = 0  # Dad's Obligations
# salaried_son_obligations = 0    # Son's Obligations
# salaried_son_ROI = 9           # Son's Rate of Interest
# salaried_dad_ROI = 9           # Dad's Rate of Interest
#
# # Running the function with the sample data
# result = step_down(customer_type, salaried_son_dob, salaried_dad_dob,
#                    salaried_son_current_net_monthly_income,
#                    salaried_dad_current_net_monthly_income,
#                    salaried_dad_obligations, salaried_son_obligations,
#                    salaried_son_ROI, salaried_dad_ROI)

# # Printing the result
# print(result)