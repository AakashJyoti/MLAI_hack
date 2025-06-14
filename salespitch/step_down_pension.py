from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

def calculate_age(dob):
    today = datetime.today()
    dob = datetime.strptime(dob, "%d %B %Y")
    age = relativedelta(today, dob)
    return round(age.years + age.months / 12 + age.days / 365.25)

def get_foir(net_monthly_income):
    annual_income = net_monthly_income * 12
    if annual_income <= 0:
        return "Require Monthly Income"
    elif annual_income <= 500000:
        return 60
    elif annual_income <= 1200000:
        return 65
    elif annual_income <= 2400000:
        return 70
    else:
        return 75


def get_eligible_emi(net_monthly_income, obligations, FOIR):
    if net_monthly_income != 0:
        eligible_emi = (net_monthly_income * (FOIR / 100)) - obligations
        return eligible_emi
    else:
        return "please provide a positive income"


def salaried_total_tenure(age, requested_tenure):
    # Calculate the first expression
    expression = (60 - age) * 12

    # Apply the conditional logic
    if expression > 300:
        conditional_result = 300
    else:
        conditional_result = expression

    # Calculate the minimum between conditional_result and B15
    # minimum_value = min(conditional_result, requested_tenure)

    return conditional_result


def pension_total_tenure(age, IMGC, requested_tenure):
    # Check the condition based on B8
    if IMGC == "Yes":
        conditional_result = (70 - age) * 12
    else:
        conditional_result = (65 - age) * 12

    # Calculate the minimum between conditional_result and C15
    # minimum_value = min(conditional_result, requested_tenure)

    return conditional_result


def calculate_present_value(roi, total_tenure, eligible_emi):
    # Convert ROI from percentage to decimal and calculate monthly rate
    monthly_rate = (roi / 100) / 12

    # Calculate total number of payments (months)
    total_payments = total_tenure

    # Calculate Present Value using the formula
    if monthly_rate == 0:  # Handle case where ROI is 0
        present_value = eligible_emi * total_payments
    else:
        present_value = (eligible_emi * (1 - (1 + monthly_rate) ** -total_payments)) / monthly_rate

    return present_value




def step_down_pension(dob_of_person=None, monthly_income_from_salary=None,monthly_income_from_pension=None,
              salaried_obligations=None, pension_obligations=None, salaried_requested_tenure=None, pension_requested_tenure=None,
              pension_ROI=None, salaried_ROI=None, age_of_person=None, IMGC=None):
    output = {
        "salaried": {
        },
        "pension": {
        }
    }

    if age_of_person:
        output["Age"] = age_of_person
    elif dob_of_person:
        age = calculate_age(dob=dob_of_person)
        output["Age"] = age

    output['salaried']['monthly_income'] = monthly_income_from_salary
    output['pension']['monthly_income'] = monthly_income_from_pension

    salaried_eligible_monthly_income = monthly_income_from_salary - monthly_income_from_pension
    pension_eligible_monthly_income = monthly_income_from_pension
    output['salaried']['eligible_monthly_income'] = salaried_eligible_monthly_income
    output['pension']['eligible_monthly_income'] = pension_eligible_monthly_income

    salaried_annual_income = monthly_income_from_salary * 12
    pension_annual_income = monthly_income_from_pension * 12
    output['salaried']['annual_income'] = salaried_annual_income
    output['pension']['annual_income'] = pension_annual_income

    salaried_foir = get_foir(monthly_income_from_salary)
    pension_foir = get_foir(monthly_income_from_pension)
    output['salaried']['foir'] = salaried_foir
    output['pension']['foir'] = pension_foir

    output['salaried']['obligations'] = salaried_obligations
    output['pension']['obligations'] = pension_obligations

    salaried_eligible_emi = get_eligible_emi(salaried_eligible_monthly_income, salaried_obligations, salaried_foir)
    pension_eligible_emi = get_eligible_emi(pension_eligible_monthly_income, pension_obligations, pension_foir)
    output['salaried']['eligible_emi'] = salaried_eligible_emi
    output['pension']['eligible_emi'] = pension_eligible_emi

    output['salaried']['requested_tenure'] = salaried_requested_tenure
    output['pension']['requested_tenure'] = pension_requested_tenure

    persons_age = output["Age"]
    maximum_salaried_total_tenure = salaried_total_tenure(persons_age,salaried_requested_tenure)
    maximum_pension_total_tenure = pension_total_tenure(persons_age,IMGC, pension_requested_tenure)
    output['salaried']['total_tenure'] = maximum_salaried_total_tenure
    output['pension']['total_tenure'] = maximum_pension_total_tenure

    output['salaried']['ROI'] = salaried_ROI
    output['pension']['ROI'] = pension_ROI

    salaried_loan_amount = calculate_present_value(salaried_ROI, maximum_salaried_total_tenure, salaried_eligible_emi)
    pension_loan_amount = calculate_present_value(pension_ROI, maximum_pension_total_tenure, pension_eligible_emi)
    output['salaried']['loan_amount'] = salaried_loan_amount
    output['pension']['loan_amount'] = pension_loan_amount

    print(json.dumps(output, indent=4))
    return output


# # Sample input data
# dob_of_person = "06 June 1977"
# monthly_income_from_salary = 73848
# monthly_income_from_pension = 38000
# salaried_obligations = 16500
# pension_obligations = 10000
# salaried_requested_tenure= 0
# pension_requested_tenure = 0
# pension_ROI = 9
# salaried_ROI = 9
# age_of_person = 47
# IMGC = "No"
#
#
# # Running the function with the sample data
# result = step_down_pension(dob_of_person, monthly_income_from_salary,monthly_income_from_pension,
#               salaried_obligations, pension_obligations, salaried_requested_tenure, pension_requested_tenure,
#               pension_ROI, salaried_ROI, age_of_person, IMGC)
#
# # Printing the result
# print(result)