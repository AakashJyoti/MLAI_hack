import numpy_financial as npf
import json

def calculate_eligible_emi(total_income, total_obligations, foir):
    net_available_income = total_income * (foir / 100) - total_obligations
    eligible_emi = net_available_income 
    return round(eligible_emi)

def calculate_approx_eligible_loan_amount(eligible_emi, roi, tenure_months):
    loan_amount = npf.pv(roi / 12, tenure_months, -eligible_emi)
    return round(loan_amount)


def loan_eligibility(total_income=None, total_obligations=None, customer_profile=None, tenure_months=None, roi=None, foir=None):
    output = {}

    if total_income is not None:
        output["Total Income"] = f"₹ {total_income:,.2f}"

    if total_obligations is not None:
        output["Total Obligations"] = f"₹ {total_obligations:,.2f}"

    if foir is not None:
        output["FOIR"] = f"{foir}%"

    if total_income is not None and total_obligations is not None and foir is not None:
        eligible_emi = calculate_eligible_emi(float(total_income), float(total_obligations), foir)
        output["Eligible EMI"] = f"₹ {eligible_emi:,.2f}"

        if roi is not None and tenure_months is not None:
            approx_eligible_loan_amount = calculate_approx_eligible_loan_amount(eligible_emi, roi, tenure_months)
            output["Approx. Eligible Loan Amount"] = f"₹ {approx_eligible_loan_amount:,.2f}"

    if customer_profile is not None:
        output["Customer Profile"] = customer_profile

    if tenure_months is not None:
        output["Tenure (months)"] = tenure_months

    if roi is not None:
        output["ROI"] = f"{roi}%"

    print(json.dumps(output, indent=4))
    return output



# def loan_eligibility(total_income, total_obligations,
#          customer_profile, tenure_months, roi, foir):
#     # Calculate total income and total obligations
#     print("function Called", total_income, total_obligations,
#          customer_profile, tenure_months, roi, foir)
#     # total_income = net_salary_applicant + net_salary_coapplicant_1 + net_salary_coapplicant_2 + net_salary_coapplicant_3
#     # total_obligations = obligations_applicant + obligations_coapplicant_1 + obligations_coapplicant_2 + obligations_coapplicant_3
#
#     # Calculate Eligible EMI
#     eligible_emi = calculate_eligible_emi(float(total_income), float(total_obligations), foir)
#
#     # Calculate Approximate Eligible Loan Amount
#     approx_eligible_loan_amount = calculate_approx_eligible_loan_amount(eligible_emi, roi, tenure_months)
#
#     # Prepare JSON output
#     output = {
#         "Total Income": f"₹ {total_income:,.2f}",
#         "Total Obligations": f"₹ {total_obligations:,.2f}",
#         "FOIR": f"{foir}%",
#         "Eligible EMI": f"₹ {eligible_emi:,.2f}",
#         "Approx. Eligible Loan Amount": f"₹ {approx_eligible_loan_amount:,.2f}"
#     }
#
#     print(output)
#     return output



# def main():
#     # Input values
#     net_salary_applicant = 50000.00
#     net_salary_coapplicant_1 = 25000.00
#     net_salary_coapplicant_2 = 12000.00
#     net_salary_coapplicant_3 = 21000.00
#     total_income = net_salary_applicant + net_salary_coapplicant_1 + net_salary_coapplicant_2 + net_salary_coapplicant_3

#     obligations_applicant = 21000.00
#     obligations_coapplicant_1 = 7000.00
#     obligations_coapplicant_2 = 7000.00
#     obligations_coapplicant_3 = 8000.00
#     total_obligations = obligations_applicant + obligations_coapplicant_1 + obligations_coapplicant_2 + obligations_coapplicant_3

#     customer_profile = "Salaried Bank"  # Assuming input as a string
#     tenure_months = 360  # 360 months for Salaried Bank
#     roi = 10 / 100  # 10%
#     foir = 70  # 70%

#     # Calculate Eligible EMI
#     eligible_emi = calculate_eligible_emi(total_income, total_obligations, foir)
    
#     # Calculate Approximate Eligible Loan Amount
#     approx_eligible_loan_amount = calculate_approx_eligible_loan_amount(eligible_emi, roi, tenure_months)

#     print(f"Total Income: ₹ {total_income:,.2f}")
#     print(f"Total Obligations: ₹ {total_obligations:,.2f}")
#     print(f"FOIR: {foir}%")
#     print(f"Eligible EMI: ₹ {eligible_emi:,.2f}")
#     print(f"Approx. Eligible Loan Amount: ₹ {approx_eligible_loan_amount:,.2f}")

# if __name__ == "__main__":
#     main()