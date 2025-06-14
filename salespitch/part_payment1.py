import numpy_financial as npf
import json
import math
def calculate_revised_tenure(loan_outstanding, part_payment_amount, emi, roi, tenure_months):
    # New loan outstanding after part payment
    new_loan_outstanding = loan_outstanding - part_payment_amount
    
    # Calculate the revised tenure using the PMT function
    revised_tenure = npf.nper(roi / 12, -emi, new_loan_outstanding)
    # print(emi)
    revised_tenure = math.ceil(float(revised_tenure))
    
    # Calculate the reduction in tenure
    reduction_in_tenure = tenure_months - revised_tenure
    
    return revised_tenure, reduction_in_tenure

def calculate_revised_emi(loan_outstanding, part_payment_amount, roi, tenure_months):
    # New loan outstanding after part payment
    new_loan_outstanding = loan_outstanding - part_payment_amount
    
    # Calculate the revised EMI using the PMT function
    revised_emi = npf.pmt(roi / 12, tenure_months, -new_loan_outstanding)
    revised_emi = math.ceil(revised_emi)
    
    # Calculate the change in EMI
    current_emi = npf.pmt(roi / 12, tenure_months, -loan_outstanding)
    change_in_emi = revised_emi - current_emi
    
    return revised_emi, change_in_emi


def part_payment(loan_outstanding=None, tenure_months=None, roi=None, part_payment_amount=None, current_emi=None):
    output = {}

    if loan_outstanding is not None:
        output["Loan Outstanding"] = f"₹ {loan_outstanding:,.2f}"

    if tenure_months is not None:
        output["Balance Tenure (months)"] = tenure_months

    if roi is not None:
        roi = roi / 100
        output["ROI"] = f"{roi * 100}%"

    if part_payment_amount is not None:
        output["Part Payment Amount"] = f"₹ {part_payment_amount:,.2f}"

    if current_emi is not None:
        output["Current EMI"] = f"₹ {current_emi:,.2f}"

    if loan_outstanding is not None and part_payment_amount is not None:
        revised_outstanding = loan_outstanding - part_payment_amount
        output["Revised Outstanding after Prepayment"] = f"₹ {revised_outstanding:,.2f}"

        if current_emi is not None and roi is not None and tenure_months is not None:
            # Option A: Calculate revised tenure and reduction in tenure
            revised_tenure, reduction_in_tenure = calculate_revised_tenure(loan_outstanding, part_payment_amount, current_emi, roi, tenure_months)
            output["Option A"] = {
                "Description": "If EMI is kept same & impact on tenure",
                "Revised Tenure (months)": revised_tenure,
                "Reduction in Tenure (months)": reduction_in_tenure
            }

            # Option B: Calculate revised EMI and change in EMI
            revised_emi, change_in_emi = calculate_revised_emi(loan_outstanding, part_payment_amount, roi, tenure_months)
            output["Option B"] = {
                "Description": "If Tenure is kept constant & EMI is revised",
                "Revised EMI": f"₹ {revised_emi:,.2f}",
                "Change in EMI": f"₹ {change_in_emi:,.2f}"
            }

    print(json.dumps(output, indent=4))
    return output

# print(part_payment(loan_outstanding=40000000, tenure_months=240, roi=10, part_payment_amount=700000, current_emi=38601))
# def part_payment(loan_outstanding, tenure_months, roi, part_payment_amount, current_emi):
#     # Revised o/s after prepayment
#     revised_outstanding = loan_outstanding - part_payment_amount
#     roi = roi /100
#     # Option A: Calculate revised tenure and reduction in tenure
#     revised_tenure, reduction_in_tenure = calculate_revised_tenure(loan_outstanding, part_payment_amount, current_emi, roi, tenure_months)
#
#     # Option B: Calculate revised EMI and change in EMI
#     revised_emi, change_in_emi = calculate_revised_emi(loan_outstanding, part_payment_amount, roi, tenure_months)
#
#     # Prepare JSON output
#     output = {
#         "Loan Outstanding": f"₹ {loan_outstanding:,.2f}",
#         "Balance Tenure (months)": tenure_months,
#         "ROI": f"{roi * 100}%",
#         "Part Payment Amount": f"₹ {part_payment_amount:,.2f}",
#         "Current EMI": f"₹ {current_emi:,.2f}",
#         "Revised Outstanding after Prepayment": f"₹ {revised_outstanding:,.2f}",
#         "Option A": {
#             "Description": "If EMI is kept same & impact on tenure",
#             "Revised Tenure (months)": revised_tenure,
#             "Reduction in Tenure (months)": reduction_in_tenure
#         },
#         "Option B": {
#             "Description": "If Tenure is kept constant & EMI is revised",
#             "Revised EMI": f"₹ {revised_emi:,.2f}",
#             "Change in EMI": f"₹ {change_in_emi:,.2f}"
#         }
#     }
#     print(output)
#     return output

# Example usage
# loan_outstanding = 4000000  # ₹ 40,00,000
# tenure_months = 240  # 240 months
# roi = 10 / 100  # 10%
# part_payment_amount = 700000  # ₹ 7,00,000
# current_emi = 38601  # ₹ 38,601

# part_payment(loan_outstanding, tenure_months, roi, part_payment_amount, current_emi)
# def main():
#     # Input values
#     loan_outstanding = 4000000  # ₹ 40,00,000
#     tenure_months = 240  # 240 months
#     roi = 10 / 100  # 10%
#     part_payment_amount = 700000  # ₹ 7,00,000
#     current_emi = 38601  # ₹ 38,601
    
#     # Revised o/s after prepayment
#     revised_outstanding = loan_outstanding - part_payment_amount
    
#     # Option A: Calculate revised tenure and reduction in tenure
#     revised_tenure, reduction_in_tenure = calculate_revised_tenure(loan_outstanding, part_payment_amount, current_emi, roi, tenure_months)
    
#     # Option B: Calculate revised EMI and change in EMI
#     revised_emi, change_in_emi = calculate_revised_emi(loan_outstanding, part_payment_amount, roi, tenure_months)
    
#     print(f"Loan o/s: ₹ {loan_outstanding:,.2f}")
#     print(f"Balance tenure (months): {tenure_months}")
#     print(f"ROI: {roi * 100}%")
#     print(f"Part payment amount: ₹ {part_payment_amount:,.2f}")
#     print(f"Current EMI: ₹ {current_emi:,.2f}")
#     print(f"Revised o/s after prepayment: ₹ {revised_outstanding:,.2f}")
    
#     print("\nOption A: if EMI is kept same & impact on tenure")
#     print(f"Revised tenure: {revised_tenure} months")
#     print(f"Reduction in Tenure: {reduction_in_tenure} months")
    
#     print("\nOption B: if Tenure is kept constant & EMI is revised")
#     print(f"Revised EMI: ₹ {revised_emi:,.2f}")
#     print(f"Change in EMI: ₹ {change_in_emi:,.2f}")

# if __name__ == "__main__":
#     main()
