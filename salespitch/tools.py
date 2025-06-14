import os
import json
from langchain.tools import StructuredTool
from .eligibility_calcultor1 import home_loan_eligibility
from .part_payment1 import part_payment
from .emi_cal2 import emi_calc
from .loan_eligibility1 import loan_eligibility
from .step_up_calculator import step_up_calculator
from .BTS_calculator2 import bts_calc
from .step_down_joint import step_down
from .step_down_pension import step_down_pension
from .login_checklist import logincheck_documents
from .csv_agnet import filter_csv
from .Select_calculator import select_calculator
from .property_faq import get_qna_by_location_from_file
from .location_cat import get_location_details


def home_loan_eligibility_tool(
    customer_type, dob, net_monthly_income, current_monthly_emi, roi
):
    """Calculate the maximum home loan amount a customer is eligible for."""
    return home_loan_eligibility(
        customer_type, dob, net_monthly_income, current_monthly_emi, roi
    )


def part_payment_tool(
    loan_outstanding, tenure_total_months, roi, part_payment_amount, current_emi
):
    """Calculate the impact of part payment on the loan."""
    return part_payment(
        loan_outstanding, tenure_total_months, roi, part_payment_amount, current_emi
    )


def emi_calc_tool(principal, tenure_total_months, roi, emi, percentage):
    """Calculate the EMI, interest, principal, or tenure of a loan."""
    return emi_calc(principal, tenure_total_months, roi, emi, percentage)


def loan_eligibility_tool(
    total_income, total_obligations, customer_profile, tenure_total_years, roi, foir
):
    """Determine the total loan amount a customer is eligible for."""
    return loan_eligibility(
        total_income,
        total_obligations,
        customer_profile,
        tenure_total_years * 12,
        roi / 100,
        foir,
    )


def bts_calc_tool(
    sanction_amount,
    tenure_remaining_months,
    existing_roi,
    abhfl_roi,
    month_of_disbursement,
):
    """Calculate the benefit of transfer of sanction (BTS) value."""
    return bts_calc(
        sanction_amount,
        tenure_remaining_months,
        existing_roi,
        abhfl_roi,
        month_of_disbursement,
    )


def step_up_calculator_tool(
    net_monthly_income,
    obligations,
    working_sector,
    total_tenure_months,
    rate,
    primary_tenure_months,
):
    """Calculate the step-up loan amount."""
    return step_up_calculator(
        net_monthly_income,
        obligations,
        working_sector,
        total_tenure_months,
        rate,
        primary_tenure_months,
    )


def step_down_joint_income_calculator_tool(
    customer_type,
    salaried_son_dob,
    salaried_dad_dob,
    salaried_son_current_net_monthly_income,
    salaried_dad_current_net_monthly_income,
    salaried_dad_obligations,
    salaried_son_obligations,
    salaried_son_ROI,
    salaried_dad_ROI,
    salaried_dad_age,
    salaried_son_age,
):
    """Determine the total loan eligibility based on son's and dad's financial profiles."""
    return step_down(
        customer_type,
        salaried_son_dob,
        salaried_dad_dob,
        salaried_son_current_net_monthly_income,
        salaried_dad_current_net_monthly_income,
        salaried_dad_obligations,
        salaried_son_obligations,
        salaried_son_ROI,
        salaried_dad_ROI,
        salaried_dad_age,
        salaried_son_age,
    )


def step_down_pension_income_calculator_tool(
    dob_of_person,
    monthly_income_from_salary,
    monthly_income_from_pension,
    salaried_obligations,
    pension_obligations,
    salaried_requested_tenure,
    pension_requested_tenure,
    pension_ROI,
    salaried_ROI,
    age_of_person,
    IMGC,
):
    """Calculate the pension income eligibility."""
    return step_down_pension(
        dob_of_person,
        monthly_income_from_salary,
        monthly_income_from_pension,
        salaried_obligations,
        pension_obligations,
        salaried_requested_tenure,
        pension_requested_tenure,
        pension_ROI,
        salaried_ROI,
        age_of_person,
        IMGC,
    )


def logincheck_documents_tool(
    employment, eligibility_method=None, rental_income=False, other_income=False
):
    """Get the list of required documents for login."""
    return logincheck_documents(
        employment.lower(), eligibility_method.lower(), rental_income, other_income
    )


def branches_list_tool(hfc_name, state, district=None, pincode=None):
    """Filter branch details of Housing Finance Companies (HFCs)."""
    return filter_csv(hfc_name, state, district, pincode)


def location_cat_affordable_tool(location):
    """Retrieve location category and cap in cr for affordable product program."""
    return get_location_details(location)


def properties_faq_tool(location_input):
    """Retrieve Q&A data based on location input."""
    return get_qna_by_location_from_file(location_input)


def get_product_info(product_name):
    """Retrieve product information from a text file."""
    try:
        with open(f"prompts/{product_name}.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Product information for '{product_name}' not found."


def get_product_descriptions():
    """Load product descriptions from a JSON file."""
    try:
        with open("prompts/product_descriptions.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def create_product_info_tool(product_name, abhfl_instance):
    """Create a tool to retrieve information about a specific product."""
    tool_name = product_name.lower().replace(" ", "_") + "_info"
    product_descriptions = get_product_descriptions()

    if product_name in product_descriptions:
        tool_description = product_descriptions[product_name]
    else:
        tool_description = (
            f"Retrieve detailed information about the {product_name} product."
        )

    def product_info_tool():
        product_info = get_product_info(product_name)
        with open("prompts/main_prompt2.txt", "r", encoding="utf-8") as f:
            text = f.read()
        # Append product info to system message
        abhfl_instance.append_to_system_message(text + product_info)
        return product_info

    return StructuredTool.from_function(
        func=product_info_tool,
        name=tool_name,
        description=tool_description,
    )


def create_tools(abhfl_instance):
    """Create and return a list of StructuredTools."""
    tools = [
        StructuredTool.from_function(
            func="Automated_Data_Flow",
            name="Automated_Data_Flow",
            description="""RBI FAQs on ADF implementation by banks.""",
        ),
        StructuredTool.from_function(
            func="Paytm_Payments_Bank_Restrictions_2024",
            name="Paytm_Payments_Bank_Restrictions_2024",
            description="""RBI press release on business restrictions for Paytm Payments Bank.""",
        ),
        StructuredTool.from_function(
            func="Private_Sector_Bank_Licensing_Clarifications",
            name="Private_Sector_Bank_Licensing_Clarifications",
            description="""RBI clarifications on licensing guidelines for new private sector banks.""",
        ),
        StructuredTool.from_function(
            func="Payments_Bank_Licensing_Clarifications",
            name="Payments_Bank_Licensing_Clarifications",
            description="""RBI clarifications on guidelines for licensing payments banks.""",
        ),
        StructuredTool.from_function(
            func="Small_Finance_Bank_Licensing_Clarifications",
            name="Small_Finance_Bank_Licensing_Clarifications",
            description="""RBI clarifications on licensing guidelines for small finance banks.""",
        ),
        StructuredTool.from_function(
            func="Compliance_Functions_and_CCO_Role",
            name="Compliance_Functions_and_CCO_Role",
            description="""RBI guidelines on compliance functions and the role of the Chief Compliance Officer in banks.""",
        ),
        StructuredTool.from_function(
            func="CRR_Exemption",
            name="CRR_Exemption",
            description=""" RBI press release on CRR exemption for certain banks.""",
        ),
        StructuredTool.from_function(
            func="DEA_Fund_Scheme_2014",
            name="DEA_Fund_Scheme_2014",
            description="""RBI guidelines on the DEA Fund Scheme 2014.""",
        ),
        StructuredTool.from_function(
            func="Digital_Lending_Guidelines",
            name="Digital_Lending_Guidelines",
            description="""RBI guidelines on digital lending practices.""",
        ),
        StructuredTool.from_function(
            func="SARFAESI_Secured_Assets_Display",
            name="SARFAESI_Secured_Assets_Display",
            description="""RBI guidelines on the display of secured assets under SARFAESI Act.""",
        ),
        StructuredTool.from_function(
            func="Credit_Supply_Large_Borrowers",
            name="Credit_Supply_Large_Borrowers",
            description="""RBI guidelines on credit supply to large borrowers.""",
        ),
        StructuredTool.from_function(
            func="Fair_Lending_Penal_Charges",
            name="Fair_Lending_Penal_Charges",
            description="""RBI guidelines on fair lending practices and penal charges.""",
        ),
        StructuredTool.from_function(
            func="RBI_Deposit_Interest_FAQs_2025",
            name="RBI_Deposit_Interest_FAQs_2025",
            description="""RBI FAQs on deposit interest rates for 2025.""",
        ),
        StructuredTool.from_function(
            func="RBI_Loan_Exposure_Transfer_FAQs_2021",
            name="RBI_Loan_Exposure_Transfer_FAQs_2021",
            description="""RBI FAQs on loan exposure transfer guidelines for 2021.""",
        ),
        StructuredTool.from_function(
            func="RBI_Card_Issuance_Conduct_FAQs_2022",
            name="RBI_Card_Issuance_Conduct_FAQs_2022",
            description="""RBI FAQs on card issuance and conduct guidelines for 2022.""",
        ),
        StructuredTool.from_function(
            func="RBI_KYC_Master_Direction_FAQs_2016",
            name="RBI_KYC_Master_Direction_FAQs_2016",
            description="""RBI FAQs on KYC Master Direction for 2016.""",
        ),
        StructuredTool.from_function(
            func="Fraud_Risk_Management_FAQs_REs_2024",
            name="Fraud_Risk_Management_FAQs_REs_2024",
            description="""RBI FAQs on fraud risk management for regulated entities in 2024.""",
        ),
        StructuredTool.from_function(
            func="COVID19_Resolution_Framework_FAQs_2022",
            name="COVID19_Resolution_Framework_FAQs_2022",
            description="""RBI FAQs on the COVID-19 resolution framework for 2022.""",
        ),
        StructuredTool.from_function(
            func="EMI_Floating_Rate_Reset_FAQs",
            name="EMI_Floating_Rate_Reset_FAQs",
            description="""RBI FAQs on EMI floating rate reset guidelines.""",
        ),
        StructuredTool.from_function(
            func="Green_Deposits_Framework",
            name="Green_Deposits_Framework",
            description="""RBI guidelines on the Green Deposits Framework.""",
        ),
        StructuredTool.from_function(
            func="Compromise_Settlements_Writeoffs_Framework",
            name="Compromise_Settlements_Writeoffs_Framework",
            description="""RBI guidelines on compromise settlements and write-offs framework.""",
        ),
        StructuredTool.from_function(
            func="Gold_Monetization_Scheme_2015",
            name="Gold_Monetization_Scheme_2015",
            description="""RBI guidelines on the Gold Monetization Scheme 2015.""",
        ),
        StructuredTool.from_function(
            func="Guidelines_SCA_SA_Appointment_Banks_NBFCs",
            name="Guidelines_SCA_SA_Appointment_Banks_NBFCs",
            description="""RBI guidelines on the appointment of SCA and SA for banks and NBFCs.""",
        ),
        StructuredTool.from_function(
            func="Default_Loss_Guarantee_Digital_Lending_2024",
            name="Default_Loss_Guarantee_Digital_Lending_2024",
            description="""RBI guidelines on default loss guarantee in digital lending for 2024.""",
        ),
        StructuredTool.from_function(
            func="MCLR_Guidelines",
            name="MCLR_Guidelines",
            description="""RBI guidelines on the Marginal Cost of Funds based Lending Rate (MCLR).""",
        ),
        StructuredTool.from_function(
            func="Partial_Credit_Guarantee_GoI_PSBs",
            name="Partial_Credit_Guarantee_GoI_PSBs",
            description="""RBI guidelines on the Partial Credit Guarantee Scheme for Public Sector Banks.""",
        ),
        StructuredTool.from_function(
            func="Govt_Pension_Payment_2025",
            name="Govt_Pension_Payment_2025",
            description="""RBI guidelines on government pension payments for 2025.""",
        ),
        StructuredTool.from_function(
            func="QA_22_Accounts",
            name="QA_22_Accounts",
            description="""RBI guidelines on quality assurance for accounts.""",
        ),
        StructuredTool.from_function(
            func="Microfinance_Loans_Regulatory_Framework_2025",
            name="Microfinance_Loans_Regulatory_Framework_2025",
            description="""RBI guidelines on the regulatory framework for microfinance loans in 2025.""",
        ),
        StructuredTool.from_function(
            func="BSBDA_FAQs_RRBs_StCBs_DCCBs",
            name="BSBDA_FAQs_RRBs_StCBs_DCCBs",
            description="""RBI FAQs on Basic Savings Bank Deposit Accounts for RRBs, StCBs, and DCCBs.""",
        ),
        StructuredTool.from_function(
            func="UDGAM_Portal",
            name="UDGAM_Portal",
            description="""RBI guidelines on the UDGAM portal for grievance redressal.""",
        ),
    ]

    # Add product information tools
    product_names = [
        "Automated_Data_Flow",
        "Paytm_Payments_Bank_Restrictions_2024",
        "Private_Sector_Bank_Licensing_Clarifications",
        "Payments_Bank_Licensing_Clarifications",
        "Small_Finance_Bank_Licensing_Clarifications",
        "Compliance_Functions_and_CCO_Role",
        "CRR_Exemption",
        "DEA_Fund_Scheme_2014",
        "Digital_Lending_Guidelines",
        "SARFAESI_Secured_Assets_Display",
        "Credit_Supply_Large_Borrowers",
        "Fair_Lending_Penal_Charges",
        "RBI_Deposit_Interest_FAQs_2025",
        "RBI_Loan_Exposure_Transfer_FAQs_2021",
        "RBI_Card_Issuance_Conduct_FAQs_2022",
        "RBI_KYC_Master_Direction_FAQs_2016",
        "Fraud_Risk_Management_FAQs_REs_2024",
        "COVID19_Resolution_Framework_FAQs_2022",
        "EMI_Floating_Rate_Reset_FAQs",
        "Green_Deposits_Framework",
        "Compromise_Settlements_Writeoffs_Framework",
        "Gold_Monetization_Scheme_2015",
        "Guidelines_SCA_SA_Appointment_Banks_NBFCs",
        "Default_Loss_Guarantee_Digital_Lending_2024",
        "MCLR_Guidelines",
        "Partial_Credit_Guarantee_GoI_PSBs",
        "Govt_Pension_Payment_2025",
        "QA_22_Accounts",
        "Microfinance_Loans_Regulatory_Framework_2025",
        "BSBDA_FAQs_RRBs_StCBs_DCCBs",
        "UDGAM_Portal",
    ]
    for product_name in product_names:
        tools.append(create_product_info_tool(product_name, abhfl_instance))

    return tools
