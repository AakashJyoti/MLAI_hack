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
            func=home_loan_eligibility_tool,
            name="home_loan_eligibility_tool",
            description="""Calculate the maximum home loan amount a customer is eligible for based on their profile.
            Parameters:
            customer_type (str, required): Type of the customer (e.g., salaried, self-employed).
            dob (str, required): Date of birth of the customer in dd Month yyyy format.
            net_monthly_income (float, required): The customer's net monthly income.
            current_monthly_emi (float, required): The customer's current monthly financial obligations.
            roi (float, required): Rate of interest for the loan.""",
        ),
        StructuredTool.from_function(
            func=part_payment_tool,
            name="part_payment_tool",
            description="""Calculate the impact of part payment on the loan, including the reduction in tenure or EMI.
            Parameters:
            loan_outstanding (float,required): The current outstanding loan amount.
            tenure_months (int,required): Remaining tenure of the loan in months.
            roi (float,required): Rate of interest for the loan.
            part_payment_amount (float,required): The amount of part payment being made.
            current_emi (float,required): The current EMI amount.""",
        ),
        StructuredTool.from_function(
            func=emi_calc_tool,
            name="emi_calc_tool",
            description="""Calculate the EMI, interest, principal, or tenure of a loan based on the provided inputs.
            Parameters:
            principal (float,required): The principal loan amount.
            tenure_months (int,required): The loan tenure in months.
            roi (float,required): Rate of interest for the loan.
            emi (float,required): The equated monthly installment amount.
            percentage (float,required): Percentage adjustment for calculating EMI.""",
        ),
        StructuredTool.from_function(
            func=bts_calc_tool,
            name="bts_calc_tool",
            description="""Calculate the benefit of transfer of sanction (BTS) value based on the loan parameters.
            Parameters:
            sanction_amount (float,required): The sanctioned loan amount in integer.
            tenure_remaining_months (int,required): Remaining loan tenure in months.
            existing_roi (float,required): The current rate of interest on the loan.
            abhfl_roi (float,required): The proposed rate of interest after transfer.
            month_of_disbursement (str, required): The month of loan disbursement in %b-%y format (e.g., 'Aug-23').""",
        ),
        StructuredTool.from_function(
            func=step_up_calculator_tool,
            name="step_up_calculator_tool",
            description="""Calculate the step-up loan amount based on various income and loan parameters.
            Parameters:
            net_monthly_income (float, required): The net monthly income of the applicant.
            obligations (float, required): Monthly financial obligations.
            working_sector (str, required): The sector in which the applicant is employed.
            total_tenure_months (int, required): The total loan tenure in months.
            rate (float, required): The applicable rate of interest on the loan.
            primary_tenure_months (int, required): The tenure for the primary loan phase.""",
        ),
        StructuredTool.from_function(
            func=step_down_joint_income_calculator_tool,
            name="step_down_joint_income_calculator_tool",
            description="""Determine the total loan eligibility based on the son's and dad's financial profiles.
            Parameters:
            customer_type (str, required): Type of customer.
            salaried_son_dob (str, required): Son's date of birth.
            salaried_dad_dob (str, required): Dad's date of birth.
            salaried_son_current_net_monthly_income (float, required): Son's net monthly income.
            salaried_dad_current_net_monthly_income (float, required): Dad's net monthly income.
            salaried_dad_obligations (float, required): Dad's monthly financial obligations.
            salaried_son_obligations (float, required): Son's monthly financial obligations.
            salaried_son_ROI (float, required): Son's applicable loan rate of interest.
            salaried_dad_ROI (float, required): Dad's applicable loan rate of interest.""",
        ),
        StructuredTool.from_function(
            func=step_down_pension_income_calculator_tool,
            name="step_down_pension_income_calculator_tool",
            description="""Calculate the pension income eligibility based on various financial parameters.
            Parameters:
            dob_of_person (str, required): Date of birth of the person.
            monthly_income_from_salary (float, required): Monthly income from salary.
            monthly_income_from_pension (float, required): Monthly income from pension.
            salaried_obligations (float, required): Monthly financial obligations from salary.
            pension_obligations (float, required): Monthly financial obligations from pension.
            salaried_requested_tenure (int, required): Requested tenure for salaried income.
            pension_requested_tenure (int, required): Requested tenure for pension income.
            pension_ROI (float, required): Applicable rate of interest for the pension loan.
            salaried_ROI (float, required): Applicable rate of interest for the salaried loan.""",
        ),
        StructuredTool.from_function(
            func=logincheck_documents_tool,
            name="logincheck_documents_tool",
            description="""Get the list of required documents for login based on customer type, eligibility method, and income considerations.
            Parameters:
            employment (str, required): The employment type (e.g., 'salaried', 'self employed').
            eligibility_method (str, required): The eligibility method based on the customer type.
            rental_income (bool, required): Whether rental income is considered.
            other_income (bool, required): Whether other income is considered.""",
        ),
        StructuredTool.from_function(
            func=branches_list_tool,
            name="branches_list_tool",
            description="""Filter branch details of Housing Finance Companies (HFCs).
            Parameters:
            hfc (str, required): The HFC name to filter. Must be one of the following:
            Aadhar
            Aavas
            ABHFL
            Aptus
            Godrej
            Home First
            ICICI HFC
            IIFL
            PNB
            Shriram
            Tata Capital
            Vaastu
            state (str, required): The state name to filter by.(note: pass delhi / ncr insted of delhi in state and New Delhi in district)
            district (str, optional): The district name to filter by.
            pincode (int/str, optional): The pincode to filter by.""",
        ),
        # StructuredTool definition for this functionality
        StructuredTool.from_function(
            func=select_calculator,
            name="business_points_ils_qualification_tool",
            description="""This tool calculates business points and ILS qualification based on financial and product parameters. It ensures proper validation and retrieval of product details when needed.
            Parameters:
            secondary_lob (str, required): Line of Business identifier must be one of this four only ['ABSLI', 'ABML', 'ABHI', 'ABFL']
            product (str, required if available, else dynamically fetched): Full product name must be provided exactly as listed under the selected LOB. e.g. 'ABSLI - First Year Premium'
            secondary_business_cr (float, optional): Sent only when both lob and product are provided.: Secondary business value in crores.
            primary_ytd_abhfl_business (float, optional): Considered as the primary YTD business when ABHFL business data is available.
            """,
        ),
        StructuredTool.from_function(
            func=properties_faq_tool,
            name="get_qna_by_location_from_file",
            description="""Tool provide a details about properties faq based on locations
            Parameters:
            location_input (str, required): The location input to search for in the Q&A data.((Location input is required and must be specified by the user.))
            Location must be Following only:
            AP & Telangana
            Chhattisgarh
            Delhi NCR
            Gujarat
            Jharkhand
            Karnataka
            Kolkata & Siliguri
            MMR / Mumbai
            MP
            Odisha
            PCH
            Pune+ROM
            Rajasthan
            Tamil Nadu
            UP & UK
            ⚠ Note: Even if you know the full form (like Punjab/Chandigarh/Haryana for PCH, or Rest of Maharashtra for ROM), please enter the location exactly as given above — full forms will not be accepted.
            """,
        ),
        StructuredTool.from_function(
            func=location_cat_affordable_tool,
            name="location_cat_affordable_tool",
            description="""Tool provide a details about location category and cap in cr for affordable product program
            Parameters:
            location (str, required): Location should be city name not a state name.((Location input is required and must be specified by the user.))""",
        ),
    ]

    # Add product information tools
    product_names = [
        "Collateral",
        "salary_income_method",
        "cash_profit_method",
        "gross_turnover_method",
        "average_banking_program",
        "gross_profit_method",
        "gross_receipt_method",
        "gst_program",
        "pure_rental_program",
        "mortgage_product",
        "low_LTV_method",
        "credit_manager_assessed_income_program",
        "ABHFL_branch_categorization",
        "pragati_home_loan",
        "pragati_plus",
        "pragati_aashiyana",
        "general_purpose_loan",
        "micro_LAP",
        "micro_CF",
        "step_up",
        "step_down",
        "extended_tenure",
        "lease_rental_discounting",
        "express_balance_transfer_program",
        "prime_hl",
        "prime_lap",
        "priority_balance_transfer",
        "semi_fixed",
        "staff_loan_price",
        "power_pitches",
        "nri_assesment_criteria",
        "PMAY",
        "Competitors",
        "home_loan_ltv",
        "ftnr_queries",
        "select",
        "deviation_matrix",
        "credit_approval_authority",
        "Mitigation",
        "APF",
        "technical_deviation",
        "deviation_delegation_matrix_affordable",
    ]
    for product_name in product_names:
        tools.append(create_product_info_tool(product_name, abhfl_instance))

    return tools
