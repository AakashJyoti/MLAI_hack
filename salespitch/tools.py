import os
import json
from langchain.tools import StructuredTool


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
    tools = []

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
