
Eligibility_Method = 	"GST"
Rental_income_to_be_considered = "Yes"


def self_employed(Eligibility_Method, Rental_income_to_be_considered):
    Application_Form_Details_of_Applicant = True
    Application_Form_Customer_Profile = True

    KYC_individual = True
    KYC_non_individual = True
    KYC_business_proof = True

    # Financial_Documents_SENP_Case_1
    if Eligibility_Method == "Micro CF/Builder LAP" or Eligibility_Method == "CM AIP":
        Financial_Documents_SENP_Case_1 = False
    else:
        Financial_Documents_SENP_Case_1 = True

    # Financial_Documents_SENP_Case_2
    yes_conditions_for_case_2 = [
        "Cash Profit Method",
        "Gross Turnover",
        "Gross Receipt",
        "Gross Profit",
        "Lease Rental Discounting",
        "Micro CF/Builder LAP",
        "Express BT"
    ]
    Financial_Documents_SENP_Case_2 = True if Eligibility_Method in yes_conditions_for_case_2 else False

    #Financial_Documents_For_exposure_upto_1_Cr
    if Eligibility_Method == "Lease Rental Discounting" or Eligibility_Method == "GST":
        Financial_Documents_For_exposure_upto_1_Cr = True
    else:
        Financial_Documents_For_exposure_upto_1_Cr = False

    #Financial_Documents_For_exposure_more_than_1_Cr
    yes_conditions_for_exposure = [
        "Cash Profit Method",
        "Gross Turnover",
        "GST",
        "Gross Receipt",
        "Gross Profit"
    ]

    # Check if B2_value matches any of the conditions
    Financial_Documents_For_exposure_more_than_1_Cr = True if Eligibility_Method in yes_conditions_for_exposure else False

    # Cibil_Experian_Equifax
    Cibil_Experian_Equifax_Debt_Chart_Sheet = True
    Cibil_Experian_Equifax_Last_6_months_banking_reflecting = True
    Cibil_Experian_Equifax_In_case_of_Express_BT = True if Eligibility_Method in ["Priority BT", "Express BT"] else False

    # Banking Details
    Banking_Details_6_Months_Banking_of_Financial_Applicant = False if Eligibility_Method in ["ABB", "GST"] else True
    Banking_Details_12_Months_Banking_of_Financial_Applicant = True if Eligibility_Method in ["ABB", "GST"] else False

    # Property Document
    Property_Document_Purchase_Transaction = True

    # Others
    Others_Rental_Agreement_and_validation_of_Rental = True if Rental_income_to_be_considered == "Yes" else False
    Others_LOI_and_LRD_agreement = True if Eligibility_Method == "Lease Rental Discounting" else False
    Others_Enterprise_date = True if Eligibility_Method == "Micro CF/Builder LAP" else False


    print(Application_Form_Details_of_Applicant)
    print(Application_Form_Customer_Profile)

    print(KYC_individual)
    print(KYC_non_individual)
    print(KYC_business_proof)

    print(Financial_Documents_SENP_Case_1)
    print(Financial_Documents_SENP_Case_2)
    print(Financial_Documents_For_exposure_upto_1_Cr)
    print(Financial_Documents_For_exposure_more_than_1_Cr)

    print(Cibil_Experian_Equifax_Debt_Chart_Sheet)
    print(Cibil_Experian_Equifax_Last_6_months_banking_reflecting)
    print(Cibil_Experian_Equifax_In_case_of_Express_BT)

    print(Banking_Details_6_Months_Banking_of_Financial_Applicant)
    print(Banking_Details_12_Months_Banking_of_Financial_Applicant)

    print(Property_Document_Purchase_Transaction)

    print(Others_Rental_Agreement_and_validation_of_Rental)
    print(Others_LOI_and_LRD_agreement)
    print(Others_Enterprise_date)


self_employed(Eligibility_Method, Rental_income_to_be_considered)


