Login_Check_list = {
	
	"Application Form" :[ {
			"Documents" : '''Details of Applicant & Co-applicants & Guarantor ( Including Personal & official E-Mail ID and address matching with the KYC Policy)''',
            "Advisory / Remarks" : """Unique Mobile Number of all Applicant,co-applicant(s) and Guarantor to be captured.""",
            "Eligibility" : {
                "Employment" : ["salaried" , "self-employed"]
            }
        },
        {
            "Documents" : '''Customer Profile''',
            "Advisory / Remarks" : """No particular format- Its should contain nature of business of self-employed borrowers""",
            "Eligibility" : {
                "Employment" : ["self-employed"]
            }

        }],

    "KYC (Applicant & Co-applicants & Guarantor)" :[ {
        "Documents" : '''Individual :- KYC matching with the KYC Policy
1. Adhar Card/Passport/Voter ID/ Driving Licence ( for NRI - Passport is mandatory)
2. PAN / form 60 in case of non financials Applicant''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried" , "self-employed"]
            }
    },
    {
    "Documents" : '''Non-Individual :
KYC document (PAN & AOA/MOA/Partnership Deed/LLP Deed/HUF Deed/Trust Deed & Utility Bill)''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["self-employed"]
    }
    },
    {
    "Documents" : '''Business Proof (Shop Act/GST Certificate/Udyam Adhar/IEC Code/FDA Licence/Certificate of Incorporation)''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["self-employed"]
    }
    },
    {
    "Documents" : '''Photocopy of Valid work Permit / Employment Visa''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                "Eligibility Method" : ["nri bank salaried"]
    }
    },
    ],

    "Financial Documents" :[
        {
            "Documents" : '''Form 16 / ITR of salaried applicant or W2 Form in case of NRI Cases (US, Canada)''',
            "Advisory / Remarks" : """Document for NRI applicants working in USA / Canada updated""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                "Eligibility Method" : ["nri bank salaried" , "bank salaried"]
    }
        },
        {
            "Documents" : '''Job Vintage Proof of last 3 years (Appointment letter/Experience Letter)''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                "Eligibility Method" : ["nri bank salaried"]
    }
        },
        {
            "Documents" : '''Salary Slip /Salary Certificate of last 3 months ''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                
    }
        },
        {
            "Documents" : '''Last 2 years ITR of Financial applicants (Saral Page, COI, P&L and B/s) -Fetched through Online integration -  Finfort''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                "Other Income to be considered" : "Yes",

            }
        },
        
        {
            "Documents" : '''Oveseas Bureau Report ( Only for NRI cases . Excluding Merchnant Navy)''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                "Eligibility Method" : ["nri bank salaried"]

            }
        },
        {
            "Documents" : '''2 Years Income Proof & CDC for Merchant Navy Salaried (NRI) including ITR''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                "Eligibility Method" : ["nri bank salaried"]

            }
        },
        {
            "Documents" : '''SENP Case - 
Latest Digital ITR and Financials throught Online fetch 
( Exception : In Program variant of ABB+ GST + Rental + Low LTV  - Even Physical ITR for latest year will suffice as per Program Norm)''',
            "Advisory / Remarks" : """Online fetch to be done through Finverse only & to be completed by customer before CPH stage""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["cash profit method","gross turnover","gross receipt","gross profit","abb" , "gst" , "low ltv" , "pure rental" , "lease rental discounting" , "express bt"]
            }
        },
        {
            "Documents" : '''SENP Case
Balance sheet for last 2 years with Annexures and Audit Report for cases above 1 Cr''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["lease rental discounting" , "express bt" , "micro cf/builder lap"]

            }
        },
        {
            "Documents" : '''For exposure upto ₹1 Cr, Last 12 months GST returns required''',
            "Advisory / Remarks" : """1. Online Fetch to be done through Finverse only 
2. Customer Leg to be completed before CPH stage
3. If the case is under GST Program online fetch to be done for all  cases irrespective of the tickt size""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["lease rental discounting" , "gst"]

            }
        },
        {
            "Documents" : '''For Exposure > ₹1 Cr - Online GST fetch from GST Finfort Through Finverse''',
            "Advisory / Remarks" : """1. Online Fetch to be done through Finverse only 
2. Customer Leg to be completed before CPH stage
3. If the case is under GST Program online fetch to be done for all  cases irrespective of the tickt size""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["cash profit method", "gross turnover", "gst", "gross receipt", "gross profit"]
            }
        },
    ],
    "Cibil / Experian / Equifax Details" : [
        {
            "Documents" : '''Debt Chart Sheet - in attached format''',
            "Advisory / Remarks" : """In case of BT transaction- validation of BT Track for 12 months to be done post login,but before sanction.""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
            }
        },
        {
            "Documents" : '''Last 6 months banking reflecting existing obligations''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
            }
        },
        {
            "Documents" : '''In case of Express BT Last 12 months banking /or SOA for Repayment''',
            "Advisory / Remarks" : """Document with total EMI paid, security details, applicant & co applicant required if not available in SOA""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["express bt"]
            }
        },
        {
            "Documents" : '''Debt Chart Sheet - in attached format''',
            "Advisory / Remarks" : """In case of BT transaction- validation of BT Track for 12 months to be done post login,but before sanction.""",
            "Eligibility" : {
                "Employment" : ["salaried"]
            }
        },
        {
            "Documents" : '''In case of Express BT/Priority BT Last 12 months banking /or SOA for Repayment''',
            "Advisory / Remarks" : """Document with total EMI paid, security details, applicant & co applicant required if not available in SOA""",
            "Eligibility" : {
                "Employment" : ["salaried"]
            }
        },
    ],
    "Banking Details" : [
        {
            "Documents" : '''6 Months Banking of Financial Applicant in PDF''',
            "Advisory / Remarks" : """Banking fetch through Account Aggregator whereever applicable and banker is participating in account aggregator""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["cash profit method","gross turnover","gross receipt","gross profit", "low ltv" , "pure rental" , "lease rental discounting" , "express bt" ,"micro cf/builder lap" , "cm aip" ]


            }
        },
        {
            "Documents" : '''12 Months Banking of Financial Applicant in PDF for Business Account.''',
            "Advisory / Remarks" : """Banking fetch through Account Aggregator whereever applicable and banker is participating in account aggregator""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["abb" , "gst"]
            }
        },
        {
            "Documents" : '''6 Months Banking of Financial Applicant- NRE/NRO Account ( For NRI cases only)''',
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                "Eligibility Method" : ["nri bank salaried"]
            }
        },
        {
            "Documents" : '''Salary Credit banking of last 6 months / If cash salaried then saving account to be doucmented- AA fetch /Finfort through Finverse to be done for relevant Bank ''',
            "Advisory / Remarks" : """Banking to reflect existing loan EMIs""",
            "Eligibility" : {
                "Employment" : ["salaried"],
            }
        },
        {
            "Documents" : '''SOA for Repayment (incase of Express BT only) ''',
            "Advisory / Remarks" : """Document with total EMI paid, security details, applicant & co applicant required if not available in SOA""",
            "Eligibility" : {
                "Employment" : ["salaried"],
                "Eligibility Method" : ["express bt" ,"priority bt"]
            }
        },

    ],

    "Property Document - Property Identified case for Technical initiation (Mandatory during Login for all case under BT ,LAP, Micro CF cases and all other cases above 3 Cr )":
    [  {
            "Documents" : '''Purchase Transaction or LAP or BT Transaction(Latest Ownership Agreement copy & Sanction Plan & OC/CC Copy/ BUC Copy) ''',
            "Advisory / Remarks" : """Draft Agreement/ATS in case available""",
            "Eligibility" : {
                "Employment" : ["salaried" , "self-employed"]
            }
        },
    ],
    "others" :[
        {
            "Documents" : "Rental Agreement and validation of Rental from Bank statement ( Mandaotry for LRD products , it is required for other method if rental income considered",
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["salaried" ,"self-employed"],
                "Rental income to be considered" : "Yes"
            }
        },
        {
            "Documents" : "LOI and LRD agreement",
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["lease rental discounting"]                
            }
        },
        {
            "Documents" : "Enterprise date in the standard format for Micro CF case",
            "Advisory / Remarks" : """""",
            "Eligibility" : {
                "Employment" : ["self-employed"],
                "Eligibility Method" : ["micro cf/builder lap"]               
            }
        },

    ]
}

import json
# "Eligibility Method" : ["Cash Profit Method","Gross Turnover","Gross Receipt","Gross Profit","ABB" , "GST" , "Low LTV" , "Pure Rental" , "lease rental discounting" , "express BT" ,"Micro CF/Builder LAP" , "CM AIP" ]
def logincheck_documents(employment, eligibility_method=None, rental_income=False, other_income=False):
    """
    Filters the documents and remarks based on the inputs provided.

    :param employment: Employment type (salaried/self-employed)
    :param eligibility_method: Specific eligibility method if applicable
    :param rental_income: Whether rental income is considered (boolean)
    :param other_income: Whether other income is considered (boolean)
    :return: Filtered documents and remarks in JSON format
    """
    result = []
    if employment == 'self employed':
        employment = "self-employed"

    print(employment , eligibility_method , rental_income)
    for category, items in Login_Check_list.items():
        for item in items:
            # Check employment type
            if employment not in item["Eligibility"]["Employment"]:
                continue
            
            # Check eligibility method if specified
            if "Eligibility Method" in item["Eligibility"]:
                if eligibility_method and eligibility_method not in item["Eligibility"]["Eligibility Method"]:
                    continue

            # Check rental income condition
            if "Rental income to be considered" in item["Eligibility"]:
                if rental_income:
                    if item["Eligibility"]["Rental income to be considered"] != "Yes":
                        continue
                else:
                    if item["Eligibility"]["Rental income to be considered"] == "Yes":
                        continue

            # Check other income condition (specific to salaried)
            # Check other income condition (must match exactly if specified)
            if "Other Income to be considered" in item["Eligibility"]:
                if other_income:
                    if item["Eligibility"]["Other Income to be considered"] != "Yes":
                        continue
                else:
                    if item["Eligibility"]["Other Income to be considered"] == "Yes":
                        continue
                        

            # Append matching items to result
            result.append({
                "Category": category,
                "Documents": item["Documents"],
                "Advisory / Remarks": item["Advisory / Remarks"]
            })

    return json.dumps(result, indent=4)

# Example usage
# employment = "salaried"
# eligibility_method = "bank salaried"
# rental_income = True
# other_income = False

# filtered_output = logincheck_documents(employment, eligibility_method, rental_income, other_income)
# print(filtered_output)