from django.http import HttpResponse5
from django.shortcuts import render
import os
import pickle
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
import networkx as nx
from matplotlib import pyplot as plt




MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_model_wrapped.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


domain_knowledge = {
    'Account_status':
    {
        "description": "Status of existing checking account",
        "values": {
            "A11": "< 0 DM",
            "A12": "0 <= ... < 200 DM",
            "A13": ">= 200 DM",
            "A14": "no checking account",
        }
    },
    'Months':
    {
        "description": "Duration in months",
    },
    'Credit_history':
    {
        "description": "Credit history",
        "values": {
            "A30": "no credits taken/ all credits paid back duly",
            "A31": "all credits at this bank paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying off in the past",
            "A34": "critical account/ other credits existing (not at this bank)",
        }
    },
    'Purpose':
    {
        "description": "Purpose",
        "values": {
            "A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",
        }
    },
    'Credit_amount':
    {
        "description": "Credit amount",
    },
    'Savings':
    {
        "description": "Savings account/bonds",
        "values": {
            "A61": "< 100 DM",
            "A62": "100 <= ... < 500 DM",
            "A63": "500 <= ... < 1000 DM",
            "A64": ">= 1000 DM",
            "A65": "unknown/ no savings account",
        }
    },
    'Employment':
    {
        "description": "Present employment since",
        "values": {
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1 <= ... < 4 years",
            "A74": "4 <= ... < 7 years",
            "A75": ">= 7 years",
        }
    },
    'Installment_rate':
    {
        "description": "Installment rate in percentage of disposable income",
    },
    'Personal_status':
    {
        "description":"Personal status and sex",
        "values": {
            	"A91" : "male: divorced/separated",
                "A92" : "female: divorced/separated/married",
                "A93" : "male: single",
                "A94": "male: married/widowed",
                "A95":"female: single",
        }

    },
    "Other_debtors":
    {
        "description": "Other debtors/ guarantors",
        "values": {
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor",
        }
    },
    "Residence":
    {
        "description": "Present residence since",
    },
    "Property":
    {
        "description": "Property",
        "values": {
            "A121": "real estate",
            "A122": "if not real estate: building society savings agreement/ life insurance",
            "A123": "if not real estate/building society savings agreement/ life insurance:  car or other, not in Savings account/bonds",
            "A124": "unknown / no property",
        }
    },
    "Age":
    {
        "description": "Age in years",
    },
    "Other_installments":
    {
        "description": "Other installments",
        "values": {
            "A141": "bank",
            "A142": "stores",
            "A143": "none",
        }
    },
    "Housing":
    {
        "description": "Housing",
        "values": {
            "A151": "rent",
            "A152": "own",
            "A153": "for free",
        }
    },
    "Number_credits":
    {
        "description": "Number of existing credits at this bank",
    },
    "Job":
    {
        "description": "Job",
        "values": {
            "A171": "unemployed/ unskilled - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/ self-employed/ highly qualified employee/ officer",
        }
    },
    "Number_dependents":
    {
        "description": "Number of people being liable to provide maintenance for",
    },
    "Telephone":
    {
        "description": "Telephone",
        "values": {
            "A191": "none",
            "A192": "yes",
        }
    },
    "Foreign_worker":
    {
        "description": "Foreign worker",
        "values": {
            "A201": "yes",
            "A202": "no",
        }
    }


}


def identify_changes(counterfactual_instance, input_df):
    changes = {}
    for feature in input_df.columns:
        original_value = input_df[feature].iloc[0]
        new_value = counterfactual_instance[feature]
        if original_value != new_value:
            changes[feature] = (original_value, new_value)
    return changes

def create_explanation_graph(counterfactual_instance, prediction, input_df):
    changes = identify_changes(counterfactual_instance, input_df)
    G = nx.DiGraph()
    G.add_node("Explanation", description="Explanation of model prediction changes")
    G.add_node(prediction, description=f"Model predicted '{prediction}'")
    G.add_edge("Explanation", prediction, relation="ModelOutput")

    for feature, (original_value, new_value) in changes.items():
        G.add_node(f"{feature}", description = f"{feature}")
        feature_descr = domain_knowledge[feature]["description"]
        G.add_node(f"{feature}: {feature_descr}", description=feature_descr)
        G.add_edge(f"{feature}: {feature_descr}", f"{feature}", relation="Describes")
        G.add_node(f"{feature}: {original_value}", description=original_value)
        G.add_node(f"{feature}: {new_value}", description=new_value)
        G.add_edge("Explanation", f"{feature}", relation="Changed")
        G.add_edge(f"{feature}", f"{feature}: {original_value}", relation="Original Value")
        G.add_edge(f"{feature}", f"{feature}: {new_value}", relation="New Value")
        G.add_edge(f"{feature}: {original_value}", f"{feature}: {new_value}", relation="Changed To")
        if "values" in domain_knowledge[feature]:
            
            feature_val_original = domain_knowledge[feature]["values"][original_value]
            G.add_node(f"{feature}: {feature_val_original}", description=feature_val_original)
            G.add_edge(f"{feature}: {original_value}", f"{feature}: {feature_val_original}", relation="Describes")

            feature_val_new = domain_knowledge[feature]["values"][new_value]
            G.add_node(f"{feature}: {feature_val_new}", description=feature_val_new)
            G.add_edge(f"{feature}: {new_value}", f"{feature}: {feature_val_new}", relation="Describes")

            G.add_edge(f"{feature}: {feature_val_original}", f"{feature}: {feature_val_new}", relation="Changed To")

    
    return G

def depth_first_search(graph, node, visited, explanation):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        relation = graph.get_edge_data(node, neighbor)["relation"]
        if relation == "ModelOutput":
            explanation.insert(0, f"The prediction of '{neighbor}' is based on the following factors:")
        if relation == "Changed":
            explanation.append(f"The {node} has changed to {neighbor}")
        if relation == "Changed To":
            explanation.append(f"The {node} has changed to {neighbor}")
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited, explanation)
    return explanation

def generate_nle(graph, root_node):
    explanation = []
    explanation = depth_first_search(graph, root_node, set(), explanation)
    print(explanation)
    
    return "\n".join(explanation)

def index(request):
    prediction = None
    exp_df = pd.DataFrame()
    if request.method == "POST":
        try:
            Account_status = request.POST["Account_status"]
            Months =int(request.POST["Months"])
            Credit_history = request.POST["Credit_history"]
            Purpose = request.POST["Purpose"]
            Credit_amount = int(request.POST["Credit_amount"])
            Savings = request.POST["Savings"]
            Employment = request.POST["Employment"]
            Installment_rate = int(request.POST["Installment_rate"])
            Personal_status = request.POST["Personal_status"]
            Other_debtors = request.POST["Other_debtors"]
            Residence = int(request.POST["Residence"])
            Property = request.POST["Property"]
            Age = int(request.POST["Age"])
            Other_installments = request.POST["Other_installments"]
            Housing = request.POST["Housing"]
            Number_credits = int(request.POST["Number_credits"])
            Job = request.POST["Job"]
            Number_dependents = int(request.POST["Number_dependents"])
            Telephone = request.POST["Telephone"]
            Foreign_worker = request.POST["Foreign_worker"]

            input_data = {
                'Account_status': Account_status,
                'Months': Months,
                'Credit_history': Credit_history,
                'Purpose': Purpose,
                'Credit_amount': Credit_amount,
                'Savings': Savings,         
                'Employment': Employment,
                'Installment_rate': Installment_rate,
                'Personal_status': Personal_status,
                'Other_debtors': Other_debtors,
                'Residence': Residence,
                'Property': Property,
                'Age': Age,
                'Other_installments': Other_installments,
                'Housing': Housing,
                'Number_credits': Number_credits,
                'Job': Job,
                'Number_dependents': Number_dependents,
                'Telephone': Telephone,
                'Foreign_worker': Foreign_worker
            }
            # print(request.POST) 

            # print(input_data)  # Debugging step
            # print("Before creating DataFrame")  # Debugging step
            input_df = pd.DataFrame(input_data, index=[0])

            # print(input_df)
            prediction = model.predict(input_df)
            # print("Prediction made")  # Debugging step
            # print(prediction)
            prediction = "Creditworthy" if prediction[0] == 1 else "Non-Creditworthy"
            data = dice_ml.Data(features={'Account_status': ['A11', 'A12', 'A13', 'A14'],
                            'Months': [1, 90],
                            'Credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
                            'Purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
                            'Credit_amount': [10, 30000],
                            'Savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
                            'Employment': ['A71', 'A72', 'A73', 'A74', 'A75'],
                            'Installment_rate': [1, 4],
                            'Personal_status': ['A91', 'A92', 'A93', 'A94'],
                            'Other_debtors': ['A101', 'A102', 'A103'],
                            'Residence': [1, 4],
                            'Property': ['A121', 'A122', 'A123', 'A124'],
                            'Age': [19, 90],
                            'Other_installments': ['A141', 'A142', 'A143'],
                            'Housing': ['A151', 'A152', 'A153'],
                            'Number_credits': [1, 4],
                            'Job': ['A171', 'A172', 'A173', 'A174'],
                            'Number_dependents': [1, 5],
                            'Telephone': ['A191', 'A192'],
                            'Foreign_worker': ['A201', 'A202']    
                           },
                 outcome_name='target')
            model_exp = dice_ml.Model(model=model, backend="sklearn", model_type='classifier')
            exp = dice_ml.Dice(data, model_exp, method="random")
            exp_user = exp.generate_counterfactuals(input_df, total_CFs=2, desired_class="opposite")
            # exp_user.visualize_as_dataframe(show_only_changes=True)
            exp_df = exp_user.cf_examples_list[0].final_cfs_df
            print(exp_df)
            G = create_explanation_graph(exp_df.iloc[0], prediction, input_df)
            # pos = nx.spring_layout(G)
            # nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10)
            # edge_labels = nx.get_edge_attributes(G, 'relation')
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            # plt.show()
            
            nle = generate_nle(G, "Explanation")
            # print(nle)

            #print(identify_changes(exp_df.iloc[0], input_df))

            context = {
                'prediction': prediction,
                'nle': nle,
                
            }

            return render(request, "index.html", context)

            
        except Exception as e:
            print(e)
            return HttpResponse("Invalid input")
    



    return render(request, "index.html")

