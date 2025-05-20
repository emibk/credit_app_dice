from django.http import HttpResponse
from django.shortcuts import render
import os
import pickle
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
import pygraphviz as pgv
from matplotlib import pyplot as plt

# Load first model
MODEL_PATH1 = os.path.join(os.path.dirname(__file__), "catboost_model_1.pkl")
with open(MODEL_PATH1, "rb") as f:
    model1 = pickle.load(f)




domain_knowledge = {
    'Account_status': {
        "description": "Status of existing checking account",
        "values": {
            "A11": "< 0 DM",
            "A12": "0 <= ... < 200 DM",
            "A13": ">= 200 DM",
            "A14": "no checking account",
        }
    },

    'Months':{
        "description": "Duration in months",
    },

    'Credit_history':{
        "description": "Credit history",
        "values": {
            "A30": "no credits taken/ all credits paid back duly",
            "A31": "all credits at this bank paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying off in the past",
            "A34": "critical account/ other credits existing (not at this bank)",
        }
    },

    'Purpose':{
        "description": "Loan Purpose",
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

    'Credit_amount':{
        "description": "Credit amount",
    },

    'Savings':{
        "description": "Savings account/bonds",
        "values": {
            "A61": "< 100 DM",
            "A62": "100 <= ... < 500 DM",
            "A63": "500 <= ... < 1000 DM",
            "A64": ">= 1000 DM",
            "A65": "unknown/ no savings account",
        }
    },

    'Employment':{
        "description": "Present employment since",
        "values": {
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1 <= ... < 4 years",
            "A74": "4 <= ... < 7 years",
            "A75": ">= 7 years",
        }
    },

    'Installment_rate':{
        "description": "Installment rate in percentage of disposable income",
    },

    'Personal_status':{
        "description":"Personal status and sex",
        "values": {
            	"A91" : "male: divorced/separated",
                "A92" : "female: divorced/separated/married",
                "A93" : "male: single",
                "A94": "male: married/widowed",
                "A95":"female: single",
        }

    },

    "Other_debtors":{
        "description": "Other debtors/ guarantors",
        "values": {
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor",
        }
    },

    "Residence":{
        "description": "Present residence since",
    },

    "Property":{
        "description": "Property type",
        "values": {
            "A121": "real estate",
            "A122": "if not real estate: building society savings agreement/ life insurance",
            "A123": "if not real estate/building society savings agreement/ life insurance:  car or other, not in Savings account/bonds",
            "A124": "unknown / no property",
        }
    },

    "Age":{
        "description": "Age in years",
    },

    "Other_installments":{
        "description": "Other installments",
        "values": {
            "A141": "bank",
            "A142": "stores",
            "A143": "none",
        }
    },

    "Housing":{
        "description": "Housing situation",
        "values": {
            "A151": "rent",
            "A152": "own",
            "A153": "for free",
        }
    },

    "Number_credits":{
        "description": "Number of existing credits at this bank",
    },

    "Job":{
        "description": "Employment status",
        "values": {
            "A171": "unemployed/ unskilled - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/ self-employed/ highly qualified employee/ officer",
        }
    },

    "Number_dependents":{
        "description": "Number of people being liable to provide maintenance for",
    },

    "Telephone":{
        "description": "Telephone registration",
        "values": {
            "A191": "none",
            "A192": "yes",
        }
    },

    "Foreign_worker":{
        "description": "Foreign worker",
        "values": {
            "A201": "yes",
            "A202": "no",
        }
    },

    "prediction": {
        "values": {
            "1": "Creditworthy",
            "0": "Non-Creditworthy",
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

def create_explanation_graph(counterfactual_instances, prediction, input_df):
    no_cf = len(counterfactual_instances)
    G = pgv.AGraph(directed=True)
    G.add_node("Explanation", label = "Explanation")
    G.add_node(prediction, label = f"Model Output: {prediction}")
    G.add_edge("Explanation", prediction, label = "Model Prediction")

    pred_meaning = domain_knowledge["prediction"]["values"][str(prediction)]
    G.add_node(pred_meaning, label = pred_meaning)
    G.add_edge(prediction, pred_meaning, label = "Prediction Meaning")

    G.add_node("Current Features", label = "Current Features")
    G.add_edge("Explanation", "Current Features", label = "Current Situation")

    for i in range(no_cf):
        changes = identify_changes(counterfactual_instances.iloc[i], input_df)

        G.add_node(f"Counterfactual{i+1} Features", label = f"Counterfactual{i+1} Features")
        G.add_edge("Explanation", f"Counterfactual{i+1} Features", label = "Alternative Situation")


        for feature, (original_value, new_value) in changes.items():
            G.add_node(f"Current {feature}", label= feature)
            G.add_edge("Current Features", f"Current {feature}", label="Current Feature")

            feature_descr = domain_knowledge[feature]["description"]
            G.add_node(f"Current {feature_descr}", label=feature_descr)
            G.add_edge(f"Current {feature}", f"Current {feature_descr}", label = "Description")


            G.add_node(f"New cf{i+1} {feature}", label=feature)
            G.add_edge(f"Counterfactual{i+1} Features", f"New cf{i+1} {feature}", label="Changed Feature")
            G.add_node(f"New cf{i+1} {feature_descr}", label=feature_descr)
            G.add_edge(f"New cf{i+1} {feature}", f"New cf{i+1} {feature_descr}", label = "Description")

            G.add_node(f"Current {original_value}", label=original_value)
            G.add_edge(f"Current {feature}", f"Current {original_value}", label="Current Value")
            G.add_node(f"New cf{i+1} {new_value}", label=new_value)
            G.add_edge(f"New cf{i+1} {feature}", f"New cf{i+1} {new_value}", label="Changed Value")

           
            if "values" in domain_knowledge[feature]:
                original_val_descr = domain_knowledge[feature]["values"][original_value]
                G.add_node(f"Current {original_val_descr}", label = original_val_descr)
                G.add_edge(f"Current {original_value}", f"Current {original_val_descr}", label="Description")

                new_val_descr = domain_knowledge[feature]["values"][new_value]
                G.add_node(f"New cf{i+1} {new_val_descr}", label = new_val_descr)
                G.add_edge(f"New cf{i+1} {new_value}", f"New cf{i+1} {new_val_descr}", label="Description")
    return G


def depth_first_search(graph, node, visited, explanation, idx = 0):
    visited.add(node)
    print(node)
    print(graph.neighbors(node))
    for neighbor in graph.neighbors(node):
        if graph.has_edge(node, neighbor):
            print(f"Edge from {node} to {neighbor}")
        else: 
            continue
        relation = graph.get_edge(node, neighbor).attr['label']

        if relation == "Prediction Meaning":
            explanation.insert(0, f"The prediction is '{neighbor}' <br><br>")
        else:
            if relation == "Current Situation":
                explanation.append(f"<br> Current Situation: <br><ul>")
            if relation == "Alternative Situation":
                explanation.append(f"</ul><br> Alternative Situation: <br><ul>")
            if len(graph.out_edges(neighbor)) == 0:
                idx += 1 
                neighbor_label = graph.get_node(neighbor).attr['label']
               
                if idx % 2 == 0:
                    explanation.append(f"{neighbor_label} </li>")
                if idx % 2 != 0:
                    explanation.append(f"<li>{neighbor_label}: ")
        if neighbor not in visited:
            depth_first_search(graph, neighbor, visited, explanation, idx)
    return explanation

def generate_nle(graph, root_node):
    explanation = []
    explanation = depth_first_search(graph, root_node, set(), explanation)
    print(explanation)
    
    return "".join(explanation)

def index(request):
    prediction = None
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
            
            input_df = pd.DataFrame(input_data, index=[0])

            prediction = model1.predict(input_df)
 
            data = dice_ml.Data(features={'Account_status': ['A11', 'A12', 'A13', 'A14'],
                            'Months': [4, 72],
                            'Credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
                            'Purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
                            'Credit_amount': [250, 18424],
                            'Savings': ['A61', 'A62', 'A63', 'A64', 'A65'],
                            'Employment': ['A71', 'A72', 'A73', 'A74', 'A75'],
                            'Installment_rate': [1, 4],
                            'Personal_status': ['A91', 'A92', 'A93', 'A94'],
                            'Other_debtors': ['A101', 'A102', 'A103'],
                            'Residence': [1, 4],
                            'Property': ['A121', 'A122', 'A123', 'A124'],
                            'Age': [19, 75],
                            'Other_installments': ['A141', 'A142', 'A143'],
                            'Housing': ['A151', 'A152', 'A153'],
                            'Number_credits': [1, 4],
                            'Job': ['A171', 'A172', 'A173', 'A174'],
                            'Number_dependents': [1, 2],
                            'Telephone': ['A191', 'A192'],
                            'Foreign_worker': ['A201', 'A202']    
                           },
                 outcome_name='target')
            
            fixed_features = ["Credit_history", "Personal_status", "Age", "Number_dependents", "Foreign_worker"]
            features_to_vary = list(set(data.feature_names) - set(fixed_features))
            

            m = dice_ml.Model(model=model1, backend="sklearn", model_type='classifier')
            exp = dice_ml.Dice(data, m, method="random")
            e = exp.generate_counterfactuals(input_df, total_CFs=1, desired_class="opposite", features_to_vary=features_to_vary)
            
            e_df = e.cf_examples_list[0].final_cfs_df

            
            G = create_explanation_graph(e_df, prediction[0], input_df)
            
            nle = generate_nle(G, "Explanation")
            context = {}

            if prediction[0] == 1:
                context = {'exp': "Creditworthy"}
            else:
                context = {'exp': nle}

            return render(request, "index.html", context)

            
        except Exception as e:
            print(e)
            return HttpResponse("Invalid input")
    
    return render(request, "index.html")

