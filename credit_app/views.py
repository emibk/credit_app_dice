from django.http import HttpResponse
from django.shortcuts import render
import os
import pickle
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
import networkx as nx
import pygraphviz as pgv
from matplotlib import pyplot as plt

# Load first model
MODEL_PATH1 = os.path.join(os.path.dirname(__file__), "catboost_model_1.pkl")
with open(MODEL_PATH1, "rb") as f:
    model1 = pickle.load(f)

#Load second model
MODEL_PATH2 = os.path.join(os.path.dirname(__file__), "catboost_model_2.pkl")
with open(MODEL_PATH2, "rb") as f:
    model2 = pickle.load(f)


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
    G = nx.DiGraph()
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
                G.add_edge(f"Current {original_value}", f"Current {original_val_descr}", label="Describes")

                new_val_descr = domain_knowledge[feature]["values"][new_value]
                G.add_node(f"New cf{i+1} {new_val_descr}", label = new_val_descr)
                G.add_edge(f"New cf{i+1} {new_value}", f"New cf{i+1} {new_val_descr}", label="Describes")
    return G

'''

def create_explanation_graph(counterfactual_instances, prediction, input_df):
    changes1 = identify_changes(counterfactual_instances.iloc[0], input_df)
    changes2 = identify_changes(counterfactual_instances.iloc[1], input_df)
   
   
    G = nx.DiGraph()

    G.add_node("Explanation", label = "Explanation")
    G.add_node(prediction, label = f"Model Output: {prediction}")
    G.add_edge("Explanation", prediction, label = "Model Prediction")

    pred_meaning = domain_knowledge["prediction"]["values"][str(prediction)]
    G.add_node(pred_meaning, label = pred_meaning)
    G.add_edge(prediction, pred_meaning, label = "Prediction Meaning")

    G.add_node("Current Features", label = "Current Features")
    G.add_edge("Explanation", "Current Features", label = "Current Situation")

    G.add_node("Counterfactual1 Features", label = "Counterfactual1 Features")
    G.add_edge("Explanation", "Counterfactual1 Features", label = "Alternative Situation")

    G.add_node("Counterfactual2 Features", label = "Counterfactual2 Features")
    G.add_edge("Explanation", "Counterfactual2 Features", label = "Alternative Situation")

    

    for feature, (original_value, new_value) in changes1.items():
        G.add_node(f"Current {feature}", label= feature)
        G.add_edge("Current Features", f"Current {feature}", label="Current Feature")

        feature_descr = domain_knowledge[feature]["description"]
        G.add_node(f"Current {feature_descr}", label=feature_descr)
        G.add_edge(f"Current {feature}", f"Current {feature_descr}", label = "Description")


        G.add_node(f"New cf1 {feature}", label=feature)
        G.add_edge("Counterfactual1 Features", f"New cf1 {feature}", label="Changed Feature")
        G.add_node(f"New cf1 {feature_descr}", label=feature_descr)
        G.add_edge(f"New cf1 {feature}", f"New cf1 {feature_descr}", label = "Description")

        G.add_node(f"Current {original_value}", label=original_value)
        G.add_edge(f"Current {feature}", f"Current {original_value}", label="Current Value")
        G.add_node(f"New cf1 {new_value}", label=new_value)
        G.add_edge(f"New cf1 {feature}", f"New cf1 {new_value}", label="Changed Value")

       
        if "values" in domain_knowledge[feature]:
            original_val_descr = domain_knowledge[feature]["values"][original_value]
            G.add_node(f"Current {original_val_descr}", label = original_val_descr)
            G.add_edge(f"Current {original_value}", f"Current {original_val_descr}", label="Describes")

            new_val_descr = domain_knowledge[feature]["values"][new_value]
            G.add_node(f"New cf1 {new_val_descr}", label = new_val_descr)
            G.add_edge(f"New cf1 {new_value}", f"New cf1 {new_val_descr}", label="Describes")

    for feature, (original_value, new_value) in changes2.items():
        G.add_node(f"Current {feature}", label= feature)
        G.add_edge("Current Features", f"Current {feature}", label="Current Feature")
        feature_descr = domain_knowledge[feature]["description"]
        G.add_node(f"Current {feature_descr}", label=feature_descr)
        G.add_edge(f"Current {feature}", f"Current {feature_descr}", label = "Description")


        G.add_node(f"New cf2 {feature}", label=feature)
        G.add_edge("Counterfactual2 Features", f"New cf2 {feature}", label="Changed Feature")
        G.add_node(f"New cf2 {feature_descr}", label=feature_descr)
        G.add_edge(f"New cf2 {feature}", f"New cf2 {feature_descr}", label = "Description")

        G.add_node(f"Current {original_value}", label=original_value)
        G.add_edge(f"Current {feature}", f"Current {original_value}", label="Current Value")
        G.add_node(f"New cf2 {new_value}", label=new_value)
        G.add_edge(f"New cf2 {feature}", f"New cf2 {new_value}", label="New Value")

       
        if "values" in domain_knowledge[feature]:
            original_val_descr = domain_knowledge[feature]["values"][original_value]
            G.add_node(f"Current {original_val_descr}", label = original_val_descr)
            G.add_edge(f"Current {original_value}", f"Current {original_val_descr}", label="Describes")

            new_val_descr = domain_knowledge[feature]["values"][new_value]
            G.add_node(f"New cf2 {new_val_descr}", label = new_val_descr)
            G.add_edge(f"New cf2 {new_value}",  f"New cf2 {new_val_descr}", label="Describes")  


    return G
'''
def depth_first_search(graph, node, visited, explanation, idx = 0):
    visited.add(node)
    for neighbor in graph.neighbors(node):
        relation = graph.get_edge_data(node, neighbor)["label"]

        if relation == "Prediction Meaning":
            explanation.insert(0, f"The prediction is '{neighbor}' <br><br>")
        else:
            if relation == "Current Situation":
                explanation.append(f"<br> Current Situation: <br><ul>")
            if relation == "Alternative Situation":
                explanation.append(f"</ul><br> Alternative Situation: <br><ul>")
            if graph.out_degree(neighbor) == 0:
                idx += 1 
                neighbor_label = graph.nodes[neighbor]["label"]
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
            # print(request.POST) 

            # print(input_data)  # Debugging step
            # print("Before creating DataFrame")  # Debugging step
            input_df = pd.DataFrame(input_data, index=[0])

            # print(input_df)
            prediction = model1.predict(input_df)
            # print("Prediction made")  # Debugging step
            # print(prediction)
            # prediction = "Creditworthy" if prediction[0] == 1 else "Non-Creditworthy"
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
            e = exp.generate_counterfactuals(input_df, total_CFs=2, desired_class="opposite", features_to_vary=features_to_vary)
            
            # e.visualize_as_dataframe(show_only_changes=True)
            e_df = e.cf_examples_list[0].final_cfs_df

            # Saving counterfactuals as images
            '''
            
            changed_cols= []
            input_pred = input_df.copy()
            input_pred['target'] = prediction[0]
            exp1 = e_df.iloc[[0]]

          
    

            for idx, row in exp1.iterrows():
                print(row.index)
                print(input_pred.iloc[0].index)
                
                changes = row != input_pred.iloc[0]
                
                changed_cols.append(row[changes])
            print("DEBUGGING")
            changes_e = pd.DataFrame(changed_cols)
            intersection = exp1.columns.intersection(changes_e.columns)
            changes_e = pd.concat([input_pred[intersection].iloc[[-1]], changes_e], ignore_index=True)
            changes_e = changes_e.fillna("-")
            changes_e.index = ["Original", "Counterfactual 1"]
            changes_e = changes_e.style.apply(lambda x: ["background: red" if x[col] != "-" and 
                                                           x.name != "Original" else "" for col in x.index], axis = 1)
            print("DEBUGGING2")
            import dataframe_image as dfi
            dfi.export(changes_e, 'counterfactuals_highlighted.png')
            '''





            
            # G = create_explanation_graph(e_df, prediction[0], input_df)
            G = create_explanation_graph(e_df, prediction[0], input_df)
            pos = nx.spring_layout(G) 
            
            edge_labels = nx.get_edge_attributes(G, 'label')
            
            nx.draw(G, pos, with_labels=True, node_size = 900, font_size = 10)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            plt.savefig("explanation_graph_nx.png")
            plt.close()

            A = nx.nx_agraph.to_agraph(G)
            A.layout(prog='dot')
            A.draw('explanation_graph.pdf')
            
            
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

