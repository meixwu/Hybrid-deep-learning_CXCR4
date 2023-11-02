from chembl_webresource_client.new_client import new_client
import pandas as pd

def filter_activities(target_id, standard_types, max_standard_value):

    try:
        activities = new_client.activity.filter(
            target_chembl_id=target_id,
            standard_type__in=standard_types,
            standard_value__lt=max_standard_value,
            standard_units="nM"  # Filter by unit (nanomolar)
        ).only(
            "molecule_chembl_id",
            "standard_type",
            "standard_relation",
            "standard_value",
            "standard_units",
        )
        return activities

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []



############################################################################################################


# Define the target ChEMBL ID, standard types, and maximum standard value
target_id = "CHEMBL2107"  #CXCR4 Homo sapiens
standard_types = ["EC50", "IC50", "Ki"]
max_standard_value = 10000  # Maximum standard value in 10 microM

filtered_activities = filter_activities(target_id, standard_types, max_standard_value)

filtered_activities = pd.DataFrame(filtered_activities)



print(filtered_activities.head())