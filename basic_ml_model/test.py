from chembl_webresource_client.new_client import new_client
activities = new_client.activity.filter( standard_type__in=["IC50"], standard_units="nM")
