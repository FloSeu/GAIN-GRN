## scripts/request_utils.py
#   Handles and extracts HTTP requests from the UniProtKB database and AlphafoldDB

import requests, os
import gaingrn.scripts.io

# requests

def request_uniprot(accession:str):
    # Make the API call to retrieve the protein information in JSON format using the accession number as query
    response = requests.get(f"https://rest.uniprot.org/uniprotkb/search?query=accession:{accession}&format=json")
    # Check if the API call was successful (HTTP status code 200 means success)
    if response.status_code == 200:
        # decode via json(), only return "results" (no other entry anyway.)
        return response.json()["results"]
    
    print("Failed to retrieve protein information.")
    return None

def get_uniprot_seq(uniprot_info, uniprot_accession, c_end=None):
    if len(uniprot_info) > 1: # If more than one match, find the correct one in via accession. More of a failsafe
        for entry in uniprot_info:
            if entry['primaryAccession'] == uniprot_accession:
                target_info = entry
    else:
        target_info = uniprot_info[0]

    if c_end is not None:
        return target_info["sequence"]["value"][:c_end] # zero_indexed!
    
    return target_info["sequence"]["value"]

def extract_gain_end(uniprot_accession:str, uniprot_info:list):
    # Takes the result of the API call (from response.json) and tries to extract the GPS record (ProSite Rule), returns the end value
    # This might in the future be updated to include set GAIN domain boundaries, if UniProtKB has then been updated with them.
    # Returns also the sequence until including the gps_end residue number
    if len(uniprot_info) > 1: # If more than one match, find the correct one in via accession. More of a failsafe
        for entry in uniprot_info:
            if entry['primaryAccession'] == uniprot_accession:
                target_info = entry
    else:
        target_info = uniprot_info[0]
    try: 
        protein_name = target_info['proteinDescription']['recommendedName']['fullName']['value']
    except: 
        protein_name='unnamed_protein'
    # Parse the entry features and find the "GPS domain" (which is of course not a domain...)
    if 'features' not in target_info.keys():
        print("FEATURES entry not found in target UniProt entry. Continuing to manual GPS/GAIN boundary specification.")
        return None, None, protein_name
    
    for feature in target_info["features"]:
        if "description" in feature.keys() and feature["description"].upper() == "GPS":
            end_res = feature["location"]["end"]["value"]
            print(f"[NOTE] Found GPS entry in the UniProtKB accession entry ending at residue {end_res}")
            truncated_sequence = target_info["sequence"]["value"][:end_res] # zero_indexed!
            return end_res, truncated_sequence, protein_name
        
    print("No GPS entry found. Continuing to manual GPS/GAIN boundary specification.")
    return None, None, protein_name

def request_alphafolddb_model(uniprot_accession, tmp_dir):
    alphafold_id = f"AF-{uniprot_accession}-F1"
    database_version = "v4"
    model_url = f'https://alphafold.ebi.ac.uk/files/{alphafold_id}-model_{database_version}.pdb'
    error_url = f'https://alphafold.ebi.ac.uk/files/{alphafold_id}-predicted_aligned_error_{database_version}.json'

    # This creates a PDB and a JSON file for the target protein. They need to be read.
    gaingrn.scripts.io.run_command(cmd = f'curl {model_url} -o {tmp_dir}/{alphafold_id}.pdb')
    gaingrn.scripts.io.run_command(cmd = f'curl {error_url} -o {tmp_dir}/{alphafold_id}.json')
    print(f"[NOTE] Done retrieving the model for {uniprot_accession} with the corresponding AlphaFoldDB accession {alphafold_id}.")
    
    data = open(f'{tmp_dir}/{alphafold_id}.pdb').readlines()[0]
    if "Error" in data:
        print(data[0])
        raise FileNotFoundError(f"The Model is invalid. Target URL = https://alphafold.ebi.ac.uk/files/{alphafold_id}-model_{database_version}.pdb")