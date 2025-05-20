import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs import Pairs

# Recebe SMILES
# Calcula ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'MACCS', 'RDK', 'AVALON', 'ATOMPAIR', 'TOPTOR', 'MW', 'ALOGP']
# Salva como CSV


def validate_smiles(smiles):
    if not smiles or not isinstance(smiles, str):
        raise ValueError("Invalid SMILES string")


def smiles_to_mol(smiles):
    validate_smiles(smiles)
    rdkit_mol = Chem.MolFromSmiles(smiles)
    if rdkit_mol is None:
        raise ValueError("Invalid SMILES string")
    return rdkit_mol


def get_morgan_fingerprint_as_bitvect(identifier, radius=2, nBits=2048, useFeatures=False):
    try:
        if isinstance(identifier, str):
            rdkit_mol = smiles_to_mol(identifier)
        elif isinstance(identifier, Chem.Mol):
            rdkit_mol = identifier
        else:
            raise ValueError("Input must be a SMILES string or an RDKit Mol object")
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(rdkit_mol, radius=radius, nBits=nBits, useFeatures=useFeatures)
    except Exception as ex:
        raise ValueError(f"Error generating fingerprint: {str(ex)}")
    return fingerprint


def get_ECFP4(identifier, radius=2, nBits=2048):
    return get_morgan_fingerprint_as_bitvect(identifier, radius=radius, nBits=nBits, useFeatures=False)


def get_ECFP6(identifier, radius=3, nBits=2048):
    return get_morgan_fingerprint_as_bitvect(identifier, radius=radius, nBits=nBits, useFeatures=False)


def get_FCFP4(identifier, radius=2, nBits=2048):
    return get_morgan_fingerprint_as_bitvect(identifier, radius=radius, nBits=nBits, useFeatures=True)


def get_FCFP6(identifier, radius=3, nBits=2048):
    return get_morgan_fingerprint_as_bitvect(identifier, radius=radius, nBits=nBits, useFeatures=True)

def get_MACCS(identifier):
    try:
        rdkit_mol = smiles_to_mol(identifier) if isinstance(identifier, str) else identifier
        fingerprint = rdMolDescriptors.GetMACCSKeysFingerprint(rdkit_mol)
    except Exception as ex:
        raise ValueError(f"Error generating MACCS fingerprint: {str(ex)}")
    return fingerprint


def get_RDK(identifier, nBits=2048):
    try:
        rdkit_mol = smiles_to_mol(identifier) if isinstance(identifier, str) else identifier
        fingerprint = Chem.RDKFingerprint(rdkit_mol, fpSize=nBits)
    except Exception as ex:
        raise ValueError(f"Error generating RDK fingerprint: {str(ex)}")
    return fingerprint


def get_AVALON(identifier, nBits=2048):
    try:
        rdkit_mol = smiles_to_mol(identifier) if isinstance(identifier, str) else identifier
        fingerprint = pyAvalonTools.GetAvalonFP(rdkit_mol, nBits=nBits)
    except Exception as ex:
        raise ValueError(f"Error generating AVALON fingerprint: {str(ex)}")
    return fingerprint


def get_ATOMPAIR(identifier):
    try:
        rdkit_mol = smiles_to_mol(identifier) if isinstance(identifier, str) else identifier
        fingerprint = Pairs.GetAtomPairFingerprintAsBitVect(rdkit_mol)
    except Exception as ex:
        raise ValueError(f"Error generating ATOMPAIR fingerprint: {str(ex)}")
    return fingerprint


def get_TOPTOR(identifier):
    try:
        rdkit_mol = smiles_to_mol(identifier) if isinstance(identifier, str) else identifier
        descriptor = Descriptors.TPSA(rdkit_mol)
    except Exception as ex:
        raise ValueError(f"Error calculating TOPTOR descriptor: {str(ex)}")
    return descriptor


def get_MW(identifier):
    try:
        rdkit_mol = smiles_to_mol(identifier) if isinstance(identifier, str) else identifier
        descriptor = Descriptors.MolWt(rdkit_mol)
    except Exception as ex:
        raise ValueError(f"Error calculating MW descriptor: {str(ex)}")
    return descriptor


def get_ALOGP(identifier):
    try:
        rdkit_mol = smiles_to_mol(identifier) if isinstance(identifier, str) else identifier
        descriptor = Descriptors.MolLogP(rdkit_mol)
    except Exception as ex:
        raise ValueError(f"Error calculating ALOGP descriptor: {str(ex)}")
    return descriptor

def calculate_descriptors_from_csv(input_csv, output_csv):
    try:
        df = pd.read_csv('data/14_public_domain_WDR91_ligands.csv')
        if 'smiles' not in df.columns:
            raise ValueError("Input CSV must contain a 'smiles' column")

        descriptors = ['ECFP4', 'ECFP6', 'FCFP4', 'FCFP6', 'MACCS', 'RDK', 'AVALON', 'ATOMPAIR', 'TOPTOR', 'MW', 'ALOGP']
        for descriptor in descriptors:
            df[descriptor] = None

        for index, row in df.iterrows():
            smiles = row['smiles']
            try:
                df.at[index, 'ECFP4'] = get_ECFP4(smiles).ToBitString()
                df.at[index, 'ECFP6'] = get_ECFP6(smiles).ToBitString()
                df.at[index, 'FCFP4'] = get_FCFP4(smiles).ToBitString()
                df.at[index, 'FCFP6'] = get_FCFP6(smiles).ToBitString()
                df.at[index, 'MACCS'] = get_MACCS(smiles).ToBitString()
                df.at[index, 'RDK'] = get_RDK(smiles).ToBitString()
                df.at[index, 'AVALON'] = get_AVALON(smiles).ToBitString()
                df.at[index, 'ATOMPAIR'] = get_ATOMPAIR(smiles).ToBitString()
                df.at[index, 'TOPTOR'] = get_TOPTOR(smiles)
                df.at[index, 'MW'] = get_MW(smiles)
                df.at[index, 'ALOGP'] = get_ALOGP(smiles)
            except Exception as ex:
                print(f"Error processing SMILES '{smiles}': {str(ex)}")

        #Salva os descritores calculados no arquivo CSV
        df.to_csv(output_csv, index=False)
        print(f"Descriptors calculated and saved to {output_csv}")
    except Exception as ex:
        print(f"Error processing CSV: {str(ex)}")

"""Salva o arquivo CSV com os descritores calculados"""
if __name__ == "__main__":
    input_csv = 'data/14_public_domain_WDR91_ligands.csv'
    output_csv = 'data/ligands_descriptors.csv'
    calculate_descriptors_from_csv(input_csv, output_csv)


"""Teste de execução
if __name__ == "__main__":
    # smiles = input("Enter a SMILES string: ")
    # molfile = smiles_to_mol(smiles)
    molfile = Chem.MolFromSmiles('CC(=O)Nc1ccc(O)cc1')
    print(f"\nECFP4:", get_ECFP4(molfile).ToBitString(),
        "\nECFP6:", get_ECFP6(molfile).ToBitString(),
        "\nFCFP4:", get_FCFP4(molfile).ToBitString(),
        "\nFCFP6:", get_ECFP6(molfile).ToBitString(),
        "\nMACCS:", get_MACCS(molfile).ToBitString(),
        "\nRDK:", get_RDK(molfile).ToBitString(),
        "\nAVALON:", get_AVALON(molfile).ToBitString(),
        "\nTOPTOR:", get_TOPTOR(molfile),
        "\nATOMPAIR:", get_ATOMPAIR(molfile).ToBitString(),
        "\nMW:", get_MW(molfile),
        "\nALOGP:", get_ALOGP(molfile),
        "\nSMILES:", Chem.MolToSmiles(molfile))
"""