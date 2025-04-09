# src/alphafold3/data/custom_utils.py

import re
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from absl import logging

# --- Dependency Imports (Ensure these paths/modules exist) ---
try:
    # Assuming FoldingInput is defined here and has a 'chain_ids' attribute (tuple/list)
    from alphafold3.common import folding_input
except ImportError:
    logging.warning("Could not import FoldingInput from alphafold3.common.folding_input")
    # Define a dummy class or use Any if FoldingInput structure is unknown
    class FoldingInput: # Dummy class - replace with actual import or structure
        sequences: Tuple[str, ...] = ()
        chain_ids: Optional[Tuple[str, ...]] = None # Example attribute
        # Add other attributes as needed for type hinting or dummy structure

try:
    # Assuming constants are defined here
    from alphafold3.model import data_constants
    # Verify these constant names exist in the actual file
    PROTEIN_RESTYPES = getattr(data_constants, 'protein_restypes', list('ACDEFGHIKLMNPQRSTVWY'))
    PROTEIN_RESTYPES_WITH_UNK_AND_GAP = getattr(data_constants, 'protein_restypes_with_unk_and_gap', list('ACDEFGHIKLMNPQRSTVWYX-'))
except ImportError:
    logging.error("Could not import data_constants from alphafold3.model.data_constants. Using fallback constants.")
    PROTEIN_RESTYPES = list('ACDEFGHIKLMNPQRSTVWY')
    PROTEIN_RESTYPES_WITH_UNK_AND_GAP = list('ACDEFGHIKLMNPQRSTVWYX-')
# -------------------------------------------------------------


def parse_mutation_string(mutation_str: Optional[str]) -> List[Tuple[Optional[str], int, str, str]]:
    """Parses mutation string like 'A123G,R45C' or 'A:G10C,B:R20D'.

    Returns:
        List of tuples: (chain_id, zero_based_pos, orig_aa, new_aa).
        Chain ID is None if not specified.
    """
    if not mutation_str:
        return []

    mutations = []
    # Regex to capture optional chain ID (single uppercase letter followed by ':')
    # and the standard mutation format.
    pattern = re.compile(r"(?:([A-Z]):)?([A-Z])(\d+)([A-Z])") # Chain ID limited to single uppercase letter

    for mut in mutation_str.split(','):
        mut = mut.strip()
        match = pattern.fullmatch(mut) # Use fullmatch for stricter format checking
        if not match:
            logging.error(f"Invalid mutation format: '{mut}'. Expected format like 'G10C' or 'A:G10C'.")
            raise ValueError(f"Invalid mutation format: {mut}")

        chain_id, orig_aa, pos_str, new_aa = match.groups()
        pos_zero_based = int(pos_str) - 1 # Convert to 0-based index

        if pos_zero_based < 0:
            logging.error(f"Invalid position '{pos_str}' in mutation '{mut}'. Position must be 1 or greater.")
            raise ValueError(f"Invalid position {pos_str} (must be >= 1)")
        if orig_aa not in PROTEIN_RESTYPES:
             logging.warning(f"Original residue '{orig_aa}' in mutation '{mut}' is not a standard protein residue.")
        if new_aa not in PROTEIN_RESTYPES:
             logging.error(f"New residue '{new_aa}' in mutation '{mut}' is not a standard protein residue.")
             raise ValueError(f"Invalid new residue '{new_aa}' in mutation '{mut}'.")

        mutations.append((chain_id, pos_zero_based, orig_aa, new_aa))

    logging.info(f"Parsed {len(mutations)} mutations.")
    return mutations


def apply_mutations_to_input(
    input_obj: folding_input.Input,
    parsed_mutations: List[Tuple[Optional[str], int, str, str]]
) -> folding_input.Input:
    """Applies parsed mutations to the sequences in FoldingInput.

    Args:
        input_obj: The FoldingInput object containing sequences and potentially chain_ids.
        parsed_mutations: List of parsed mutation tuples.

    Returns:
        The potentially modified FoldingInput object.

    Raises:
        ValueError: If a mutation is invalid (position out of bounds, original AA mismatch,
                    missing chain ID for multi-sequence input, or invalid chain ID).
        IndexError: If sequence index derived from chain ID is out of bounds.
    """
    if not parsed_mutations:
        return input_obj

    # Create a mutable list from the sequences tuple
    try:
        new_sequences = list(input_obj.sequences)
        num_sequences = len(new_sequences)
        if num_sequences == 0:
             logging.error("Cannot apply mutations: FoldingInput object has no sequences.")
             raise ValueError("Input object has no sequences.")
    except AttributeError:
         logging.error("Cannot apply mutations: Invalid input_obj structure (missing 'sequences' attribute?).")
         raise ValueError("Invalid input object structure for mutation.")

    # Check for chain IDs attribute (verify 'chain_ids' is the correct name)
    input_chain_ids = None
    if hasattr(input_obj, 'chain_ids') and isinstance(input_obj.chain_ids, (list, tuple)):
         input_chain_ids = input_obj.chain_ids
         if len(input_chain_ids) != num_sequences:
              logging.warning(f"Mismatch between number of sequences ({num_sequences}) and chain IDs ({len(input_chain_ids)}). Chain ID mapping might be unreliable.")
              # Decide if this should be a fatal error or just a warning
              # raise ValueError("Mismatch between number of sequences and chain IDs.")


    applied_count = 0
    for chain_id, pos, orig_aa, new_aa in parsed_mutations:
        target_sequence_index = -1 # Initialize to invalid index

        if chain_id is not None:
            # Chain ID provided in mutation: Find corresponding sequence index
            if input_chain_ids is None:
                logging.error(f"Chain ID '{chain_id}' provided for mutation, but FoldingInput object does not have chain ID information (expected 'chain_ids' attribute).")
                raise ValueError(f"Chain ID '{chain_id}' provided, but input object lacks chain information.")
            try:
                target_sequence_index = input_chain_ids.index(chain_id)
                logging.info(f"Mutation targets chain '{chain_id}' at sequence index {target_sequence_index}.")
            except ValueError:
                logging.error(f"Chain ID '{chain_id}' provided in mutation ('{chain_id}:{orig_aa}{pos+1}{new_aa}') not found in input chain IDs: {input_chain_ids}")
                raise ValueError(f"Chain ID '{chain_id}' not found in input chains: {input_chain_ids}")

        else:
            # No Chain ID provided in mutation
            if num_sequences == 1:
                # Only one sequence, assume it's the target
                target_sequence_index = 0
                logging.info(f"Mutation '{orig_aa}{pos+1}{new_aa}' applied to the single input sequence (index 0).")
            else:
                # Multiple sequences, but no chain ID specified - ambiguous
                logging.error(f"Mutation '{orig_aa}{pos+1}{new_aa}' lacks a chain ID prefix (e.g., 'A:'), which is required for inputs with multiple sequences ({num_sequences} sequences found).")
                raise ValueError(f"Mutation '{orig_aa}{pos+1}{new_aa}' requires a chain ID for multi-sequence inputs.")

        # --- Apply mutation to the determined target sequence ---
        if target_sequence_index < 0 or target_sequence_index >= num_sequences:
            # This should theoretically be caught by earlier checks, but added for safety
            logging.error(f"Internal error: Invalid target sequence index {target_sequence_index} derived for chain '{chain_id}'.")
            raise IndexError(f"Internal error: Invalid target sequence index {target_sequence_index}.")

        current_sequence_list = list(new_sequences[target_sequence_index])

        if pos >= len(current_sequence_list):
            logging.error(f"Mutation error: Position {pos + 1} is out of bounds for sequence index {target_sequence_index} (Chain: {chain_id or input_chain_ids[target_sequence_index] if input_chain_ids else 'N/A'}, length {len(current_sequence_list)}). Mutation: '{orig_aa}{pos+1}{new_aa}'.")
            raise ValueError(f"Position {pos + 1} out of bounds for sequence index {target_sequence_index} (length {len(current_sequence_list)}).")

        actual_orig_aa = current_sequence_list[pos]
        if actual_orig_aa != orig_aa:
            logging.error(f"Mutation error: Expected original residue '{orig_aa}' at position {pos + 1} but found '{actual_orig_aa}' in sequence index {target_sequence_index} (Chain: {chain_id or input_chain_ids[target_sequence_index] if input_chain_ids else 'N/A'}). Mutation: '{orig_aa}{pos+1}{new_aa}'.")
            raise ValueError(f"Original residue mismatch at position {pos + 1} in sequence index {target_sequence_index}: expected '{orig_aa}', found '{actual_orig_aa}'.")

        # Apply mutation
        current_sequence_list[pos] = new_aa
        new_sequences[target_sequence_index] = "".join(current_sequence_list)
        logging.info(f"Applied mutation: {orig_aa}{pos + 1}{new_aa} to sequence index {target_sequence_index} (Chain: {chain_id or input_chain_ids[target_sequence_index] if input_chain_ids else 'N/A'})")
        applied_count += 1

    # Update the input_obj with the modified sequences tuple
    # This depends on how FoldingInput is structured (e.g., dataclass, attrs, simple class)
    # If immutable (like a standard tuple attribute or dataclass(frozen=True)):
    # return dataclasses.replace(input_obj, sequences=tuple(new_sequences)) # Example for dataclasses
    # If mutable:
    input_obj.sequences = tuple(new_sequences) # Directly update if mutable

    logging.info(f"Applied {applied_count} mutations successfully.")
    return input_obj


def parse_masking_positions(pos_str: Optional[str]) -> List[int]:
    """Parses position string like '10-15,20,30' into list of 0-based indices."""
    if not pos_str:
        return []

    indices = set()
    for part in pos_str.split(','):
        part = part.strip()
        if not part: continue # Skip empty parts

        if '-' in part:
            try:
                start_str, end_str = part.split('-')
                start = int(start_str)
                end = int(end_str) # Input range is inclusive
                if start < 1 or end < start:
                     logging.error(f"Invalid range: '{part}'. Start must be >= 1 and end >= start.")
                     raise ValueError(f"Invalid range: {part}")
                # Convert inclusive 1-based range to 0-based indices
                indices.update(range(start - 1, end))
            except ValueError:
                 logging.error(f"Invalid range format: '{part}'. Expected format like '10-15'.")
                 raise ValueError(f"Invalid range format: {part}")
        else:
            try:
                pos = int(part)
                if pos < 1:
                     logging.error(f"Invalid position: '{part}'. Position must be 1 or greater.")
                     raise ValueError(f"Invalid position: {part}")
                # Convert 1-based position to 0-based index
                indices.add(pos - 1)
            except ValueError:
                 logging.error(f"Invalid position format: '{part}'. Expected an integer.")
                 raise ValueError(f"Invalid position format: {part}")

    sorted_indices = sorted(list(indices))
    logging.info(f"Parsed {len(sorted_indices)} unique 0-based indices for masking.")
    return sorted_indices


def apply_masking_to_features(
    feature_dict: Dict[str, np.ndarray],
    masking_config: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Applies MSA and/or Deletion Matrix masking based on config.

    Args:
        feature_dict: The dictionary containing features (e.g., 'msa_feat', 'deletion_matrix').
        masking_config: Dictionary containing masking parameters like
                        'positions_str', 'mask_msa', 'mask_deletion_matrix', 'mask_token'.

    Returns:
        The potentially modified feature dictionary.
    """
    if not masking_config or not masking_config.get('positions_str'):
        logging.info("No masking positions provided. Skipping masking.")
        return feature_dict

    # Check if any masking is enabled
    mask_msa_flag = masking_config.get('mask_msa', False)
    mask_del_mat_flag = masking_config.get('mask_deletion_matrix', False)
    if not mask_msa_flag and not mask_del_mat_flag:
        logging.info("MSA and Deletion Matrix masking are both disabled. Skipping.")
        return feature_dict

    # Parse positions only if masking is enabled and positions are specified
    zero_based_indices = parse_masking_positions(masking_config['positions_str'])
    if not zero_based_indices:
        logging.info("Parsed masking positions list is empty. Skipping masking.")
        return feature_dict

    logging.info(f"Applying masking to 0-based indices: {zero_based_indices}")

    # --- MSA Masking ---
    if mask_msa_flag:
        # TODO: Verify this key is correct for AF3
        msa_key = 'msa'
        if msa_key in feature_dict and isinstance(feature_dict[msa_key], np.ndarray):
            msa_feat = feature_dict[msa_key].copy() # Work on a copy
            mask_token = masking_config.get('mask_token', 'X')

            try:
                # Find the numerical index for the mask token
                mask_index = PROTEIN_RESTYPES_WITH_UNK_AND_GAP.index(mask_token)
            except ValueError:
                logging.error(f"Invalid mask_token '{mask_token}'. Must be one of {PROTEIN_RESTYPES_WITH_UNK_AND_GAP}")
                raise ValueError(f"Invalid mask_token '{mask_token}'")

            # Ensure indices are within bounds for the sequence length dimension
            num_seq, seq_len = msa_feat.shape[:2] # Assuming shape (num_seq, seq_len, ...)
            valid_indices = [idx for idx in zero_based_indices if idx < seq_len]

            if len(valid_indices) != len(zero_based_indices):
                logging.warning(f"Some mask positions were out of bounds for MSA (length {seq_len}). Applying only to valid positions.")

            if valid_indices:
                # Apply mask index to specified columns (valid_indices) for all sequences *except the first* (target)
                msa_feat[1:, valid_indices] = mask_index
                feature_dict[msa_key] = msa_feat # Update the dictionary with the modified array
                logging.info(f"Masked MSA positions (1-based): {[i+1 for i in valid_indices]} with token '{mask_token}' (index {mask_index}).")
            else:
                 logging.info("No valid indices found for MSA masking within sequence length.")
        else:
            logging.warning(f"MSA feature key '{msa_key}' not found or not a NumPy array in feature_dict. Cannot apply MSA mask.")

    # --- Deletion Matrix Masking ---
    if mask_del_mat_flag:
        # TODO: Verify this key is correct for AF3
        del_mat_key = 'deletion_matrix'
        if del_mat_key in feature_dict and isinstance(feature_dict[del_mat_key], np.ndarray):
            del_mat = feature_dict[del_mat_key].copy() # Work on a copy

            # Ensure indices are within bounds
            num_seq, seq_len = del_mat.shape # Assuming shape (num_seq, seq_len)
            valid_indices = [idx for idx in zero_based_indices if idx < seq_len]

            if len(valid_indices) != len(zero_based_indices):
                 logging.warning(f"Some mask positions were out of bounds for Deletion Matrix (length {seq_len}). Applying only to valid positions.")

            if valid_indices:
                # Set specified columns (valid_indices) to 0 for all sequences *except the first* (target)
                del_mat[1:, valid_indices] = 0
                feature_dict[del_mat_key] = del_mat # Update the dictionary
                logging.info(f"Masked (zeroed) Deletion Matrix positions (1-based): {[i+1 for i in valid_indices]}.")
            else:
                 logging.info("No valid indices found for Deletion Matrix masking within sequence length.")
        else:
            logging.warning(f"Deletion matrix key '{del_mat_key}' not found or not a NumPy array in feature_dict. Cannot apply deletion matrix mask.")

    return feature_dict 