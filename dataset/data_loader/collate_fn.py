import torch

def custom_collate_fn(batch):
    """
    Custom collate function for session-multi-batch where each item in the batch 
    corresponds to a single session composed of multiple chunks (variable-length).
    
    This function avoids stacking chunk tensors across sessions (which can fail 
    when num_chunks is variable), and instead returns a list of tensors per session.

    Args:
        batch (List[Tuple[Tensor, int, str, str]]): 
            Each item is a tuple:
              - chunks: Tensor of shape (num_chunks, C, T, H, W)
              - label: int (class label)
              - session_id: str (unique identifier)
              - filepath: str or metadata

    Returns:
        Tuple[List[Tensor], List[int], List[str], List[str]]:
            - batch_chunks: list of Tensors (num_chunks, C, T, H, W)
            - batch_labels: list of ints
            - batch_session_ids: list of session IDs
            - batch_filepaths: list of strings
    """
    # print("[DEBUG] custom_collate_fn called")
    batch_chunks = []
    batch_labels = []
    batch_session_ids = []
    batch_filepaths = []

    for idx, item in enumerate(batch):
        try:
            chunks, label, sess_id, filepath = item

            # Sanity check for chunk tensor shape
            assert isinstance(chunks, torch.Tensor), f"[Batch {idx}] 'chunks' must be Tensor but got {type(chunks)}"
            assert chunks.ndim == 5, f"[Batch {idx}] chunks tensor must be 5D but got shape {chunks.shape}"

            batch_chunks.append(chunks)
            batch_labels.append(label)
            batch_session_ids.append(sess_id)
            batch_filepaths.append(filepath)
        except Exception as e:
            print(f"[ERROR] Failed to process sample at index {idx}: {e}")
            raise e  # Let DataLoader surface the full stack trace for debugging

    return batch_chunks, batch_labels, batch_session_ids, batch_filepaths
