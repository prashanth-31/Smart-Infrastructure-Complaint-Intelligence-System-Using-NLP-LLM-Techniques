import torch
from config import MODEL_FILES

model_path = MODEL_FILES['multi_task_classifier']
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

state_dict = checkpoint.get('model_state_dict', {})
bert_keys = [k for k in state_dict.keys() if k.startswith('bert.')]

print(f"Total keys in state_dict: {len(state_dict)}")
print(f"Keys starting with 'bert.': {len(bert_keys)}")

if bert_keys:
    print("\nFirst 10 BERT keys:")
    for key in bert_keys[:10]:
        print(f"  {key}")
    print(f"\n✓ Checkpoint includes BERT weights")
else:
    print("\n✗ Checkpoint does NOT include BERT weights")
    print("\nAll keys in checkpoint:")
    for key in list(state_dict.keys())[:20]:
        print(f"  {key}")
