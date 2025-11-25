from transformers import AutoModel, AutoTokenizer
import sys

print("=" * 70)
print("Downloading BERT base model (bert-base-uncased)")
print("This may take 2-5 minutes depending on your internet connection...")
print("=" * 70)

try:
    print("\n1. Downloading model...")
    model = AutoModel.from_pretrained('bert-base-uncased')
    print(f"   ✓ Model downloaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    print("\n2. Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    print(f"   ✓ Tokenizer downloaded ({len(tokenizer)} vocab size)")
    
    print("\n" + "=" * 70)
    print("✓ SUCCESS: BERT model cached locally")
    print("=" * 70)
    print("\nYou can now restart Streamlit and the model should load without issues.")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
