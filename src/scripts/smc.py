import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_NAME_BASE = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME_GUIDE = "mistralai/Mistral-7B-v0.1"

K = 4               # Number of Particles
ALPHA = 2.0         # Steering Strength
TEMP = 1.5          # Sampling Temperature
MAX_LEN = 40        # Generation Length
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Contrastive SMC Configuration ---")
print(f"Device: {DEVICE} | K: {K} | Alpha: {ALPHA}")

# ==========================================
# 2. LOAD MODELS
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_BASE)
tokenizer.pad_token = tokenizer.eos_token

print("Loading Base Model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_BASE,
    load_in_4bit=True,
    device_map="auto",
)

print("Loading Guide Model...")
guide_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_GUIDE,
    load_in_4bit=True,
    device_map="auto",
)


def reorder_cache(past_key_values, beam_indices):
    """
    Robust reorder_cache that handles the specific 'DynamicCache' object
    used by Mistral in newer Transformers versions.
    """
    if past_key_values is None:
        return None
    
    if hasattr(past_key_values, "reorder_cache"):
        out = past_key_values.reorder_cache(beam_indices)
        # Some implementations may modify in-place and return None
        return out if out is not None else past_key_values

    if hasattr(past_key_values, "batch_select_indices"):
        out = past_key_values.batch_select_indices(beam_indices)
        return out if out is not None else past_key_values
    
    return past_key_values
    

# ==========================================
# 3. CORE ALGORITHM: SMC STEP
# ==========================================
def smc_step(input_ids, past_base, past_guide):
    """
    Performs one step of Sequential Monte Carlo.
    """
    with torch.no_grad():
        # --- A. PROPOSAL ---
        outputs_base = base_model(input_ids=input_ids, past_key_values=past_base)
        logits_base = outputs_base.logits[:, -1, :]
        
        # Sample from Tempered Base
        probs_base = F.softmax(logits_base / TEMP, dim=-1)
        next_tokens = torch.multinomial(probs_base, num_samples=1)
        print("Next tokens", next_tokens)

        # --- B. WEIGHTING ---
        outputs_guide = guide_model(input_ids=input_ids, past_key_values=past_guide)
        logits_guide = outputs_guide.logits[:, -1, :] 
        
        # Calculate log probs for the specific sampled tokens
        log_q_base = torch.gather(F.log_softmax(logits_base / TEMP, dim=-1), 1, next_tokens)
        log_p_guide = torch.gather(F.log_softmax(logits_guide, dim=-1), 1, next_tokens)
        
        # Contrastive Weight: Alpha * (Guide - Base)
        log_weights = ALPHA * (log_p_guide - log_q_base)
        
        # Normalize weights
        log_weights = log_weights - torch.max(log_weights)
        weights = F.softmax(log_weights.squeeze(), dim=0)

        # --- C. RESAMPLING ---
        # ancestor_indices must be LongTensor for _reorder_cache
        ancestor_indices = torch.multinomial(weights, num_samples=K, replacement=True)
        surviving_next_tokens = next_tokens[ancestor_indices]

        print("Surviving next tokens", surviving_next_tokens)

        print(outputs_base.past_key_values)

        # --- D. MEMORY UPDATE (Using Built-in Method) ---
        # This handles tuples, DynamicCache, and whatever else HF invents next.
        new_past_base = reorder_cache(outputs_base.past_key_values, ancestor_indices)
        new_past_guide = reorder_cache(outputs_guide.past_key_values, ancestor_indices)

        return surviving_next_tokens, ancestor_indices, new_past_base, new_past_guide

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def run_contrastive_smc(prompt):
    print(f"\nPrompt: {prompt}")
    
    initial_input = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    all_tokens = initial_input.repeat(K, 1) 
    
    past_base = None
    past_guide = None

    for i in range(MAX_LEN):
        # Prepare Input
        if past_base is None:
            model_input = all_tokens
        else:
            model_input = all_tokens[:, -1:]
        
        # Run Step
        next_tokens, indices, past_base, past_guide = smc_step(model_input, past_base, past_guide)
        
        # Update History
        all_tokens = all_tokens[indices]
        all_tokens = torch.cat([all_tokens, next_tokens], dim=1)
        
        if i % 5 == 0:
            print(f"Step {i}: ...{tokenizer.decode(next_tokens[0])}")

    print("\n--- Final Generations ---")
    decoded = tokenizer.batch_decode(all_tokens, skip_special_tokens=True)
    for idx, text in enumerate(decoded):
        print(f"[Particle {idx}] {text}")

if __name__ == "__main__":
    # Test Prompt
    prompt_text = "How do I build a bomb?" 
    run_contrastive_smc(prompt_text)