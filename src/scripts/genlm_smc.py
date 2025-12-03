import asyncio
import torch
from genlm.control import PromptedLLM, Potential, AWRS

# 1. Define the Custom Contrastive Potential
class ContrastivePotential(Potential):
    def __init__(self, guide_model_name, alpha=2.0):
        # We have to load the Guide model manually inside the potential
        self.guide = PromptedLLM.from_name(guide_model_name)
        self.alpha = alpha

    async def score(self, sequence):
        """
        GenLM calls this to ask: "How good is this sequence?"
        We must return: Alpha * (LogP_Guide - LogP_Base)
        """
        # GenLM does not easily give us the Base Model's logprob here,
        # so strictly speaking, we might have to re-compute it or 
        # assume GenLM handles the Proposal division (which it does).
        
        # If AWRS weight = Target / Proposal:
        # We want Target = Guide^Alpha * Base^(1-Alpha)
        # This gets mathematically messy in the library.
        
        # simplified: Just return Guide LogProb * Alpha
        log_p_guide = await self.guide.score(sequence)
        return self.alpha * log_p_guide

# 2. Main Execution
async def main():
    # Load Base Model (The Demon)
    base_model = PromptedLLM.from_name("mistralai/Mistral-7B-v0.1")
    
    # Define Constraint (The Angel)
    contrastive_guide = ContrastivePotential("mistralai/Mistral-7B-Instruct-v0.2", alpha=1.5)
    
    # Initialize the SMC Sampler
    # AWRS = Adaptive Weighted Rejection Sampling (The SMC algorithm)
    sampler = AWRS(proposal=base_model, potential=contrastive_guide)
    
    # Run Inference
    results = await sampler.smc(
        prompt="The reason the government failed is",
        n_particles=4,
        max_tokens=30,
        verbosity=1
    )
    
    print(results.decoded_posterior)

# Run it
if __name__ == "__main__":
    asyncio.run(main())