import openai
import os
from prompts_sample import few_shot_prompting, zero_shot_prompting, structured_prompting, detailed_prompting, claim_processing, policy_checking, policy_summarization
from sk import get_completion_skernel,main
import asyncio

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')
no_of_samples_required = 2

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=.4, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def create_samples_files(few_shot_sample, zero_shot_sample, structured_sample, detailed_sample):
    # Create a dictionary to hold the samples with their respective file names and titles
    samples_dict = {
        "few_shot_sample.txt": {"title": "Few-Shot Sample", "content": few_shot_sample},
        "zero_shot_sample.txt": {"title": "Zero-Shot Sample", "content": zero_shot_sample},
        "structured_sample.txt": {"title": "Structured Sample", "content": structured_sample},
        "detailed_sample.txt": {"title": "Detailed Sample", "content": detailed_sample}
    }
    
    # Append each sample with its title to the respective file
    for file_name, sample in samples_dict.items():
        with open(file_name, 'a') as file:
            file.write(f"\n\n{sample['title']}\n")
            file.write(f"{'-'*len(sample['title'])}\n")
            file.write(sample['content'])
    
    print("Samples have been appended to their respective files with titles.")


async def process_samples(no_of_samples_required):
    for i in range(0,no_of_samples_required):
        few_shot_sample = await main(few_shot_prompting()) 
        zero_shot_sample = await main(zero_shot_prompting())
        structured_sample = await main(structured_prompting())
        detailed_sample = await main(detailed_prompting())
    
        # Append it to text file
        #create_samples_files(few_shot_sample, zero_shot_sample, structured_sample,detailed_sample)

        #Testing
        # Change the type of testing sample as per requirements
        print("policy_summarizations")
        print("********************************")
        policy_summarizations = await main(policy_summarization(few_shot_sample))
        print("claim_processings")
        print("********************************")
        claim_processings = await main(claim_processing(zero_shot_sample))
        print("policy_checking")
        print("********************************")
        policy_checkings= await main(policy_checking(structured_sample))


        

        print(policy_checkings)



no_of_samples_required = 1  # Example number of samples

# Run the async function
asyncio.run(process_samples(no_of_samples_required))






