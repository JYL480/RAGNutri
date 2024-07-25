import random
import torch
import numpy as np
import pandas as pd
from time import perf_counter as timer
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"

# Import texts and embedding df
file_path = "app/text_chunks_and_embeddings_df.csv"
text_chunks_and_embedding_df = pd.read_csv(file_path)

def read_embeddings() -> torch.Tensor:
    # Check if the first element in the embedding column is a string
    if isinstance(text_chunks_and_embedding_df["embedding"].iloc[0], str):
        # Convert embedding column back to np.array
        text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    
    # Convert texts and embedding df to list of dicts
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
    return embeddings, pages_and_chunks



# The get the relevant scores and indices of the source
embedding_model = SentenceTransformer('all-mpnet-base-v2')

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int = 5,
                                print_time: bool = True):

  query_embedding = model.encode(query, convert_to_tensor=True)
#   print(query_embedding.shape)
#   print(embeddings.shape)
  # Get the time to do the semantic search, which compares to our source PDF embeddings!
  start_time = timer()
  dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0] # The index zero is just to remove the outer list
  end_time = timer()
  time_taken = end_time-start_time

#   if print_time:
    # print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")

  # Get the top k scores of the semantic search
  score, indices = torch.topk(dot_scores, n_resources_to_return);
  return score, indices, time_taken

def prompt_formatter(query: str,
                     context_items: list[dict],) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Join context items into one dotted paragraph
    print(context_items[0]["page_number"])
    context = "- " + "\n- ".join([f"Page {item['page_number']}: {item['sentence_chunk']}" for item in context_items])
    print(context)
    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
Use the following examples as reference for the ideal answer style.
\nExample 1:
Query: What are the fat-soluble vitamins?
Answer: (Page number 100) The fat-soluble vitamins include Vitamin A, Vitamin D, Vitamin E, and Vitamin K. \n (Page number 105) These vitamins are absorbed along with fats in the diet and can be stored in the body's fatty tissue and liver for later use. Vitamin A is important for vision, immune function, and skin health. Vitamin D plays a critical role in calcium absorption and bone health. \n (Page number 155) Vitamin E acts as an antioxidant, protecting cells from damage. Vitamin K is essential for blood clotting and bone metabolism.
\nExample 2:
Query: What are the causes of type 2 diabetes?
Answer: (Page number 60) Type 2 diabetes is often associated with overnutrition, particularly the overconsumption of calories leading to obesity. Factors include a diet high in refined sugars and saturated fats, which can lead to insulin resistance, a condition where the body's cells do not respond effectively to insulin.\n (Page number 700) Over time, the pancreas cannot produce enough insulin to manage blood sugar levels, resulting in type 2 diabetes. Additionally, excessive caloric intake without sufficient physical activity exacerbates the risk by promoting weight gain and fat accumulation, particularly around the abdomen, further contributing to insulin resistance.
\nExample 3:
Query: What is the importance of hydration for physical performance?
Answer: (Page number 15) Hydration is crucial for physical performance because water plays key roles in maintaining blood volume, regulating body temperature, and ensuring the transport of nutrients and oxygen to cells. Adequate hydration is essential for optimal muscle function, endurance, and recovery. \n (Page number 189) Dehydration can lead to decreased performance, fatigue, and increased risk of heat-related illnesses, such as heat stroke. Drinking sufficient water before, during, and after exercise helps ensure peak physical performance and recovery.
\nNow use the following context items to answer the user query and I want each new page to place in a new bullet point.
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query
    prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model


    # Apply the chat template
  
    return prompt