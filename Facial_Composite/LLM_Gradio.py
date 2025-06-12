from copy import deepcopy
import gradio as gr

if gr.NO_RELOAD:
    import torch
    import transformers
    from transformers import BitsAndBytesConfig

    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16, "quantization_config": quantization_config},
        device_map="auto",
    )

    pipeline.model.generation_config.cache_implementation = "static"
    quantized_model = torch.compile(pipeline.model, mode="reduce-overhead", fullgraph=True)
    pipeline.model = quantized_model


def test_streaming_response(message, history):
    task_desc = (
        """
        You are a forensic artist working for law enforcement, and your job is to combine single facial characteristic descriptions from witness testimonies about a suspect, into a single cohesive suspect characteristic description.
        
        Final characteristic description requirements:
        - The witness testimony characteristic descriptions must all have been examined very thoroughly and in minute detail before writing the final characteristic description.
        - The final characteristic description must contain all the reframed similarities and contradictions, including those details not part of any reframe actions.  
        - The final characteristic description should be a single sentence.
        - Only the present characteristic descriptors are to be discussed, no other body details.
        - There should be as few redundant words as possible in the final description.

        You will follow these steps, and their bullet points:

        Merging similarities between descriptions:
        1. Discuss what the similarities between the testimonies are.
        2. Discuss how the similarities can be reframed into one, without losing important details.

        Interpret and resolve contradictions between characteristic descriptions:
        3. Discuss what it is about the characteristic descriptions that contradict each other, if there are any contradictions.
        4. Discuss as to what detail(s) the contradictions could actually be referring to.
        5. For all contradictions, turn them into the new details you think they might be referring to.

        Putting together the new characteristic description:
        - Combine all witness testimony characteristic description details, including new ones, into a single suspect characteristic description, except the original details that were reframed because of similarity or contradictions.
        """)

    system_prompt = {"role": "system", "content": task_desc}

    if not history:
        history = [system_prompt]
    else: history = [system_prompt] + history

    tmp_history = deepcopy(history)
    tmp_history.append({"role": "user", "content": message})

    # Diverse 8-Beam Search params:
    outputs = pipeline(
        tmp_history,
        max_new_tokens=256 * 10,
        do_sample=False,
        num_beams=8,
        num_beam_groups=4,
        temperature=None,
        top_p=None,
        early_stopping=True,
        repetition_penalty=1.2,
        diversity_penalty=0.5, # 'Diverse Beam Search' research paper recommended 0.2 - 0.8
    )

    # Below are other pipeline parameters used in experiments:

    # Params for more accurate responses:
    """
    outputs = pipeline(
        tmp_history,
        max_new_tokens=256 * 10,
        do_sample=True,
        temperature=0.2,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.2,
    )
    """

    # Accuracy-params with 2-beam:
    """
    outputs = pipeline(
        tmp_history,
        max_new_tokens=256 * 10,
        do_sample=True, 
        num_beams=2, 
        temperature=0.2, 
        top_p=0.95, 
        top_k=20,
        early_stopping=True,
        repetition_penalty=1.2,
    )
    """

    tmp_history.clear()
    response = outputs[0]["generated_text"][-1]['content']

    return response


demo = gr.ChatInterface(
    fn=test_streaming_response,
    type="messages",
    stop_btn=True,
    editable=True,
    #save_history=True,
    examples=["Hello."],
    analytics_enabled=False
)

if __name__ == "__main__":
    demo.launch(
        #share=True
    )
    # gradio gradio_test.py