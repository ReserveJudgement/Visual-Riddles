import textwrap
import os
import google.generativeai as genai
import pandas as pd
from PIL import Image
import urllib.request
from IPython.display import display
from IPython.display import Markdown
from tqdm import tqdm
import time
import json
from datasets import load_dataset


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def list_models():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
    return


# baseline for Gemini zero-shot
def solve_from_image(modelname, filename, safety_config, systemprompt, tag):
    model = genai.GenerativeModel(modelname, safety_settings=safety_config)
    data = pd.read_csv(filename)
    print("Samples: ", len(data.index))
    responses = []
    tokencount = 0
    request = 0
    for i, sample in tqdm(data.iterrows()):
        question = sample['question']
        img = Image.open(f"./images/image-{i}.jpg")
        problem = [img, systemprompt + question]
        tokencount += model.count_tokens(problem).total_tokens
        request += 1
        # handle request limits
        if tokencount > 32000 or request > 2:
            time.sleep(60)
            tokencount = 0
            request = 0
        response = model.generate_content(problem)
        responses.append({"idx": i, "question": question, "response": response.text, "ground-truth": sample['ground_truth_answer']})
    df = pd.DataFrame(responses)
    df.to_csv(f"./gemini_openended_zeroshot_{tag}.csv", index=False)
    return

############################
# Functions for evaluation #
############################

# Automatic evaluation using Gemini
def autojudge(modelname, answerfile, safety_config):
    model = genai.GenerativeModel(modelname, safety_settings=safety_config)
    riddles = pd.read_csv(answerfile).to_dict('records')
    judgements = []
    tokencount = 0
    request = 0
    for i, riddle in tqdm(enumerate(riddles)):
        img = Image.open(f"./images/image-{i}.jpg")
        question = riddle["question"]
        ground_truth = riddle["ground-truth"]
        candidate_answer = riddle["response"]
        prompt = [img, f"Answer with only Yes OR No. Given the image, the question and the ground-truth answer, is the candidate answer correct?\nQuestion: {question}\nGround-Truth Answer:{ground_truth}\nCandidate Answer: {candidate_answer}\n\nThe ground-truth answer is the real and correct full-answer for the visual riddle containing the question and the image - USE it when you decide if the candidate answer address the riddle correctly. Do not forget, answer with only Yes OR No."]
        tokencount += model.count_tokens(prompt).total_tokens
        request += 1
        if tokencount > 32000 or request > 2:
            time.sleep(60)
            tokencount = 0
            request = 0
        response = model.generate_content(prompt)
        result = response.text.split()[0]
        if result[:3] == "Yes":
            answer = True
        elif result[:2] == "No":
            answer = False
        else:
            answer = "undetermined"
        judgements.append({"idx": i, "question": question, "ground-truth": ground_truth, "candidate-answer": candidate_answer, "evaluation": response.text, "result": answer})
    judgements = pd.DataFrame(judgements)
    judgements.to_csv(f"./eval_{answerfile}", index=False)
    return


def autojudge_explanations(modelname, dataset, cluesfile, answerfile, safety_config):
    model = genai.GenerativeModel(modelname, safety_settings=safety_config)
    riddles = pd.read_csv(answerfile)
    riddles["judge"] = riddles["judge"].str.strip()
    riddles = riddles[(riddles["judge"] == "No") | (riddles["judge"] == "No.") | (riddles["judge"] == "Answer:No.")]
    print("Wrong answers: ", len(riddles.index))
    judgements = []
    tokencount = 0
    request = 0
    for i, riddle in tqdm(riddles.iterrows()):
        sample = dataset.iloc[i, :]
        question = riddle["question"]
        assert question == sample["question"]
        ground_truth = riddle["ground-truth"]
        assert ground_truth == sample["ground_truth_answer"]
        candidate_answer = riddle["candidate-answer"]
        img = sample["image"]
        # construct prompt
        txt = f"Answer with only 1, 2, 3. Given the image, the question, the detected evidence and the ground-truth answer, why was the candidate answer incorrect?\nQuestion: {question}\nCandidate Answer: {candidate_answer}\nGround-Truth Answer:{ground_truth}\nThe ground-truth answer is the real and correct full-answer for the visual riddle containing the question and the image - USE it to decide why the candidate answer was incorrect.\nOptions:\n1. A necessary piece of evidence was not detected in the image.\n2. A required piece of external information was missed (for example cultural or scientific background knowledge).\n3. All necessary information was available but there was a failure in reasoning.\nDo not forget, answer with only 1, 2 or 3."
        prompt = [img, txt]
        tokencount += model.count_tokens(prompt).total_tokens
        request += 1
        if tokencount >= 1000000 or request >= 15:
            time.sleep(60)
            tokencount = 0
            request = 0
        # get response from model
        response = model.generate_content(prompt)
        result = response.text.split()[0]
        if (result[0] == "1") or (result[0] == "2") or (result[0] == "3"):
            answer = int(result[0])
        else:
            answer = "undetermined"
        judgements.append({"idx": i, "question": question, "candidate_answer": candidate_answer, "ground-truth": ground_truth, "evaluation": response.text, "reason": answer})
    judgements = pd.DataFrame(judgements)
    judgements.to_csv(f"./matan/failure_analysis_flash_fewshotclues.csv", index=False)
    total = len(judgements.index)
    missed_evidence = len(judgements[judgements["reason"] == 1].index)
    missed_knowledge = len(judgements[judgements["reason"] == 2].index)
    bad_reasoning = len(judgements[judgements["reason"] == 3].index)
    print("total: ", total)
    print(f"missed evidence: {missed_evidence}, {missed_evidence/total}")
    print(f"missed knowledge: {missed_knowledge}, {missed_knowledge / total}")
    print(f"bad reasoning: {bad_reasoning}, {bad_reasoning / total}")
    return



###################################
# Functions for chain of thoughts #
###################################

# Extract clues from image in json format as first step in chain-of-thought process
def extract_clues(modelname, data, safety_config, systemprompt, tag, func=None, mode="from_image"):
    model = genai.GenerativeModel(modelname, safety_settings=safety_config, system_instruction=systemprompt, tools=[func], tool_config={'function_calling_config':'ANY'})
    print("Samples: ", len(data.index))
    clues = []
    tokencount = 0
    request = 0
    for i, sample in tqdm(data.iterrows()):
        question = sample['question']
        answer = ""
        # option to extract actual clues required for ground truth answer, can be used for evaluating this step in chain-of-thought seperately
        if mode == "ground_truth":
            answer = " Answer: " + sample['ground_truth_answer']
        img = sample["image"]
        problem = [img, "Question: " + question + answer + " ### Response: Here is the set of all clues in JSON format.\n"]
        tokencount += model.count_tokens(problem).total_tokens
        request += 1
        if tokencount > 32000 or request > 2:
            time.sleep(60)
            tokencount = 0
            request = 0
        response = model.generate_content(problem)
        structured = response.candidates[0].content.parts[0].function_call
        clues.append({"idx": i, "question": question, "clues": type(structured).to_dict(structured)})
    with open(f'clues_{tag}.json', 'w', encoding='utf-8') as f:
        json.dump(clues, f, ensure_ascii=False, indent=4)
    return


# attempt to get even better list of clues by focusing seperately on one quadrant of the image at a time
def extract_clues_quarters(modelname, data, safety_config, systemprompt, tag, func=None):
    model = genai.GenerativeModel(modelname, safety_settings=safety_config, system_instruction=systemprompt, tools=[func], tool_config={'function_calling_config': 'ANY'})
    print("Samples: ", len(data.index))
    clues = []
    tokencount = 0
    request = 0
    #data = data[:200]
    for i, sample in tqdm(data.iterrows()):
        question = sample['question']
        evidence = []
        img = sample["image"]
        w, h = img.size
        img1 = img.crop((0, 0, w/2, h/2))
        img2 = img.crop((w/2, 0, w, h/2))
        img3 = img.crop((0, h/2, w/2, h))
        img4 = img.crop((w/2, h/2, w, h))
        for im in [img1, img2, img3, img4]:
            problem = [im, "Question: " + question + " ### Response: Here is the set of clues in JSON format.\n"]
            tokencount += model.count_tokens(problem).total_tokens
            request += 1
            if tokencount >= 1000000 or request >= 15:
                time.sleep(60)
                tokencount = 0
                request = 0
            response = model.generate_content(problem)
            structured = response.candidates[0].content.parts[0].function_call
            evidence.extend(type(structured).to_dict(structured))
        clues.append({"idx": i, "question": question, "clues": evidence})
    with open(f'./clues/clues_{tag}.json', 'w', encoding='utf-8') as f:
        json.dump(clues, f, ensure_ascii=False, indent=4)
    return


def JSON_clue_format():
    """
    Helper function to provide function-call tool for genai object to give JSON format to clues
    :return: function-call tool
    """
    clue = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            'name': genai.protos.Schema(type=genai.protos.Type.STRING),
            'description': genai.protos.Schema(type=genai.protos.Type.STRING),
            'location': genai.protos.Schema(type=genai.protos.Type.STRING),
            'peculiarities': genai.protos.Schema(type=genai.protos.Type.STRING)
        },
        required=['name', 'description', 'location']
    )
    clues = genai.protos.Schema(
        type=genai.protos.Type.ARRAY,
        items=clue
    )
    func = genai.protos.FunctionDeclaration(
        name="add_to_database",
        description=textwrap.dedent("""\
                Adds entities to the database.
                """),
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                'clues': clues,
            }
        )
    )
    return func


# Second step in chain-of though process: solve riddle based on textual analysis of image as encoded in json file of clues
def solve_from_clues(modelname, data, cluesfile, safety_config, systemprompt, tag, include_img=False):
    model = genai.GenerativeModel(modelname, safety_settings=safety_config)
    f = open(cluesfile, encoding='utf-8')
    clues = json.load(f)
    f.close()
    print("Samples: ", len(data.index))
    responses = []
    tokencount = 0
    request = 0
    for i, sample in tqdm(data.iterrows()):
        question = str(sample['question'])
        assert question == str(clues[i]["question"])
        evidence = str(clues[i]["clues"]["args"]["clues"])
        prompt = str(systemprompt + "\nQuestion: " + question + "\nThe riddle can be solved from this set of clues: " + evidence + "\nSolution:")
        if include_img is True:
            img = sample["image"]
            prompt = [img, prompt]
        tokencount += model.count_tokens(prompt).total_tokens
        request += 1
        if tokencount > 32000 or request > 2:
            time.sleep(60)
            tokencount = 0
            request = 0
        response = model.generate_content(prompt)
        print(response.text)
        responses.append({"idx": i, "question": question, "response": response.text, "ground-truth": sample['ground_truth_answer']})
    df = pd.DataFrame(responses)
    df.to_csv(f"./gemini_solutions_from_clues{tag}.csv", index=False)
    return


# prompt opimization experiment, using another LM as prompt generator
def optimize_prompt(modelname, promptername, data, evalset, safety_config):
    model = genai.GenerativeModel(modelname, safety_settings=safety_config)
    prompter = genai.GenerativeModel(promptername, safety_settings=safety_config)
    evals = pd.read_csv(evalset)
    evals = evals[evals["result"] == False]
    evals = evals.sample(n=20)
    data = data.iloc[evals.index, :]
    tokencount = 0
    requests = 0
    results = [{"Prompt": "This is a visual riddle, look for clues and think step by step.", "Accuracy": 0}]
    basegetprompt = "We are solving visual riddles. They each involve an image with subtle clues and a question. The solution requires keen visual observation, general knowledge and common sense reasoning. We have a large vision-language model and want to prompt it to solve the riddles. The prompt will be of the form '<image> <system prompt> <riddle question>' . Based on previous attempts and their results, suggest just one improved <system prompt> that is likely to work. Make sure to try new and novel prompts compared to previous attempts. Results from previous attempts: "
    # iterate for optimization
    for i in tqdm(range(10)):
        getprompt = basegetprompt + str(results)
        response = prompter.generate_content(getprompt)
        suggested_prompt = response.text
        ### iterate over samples
        correct = 0
        total = 0
        for j, sample in data.iterrows():
            question = sample["question"]
            img = sample["image"]
            prompt = [img, suggested_prompt + " ### Riddle Question: " + question]
            tokencount += model.count_tokens(prompt).total_tokens
            requests += 1
            if tokencount > 32000 or requests > 2:
                time.sleep(60)
                tokencount = 0
                requests = 0
            response = model.generate_content(prompt)
            candidate_answer = response.text
            ### evaluate
            ground_truth = sample["ground_truth_answer"]
            evalprompt = [img, f"Answer with only Yes OR No. Given the image, the question and the ground-truth answer, is the candidate answer correct?\nQuestion: {question}\nGround-Truth Answer:{ground_truth}\nCandidate Answer: {candidate_answer}\n\nThe ground-truth answer is the real and correct full-answer for the visual riddle containing the question and the image - USE it when you decide if the candidate answer address the riddle correctly. Do not forget, answer with only Yes OR No."]
            tokencount += model.count_tokens(prompt).total_tokens
            requests += 1
            if tokencount > 32000 or requests > 2:
                time.sleep(60)
                tokencount = 0
                requests = 0
            response = model.generate_content(evalprompt)
            result = response.text.split()[0]
            if result[:3] == "Yes":
                total += 1
                correct += 1
            elif result[:2] == "No":
                total += 1
        # add suggest prompt and accuracy to list
        results.append({"Prompt": suggested_prompt, "Accuracy": correct/total})
        # cycle to next suggested prompt
    outcome = pd.DataFrame(results)
    outcome.to_csv("./prompt_trials2.csv", index=False)
    print(outcome)
    return

##########################
# functions for few-shot #
##########################

def compose_grid(images, separator_width=10):
    """
    Composes a 2x2 grid from a list of 4 images with a white separator between images,
    ensuring all images are resized to the same dimensions.

    Args:
    images (list): List of 4 PIL Image objects.
    separator_width (int): Width of the separator line between images.
    size (tuple): The desired size (width, height) for each image.

    Returns:
    Image: A PIL Image object with the 2x2 grid.
    """
    if len(images) != 4:
        raise ValueError("The list must contain exactly 4 images.")

    size = images[-1].size
    # Resize all images to the same size
    resized_images = [img.resize(size) for img in images]

    # Get the size of the resized images
    width, height = size

    # Calculate new size considering separators
    total_width = width * 2 + separator_width
    total_height = height * 2 + separator_width

    # Create a blank image with size to fit 2x2 grid with separators
    grid_image = Image.new('RGB', (total_width, total_height), 'white')

    # Paste images into the grid with separators
    grid_image.paste(resized_images[0], (0, 0))
    grid_image.paste(resized_images[1], (width + separator_width, 0))
    grid_image.paste(resized_images[2], (0, height + separator_width))
    grid_image.paste(resized_images[3], (width + separator_width, height + separator_width))

    return grid_image


def get_shots_imgs(current_id, dataset, num_of_shots=3):
    img_list = []
    text = ""
    rand_shots = [current_id]
    while current_id in rand_shots:
        rand_shots = np.random.permutation(len(dataset.index))[:num_of_shots]
    for i, loc in zip(range(1, num_of_shots + 1), ['Top-Left', 'Top-Right', 'Bottom-Left']):
        sample = dataset.iloc[int(rand_shots[i - 1]), :]
        text += f"\n[{i}.{loc} image] \nQuestion: {sample['question']}"
        text += f"\nAnswer: {sample['ground_truth_answer']}\n\n"
        img_list.append(sample['image'])
    return text, img_list


def get_prompt_img_fs(current_id, sample, dataset, num_of_shots=3):
    text = f"This is a visual riddle. Here are {num_of_shots} examples to give you an idea of what is required:\n"
    shots, img_list = get_shots_imgs(current_id, dataset)
    img_list.append(sample['image'])

    fs_img = compose_grid(img_list)
    text += shots
    text += f"\nNow your turn! Take a deep breath, look for clues and think step by step. If you are not sure, give the most likely answer. \n[4.Bottom-Right image]\nQuestion: {sample['question']}\nAnswer:"
    return text, fs_img


def fewshot(modelname, dataset, safety):
    model = genai.GenerativeModel(modelname, safety_settings=safety)
    tokencount = 0
    request = 0
    results = []
    dataset = dataset.iloc[50:100, :]
    for i, sample in tqdm(dataset.iterrows()):
        text, img = get_prompt_img_fs(i, sample, dataset)
        prompt = [img, text]
        tokencount += model.count_tokens(prompt).total_tokens
        request += 1
        if tokencount > 32000 or request > 2:
            time.sleep(60)
            tokencount = 0
            request = 0
        response = model.generate_content(prompt)

        results.append({"idx": i,
                        "prompt": text,
                        "response": response.text,
                        "ground-truth": sample['ground_truth_answer']})

    results = pd.DataFrame(results)
    results.to_csv("./gemini_solutions_fewshot2.csv", index=False)
    return results



if __name__ == '__main__':

    device = "cuda"
    torch.set_default_device(device)

    sysprompt = "This is a visual riddle. Look for clues and think step by step."
    tag = "latest"
    cluefile = "./clues/clues_gemini_latest.json"
    
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)

    safesettings = {"HARM_CATEGORY_HARASSMENT": "block_none",
              "HARM_CATEGORY_DANGEROUS": "block_none",
              "HARM_CATEGORY_HATE_SPEECH": "block_none",
              "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none"}
    
    modelname = "gemini-1.5-pro"
    #modelname = "gemini-1.5-flash-latest"
    
    dataset = load_dataset("visual-riddles/visual-riddles", token=token, trust_remote_code=True)["test"]
    df = pd.DataFrame(dataset)

    # zero-shot for baseline
    solve_from_image(modelname, df, safesettings, sysprompt, tag)

    # Chain-of-thought via clue-extraction
    #clueformat = JSON_clue_format()
    #extract_clues(modelname, df, safesettings, sysprompt, tag, clueformat)
    #solve_from_clues(modelname, df, clues, safesettings, sysprompt, tag)  

