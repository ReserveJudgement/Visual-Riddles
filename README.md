# Visual-Riddles
This is a project experimenting with language-vision models (LVM) on a challenging visual question answering (VQA) task.
The dataset and baselines can be found at https://huggingface.co/datasets/visual-riddles/visual_riddles
It consists of images and a question with a non-trivial catch. For example:


Which chair should you sit on?


![image](https://github.com/user-attachments/assets/e78c36dd-07ed-411f-ad2e-f5562a9061aa)


The answer requires noticing that a leg is missing from one chair, and real-world knowledge to infer that that would be uncomfortable.
Even SoTA commercial LVMs like GPT4 and Gemini find this challenging.
The experiments in this repository stick to the open-ended question answering task.
Automatic evaluation is used with Gemini Pro 1.5 (which has been found to agree with human evaluators 85% if the time).

# Experiments
To gain initial improvement over baselines the following methods are tried:
- **Prompt-optimization**: try simple prompt of the form "This is a visual riddle, look for clues, use scientific or cultural knowledge, and think step by step". Also try a much more detailed version.
- **Few-shot**: since most LVMs only take one image, four riddles are combined into one image by dividing it into quarters. Three of the images are referenced in the prompt along with their ground-truth answers for in-context learning. The fourth image referenced as the riddle to be solved.
- **Chain-of-thought**: a two-stage method where the model is first asked to generate a set of clues from the image with their potential relevance to the riddle. At the second stage the model solves the riddle using the generated clues. It might also be the case that clues are missed due to regional attention. To test for this, another version of the method is tried, where clues are asked for seperately in each of four quadrants of the image, and then compiled into one list.
- **Retrieval-augmentation**: since many of the riddles require external real-world knowledge (e.g. cultural or scientific background information) which may not be encoded in the model itself, we try giving the model access to a browser to search for relevant information. The model is prompted for a search query. It is fed to the you.com API to get relevant snippets from web-sites. The snippets are summarized and fed back as part of the prompt with the riddle. 

A combination of few-shot and chain-of-thoughts is also tried.

# Failure analysis
Our aim is not only to test effect on overall performance, but to see how the above workflows affect reasoning. The following reasoning failure-modes are posited:
- **Missed Visual Clues**: Visual riddles often contain subtle hints that models may fail to detect
within the image. For instance, recognizing that a
chair is missing a leg or identifying a famous landmark in the background can be crucial for solving
the riddle.
- **Lack of World Knowledge**: Some visual riddles
demand specific scientific or cultural knowledge.
For example, identifying a country by its flag or recognizing a particular species of snake based on its
colors and patterns requires domain-specific understanding. Models might either lack this knowledge
or fail to recall it in the appropriate context.
- **Failed Reasoning**: The solution to a riddle may
involve combining different sub-conclusions. For instance,
deducing that the scene is in Spain, observing that
it is early afternoon, combining this with the cultural knowledge of a siesta, and inferring that a shop is likely closed.

# Evaluation
To automatically evaluate the cause of failure: the riddle, the model-generated answer, and the ground-truth answer are provided to the GeminiFlash-1.5-latest model, along with a concise
prompt instructing it to classify the reason for the incorrect answer from among the three options above.
To validate the model’s ability to classify cause
of failure, it was compared against a manual human
classification on a subset of 50 incorrect answers.
The model matched the manual classifications 78%
of the time (39/50).
Note: there could be multiple causes of failure for a single wrong answer.

# Hypothesis
- a) A chain-of-thought process that begins with a clue-detection phase should reduce failures
stemming from missed visual clues.
- b) Retrieval augmentation that incorporates information from
the web should mitigate failures caused by a lack of
world knowledge.
- c) Providing task demonstrations will help address failures related to complex
reasoning.

# Results
Two LVM models are tried as the base for the visual riddle solver agent: Gemini-flash and LlavaNext 32B.
These are overall accuracies over dataset of 400 clues:
| Experiment | Gemini% | Lava% |
|---|---|---|
| Vanilla baseline              | 36.25 | 31.59 |
| Short optimized prompt        | 58.25 | 35.75 |
| Detailed optimized prompt     | 46.00 |   -   |
| Web search augmented          | 47.75 |   -   |
| Chain of clues                | 50.00 | 19.08 |
| Chain of clues from quadrants | 47.50 |   -   |
| Few-shot                      | 42.00 | 28.18 |
| Few-shot + Chain of thoughts  | 46.88 |   -   |

For the failure anaylsis for the Gemini mode:

![image](https://github.com/user-attachments/assets/69bf6bb9-fc2b-4cf0-ad91-5b6edfb1d063)

# Discussion

All methods significantly outperform
the vanilla baseline. Surprisingly, the best results came from
a simple concise system prompt.
LlavaNext best performance gain was with the web
search, implying that “missing knowledge” is the
main source of failure in the model. Curiously, chain of clues and few-shot settings decreased its results
to below baseline, perhaps indicating a sensitivity
to too much information.


Regarding the Gemini failure modes,
clue extraction from the image reduced failures
due to missing clues or knowledge but increased
reasoning errors, compared to the baseline. Extracting clues from quadrants separately did not improve
compared to a single pass over the whole image, so there does not seem to be a problem with attention to image regions.
The web search experiment had the lowest number
and proportion of mistakes due to missing knowledge, but it had lower overall accuracy compared
to chain-of-thoughts or prompt engineering. Fewshot did not reduce reasoning errors, although it
reduced other errors. 


Hypotheses (a) and (b) seem to be
confirmed: the approaches reduced their target error categories. But this was achieved at the expense of other error categories. This is likely due to
multiple failure reasons per riddle, where improving one aspect is insufficient. Additionally, combining techniques, such as chain-of-thoughts with
few-shot, performed worse than chain-of-thoughts
alone, such that different methods might interfere
with each other’s effects. 
Fo
