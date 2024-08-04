# Visual-Riddles
This is a project experimenting with language-vision models (LVM) on a challenging visual question answering (VQA) task.
The dataset and baselines can be found at https://huggingface.co/datasets/visual-riddles/visual_riddles
It consists of images and a question with a non-trivial catch. For example:
![image](https://github.com/user-attachments/assets/e78c36dd-07ed-411f-ad2e-f5562a9061aa)
Which chair should you sit on?
The answer requires noticing that a leg is missing from one chair, and real-world knowledge to infer that that would be uncomfortable.
Even SoTA commercial LVMs like GPT4 and Gemini find this challenging.
The experiments in this repository stick to the open-ended question answering task.
Automatic evaluation is used with Gemini Pro 1.5 (which has been found to agree with human evaluators 85% if the time).

# Experiments
To gain initial improvement over baselines the following methods are tried:
- **Few-shot**: since most LVMs only take one image, four riddles are combined into one image by dividing it into quarters. Three of the images are referenced in the prompt along with their ground-truth answers for in-context learning. The fourth image referenced as the riddle to be solved.
- **Chain-of-thought**: a two-stage method where the model is first asked to generate a set of clues from the image with their potential relevance to the riddle. At the second stage the model solves the riddle using the generated clues.
- **Tool-use**: since many of the riddles require external real-world knowledge (e.g. cultural or scientific background information) which may not be encoded in the model itself, we try giving the model access to a browser to search for relevant information.

# Analysis
Examining the questions on which the model is wrong and asking Gemini to select the reason for failure from a set of options, gives illuminating results:
| Reason for Failure | Percentage of Wrong Answers |
|---|---|
| Salient object not detected in image | 24% |
| Missing external information | 26% |
| Problem in reasoning process | 53% |

