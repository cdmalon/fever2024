# Team Papelo's system for FEVER 2024 (AVeriTeC shared task)

This system for fact verification of claims on the open web is documented
in "Multihop Evidence Pursuit Meets the Web: Team Papelo at FEVER 2024",
to be presented at the FEVER workshop at EMNLP 2024.  Given a claim, the
system decides whether it is supported or refuted and provides a series of
questions and answers citing evidence from the open web to justify its decision.

It is a solution for the [AVeriTeC shared task](https://arxiv.org/abs/2410.23850).  Please see the [FEVER website](https://fever.ai) for more information
about the task.  This system scored highest among systems that retrieved
evidence from the open web (instead of using a provided knowledge database).

This system is intended for academic research, and has been designed
for the peculiarities of the AVeriTeC dataset rather than real world use.
See, in particular, the list of limitations below.

## Installation

First, download the [AVeriTeC baseline system](https://github.com/chenxwh/AVeriTeC) and prepare a Conda environment for it using its `requirements.txt` file.  This is needed for the scraper they have implemented and for the official scoring scripts.  Below, we assume your checkout is in `../AVeriTeC`.

Task data can be downloaded as instructed on the [AVeriTeC task page](https://fever.ai/task.html).

In your checkout directory, `mkdir pdf_dir`.

The current implementation uses GPT-4o from OpenAI and
a [Google Programmable Search Engine](https://developers.google.com/custom-search/docs/tutorial/creatingcse).  After creating your custom search engine,
from your list of search engines, select the search engine you want to edit.
Under Overview, scroll down to Search features. Beside Augment results, Toggle On or Off to enable or disable Search the entire web.

Once your OpenAI account and 
[Google Programmable Search Engine](https://developers.google.com/custom-search/docs/tutorial/creatingcse) are set up,
you need to set the environment variables `OPENAI_API_KEY`,
`GOOGLE_API_KEY`, and `GOOGLE_CONTEXT`.
If you don't want to use these services,
you may change the LLM function `camel()` and search function `google()`
to use other services or local models.

## Usage

To run the system, execute

```
PYTHONPATH=../AVeriTeC python papelo.py --in_file /AVeriTeC/data/dev.json | tee output.txt
```
To reproduce our ablation studies, add the argument `--number 200`, in
addition to changing any of the other arguments from their defaults.
Run `PYTHONPATH=../AVeriTeC python papelo.py --help` for a complete list
of options.  There is a separate script `allatonce.py` for the the
"all at once" ablation, which supports similar options to `papelo.py`.

The output file contains lots of potentially helpful information to help
you debug how the system worked on a particular claim.  To prepare
a submission file for evaluation, extract the JSONL part of the output
and prepare a JSON file as follows:

```
egrep '^\{"' output.txt | python jsonl-questions-to-evidence.py /dev/stdin >output.json
```

Then you can run the official evaluation script like:
```
PYTHONPATH=../AVeriTeC/ python ../AVeriTeC/src/prediction/evaluate_veracity.py -i output.json --label_file /AVeriTeC/data/dev.json
```
If you have used the `--number` argument, you will also need to truncate
the corresponding examples in the label file before running the evaluation,
as in:
```
python firstn.py /AVeriTeC/data/dev.json 200 >dev200.json
```

## Limitations

This system never outputs "not enough evidence" or "conflicting evidence"
as judgments because it was never able to predict these classes with
acceptable accuracy on the AVeriTeC dataset.  When "not enough evidence"
(NEI) is an option, an LLM tends to select it too often.
Given these judgment restrictions, and the fact that the overall label
accuracy is only .754, humans should be cautious in trusting this system's
output to verify a claim without reading the rationale.

LLMs have insufficient information to judge the overall credibility of
a website, and currently just the site name is given for the LLM's
consideration.  Metadata including the site name helps (to give an example
from the dev set, GPT-4o was aware or discovered through its searches that
Scoopertino was a satirical website), but generally, misinformation that is
corroborated elsewhere on the web may fool our fact checking system.

Although the LLM is always prompted to answer questions "based on
the above information" quoted from retrieved documents or its previous
answers, there is no guarantee that the LLM does not apply other,
untraceable knowledge in forming its answers.  We use a date filter
to ensure that all web searches return documents only from before each claim
date, but we use an LLM whose training cutoff is after the claim dates.

Novel information first reported, which has no basis in existing documents,
can never be fact-checked with the techniques of this system (for example,
the first report that a presidential candidate was shot).  That kind
of fact checking requires judgments of plausibility, credibility, and
consistency that are out of scope for this system.

## Copyright

Copyright (C) 2024 [NEC Laboratories America, Inc.](https://www.nec-labs.com)

