import uuid
import os
import re
import json
import sys
import tqdm
import re
import time
import argparse
import nltk.tokenize
from urllib.parse import urlparse, quote, urlencode
from urllib.request import Request, urlopen
import requests
from ner_retriever import NER_Retriever
from src.retrieval.scraper_for_knowledge_store import scrape_text_from_url
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

parser = argparse.ArgumentParser()
parser.add_argument(
  "-ne",
  "--num_desired_evidence",
  type=int,
  default=5,
  help="Number of QA pairs desired"
)
parser.add_argument(
  "-wsz",
  "--window_size",
  type=int,
  default=5,
  help="Number of sentences to extract around snippet from long document"
)
parser.add_argument(
  "-fq",
  "--first_question",
  type=str,
  default="T5",
  choices=["T5","GPT4"],
  help="Model to use for first question"
)
parser.add_argument(
  "-nq",
  "--next_question",
  type=str,
  default="GPT4",
  choices=["T5", "GPT4"],
  help="Model to use for subsequent questions"
)
parser.add_argument(
  "--paraphrase",
  action="store_true",
  help="Paraphrase questions after verdict computed until --num_desired_evidence pairs are written (retrieving new answers)"
)
parser.add_argument(
  "--no-paraphrase",
  dest="paraphrase",
  action="store_false"
)
parser.set_defaults(paraphrase=True)
parser.add_argument(
  "--long_docs",
  action="store_true",
  help="Retrieve full documents to compute answer"
)
parser.add_argument(
  "--no-long-docs",
  dest="long_docs",
  action="store_false"
)
parser.set_defaults(long_docs=True)
parser.add_argument(
  "--one_doc",
  action="store_true",
  help="LLM answers based on one best document"
)
parser.add_argument(
  "--no-one-doc",
  dest="one_doc",
  action="store_false"
)
parser.set_defaults(one_doc=True)
parser.add_argument(
  "--metadata",
  action="store_true",
  help="Provide LLM with metadata of documents"
)
parser.add_argument(
  "--no-metadata",
  dest="metadata",
  action="store_false"
)
parser.set_defaults(metadata=True)
parser.add_argument(
  "--repeat",
  action="store_true",
  help="Repeat QA pairs after verdict computed until --num_desired_evidence pairs are written"
)
parser.add_argument(
  "--no-repeat",
  dest="repeat",
  action="store_false"
)
parser.set_defaults(repeat=False)
parser.add_argument(
  "--late_verdict",
  action="store_true",
  help="Reconsider the final verdict after all QA pairs including paraphrases"
)
parser.add_argument(
  "--no-late-verdict",
  dest="late_verdict",
  action="store_false"
)
parser.set_defaults(late_verdict=True)
parser.add_argument(
  "--yesno_only",
  action="store_true",
  help="Output only supported or refuted as a verdict"
)
parser.add_argument(
  "--no-yesno-only",
  dest="yesno_only",
  action="store_false"
)
parser.set_defaults(yesno_only=True)
parser.add_argument(
  "--q1_model_name",
  default="/zdata/users/malon/fever2024/genq/q1-model",
  type=str,
  help="sequence-to-sequence model for first question"
)
parser.add_argument(
  "--q2_model_name",
  default="/zdata/users/malon/fever2024/genq/q2-model",
  type=str,
  help="sequence-to-sequence model for followup questions"
)
parser.add_argument(
  "--resume",
  type=str,
  default="/dev/null",
  help="File of partial results computed already"
)
parser.add_argument(
  "--number",
  type=int,
  default=0
)
parser.add_argument(
  "--in_file",
  type=str,
  required=True,
  help="Problem file to process"
)

prog_args = parser.parse_args()
for k in prog_args.__dict__:
  print("ARGUMENT " + k + ": " + str(prog_args.__dict__[k]))

if(prog_args.repeat and prog_args.paraphrase):
  raise ValueError("Can't both repeat and paraphrase")

num_desired_evidence = prog_args.num_desired_evidence
wsz = prog_args.window_size

if prog_args.first_question == "T5":
  q1_tokenizer = AutoTokenizer.from_pretrained(prog_args.q1_model_name)
  q1_model = AutoModelForSeq2SeqLM.from_pretrained(prog_args.q1_model_name)
if prog_args.next_question == "T5":
  q2_tokenizer = AutoTokenizer.from_pretrained(prog_args.q2_model_name)
  q2_model = AutoModelForSeq2SeqLM.from_pretrained(prog_args.q2_model_name)

in_file = prog_args.in_file
num = prog_args.number
done_already = prog_args.resume

retr = NER_Retriever()

in_fp = open(in_file, "r", encoding="utf8")
if(num > 0):
  all = json.loads(in_fp.read())[:num]
else:
  all = json.loads(in_fp.read())
in_fp.close()

already_fp = open(done_already, "r", encoding="utf8")
already = {}
for line in already_fp.readlines():
  if(re.search(r'^\{"', line)):
    example = json.loads(line)
    already[example["claim_id"]] = line
already_fp.close()

def google(m, claim_date):
  md = re.findall(r"\d+", claim_date)
  if(md is None or len(md) != 3):
    raise ValueError("Bad claim date")
  dd = md[0]
  mm = md[1]
  yyyy = md[2]
  enddate = str(yyyy)+str("%02d" % int(mm))+str("%02d" % int(dd))

  params = urlencode({'key': os.getenv('GOOGLE_API_KEY'), 'cx': os.getenv('GOOGLE_CONTEXT'), 'q': m, 'sort': "date:r:19000101:" + enddate})
  url = 'https://www.googleapis.com/customsearch/v1?%s' % params
  
  headers={'Content-Type':'application/json'}

  tries = 0
  answer = {}
  while "items" not in answer and tries < 3:
    r = requests.get(url, headers=headers)
  
    answer = json.loads(r.text)
    if "items" not in answer:
      time.sleep(5)
      tries = tries + 1

  if "items" not in answer:
    print(r.text)
    raise ValueError("Google failure")

  return answer

def camel(m):
  senddata = {"model": "gpt-4o", "seed": 42, "messages": [{"role": "user", "content": m}]}
  jsonsend=json.dumps(senddata)

  result = {}
  tries = 0
  while "choices" not in result and tries < 3:
    headers={'Content-Type':'application/json', 'Authorization': 'Bearer ' + os.getenv('OPENAI_API_KEY')}
    r = requests.post('https://api.openai.com/v1/chat/completions', json=senddata, headers=headers)
 
    try: 
      result = json.loads(r.text)
    except:
      result = {}

    if "choices" in result:
      answer = result["choices"][0]["message"]["content"]
    else:
      print("GPT-4 failure: " + r.text)
      time.sleep(5)
      tries = tries + 1

  if "choices" not in result:
    answer = "GPT-4 failure: " + r.text

  return answer

def get_first_question(claim, speaker, fecha):
  # speaker and fecha are ignored now
  if(prog_args.first_question == "T5"):
    inputs = ["question: " + claim]
    model_inputs = q1_tokenizer.batch_encode_plus(inputs, max_length=128, padding="max_length", return_tensors="pt", truncation=False)
    outputs = q1_model.generate(**model_inputs, num_beams=5, do_sample=False, length_penalty=1.0, early_stopping=False, max_new_tokens=128)
    responses = q1_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses[0]

  elif(prog_args.first_question == "GPT4"):
    if speaker:
      if fecha:
        prompt = "We are trying to verify the following claim by " + speaker + " on " + fecha + ".\n"
      else:
        prompt = "We are trying to verify the following claim by " + speaker + ".\n"
    else:
      if fecha:
        prompt = "We are trying to verify the following claim on " + fecha + ".\n"
      else:
        prompt = "We are trying to verify the following claim.\n"
    prompt = prompt + "Claim: " + claim + "\n"
    prompt = prompt + "We aren't sure yet whether this claim is true or false.  Please write one or more questions that would help us verify this claim, as a JSON list of strings.  Keep the list short.\n"
    hope = camel(prompt)
    try:
      hope = re.sub(r"^\s*", "", hope)
      hope = re.sub(r"\s*$", "", hope)
      hope = re.sub(r"^[^\[]*\[", "[", hope)
      hope = re.sub(r"\][^\]]*$", "]", hope)
      qlist = json.loads(hope)
    except:
      print("GPT-4: Result not JSON:", hope)
      qlist = [hope]
  
    return qlist[0]
  else:
    raise NotImplementedError("get_first question not implemented")

def get_next_question(claim, speaker, fecha, qa_list):
  if(prog_args.next_question == "T5"):
    src = "question: claim: " + claim
    for qa in qa_list:
      src = src + " question: " + qa["question"] + " answer: " + qa["answer"]
    inputs = [src]
    model_inputs = q2_tokenizer.batch_encode_plus(inputs, max_length=128, padding="max_length", return_tensors="pt", truncation=False)
    outputs = q2_model.generate(**model_inputs, num_beams=5, do_sample=False, length_penalty=1.0, early_stopping=False, max_new_tokens=128)
    responses = q2_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses[0]

  elif(prog_args.next_question == "GPT4"):
    if speaker:
      if fecha:
        prompt = "We are trying to verify the following claim by " + speaker + " on " + fecha + ".\n"
      else:
        prompt = "We are trying to verify the following claim by " + speaker + ".\n"
    else:
      if fecha:
        prompt = "We are trying to verify the following claim on " + fecha + ".\n"
      else:
        prompt = "We are trying to verify the following claim.\n"
    prompt = prompt + "Claim: " + claim + "\n"
    prompt = prompt + "So far we have asked the questions:\n"
    for i, qa in enumerate(qa_list):
      prompt = prompt + "Question " + str(i) + ": " + qa["question"] + " Answer: " + qa["answer"] + "\n"
    prompt = prompt + "Based on these questions and answers, can you verify whether the claim is true or false?  Please answer [[True]] or [[False]], or ask one more question that would help you verify.\n"
    hope = camel(prompt)
    m = re.search(r"\\?\[ *\\?\[\s*(\w+)\s*\\?\] *\\?\]", hope)
    if m:
      if m[1].lower() == "true":
        return m[1].lower()
      elif m[1].lower() == "false":
        return m[1].lower()
  
    next_question = just_question(hope)
    return next_question

  else:
    return NotImplementedError("get_next_question not implemented")

def paraphrase(q):
  prompt = "Please give four ways to rephrase the following question.\n"
  prompt = prompt + "Give your answer as a JSON list of strings, each string being one question.\n"
  prompt = prompt + "Question: " + q
  hope = camel(prompt)
  try:
    hope = re.sub(r"^\s*", "", hope)
    hope = re.sub(r"\s*$", "", hope)
    hope = re.sub(r"^[^\[]*\[", "[", hope)
    hope = re.sub(r"\][^\]]*$", "]", hope)
    qlist = json.loads(hope)
  except:
    print("GPT-4: Paraphrase result not JSON:", hope)
    qlist = [q for i in range(4)]

  return qlist

def just_question(r):
  sents = nltk.tokenize.sent_tokenize(r)
  for sent in sents:
    if "?" in sent:
      sent = re.sub(r"^.*:", "", sent, flags=re.MULTILINE) # Prefixes considered part of question
      return sent
  return r


def ask(q, chunks, metadata):
  prompt = "We searched the web and found the following information.\n"
  for i, chunk in enumerate(chunks):
    prompt = prompt + "Document " + str(i) + metadata[i] +  ": " + chunk + "\n"

  if(prog_args.yesno_only == False and prog_args.late_verdict == False):
    prompt = prompt + "Based on the above information, please answer the following question, referring to document numbers.  If there is conflicting or insufficient information, write [[Conflicting]] or [[Insufficient]] and explain.\n"
  elif prog_args.one_doc:
    prompt = prompt + "Based on the above information, please answer the following question, referring to the one document that best answers the question.\n"
  else:
    prompt = prompt + "Based on the above information, please answer the following question.\n"

  prompt = prompt + q + "\n"
  answer = camel(prompt)
  print(prompt)
  return answer

def ask_one_long(q, snippet, metadata):
  prompt = "We searched the web and found the following information.\n"
  prompt = prompt + "Document" + metadata +  ": " + snippet + "\n"
  prompt = prompt + "Based on the above information, please answer the following question.\n"
  prompt = prompt + q + "\n"
  answer = camel(prompt)
  print(prompt)
  return answer

def verify(claim, evidence):
  prompt = "We are trying to verify the following claim: " + claim + "\n"
  prompt = prompt + "Based on our web searches, we resolved the following questions.\n"
  for i, qa in enumerate(evidence):
    prompt = prompt + str(i+1) + ". " + qa["question"] + " " + qa["answer"] + "\n"
  if(prog_args.yesno_only == False):
    prompt = prompt + "Is the claim (A) fully supported by the evidence, (B) contradicted by the evidence, (C) insufficient information, or (D) conflicting evidence?\n"
    prompt = prompt + "Please respond in the format [[A]], [[B]], [[C]], or [[D]]."
  else:
    prompt = prompt + "Is the claim (A) fully supported by the evidence, or (B) contradicted by the evidence?\n"
    prompt = prompt + "Please respond in the format [[A]] or [[B]]."

  answer = camel(prompt)

  if(re.search(r"\[ *\[A\] *\]", answer)):
    return "Supported"
  elif(re.search(r"\[ *\[B\] *\]", answer)):
    return "Refuted"
  elif(re.search(r"\[ *\[C\] *\]", answer)):
    return "Not Enough Evidence"
  elif(re.search(r"\[ *\[D\] *\]", answer)):
    return "Conflicting Evidence/Cherrypicking"

  return "GPT-4 verify failure: " + answer


def get_best_doc(e):
  answer = re.sub(r"\n", " ", e, flags=re.MULTILINE)
  for m in re.finditer(r"Document\s+(\d+)", answer):
    return int(m[1])
  m = re.search(r"Documents([ 0-9,]+)and ([0-9]+)", answer)
  if m:
    return int(m[2])
  return None


def long_snippet(short_snippet, url, window):
  tmpname = str(uuid.uuid1())

  if(len(url) > 1000):
    print("URL longer than 1000:  " + url)
    return [short_snippet, ""]

  start_time = time.time()
  sentences = scrape_text_from_url(url, tmpname)
  scrape_time = time.time() - start_time

  if(os.path.exists(tmpname)):
    os.unlink(tmpname)

  if(len(sentences) < 1):
    return [short_snippet, ""]

  if(len(sentences) <= window):
    return [" ".join(sentences), sentences]

  short_words = short_snippet.lower().split()

  overlap = []
  for i in tqdm.tqdm(range(len(sentences)-window), desc="Sentences"):
    s = 0
    long = " ".join(sentences[i:i+window])
    swords = long.lower().split()
    for word in short_words:
      if word in long:
        s = s + 1
    if(s > .7 * len(short_words)):
      overlap.append(i)

  total_time = time.time() - start_time
  print("Time", url, scrape_time, total_time)

  if(len(overlap) == 0):
    return [short_snippet, sentences]

  r = int(len(overlap)/2)
  i = overlap[r]
  return [" ".join(sentences[i:i+window]), sentences]


def get_answer(q, example):
  s = example["claim"] + " " + q  # good for decontextualization or not?
  s = re.sub(r"\n", " ", s, flags=re.MULTILINE)
  s = re.sub(r"'s\s", " ", s)
  s = re.sub(r"[^-$.0-9a-zA-Z ]", "", s)

  try:
    struct = google(s, example["claim_date"])
  except:
    print("Google failed at first", s)
    try:
      s = " ".join(retr.lookups(s))
      struct = google(s, example["claim_date"])
    except:
      print("Google failed both times", s)
      return None

  chunks = []
  metadata = []
  for j, hit in enumerate(struct["items"]):
    m = ""
    if prog_args.metadata:
      try:
        info = []
        tags = hit["pagemap"]["metatags"][0]
        if "og:title" in tags:
          title = tags["og:title"]
          info.append(title)
        if "og:site_name" in tags:
          site = tags["og:site_name"]
          info.append("from " + site)
        if "article:published_time" in tags:
          day = re.sub(r"T.*", "", tags["article:published_time"])
          info.append("published " + day)
        if len(info) > 0:
          m = " (" + ", ".join(info) + ")"
      except:
        m = ""

    if "snippet" not in hit:
      # print("Google: Snippet not in hit", json.dumps(hit))
      continue

    doc = re.sub(r"^.*?\.\.\.", "", hit["snippet"])
    chunk = re.sub(r"\.\.\.\s*$", "", doc)
    chunks.append(chunk)
    metadata.append(m)

  answer = ask(q, chunks, metadata)

  finding = {"question": q, "answer": answer}
  finding["google"] = chunks
  finding["google_search"] = s

  # Provenance in case there is no best document and we have to use Google
  md = re.findall(r"\d+", example["claim_date"])
  if(md is None or len(md) != 3):
    raise ValueError("Bad claim date")
  dd = md[0]
  mm = md[1]
  yyyy = md[2]
  enddate = str(yyyy)+str("%02d" % int(mm))+str("%02d" % int(dd))
  nonapi_params = urlencode({'q': s, 'sort': "date:r:19000101:" + enddate})
  finding["url"] = 'https://www.google.com/search?%s' % nonapi_params
  finding["scraped_text"] = " ".join(chunks)

  if prog_args.one_doc:
    best_doc_idx = get_best_doc(answer)
    if best_doc_idx is not None:
      best_snippet = chunks[best_doc_idx]
      best_meta = metadata[best_doc_idx]
      best_url = struct["items"][best_doc_idx]["link"]

      if prog_args.long_docs:
        best_info = long_snippet(best_snippet, best_url, wsz)
      else:
        best_info = [best_snippet, ""]
      best_long = best_info[0]
      finding["initial_answer"] = answer
  
      answer = ask_one_long(q, best_long, best_meta)
  
      finding["answer"] = answer
      finding["short_snippet"] = best_snippet
      finding["long_snippet"] = best_long
      finding["scraped_text"] = best_long
      finding["long_doc"] = best_info[1]
      finding["url"] = best_url
      finding["meta"] = best_meta

  return finding


nqueries = 0
for i, example in tqdm.tqdm(enumerate(all)):
  if i in already:
    print(already[i])
    continue

  if "claim_id" in example and int(example["claim_id"]) != i:
    raise ValueError("Misnumbered claims")

  result = {"claim_id": i, "claim": example["claim"]}
  if "label" in example:
    result["label"] = example["label"]
  result["questions"] = []

  last_q = ""
  q = get_first_question(example["claim"], example["speaker"], example["claim_date"])

  while(q != "true" and q != "false" and len(result["questions"]) < prog_args.num_desired_evidence and q != last_q):
    last_q = q

    finding = get_answer(q, example)
    nqueries = nqueries + 1
    if finding is not None:
      result["questions"].append(finding)
      answer = finding["answer"]

      if(prog_args.yesno_only == False):
        m = re.search(r"\[\[\s*(\w+)\s*\]\]", answer)
        if m:
          if m[1].lower() == "conflicting":
            result["pred_label"] = "Conflicting Evidence/Cherrypicking"
            break
          if m[1].lower() == "insufficient":
            result["pred_label"] = "Not Enough Evidence"
            break

    q = get_next_question(example["claim"], example["speaker"], example["claim_date"], result["questions"])

  if q == "true":
    result["pred_label"] = "Supported"
    result["pred_method"] = "Next question"
  elif q == "false":
    result["pred_label"] = "Refuted"
    result["pred_method"] = "Next question"
  elif "pred_label" not in result:
    result["pred_label"] = verify(example["claim"], result["questions"])
    result["pred_method"] = "Verify"

  # Expand to five questions and answers
  if(prog_args.paraphrase):
    paraphrases = []
    for e in result["questions"]:
      paraphrases.append(paraphrase(e["question"]))
    nq = len(result["questions"])
    if nq > 0:
      for i in range(num_desired_evidence-nq):
        j = i % nq
        k = i // nq
        npar = len(paraphrases[j])
        q = paraphrases[j][k % npar]
        print("Paraphrased question", j, k, q)
        finding = get_answer(q, example)
        nqueries = nqueries + 1
    
        if finding is None:
          continue
        result["questions"].append(finding)

  elif(prog_args.repeat):
    nq = len(result["questions"])
    if nq > 0:
      for i in range(nq, prog_args.num_desired_evidence):
        j = (i - nq) % nq
        result["questions"].append(result["questions"][j])

  if(prog_args.late_verdict):
    if "pred_label" in result:
      result["old_pred_label"] = result["pred_label"]
    if "pred_method" in result:
      result["old_pred_method"] = result["pred_method"]

    result["pred_label"] = verify(example["claim"], result["questions"])
    result["pred_method"] = "Verify"

  print(json.dumps(result))

print("Total search queries", nqueries)

